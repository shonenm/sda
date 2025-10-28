r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import broadcast

from .nn import *


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    時刻埋め込み：連続的な時刻tを高次元ベクトルに変換
    正弦波・余弦波の周波数エンコーディングを使用（Transformerと同様）
    拡散モデルなどで時間ステップをネットワークに条件付けする際に使用
    (拡散過程の時刻のこと)

    Arguments:
        features: 埋め込み特徴の次元数
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(32, 256),      # 32次元 -> 256次元
            nn.SiLU(),               # Swish活性化関数
            nn.Linear(256, features), # 256次元 -> 出力次元
        )

        # 16種類の周波数を用意（正弦波エンコーディング用）
        self.register_buffer('freqs', torch.pi * torch.arange(1, 16 + 1))

    def forward(self, t: Tensor) -> Tensor:
        # 各周波数で正弦波・余弦波を計算
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)  # 32次元 (16*2)

        # MLPで特徴次元に変換
        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a score network.

    スコアネットワーク：拡散モデルのノイズ予測器
    時刻tとオプションで文脈cに条件付けられたスコア関数を学習
    低次元データ（ベクトル）向けのMLP実装
    スコア関数 sϕ(x(t),t|c) を近似するネット

    Arguments:
        features: 入力特徴の次元数
        context: 文脈特徴の次元数（条件付き生成用）
        embedding: 時刻埋め込みの次元数
    """

    def __init__(self, features: int, context: int = 0, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        # 入力 = データ + 時刻埋め込み + 文脈
        self.network = ResMLP(features + context + embedding, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        r"""
        x: ノイズが混ざった状態 x(t)（shape: [B, F]）
        t: 拡散時間（0〜1）。TimeEmbeddingで [B, embedding] に変換。
        c: 任意の文脈（条件）。例：観測特徴、物理パラメータ等（shape: [B, C]）
        """
        
        # 時刻を埋め込み表現に変換
        t = self.embedding(t)

        # バッチブロードキャストして連結
        if c is None:
            x, t = broadcast(x, t, ignore=1)  # 最後の次元を除いてブロードキャスト
            x = torch.cat((x, t), dim=-1)
        else:
            x, t, c = broadcast(x, t, c, ignore=1)
            x = torch.cat((x, t, c), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    U-Netベースのスコアネットワーク：画像などの高次元データ向け
    空間構造を持つデータに対してスコア関数を学習

    Arguments:
        channels: 入力チャネル数
        context: 文脈チャネル数
        embedding: 時刻埋め込みの次元数
    """

    def __init__(self, channels: int, context: int = 0, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        # 入力チャネル = データチャネル + 文脈チャネル
        self.network = UNet(channels + context, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        dims = self.network.spatial + 1  # チャネル + 空間次元

        # 文脈がある場合はチャネル方向に連結
        if c is None:
            y = x
        else:
            y = torch.cat(broadcast(x, c, ignore=dims), dim=-dims)

        # バッチと空間次元を整形
        y = y.reshape(-1, *y.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain.

    マルコフ連鎖用のスコアネットワークラッパー
    ScoreUNetを時系列データ(B, L, C, H, W)に適用できるように変換
    時間軸をチャネル軸として扱う
    """

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W) バッチ、時系列長、チャネル、高さ、幅
        t: Tensor,  # () スカラー時刻
        c: Tensor = None,
    ) -> Tensor:
        # 時系列軸とチャネル軸を交換: (B, L, C, H, W) -> (B, C, L, H, W)
        # これによりLをチャネルとして扱える
        
        # PyTorch の畳み込み層（例：nn.Conv2d）は、常に次のようなテンソル形を前提としている：
        # 入力テンソル: (B, C, H, W)
            # B: バッチ数
            # C: チャネル数（特徴の種類）
            # H, W: 空間サイズ
        # Markov Chain の時系列データは (B, L, C, H, W)
        # どのみち二つ目の次元が L × C として扱われるためそこに意味はないが、重みをかける組み合わせが変わるらしい
        # L と C を入れ替えることで L をチャネル数として扱えるようにする。
        
        
        return self.score(x.transpose(1, 2), t, c).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    マルコフ連鎖用のスコアネットワーク：時系列データに対するスコア関数
    高次マルコフ性を考慮し、過去のorder個の状態を入力に含める (擬似マルコフブランケット)
    時間的な依存関係をモデル化

    Arguments:
        features: 特徴の次元数
        context: 文脈特徴の次元数
        order: マルコフ連鎖の次数（何時刻前まで考慮するか）
    """

    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        # 空間次元があればUNet、なければMLPを使用
        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        # 入力は過去order個 + 現在 + 未来order個の状態を含む
        # 擬似マルコフブランケット
        # 2k + 1
        self.kernel = build(features * (2 * order + 1), context, **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W) 時系列データ
        t: Tensor,  # () スカラー時刻
        c: Tensor = None,  # (C', H, W) 文脈
    ) -> Tensor:
        # 時間窓を展開：各時刻について前後order個の状態を連結
        x = self.unfold(x, self.order)
        # スコア関数を適用
        s = self.kernel(x, t, c)
        # 元の時系列形状に戻す
        s = self.fold(s, self.order)

        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        # 時間軸に沿って2*order+1個の窓をスライド
        x = x.unfold(1, 2 * order + 1, 1)  # (B, L-2*order, C, H, W, 2*order+1)
        x = x.movedim(-1, 2)                # (B, L-2*order, 2*order+1, C, H, W)
        x = x.flatten(2, 3)                 # (B, L-2*order, (2*order+1)*C, H, W)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        # 展開した形状から元の時系列長に戻す
        x = x.unflatten(2, (2 * order  + 1, -1))

        # 先頭order個、中央部分、末尾order個を連結
        return torch.cat((
            x[:, 0, :order],      # 最初の窓から先頭order個
            x[:, :, order],       # 各窓の中央（order番目）
            x[:, -1, -order:],    # 最後の窓から末尾order個
        ), dim=1)


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    分散保存型確率微分方程式（VP-SDE）のノイズスケジューラ
    拡散モデルの前向き過程と逆向き過程を定義
    時刻t∈[0,1]でデータを徐々にノイズに変換し、サンプリング時は逆向きに復元

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: ノイズ推定器 :math:`\epsilon_\phi(x, t)` （ニューラルネットワーク）
        shape: データの形状（バッチサイズを除く）
        alpha: ノイズスケジュール :math:`\alpha(t)` の選択（'lin', 'cos', 'exp'）
        eta: 数値安定性のための小さな定数
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps  # ノイズ予測ネットワーク
        self.shape = shape  # データ形状
        self.dims = tuple(range(-len(shape), 0))  # イベント次元
        self.eta = eta  # 数値安定性項

        # ノイズスケジュール関数の選択
        if alpha == 'lin':
            # 線形スケジュール
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            # コサインスケジュール（より滑らかな減衰）
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            # 指数スケジュール
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        # 時刻tでの平均係数（信号の減衰）
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        # 時刻tでのノイズの標準偏差
        # 分散保存：信号^2 + ノイズ^2 ≈ 1 を維持
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`.

        前向き拡散過程：クリーンなデータxにノイズを加える
        x(t) = μ(t) * x + σ(t) * ε, where ε ~ N(0, I)
        """

        t = t.reshape(t.shape + (1,) * len(self.shape))

        # ガウシアンノイズを生成
        eps = torch.randn_like(x)
        # ノイズ付きデータを生成
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps  # 学習時はノイズも返す（ターゲットとして使用）
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        逆向きサンプリング過程：ノイズからデータを生成
        Predictor-Correctorアプローチを使用

        Arguments:
            shape: バッチ形状
            c: オプションの文脈（条件付き生成）
            steps: 離散時間ステップ数
            corrections: 各時間ステップでのLangevin補正回数
            tau: Langevinステップの振幅
        """

        # ランダムノイズから開始（t=1）
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        # 時刻1から0へ逆向きに進む
        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                # Predictor: 決定論的な逆向きステップ
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector: Langevin MCMCによる補正（オプション）
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    # ステップサイズを適応的に調整
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    # Langevinダイナミクスで補正
                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss.

        ノイズ除去損失：予測ノイズと真のノイズの二乗誤差
        拡散モデルの学習目的関数

        Arguments:
            x: クリーンなデータ
            c: 文脈（条件付き生成）
            w: サンプルごとの重み（オプション）
        """

        # ランダムな時刻tを選択
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        # ノイズを加える
        x, eps = self.forward(x, t, train=True)

        # 予測ノイズと真のノイズの誤差
        err = (self.eps(x, t, c) - eps).square()

        # 重み付き平均損失
        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()


class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    亜分散保存型SDE：VPSDEの変種
    ノイズの増加をより緩やかにしたバージョン

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        # 平方根を取らない形（より緩やかなノイズ増加）
        return 1 - self.alpha(t) ** 2 + self.eta


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    さらに緩やかなノイズスケジュール

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        # より緩やかなノイズスケジュール
        return 1 - self.alpha(t) + self.eta


class DPSGaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    ガウシアン逆問題のためのスコアモジュール（DPS法）
    観測yから未知のxを推定する問題（例：デノイジング、超解像）
    DPS（Diffusion Posterior Sampling）
    
    -sigma(t) · s(x(t), t | y)（＝“負のσ倍の事後スコア”）を返す形
    
    Chung+ (2022) の DPS の考え方で、観測とのズレの勾配をノルムで正規化してガイドする

    .. math:: p(y | x) = N(y | A(x), Σ)

    References:
        | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
        | https://arxiv.org/abs/2209.14687

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,                            # 観測データ
        A: Callable[[Tensor], Tensor],        # 順問題の演算子（例：ダウンサンプリング）
        sde: VPSDE,                           # ベースとなるSDE
        zeta: float = 1.0,                    # ガイダンスの強さ
    ):
        super().__init__()

        self.register_buffer('y', y)

        self.A = A
        self.sde = sde
        self.zeta = zeta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        # 観測条件に基づくスコア補正を計算
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            # ノイズを除去してクリーンな推定を得る
            eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu
            # 観測との誤差
            err = (self.y - self.A(x_)).square().sum()

        # 誤差に対する勾配（データ忠実度項）
        s, = torch.autograd.grad(err, x)
        s = -s * self.zeta / err.sqrt()  # 正規化されたガイダンス

        # 事前分布のスコアとデータ忠実度を組み合わせ
        # eps は「事前側」から来る方向（拡散の denoise）。
        # -σ·s は「観測（尤度）側」の誘導方向。
        # 2つを合成すると 事後の方向ベクトルに近づく（DPSの本質）。
        # 事後スコア = 事前スコア + 尤度スコア
        return eps - sigma * s


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    ガウシアン逆問題のためのスコアモジュール（より一般的な定式化）
    観測ノイズと拡散ノイズの両方を考慮
    DPS（Diffusion Posterior Sampling）
    -sigma(t) · s(x(t), t | y)（＝“負のσ倍の事後スコア”）を返す形
    
    観測ノイズと拡散ノイズの両方を考慮し、log-likelihood の勾配を明示的に使う“ベイズ的”な版

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,                            # 観測データ
        A: Callable[[Tensor], Tensor],        # 順問題の演算子
        std: Union[float, Tensor],            # 観測ノイズの標準偏差
        sde: VPSDE,                           # ベースとなるSDE
        gamma: Union[float, Tensor] = 1e-2,   # ノイズ分散の調整係数
        detach: bool = False,                 # 勾配の切断フラグ
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        # detachオプション：スコアネットワークの勾配を遮断
        if self.detach:
            eps = self.sde.eps(x, t, c)

        # 対数尤度の勾配を計算
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            # ノイズを除去
            x_ = (x - sigma * eps) / mu

            # 観測誤差と分散（時刻に依存）
            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            # 対数尤度
            log_p = -(err ** 2 / var).sum() / 2

        # 対数尤度の勾配
        s, = torch.autograd.grad(log_p, x)

        # 事前分布とデータ尤度を組み合わせた修正スコア
        # eps は「事前側」から来る方向（拡散の denoise）。
        # -σ·s は「観測（尤度）側」の誘導方向。
        # 2つを合成すると 事後の方向ベクトルに近づく（DPSの本質）。
        # 事後スコア = 事前スコア + 尤度スコア
        return eps - sigma * s
