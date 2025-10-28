r"""Lorenz experiment helpers

Lorenz 63システム実験のためのユーティリティ関数群
- グローバル/ローカルスコアネットワークの構築
- データ同化（4D-Var、粒子フィルタ）のための尤度・事後分布計算
- 3次元カオス力学系の時系列モデリング
"""

import os

from pathlib import Path
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *


# データとモデルの保存先パスの設定
# 環境変数SCRATCHが存在する場合はそちらを優先（HPC環境対応）
if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/lorenz'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_chain() -> MarkovChain:
    """ノイズ付きLorenz 63システムのマルコフ連鎖を作成

    Returns:
        NoisyLorenz63: 時間刻み0.025のLorenzシステム
                      3次元状態空間 (x, y, z)、カオス的挙動を示す
    """
    return NoisyLorenz63(dt=0.025)


def make_global_score(
    embedding: int = 32,
    hidden_channels: Sequence[int] = (64,),
    hidden_blocks: Sequence[int] = (3,),
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    """グローバルスコアネットワークを構築

    時系列全体を一度に処理するU-Netベースのスコアネットワーク
    長期依存関係を捉えるが、計算コストが高い

    Args:
        embedding: 時刻埋め込みの次元数
        hidden_channels: U-Netの各深さでのチャネル数
        hidden_blocks: 各深さでの残差ブロック数
        activation: 活性化関数名
        **absorb: 未使用の追加パラメータを吸収

    Returns:
        MCScoreWrapper: 時系列全体を処理するスコアネットワーク
    """
    return MCScoreWrapper(
        ScoreUNet(
            channels=3,              # Lorenzシステムの3次元状態 (x, y, z)
            embedding=embedding,
            hidden_channels=hidden_channels,
            hidden_blocks=hidden_blocks,
            activation=ACTIVATIONS[activation],
            spatial=1,               # 1次元時系列データ（時間軸のみ）
        )
    )


def make_local_score(
    window: int = 5,
    embedding: int = 32,
    width: int = 128,
    depth: int = 5,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    """ローカルスコアネットワークを構築

    時間窓内の局所的な依存関係のみを考慮するMLPベースのスコアネットワーク
    グローバルモデルより軽量で、長い時系列に対応可能

    Args:
        window: 時間窓のサイズ（奇数、中心時刻±order）
        embedding: 時刻埋め込みの次元数
        width: 隠れ層の幅
        depth: 隠れ層の深さ
        activation: 活性化関数名
        **absorb: 未使用の追加パラメータを吸収

    Returns:
        MCScoreNet: 局所的な時間窓を処理するスコアネットワーク
    """
    return MCScoreNet(
        features=3,                      # Lorenzシステムの3次元状態
        order=window // 2,               # 前後何ステップ見るか
        embedding=embedding,
        hidden_features=[width] * depth, # 全隠れ層で同じ幅を使用
        activation=ACTIVATIONS[activation],
    )


def load_score(
    file: Path,
    local: bool = False,
    device: str = 'cpu',
    **kwargs,
) -> nn.Module:
    """学習済みスコアネットワークをロード

    Args:
        file: state.pthファイルのパス
        local: Trueの場合ローカルモデル、Falseの場合グローバルモデル
        device: ロード先デバイス（'cpu'または'cuda'）
        **kwargs: 設定を上書きするパラメータ

    Returns:
        ロードされたスコアネットワーク
    """
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)  # 同じディレクトリからconfig.yamlを読み込む
    config.update(kwargs)

    # モデルタイプに応じてスコアネットワークを構築
    if local:
        score = make_local_score(**config)
    else:
        score = make_global_score(**config)

    score.load_state_dict(state)

    return score


def log_prior(x: Tensor) -> Tensor:
    """Lorenzシステムの事前分布（遷移確率）の対数尤度を計算

    Args:
        x: 状態の時系列 (..., T, 3)

    Returns:
        対数事前確率 (...,)
    """
    chain = make_chain()

    # 各時刻間の遷移確率を計算
    log_p = chain.log_prob(x[..., :-1, :], x[..., 1:, :])
    # 時間方向に総和（軌道全体の確率）
    log_p = log_p.sum(dim=-1)

    return log_p


def log_likelihood(
    y: Tensor,
    x: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
) -> Tensor:
    """観測尤度の対数を計算（データ同化用）

    Args:
        y: 観測データ (..., T_obs, d)
        x: 状態の時系列 (..., T, 3)
        A: 観測演算子（状態から観測への写像）
        sigma: 観測ノイズの標準偏差
        step: 観測の間隔（1なら全時刻、8なら8ステップごと）

    Returns:
        対数尤度 (...,)
    """
    # 観測時刻のみ抽出
    x = x[..., ::step, :]

    # ガウス観測モデル: y ~ N(A(x), sigma^2)
    log_p = Normal(y, sigma).log_prob(A(x))
    # 時間・変数方向に総和
    log_p = log_p.sum(dim=(-1, -2))

    return log_p


def posterior(
    y: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
    particles: int = 16384,
) -> Tensor:
    """Bootstrap粒子フィルタで事後分布からサンプリング

    観測データから状態の事後分布を推定（データ同化）
    粒子フィルタによる逐次ベイズ推定

    Args:
        y: 観測データ (T_obs, d)
        A: 観測演算子（状態から観測への写像）
        sigma: 観測ノイズの標準偏差
        step: 観測の間隔
        particles: 粒子数（多いほど精度が高いがコストも増加）

    Returns:
        事後サンプル (particles, T, 3)
    """
    chain = make_chain()

    # 事前分布からサンプリング
    x = chain.prior((particles,))
    # バーンイン期間（定常状態に到達）
    x = chain.trajectory(x, length=64, last=True)

    # 尤度関数の定義（観測との整合性）
    def likelihood(y, x):
        w = Normal(y, sigma).log_prob(A(x)).sum(dim=-1)  # 対数重み
        w = torch.softmax(w, 0)  # 正規化された重み
        return w

    # Bootstrap粒子フィルタの実行
    return bpf(x, y, chain.transition, likelihood, step)[:, step:]


def weak_4d_var(
    x: Tensor,
    y: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
    iterations: int = 16,
) -> Tensor:
    """Weak-constraint 4次元変分法（4D-Var）でデータ同化

    初期条件と軌道全体を同時最適化し、観測データとの整合性を最大化
    L-BFGS法による非線形最適化

    Args:
        x: 初期推定軌道 (T, 3)
        y: 観測データ (T_obs, d)
        A: 観測演算子
        sigma: 観測ノイズの標準偏差
        step: 観測の間隔
        iterations: 最適化イテレーション数

    Returns:
        最適化された軌道 (T, 3)
    """
    # 初期条件をバックグラウンド値として保存
    x_b = x[0]
    # 軌道全体を最適化パラメータに設定
    x = torch.nn.Parameter(x.clone())
    optimizer = torch.optim.LBFGS((x,))

    # コスト関数：バックグラウンド項 + 観測項 + モデル制約項
    def closure():
        optimizer.zero_grad()
        # J = ||x[0] - x_b||^2 - log p(x) - log p(y|x)
        loss = (x[0] - x_b).square().sum() \
               - log_prior(x) \              # モデルとの整合性
               - log_likelihood(y, x, A, sigma, step)  # 観測との整合性
        loss.backward()
        return loss

    # L-BFGS法による最適化
    for _ in range(iterations):
        optimizer.step(closure)

    return x.data
