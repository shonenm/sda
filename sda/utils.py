r"""Helpers"""

import h5py
import json
import math
import ot
import random
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from typing import *

from .score import *


# 利用可能な活性化関数の辞書
ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'ELU': torch.nn.ELU,
    'GELU': torch.nn.GELU,
    'SELU': torch.nn.SELU,
    'SiLU': torch.nn.SiLU,
}


def random_config(configs: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    """ランダムなハイパーパラメータ設定を生成

    各パラメータの候補リストからランダムに選択
    ハイパーパラメータ探索に使用

    Args:
        configs: パラメータ名と候補値のリストの辞書

    Returns:
        ランダムに選択されたパラメータの辞書
    """
    return {
        key: random.choice(values)
        for key, values in configs.items()
    }


def save_config(config: Dict[str, Any], path: Path) -> None:
    """設定をJSONファイルとして保存

    Args:
        config: 保存する設定の辞書
        path: 保存先のディレクトリパス
    """
    with open(path / 'config.json', mode='x') as f:
        json.dump(config, f)


def load_config(path: Path) -> Dict[str, Any]:
    """JSONファイルから設定を読み込む

    Args:
        path: 設定ファイルがあるディレクトリパス

    Returns:
        読み込んだ設定の辞書
    """
    with open(path / 'config.json', mode='r') as f:
        return json.load(f)


def to(x: Any, **kwargs) -> Any:
    """データを指定されたデバイス・型に再帰的に転送

    テンソル、リスト、タプル、辞書を再帰的に処理
    バッチデータをGPUに転送する際などに使用

    Args:
        x: 転送するデータ（任意の型）
        **kwargs: torch.Tensor.to()に渡す引数（例：device='cuda'）

    Returns:
        転送されたデータ
    """
    if torch.is_tensor(x):
        return x.to(**kwargs)
    elif type(x) is list:
        return [to(y, **kwargs) for y in x]
    elif type(x) is tuple:
        return tuple(to(y, **kwargs) for y in x)
    elif type(x) is dict:
        return {k: to(v, **kwargs) for k, v in x.items()}
    else:
        return x


class TrajectoryDataset(Dataset):
    """時系列軌跡データセット

    HDF5ファイルから時系列データを読み込むデータセットクラス
    ランダムな時間窓の切り出しやフラット化をサポート
    """

    def __init__(
        self,
        file: Path,             # HDF5データファイルのパス
        window: int = None,     # 時間窓のサイズ（Noneの場合は全系列を使用）
        flatten: bool = False,  # 時間軸とチャネル軸をフラット化するか
    ):
        super().__init__()

        # HDF5ファイルから軌跡データを読み込む
        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]  # 形状: (N_trajectories, T, ...)

        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        # データセットのサイズ（軌跡の数）
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        # i番目の軌跡を取得
        x = torch.from_numpy(self.data[i])

        # 時間窓が指定されている場合はランダムな部分系列を抽出
        if self.window is not None:
            i = torch.randint(0, len(x) - self.window + 1, size=())
            x = torch.narrow(x, dim=0, start=i, length=self.window)

        # フラット化オプション：時間×チャネルを1次元に
        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}


def loop(
    sde: VPSDE,                        # 学習するSDEモデル
    trainset: Dataset,                 # 訓練データセット
    validset: Dataset,                 # 検証データセット
    epochs: int = 256,                 # エポック数
    batch_size: int = 64,              # バッチサイズ
    optimizer: str = 'AdamW',          # オプティマイザの種類
    learning_rate: float = 1e-3,       # 学習率
    weight_decay: float = 1e-3,        # 重み減衰
    scheduler: float = 'linear',       # 学習率スケジューラ
    device: str = 'cpu',               # デバイス（'cpu' or 'cuda'）
    **absorb,                          # その他の引数を吸収
) -> Iterator:
    """SDEモデルの学習ループ

    訓練と検証を交互に行い、各エポックでの損失を返すジェネレータ

    Yields:
        (loss_train, loss_valid, lr): 訓練損失、検証損失、学習率のタプル
    """

    # データローダーの準備
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

    # オプティマイザの設定
    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            sde.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # L2正則化
        )
    else:
        raise ValueError()

    # 学習率スケジューラの設定
    if scheduler == 'linear':
        # 線形減衰
        lr = lambda t: 1 - (t / epochs)
    elif scheduler == 'cosine':
        # コサイン減衰（より滑らかな減衰）
        lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == 'exponential':
        # 指数減衰
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    # 学習ループ
    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train = []
        losses_valid = []

        ## 訓練フェーズ
        sde.train()  # 訓練モードに設定

        for batch in trainloader:
            x, kwargs = to(batch, device=device)  # データをデバイスに転送

            # 順伝播と損失計算
            l = sde.loss(x, **kwargs)
            l.backward()  # 逆伝播

            # パラメータ更新
            optimizer.step()
            optimizer.zero_grad()

            losses_train.append(l.detach())

        ## 検証フェーズ
        sde.eval()  # 評価モードに設定

        with torch.no_grad():  # 勾配計算を無効化
            for batch in validloader:
                x, kwargs = to(batch, device=device)
                losses_valid.append(sde.loss(x, **kwargs))

        ## 統計情報の計算と表示
        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        lr = optimizer.param_groups[0]['lr']

        # 結果をyield（ジェネレータとして動作）
        yield loss_train, loss_valid, lr

        # プログレスバーに情報を表示
        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=lr)

        ## 学習率の更新
        scheduler.step()


def bpf(
    x: Tensor,  # (M, *) 初期粒子
    y: Tensor,  # (N, *) 観測系列
    transition: Callable[[Tensor], Tensor],  # 遷移関数
    likelihood: Callable[[Tensor, Tensor], Tensor],  # 尤度関数
    step: int = 1,  # 観測間の遷移ステップ数
) -> Tensor:  # (M, N + 1, *) フィルタリングされた軌跡
    r"""Performs bootstrap particle filter (BPF) sampling

    ブートストラップ粒子フィルタによる状態推定
    観測データyから隠れ状態xの軌跡を推定する逐次モンテカルロ法

    .. math:: p(x_0, x_1, ..., x_n | y_1, ..., y_n)
        = p(x_0) \prod_i p(x_i | x_{i-1}) p(y_i | x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Particle_filter

    Arguments:
        x: 初期状態の粒子集合 :math:`x_0`
        y: 観測系列 :math:`(y_1, ..., y_n)`
        transition: 遷移関数 :math:`p(x_i | x_{i-1})`（確率的サンプリング）
        likelihood: 尤度関数 :math:`p(y_i | x_i)`（重みを返す）
        step: 観測1つあたりの遷移ステップ数

    Returns:
        フィルタリングされた粒子の軌跡
    """

    x = x[:, None]  # (M, 1, *)に拡張

    # 各観測に対して予測・更新を実行
    for yi in y:
        # 予測ステップ：遷移モデルで粒子を伝播
        for _ in range(step):
            xi = transition(x[:, -1])
            x = torch.cat((x, xi[:, None]), dim=1)

        # 更新ステップ：尤度に基づいて重みを計算し、リサンプリング
        w = likelihood(yi, xi)  # 各粒子の重み
        j = torch.multinomial(w, len(w), replacement=True)  # 重み付きリサンプリング
        x = x[j]  # 粒子を再選択

    return x


def emd(
    x: Tensor,  # (M, *) 分布pのサンプル
    y: Tensor,  # (N, *) 分布qのサンプル
) -> Tensor:
    r"""Computes the earth mover's distance (EMD) between two distributions.

    アースムーバー距離（Wasserstein距離）の計算
    2つの分布間の最適輸送コストを測定
    分布の類似度評価に使用

    Wikipedia:
        https://wikipedia.org/wiki/Earth_mover%27s_distance

    Arguments:
        x: 分布 :math:`p(x)` からのサンプル集合
        y: 分布 :math:`q(y)` からのサンプル集合

    Returns:
        アースムーバー距離
    """

    # 最適輸送ライブラリを使用してEMDを計算
    return ot.emd2(
        x.new_tensor(()),  # 均等な質量分布を仮定
        y.new_tensor(()),
        torch.cdist(x.flatten(1), y.flatten(1)),  # コスト行列（ユークリッド距離）
    )


def mmd(
    x: Tensor,  # (M, *) 分布pのサンプル
    y: Tensor,  # (N, *) 分布qのサンプル
) -> Tensor:
    r"""Computes the empirical maximum mean discrepancy (MMD) between two distributions.

    最大平均不一致（MMD）の計算
    再生核ヒルベルト空間での分布間距離
    ガウシアンカーネルを使用した非パラメトリックな2標本検定
    複数のスケールで評価（マルチスケールMMD）

    Wikipedia:
        https://wikipedia.org/wiki/Kernel_embedding_of_distributions

    Arguments:
        x: 分布 :math:`p(x)` からのサンプル集合
        y: 分布 :math:`q(y)` からのサンプル集合

    Returns:
        MMD^2の推定値
    """

    # データをフラット化
    x = x.flatten(1)
    y = y.flatten(1)

    # グラム行列の計算
    xx = x @ x.T  # p(x)内の内積
    yy = y @ y.T  # q(y)内の内積
    xy = x @ y.T  # p(x)とq(y)間の内積

    # 距離行列の計算（||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>）
    dxx = xx.diag().unsqueeze(1)
    dyy = yy.diag().unsqueeze(0)

    err_xx = dxx + dxx.T - 2 * xx  # p(x)内のペア距離
    err_yy = dyy + dyy.T - 2 * yy  # q(y)内のペア距離
    err_xy = dxx + dyy - 2 * xy    # p(x)とq(y)間のペア距離

    mmd = 0

    # マルチスケールのガウシアンカーネル（異なる帯域幅）
    for sigma in (1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3):
        # ガウシアンカーネル: k(x, y) = exp(-||x - y||^2 / σ)
        kxx = torch.exp(-err_xx / sigma)
        kyy = torch.exp(-err_yy / sigma)
        kxy = torch.exp(-err_xy / sigma)

        # MMD^2 = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
        mmd = mmd + kxx.mean() + kyy.mean() - 2 * kxy.mean()

    return mmd
