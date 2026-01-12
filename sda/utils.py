r"""Helpers"""

import functools
import json
import math
import os
import random
import socket
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

# Optional import for Optimal Transport (used for W2 distance computation)
try:
    import ot
except ImportError:
    ot = None

from .score import VPSDE

# =============================================================================
# Slack通知機能
# =============================================================================


def slack_notify(
    message: str,
    webhook_url: str | None = None,
    username: str | None = None,
    icon_emoji: str = ":robot_face:",
) -> bool:
    """Slackにメッセージを送信

    環境変数 SLACK_WEBHOOK_URL が設定されていれば自動的に使用。
    設定されていない場合は何もせず False を返す（エラーにはならない）。

    Args:
        message: 送信するメッセージ
        webhook_url: Slack Webhook URL（省略時は環境変数から取得）
        username: 表示名（省略時はホスト名）
        icon_emoji: アイコン絵文字

    Returns:
        送信成功時True、失敗または未設定時False
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False

    if username is None:
        username = f"sda@{socket.gethostname()}"

    payload = json.dumps(
        {
            "text": message,
            "username": username,
            "icon_emoji": icon_emoji,
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def slack_on_complete(
    success_msg: str | None = None,
    error_msg: str | None = None,
    include_traceback: bool = True,
):
    """関数完了時にSlack通知を送るデコレータ

    環境変数 SLACK_WEBHOOK_URL が設定されている場合のみ通知。

    Args:
        success_msg: 成功時のメッセージ（省略時は自動生成）
        error_msg: エラー時のメッセージ（省略時は自動生成）
        include_traceback: エラー時にトレースバックを含めるか

    Example:
        @slack_on_complete()
        def train():
            ...

        @slack_on_complete(success_msg="Training finished!")
        def my_job():
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                elapsed_str = _format_duration(elapsed)

                msg = success_msg or f"✅ `{func_name}` completed successfully"
                msg += f"\n⏱️ Duration: {elapsed_str}"

                slack_notify(msg)
                return result

            except Exception as e:
                elapsed = time.time() - start_time
                elapsed_str = _format_duration(elapsed)

                msg = error_msg or f"❌ `{func_name}` failed with error"
                msg += f"\n⏱️ Duration: {elapsed_str}"
                msg += f"\n🔴 Error: {type(e).__name__}: {e}"

                if include_traceback:
                    tb = traceback.format_exc()
                    # Slack message limit対策（最後の1000文字）
                    if len(tb) > 1000:
                        tb = "...\n" + tb[-1000:]
                    msg += f"\n```\n{tb}\n```"

                slack_notify(msg)
                raise

        return wrapper

    return decorator


def _format_duration(seconds: float) -> str:
    """秒数を人間が読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


def load_env_for_slurm(
    keys: list[str],
    env_file: Path | None = None,
) -> list[str]:
    """指定したキーの環境変数をSLURM用のexport文として取得

    以下の順序で値を探す:
    1. 現在の環境変数 (os.environ)
    2. .envファイル（プロジェクトルートを自動検出）

    Args:
        keys: 取得したい環境変数のキー名リスト
        env_file: .envファイルのパス（省略時は自動検出）

    Returns:
        export文のリスト（例: ['export SLACK_WEBHOOK_URL="https://..."']）
    """
    # .envファイルの自動検出
    if env_file is None:
        # このファイルの親ディレクトリから上に向かって.envを探す
        current = Path(__file__).resolve().parent
        for _ in range(5):  # 最大5階層上まで
            candidate = current / ".env"
            if candidate.exists():
                env_file = candidate
                break
            current = current.parent

    # .envファイルを読み込み（シンプルなパーサー）
    env_from_file = {}
    if env_file and env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # クォートを除去
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    env_from_file[key] = value

    # export文を生成
    exports = []
    for key in keys:
        # 現在の環境変数を優先、なければ.envから
        value = os.environ.get(key) or env_from_file.get(key, "")
        if value:
            # シェルエスケープ（シンプル版）
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            exports.append(f'export {key}="{escaped}"')

    return exports


# =============================================================================
# 利用可能な活性化関数の辞書
ACTIVATIONS = {
    "ReLU": torch.nn.ReLU,
    "ELU": torch.nn.ELU,
    "GELU": torch.nn.GELU,
    "SELU": torch.nn.SELU,
    "SiLU": torch.nn.SiLU,
}


def random_config(configs: dict[str, Sequence[Any]]) -> dict[str, Any]:
    """ランダムなハイパーパラメータ設定を生成

    各パラメータの候補リストからランダムに選択
    ハイパーパラメータ探索に使用

    Args:
        configs: パラメータ名と候補値のリストの辞書

    Returns:
        ランダムに選択されたパラメータの辞書
    """
    return {key: random.choice(values) for key, values in configs.items()}


def save_config(config: dict[str, Any], path: Path) -> None:
    """設定をJSONファイルとして保存

    Args:
        config: 保存する設定の辞書
        path: 保存先のディレクトリパス
    """
    with open(path / "config.json", mode="x") as f:
        json.dump(config, f)


def load_config(path: Path) -> dict[str, Any]:
    """JSONファイルから設定を読み込む

    Args:
        path: 設定ファイルがあるディレクトリパス

    Returns:
        読み込んだ設定の辞書
    """
    with open(path / "config.json") as f:
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
        file: Path,  # HDF5データファイルのパス
        window: int | None = None,  # 時間窓のサイズ（Noneの場合は全系列を使用）
        flatten: bool = False,  # 時間軸とチャネル軸をフラット化するか
    ):
        super().__init__()

        # HDF5ファイルから軌跡データを読み込む
        with h5py.File(file, mode="r") as f:
            self.data = f["x"][:]  # 形状: (N_trajectories, T, ...)

        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        # データセットのサイズ（軌跡の数）
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[Tensor, dict]:
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
    sde: VPSDE,  # 学習するSDEモデル
    trainset: Dataset,  # 訓練データセット
    validset: Dataset,  # 検証データセット
    epochs: int = 256,  # エポック数
    batch_size: int = 64,  # バッチサイズ
    optimizer: str = "AdamW",  # オプティマイザの種類
    learning_rate: float = 1e-3,  # 学習率
    weight_decay: float = 1e-3,  # 重み減衰
    scheduler: float = "linear",  # 学習率スケジューラ
    device: str = "cpu",  # デバイス（'cpu' or 'cuda'）
    **absorb,  # その他の引数を吸収
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
    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            sde.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # L2正則化
        )
    else:
        raise ValueError()

    # 学習率スケジューラの設定
    if scheduler == "linear":
        # 線形減衰
        lr = lambda t: 1 - (t / epochs)
    elif scheduler == "cosine":
        # コサイン減衰（より滑らかな減衰）
        lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == "exponential":
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
        lr = optimizer.param_groups[0]["lr"]

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
    err_xy = dxx + dyy - 2 * xy  # p(x)とq(y)間のペア距離

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
