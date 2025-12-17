r"""IBPM experiment helpers

IBPM（Immersed Boundary Projection Method）流体シミュレーション実験のためのユーティリティ関数群
- モデルの構築とロード（Kolmogorov流と同じアーキテクチャを使用）
- 渦度場の可視化（RGB変換、グリッド表示、GIF作成）
- 円柱周りの流れ（Re=100）の2D速度場を扱う
"""

import h5py
import matplotlib.pyplot as plt
import os
import seaborn

from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from typing import *

from sda.data.ibpm_dataset import IBPMNormalizer
from sda.score import *
from sda.utils import *

try:
    from sda.mcs import KolmogorovFlow
except ImportError:
    # KolmogorovFlow requires jax, but make_chain() is not used in training
    KolmogorovFlow = None


# データとモデルの保存先パスの設定
# 環境変数SCRATCHが存在する場合はそちらを優先（HPC環境対応）
if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/kolmogorov'  # 注: パスは'kolmogorov'のまま（コードの再利用）
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_chain() -> 'MarkovChain':
    """Kolmogorov流のマルコフ連鎖を作成

    注: IBPM実験でもKolmogorovFlowクラスを使用しているが、
        実際のデータは別途IBPMシミュレーション結果を使用
        学習スクリプトではこの関数は使用されない

    Returns:
        KolmogorovFlow: 256×256グリッド、時間刻み0.2
    """
    if KolmogorovFlow is None:
        raise ImportError("KolmogorovFlow requires jax, which is not available")
    return KolmogorovFlow(size=256, dt=0.2)


class LocalScoreUNet(ScoreUNet):
    r"""幾何条件チャネルを持つスコアU-Net（IBPM専用）

    IBPM円柱流れ用の条件付きU-Net
    - 円柱マスク + 流入プロファイル（+ オプショナルSDF）を条件として使用
    - reflect padding（非周期境界条件）

    Args:
        channels: 入力チャネル数（時間窓×速度成分数）
        cond_channels: 条件チャネル数（2 or 3）
        **kwargs: ScoreUNetの追加パラメータ
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int = 2,  # mask + inflow (+ optional sdf)
        **kwargs,
    ):
        # 条件付けチャネル数を指定
        super().__init__(channels, cond_channels, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """順伝播：幾何条件を使用

        Args:
            x: 入力速度場 (B, L-2*order, (2*order+1)*C, H, W) after unfold or (batch, channels, H, W)
            t: 拡散時刻 (batch,)
            c: 幾何条件 (batch, cond_channels, H, W) - batch dimension included

        Returns:
            スコア推定値 (same shape as x)
        """
        # c already has batch dimension from MCScoreNet
        # No need to expand manually - parent class handles broadcasting
        return super().forward(x, t, c)


def make_score(
    window: int = 5,
    cond_channels: int = 2,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (96, 192, 384),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    """IBPM用のスコアネットワークを構築

    MCScoreNet（マルコフ連鎖スコアネット）+ LocalScoreUNetの組み合わせ
    IBPM専用：reflect padding、幾何条件チャネル対応

    Args:
        window: 時間窓のサイズ（奇数、中心時刻±order）
        cond_channels: 条件チャネル数（2=mask+inflow, 3=mask+inflow+sdf）
        embedding: 時刻埋め込みの次元数
        hidden_channels: U-Netの各深さでのチャネル数
        hidden_blocks: 各深さでの残差ブロック数
        kernel_size: 畳み込みカーネルのサイズ
        activation: 活性化関数名（'SiLU', 'ReLU'など）
        **absorb: 未使用の追加パラメータを吸収

    Returns:
        MCScoreNet: 時系列を考慮したスコアネットワーク
    """
    # MCScoreNet: 2成分の速度場、order=window//2（前後何ステップ見るか）
    score = MCScoreNet(2, order=window // 2)

    # カーネルとして幾何条件付きU-Netを使用（IBPM専用）
    score.kernel = LocalScoreUNet(
        channels=window * 2,           # window時刻 × 2成分の速度場
        cond_channels=cond_channels,   # 幾何条件チャネル
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,                     # 2次元空間データ
        padding_mode='reflect',        # 非周期境界条件（IBPM用）
    )

    return score


def load_score(file: Path, device: str = 'cpu', **kwargs) -> nn.Module:
    """学習済みスコアネットワークをロード

    Args:
        file: state.pthファイルのパス
        device: ロード先デバイス（'cpu'または'cuda'）
        **kwargs: 設定を上書きするパラメータ

    Returns:
        ロードされたスコアネットワーク
    """
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)  # 同じディレクトリからconfig.yamlを読み込む
    config.update(kwargs)

    score = make_score(**config)
    score.load_state_dict(state)

    return score


def vorticity2rgb(
    w: ArrayLike,
    vmin: float = -1.25,
    vmax: float = 1.25,
) -> ArrayLike:
    """渦度場をRGB画像に変換

    渦度の値を発散カラーマップ（icefire）で可視化
    青：負の渦度（時計回り）、赤：正の渦度（反時計回り）
    円柱背後のカルマン渦の可視化に有効

    Args:
        w: 渦度場の配列
        vmin: 最小値（この値以下は青で飽和）
        vmax: 最大値（この値以上は赤で飽和）

    Returns:
        RGB画像（uint8、0-255）
    """
    w = np.asarray(w)
    # [vmin, vmax]を[0, 1]に正規化
    w = (w - vmin) / (vmax - vmin)
    # [-1, 1]に変換してガンマ補正（0.8乗）で視覚的なコントラスト向上
    w = 2 * w - 1
    w = np.sign(w) * np.abs(w) ** 0.8
    # [0, 1]に戻してカラーマップを適用
    w = (w + 1) / 2
    w = seaborn.cm.icefire(w)  # 発散カラーマップ（青-白-赤）
    w = 256 * w[..., :3]       # RGB成分のみ抽出（アルファチャネル除去）
    w = w.astype(np.uint8)

    return w


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    """渦度場をグリッド状に並べた画像を作成

    複数の渦度場スナップショットを2次元グリッドで配置し、
    白い余白で区切って表示（論文やプレゼン用）
    円柱内部など非流体領域をマスクで表示可能

    Args:
        w: 渦度場の配列（最大5次元: (M, N, H, W, ...)）
        mask: マスク配列（Trueの領域を半透明のグレーで覆う、円柱など）
        pad: 画像間の余白ピクセル数
        zoom: 拡大倍率（1以上の整数）
        **kwargs: vorticity2rgbへの追加引数（vmin, vmaxなど）

    Returns:
        グリッド配置されたPIL画像
    """
    # 渦度をRGBに変換
    w = vorticity2rgb(w, **kwargs)
    # 5次元に拡張（不足次元を先頭に追加）
    w = w[(None,) * (5 - w.ndim)]

    M, N, H, W, _ = w.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask = mask[(None,) * (4 - mask.ndim)]  # 4次元に拡張

    # 白背景のキャンバスを作成
    img = Image.new(
        'RGB',
        size=(
            N * (W + pad) + pad,  # 横: N個の画像 + (N+1)個の余白
            M * (H + pad) + pad,  # 縦: M個の画像 + (M+1)個の余白
        ),
        color=(255, 255, 255),
    )

    # M×Nグリッドで各画像を配置
    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,  # 横位置
                i * (H + pad) + pad,  # 縦位置
            )

            img.paste(Image.fromarray(w[i][j]), offset)

            # マスクがある場合は半透明グレーで覆う（円柱など固体領域）
            if mask is not None:
                img.paste(
                    Image.new('L', size=(W, H), color=240),  # 薄いグレー
                    offset,
                    Image.fromarray(~mask[i][j]),  # マスクの反転（Falseの領域を保護）
                )

    # ズーム（ニアレストネイバー補間でピクセルアートスタイル）
    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    else:
        return img


def sandwich(
    w: ArrayLike,
    offset: int = 5,
    border: int = 1,
    mirror: bool = False,
    **kwargs,
):
    """渦度場の時系列を斜めにずらして重ねた画像を作成

    時間発展を視覚的に表現するため、複数時刻のスナップショットを
    対角線上に少しずつずらして配置（サンドイッチ効果）
    カルマン渦の周期的な放出パターンの可視化に有効

    Args:
        w: 渦度場の時系列（4次元: (N, H, W, ...)）
        offset: 各画像のオフセットピクセル数（対角線方向）
        border: 画像間の白い境界線の幅
        mirror: Trueの場合、左右反転して配置
        **kwargs: vorticity2rgbへの追加引数

    Returns:
        斜めに重ねられたPIL画像
    """
    w = vorticity2rgb(w, **kwargs)
    N, H, W, _ = w.shape

    # ミラーモードの場合は左右反転
    if mirror:
        w = w[:, :, ::-1]

    # キャンバスサイズ: 最後の画像の右下隅まで
    img = Image.new(
        'RGB',
        size=(
            W + (N - 1) * offset,  # 幅: 元の幅 + オフセット × (枚数-1)
            H + (N - 1) * offset,  # 高さ: 元の高さ + オフセット × (枚数-1)
        ),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(img)

    # 古い時刻から新しい時刻へ順に配置
    for i in range(N):
        # 白い背景矩形を描画（境界線効果）
        draw.rectangle(
            (i * offset - border, i * offset - border, img.width, img.height),
            (255, 255, 255),
        )
        # 画像を対角線上に配置
        img.paste(Image.fromarray(w[i]), (i * offset, i * offset))

    # ミラーモードの場合は最終結果も左右反転
    if mirror:
        return ImageOps.mirror(img)
    else:
        return img


def save_gif(
    w: ArrayLike,
    file: Path,
    dt: float = 0.2,
    **kwargs,
) -> None:
    """渦度場の時系列をGIFアニメーションとして保存

    カルマン渦列の動的な振る舞いをアニメーション化

    Args:
        w: 渦度場の時系列（3次元以上: (T, H, W, ...)）
        file: 保存先のGIFファイルパス
        dt: 時間刻み（秒）、フレーム間隔に変換される
        **kwargs: vorticity2rgbへの追加引数（vmin, vmaxなど）
    """
    w = vorticity2rgb(w, **kwargs)

    # 各フレームをPIL画像に変換
    imgs = [Image.fromarray(img) for img in w]

    # GIFとして保存（ループ再生）
    imgs[0].save(
        file,
        save_all=True,           # 複数フレームを保存
        append_images=imgs[1:],  # 2フレーム目以降を追加
        duration=int(1000 * dt), # フレーム間隔（ミリ秒）
        loop=0,                  # 無限ループ
    )


# ===========================================================================
# 追加のユーティリティ関数（evaluate.py 用）
# ===========================================================================

def compute_vorticity(x: Tensor) -> Tensor:
    """速度場から渦度を計算

    Args:
        x: (..., 2, H, W) - 速度場 [u, v]

    Returns:
        vorticity: (..., H, W) - 渦度 (dv/dx - du/dy)
    """
    u = x[..., 0, :, :]
    v = x[..., 1, :, :]
    dvdx = torch.gradient(v, dim=-1)[0]
    dudy = torch.gradient(u, dim=-2)[0]
    return dvdx - dudy


def plot_vorticity(
    w: Tensor,
    title: str = 'Vorticity',
    vmin: float = None,
    vmax: float = None,
    use_percentile: bool = False,
    percentile_range: Tuple[float, float] = (2, 98),
    figsize: Tuple[int, int] = (15, 3),
    save_path: Path = None,
) -> plt.Figure:
    """渦度場をmatplotlibでプロット

    Args:
        w: (N, H, W) or (H, W) 渦度場
        title: プロットタイトル
        vmin, vmax: カラーバーの範囲（Noneなら自動）
        use_percentile: Trueならパーセンタイルで外れ値を無視
        percentile_range: パーセンタイル範囲 (low, high)
        figsize: 図のサイズ
        save_path: 保存先パス（Noneなら保存しない）

    Returns:
        matplotlib Figure
    """
    if w.ndim == 2:
        w = w[None, ...]

    w_np = w.numpy() if isinstance(w, torch.Tensor) else w

    if vmin is None or vmax is None:
        if use_percentile:
            low, high = np.percentile(w_np, percentile_range)
            wmax = max(abs(low), abs(high))
        else:
            wmax = float(max(abs(w_np.min()), abs(w_np.max())))
        vmin, vmax = -wmax, wmax

    n_samples = w.shape[0]
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        im = axes[i].imshow(w_np[i], cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[i].set_title(f't={i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_velocity_field(
    x: Tensor,
    title: str = 'Velocity',
    use_percentile: bool = True,
    figsize: Tuple[int, int] = (15, 6),
    save_path: Path = None,
) -> plt.Figure:
    """速度場を u, v, |v| の3行でプロット

    Args:
        x: (T, 2, H, W) or (2, H, W) 速度場
        title: プロットタイトル
        use_percentile: Trueならパーセンタイルで範囲決定
        figsize: 図のサイズ
        save_path: 保存先パス

    Returns:
        matplotlib Figure
    """
    if x.ndim == 3:
        x = x[None, ...]

    x_np = x.numpy() if isinstance(x, torch.Tensor) else x
    T = x_np.shape[0]

    fig, axes = plt.subplots(3, T, figsize=figsize)
    if T == 1:
        axes = axes[:, None]

    if use_percentile:
        u_range = np.percentile(x_np[:, 0], [2, 98])
        v_range = np.percentile(x_np[:, 1], [2, 98])
    else:
        u_range = [x_np[:, 0].min(), x_np[:, 0].max()]
        v_range = [x_np[:, 1].min(), x_np[:, 1].max()]

    mag = np.sqrt(x_np[:, 0]**2 + x_np[:, 1]**2)

    for t in range(T):
        im0 = axes[0, t].imshow(x_np[t, 0], cmap='RdBu_r', vmin=u_range[0], vmax=u_range[1], origin='lower')
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        im1 = axes[1, t].imshow(x_np[t, 1], cmap='RdBu_r', vmin=v_range[0], vmax=v_range[1], origin='lower')
        axes[1, t].axis('off')

        im2 = axes[2, t].imshow(mag[t], cmap='viridis', origin='lower')
        axes[2, t].axis('off')

    axes[0, 0].set_ylabel('u velocity')
    axes[1, 0].set_ylabel('v velocity')
    axes[2, 0].set_ylabel('|v| magnitude')

    plt.colorbar(im0, ax=axes[0, :], shrink=0.6, label='u')
    plt.colorbar(im1, ax=axes[1, :], shrink=0.6, label='v')
    plt.colorbar(im2, ax=axes[2, :], shrink=0.6, label='|v|')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def load_ibpm_data(
    data_path: Path,
    split: str = 'train',
    normalize: bool = False,
) -> Tensor:
    """IBPMデータをロード

    Args:
        data_path: h5ファイルのあるディレクトリ
        split: 'train' or 'test'
        normalize: Trueの場合、正規化して返す

    Returns:
        data: (N, T, 2, H, W) 速度場テンソル
    """
    file_path = Path(data_path) / f'{split}.h5'
    with h5py.File(file_path, 'r') as f:
        data = torch.from_numpy(f['x'][:])

    if normalize:
        normalizer = IBPMNormalizer()
        data = normalizer.normalize(data)

    return data


def load_trained_model(
    run_dir: Path,
    device: str = 'cuda',
) -> Tuple[nn.Module, dict]:
    """学習済みモデルをロード

    Args:
        run_dir: 学習済みモデルのディレクトリ（state_final.pthがある場所）
        device: ロード先デバイス

    Returns:
        score: ロードされたスコアネットワーク
        config: 設定辞書
    """
    run_dir = Path(run_dir)

    # state_final.pth または state.pth を探す
    if (run_dir / 'state_final.pth').exists():
        state_path = run_dir / 'state_final.pth'
    elif (run_dir / 'state.pth').exists():
        state_path = run_dir / 'state.pth'
    else:
        raise FileNotFoundError(f"No state file found in {run_dir}")

    config = load_config(run_dir)
    score = make_score(**config)
    score.load_state_dict(torch.load(state_path, map_location=device))
    score = score.to(device)
    score.eval()

    return score, config


def reconstruct_sparse(
    x_star: Tensor,
    score: nn.Module,
    cond: Tensor,
    subsample_rates: List[int] = [2, 4, 8, 16],
    noise_std: float = 0.1,
    steps: int = 256,
    corrections: int = 1,
    tau: float = 0.5,
) -> Dict[int, Tensor]:
    """複数のサブサンプルレートでスパース再構成を実行

    Args:
        x_star: (T, 2, H, W) 真の速度場
        score: スコアネットワーク
        cond: (1, C, H, W) 幾何条件
        subsample_rates: サブサンプルレートのリスト
        noise_std: 観測ノイズの標準偏差
        steps: サンプリングステップ数
        corrections: Langevin correctionの回数
        tau: annealing parameter

    Returns:
        results: {subsample_rate: reconstructed_tensor} の辞書
    """
    results = {}
    device = next(score.parameters()).device

    for sub in subsample_rates:
        def A(x, s=sub):
            return x[..., ::s, ::s]

        y_star = torch.normal(A(x_star), noise_std)

        # eta=0.01 で数値安定性を向上（学習時と同じ設定）
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=noise_std,
                sde=VPSDE(score, shape=(), eta=0.01),
            ),
            shape=x_star.shape,
            eta=0.01,
        ).to(device)

        x_recon = sde.sample(c=cond, steps=steps, corrections=corrections, tau=tau).cpu()
        results[sub] = x_recon

    return results
