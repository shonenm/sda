r"""IBPM experiment helpers

IBPM（Immersed Boundary Projection Method）流体シミュレーション実験のためのユーティリティ関数群
- モデルの構築とロード（Kolmogorov流と同じアーキテクチャを使用）
- 渦度場の可視化（RGB変換、グリッド表示、GIF作成）
- 円柱周りの流れ（Re=100）の2D速度場を扱う
"""

import os
import seaborn

from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *


# データとモデルの保存先パスの設定
# 環境変数SCRATCHが存在する場合はそちらを優先（HPC環境対応）
if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/kolmogorov'  # 注: パスは'kolmogorov'のまま（コードの再利用）
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_chain() -> MarkovChain:
    """Kolmogorov流のマルコフ連鎖を作成

    注: IBPM実験でもKolmogorovFlowクラスを使用しているが、
        実際のデータは別途IBPMシミュレーション結果を使用

    Returns:
        KolmogorovFlow: 256×256グリッド、時間刻み0.2
    """
    return KolmogorovFlow(size=256, dt=0.2)


class LocalScoreUNet(ScoreUNet):
    r"""強制項チャネルを持つスコアU-Net

    Kolmogorov流の強制項 sin(4x) を条件付けチャネルとして持つU-Net
    IBPM実験でも同じアーキテクチャを使用（円柱は境界条件として扱う）

    Args:
        channels: 入力チャネル数（時間窓×速度成分数）
        size: 空間グリッドサイズ（デフォルト: 64）
        **kwargs: ScoreUNetの追加パラメータ
    """

    def __init__(
        self,
        channels: int,
        size: int = 64,
        **kwargs,
    ):
        # 条件付けチャネル数を1に設定（強制項用）
        super().__init__(channels, 1, **kwargs)

        # 強制項 sin(4x) の作成（物理空間のセルセンターで評価）
        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        # バッファとして登録（学習パラメータではないが、デバイス移動時に自動で移動される）
        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        """順伝播：常に強制項を条件として使用

        Args:
            x: 入力速度場 (batch, channels, H, W)
            t: 拡散時刻 (batch,)
            c: 条件（使用されない、常にself.forcingを使用）

        Returns:
            スコア推定値 (batch, channels, H, W)
        """
        return super().forward(x, t, self.forcing)


def make_score(
    window: int = 3,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    """IBPM用のスコアネットワークを構築

    MCScoreNet（マルコフ連鎖スコアネット）+ LocalScoreUNetの組み合わせ
    Kolmogorov流と同じアーキテクチャで円柱周りの流れをモデル化

    Args:
        window: 時間窓のサイズ（奇数、中心時刻±order）
        embedding: 時刻埋め込みの次元数
        hidden_channels: U-Netの各深さでのチャネル数（例: (64, 128, 256)は3段階）
        hidden_blocks: 各深さでの残差ブロック数
        kernel_size: 畳み込みカーネルのサイズ
        activation: 活性化関数名（'SiLU', 'ReLU'など）
        **absorb: 未使用の追加パラメータを吸収

    Returns:
        MCScoreNet: 時系列を考慮したスコアネットワーク
    """
    # MCScoreNet: 2成分の速度場、order=window//2（前後何ステップ見るか）
    score = MCScoreNet(2, order=window // 2)

    # カーネルとして強制項付きU-Netを使用
    score.kernel = LocalScoreUNet(
        channels=window * 2,           # window時刻 × 2成分の速度場
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,                     # 2次元空間データ
        padding_mode='circular',       # 周期境界条件（円柱は境界で処理）
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
