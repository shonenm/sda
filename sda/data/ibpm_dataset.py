r"""IBPM Dataset for Score-Based Data Assimilation

IBPM（Immersed Boundary Projection Method）による円柱周り流れデータの
PyTorchデータセット実装

- 幾何条件（円柱マスク + 流入プロファイル）の生成
- 時系列データの時間窓切り出し
- 流体マスクの提供
"""

import h5py
import torch
import numpy as np

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import *


class IBPMNormalizer:
    """IBPM速度場の正規化・逆正規化を行うクラス

    チャネルごとに mean=0, std=1 に正規化する。
    学習データから計算した統計量をデフォルト値として保持。

    Usage:
        normalizer = IBPMNormalizer()
        x_norm = normalizer.normalize(x)      # 正規化
        x_orig = normalizer.denormalize(x_norm)  # 逆正規化
    """

    # 学習データから計算したデフォルト統計量
    # 127×127解像度データ（IBPM 128×128シミュレーション出力）
    DEFAULT_MEAN = torch.tensor([0.998540, 0.000000])  # [u_mean, v_mean]
    DEFAULT_STD = torch.tensor([0.415285, 0.207527])   # [u_std, v_std]

    def __init__(
        self,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
    ):
        """
        Args:
            mean: チャネルごとの平均 (C,)。Noneの場合はデフォルト値を使用
            std: チャネルごとの標準偏差 (C,)。Noneの場合はデフォルト値を使用
        """
        self.mean = mean if mean is not None else self.DEFAULT_MEAN.clone()
        self.std = std if std is not None else self.DEFAULT_STD.clone()

    def normalize(self, x: Tensor) -> Tensor:
        """正規化: x_norm = (x - mean) / std

        Args:
            x: (..., C, H, W) 速度場テンソル

        Returns:
            x_norm: (..., C, H, W) 正規化された速度場
        """
        # mean, std を x と同じデバイス・形状にブロードキャスト
        mean = self.mean.to(x.device).view(*([1] * (x.ndim - 3)), -1, 1, 1)
        std = self.std.to(x.device).view(*([1] * (x.ndim - 3)), -1, 1, 1)
        return (x - mean) / std

    def denormalize(self, x_norm: Tensor) -> Tensor:
        """逆正規化: x = x_norm * std + mean

        Args:
            x_norm: (..., C, H, W) 正規化された速度場

        Returns:
            x: (..., C, H, W) 元のスケールの速度場
        """
        mean = self.mean.to(x_norm.device).view(*([1] * (x_norm.ndim - 3)), -1, 1, 1)
        std = self.std.to(x_norm.device).view(*([1] * (x_norm.ndim - 3)), -1, 1, 1)
        return x_norm * std + mean

    @classmethod
    def from_data(cls, data: Tensor) -> "IBPMNormalizer":
        """データから統計量を計算してNormalizerを作成

        Args:
            data: (N, T, C, H, W) または (T, C, H, W) の速度場データ

        Returns:
            IBPMNormalizer インスタンス
        """
        # チャネル次元を特定 (後ろから3番目)
        c_dim = -3
        # チャネル以外の次元で平均・標準偏差を計算
        dims = list(range(data.ndim))
        dims.remove(data.ndim + c_dim)  # チャネル次元を除く

        mean = data.mean(dim=dims)
        std = data.std(dim=dims)

        return cls(mean=mean, std=std)


def build_cylinder_mask(
    H: int = 64,
    W: int = 64,
    center: Tuple[float, float] = (32.0, 37.0),
    radius: float = 7.5
) -> Tensor:
    """円柱マスクを生成

    Args:
        H: 高さ（ピクセル）
        W: 幅（ピクセル）
        center: 円柱中心座標 (cx, cy)
        radius: 円柱半径（ピクセル）

    Returns:
        cylinder_mask: (H, W) テンソル、1=流体、0=物体
    """
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )

    # 中心からの距離
    dist = torch.sqrt((x - center[0])**2 + (y - center[1])**2)

    # 流体領域（円柱外部）
    fluid_mask = (dist > radius).float()

    return fluid_mask


def build_inflow_profile(
    H: int = 64,
    W: int = 64,
    U: float = 1.0
) -> Tensor:
    """流入プロファイルを生成

    Args:
        H: 高さ（ピクセル）
        W: 幅（ピクセル）
        U: 流入速度（無次元）

    Returns:
        inflow_profile: (H, W) テンソル、左端列のみU、他は0
    """
    profile = torch.zeros(H, W, dtype=torch.float32)
    profile[:, 0] = U  # 左端列（x=0）のみ非ゼロ

    return profile


def build_sdf(
    H: int = 64,
    W: int = 64,
    center: Tuple[float, float] = (32.0, 37.0),
    radius: float = 7.5
) -> Tensor:
    """符号付き距離場（SDF）を生成

    Args:
        H: 高さ（ピクセル）
        W: 幅（ピクセル）
        center: 円柱中心座標 (cx, cy)
        radius: 円柱半径（ピクセル）

    Returns:
        sdf: (H, W) テンソル、外部=正、内部=負
    """
    from scipy.ndimage import distance_transform_edt

    # 円柱マスクを生成
    cylinder_mask = build_cylinder_mask(H, W, center, radius)

    # NumPy配列に変換
    fluid_region = cylinder_mask.cpu().numpy()
    solid_region = 1 - fluid_region

    # 距離場を計算
    dist_fluid = distance_transform_edt(fluid_region)   # 流体領域での距離
    dist_solid = distance_transform_edt(solid_region)   # 物体領域での距離

    # 符号付き距離
    sdf = dist_fluid - dist_solid

    return torch.from_numpy(sdf).float()


class IBPMDataset(Dataset):
    """IBPMデータセット

    HDF5ファイルから時系列速度場を読み込み、幾何条件と共に返すデータセット

    HDF5構造:
        /velocity: (T, N, C, H, W) - 時刻、サンプル、チャネル、高さ、幅

    Returns:
        x: (T, C, H, W) - 速度場の時間窓
        cond: (C_cond, H, W) - 条件チャネル（マスク、流入、SDF等）
        mask: (H, W) - 流体マスク

    Args:
        h5_path: HDF5ファイルのパス
        time_window: 時間窓のサイズ
        use_sdf: SDFを条件に含めるか（Trueの場合、C_cond=3）
        cylinder_params: 円柱パラメータ {'center': (cx, cy), 'radius': r}
        normalize: データを正規化するか（デフォルト: True）
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        time_window: int = 8,
        use_sdf: bool = False,
        cylinder_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
    ):
        super().__init__()

        self.h5_path = Path(h5_path)
        self.time_window = time_window
        self.use_sdf = use_sdf
        self.normalize = normalize

        # 正規化を行う場合はNormalizerを初期化
        if normalize:
            self.normalizer = IBPMNormalizer()

        # HDF5ファイルを開く
        self.h5_file = h5py.File(self.h5_path, 'r')
        # Try 'velocity' first, fall back to 'x' for IBPM data
        if 'velocity' in self.h5_file:
            self.data = self.h5_file['velocity']  # (T, N, C, H, W)
        elif 'x' in self.h5_file:
            self.data = self.h5_file['x']  # (T, N, C, H, W) for IBPM
        else:
            raise KeyError(f"Neither 'velocity' nor 'x' found in {self.h5_path}")

        # データ形状を取得
        self.T, self.N, self.C, self.H, self.W = self.data.shape

        # 円柱パラメータ
        # 127×127解像度、領域 x∈[-2,2], y∈[-2,2]、円柱中心(0, 0)の場合:
        # center = (0 - (-2)) / (4/127) = 63.5
        # radius = 0.5 / (4/127) = 15.875
        if cylinder_params is None:
            cylinder_params = {
                'center': (63.5, 63.5),  # (W方向, H方向)のピクセル座標
                'radius': 15.875,
            }
        self.cylinder_params = cylinder_params

        # 幾何条件を生成
        self.cylinder_mask = build_cylinder_mask(
            self.H, self.W,
            center=cylinder_params['center'],
            radius=cylinder_params['radius']
        )

        self.inflow_profile = build_inflow_profile(self.H, self.W, U=1.0)

        if use_sdf:
            self.sdf = build_sdf(
                self.H, self.W,
                center=cylinder_params['center'],
                radius=cylinder_params['radius']
            )
        else:
            self.sdf = None

    def __len__(self) -> int:
        """有効なサンプル数（時間窓を考慮）

        各サンプルから複数の時間窓を切り出せる
        """
        # 各サンプルから取れる時間窓の数
        num_windows_per_sample = max(0, self.T - self.time_window + 1)
        return self.N * num_windows_per_sample

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """データセットから1つのサンプルを取得

        Args:
            idx: インデックス

        Returns:
            x: (T, C, H, W) - 速度場の時間窓
            cond: (C_cond, H, W) - 条件チャネル
            mask: (H, W) - 流体マスク
        """
        # インデックスをサンプル番号と時刻に分解
        num_windows_per_sample = self.T - self.time_window + 1
        sample_idx = idx // num_windows_per_sample
        time_idx = idx % num_windows_per_sample

        # 時間窓を切り出し: (T, C, H, W)
        x = self.data[time_idx:time_idx + self.time_window, sample_idx, :, :, :]
        x = torch.from_numpy(np.array(x)).float()

        # 正規化（mean=0, std=1に）
        if self.normalize:
            x = self.normalizer.normalize(x)

        # 条件チャネルを生成
        if self.use_sdf:
            cond = torch.stack([
                self.cylinder_mask,
                self.inflow_profile,
                self.sdf
            ], dim=0)  # (3, H, W)
        else:
            cond = torch.stack([
                self.cylinder_mask,
                self.inflow_profile
            ], dim=0)  # (2, H, W)

        # 流体マスク
        mask = self.cylinder_mask

        return x, cond, mask

    def __del__(self):
        """デストラクタ: HDF5ファイルを閉じる"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
