#!/usr/bin/env python
"""
IBPM データ生成スクリプト

既存の199×399データを100×200にリサイズして64サンプルを生成。
時間方向のスライディングウィンドウとデータ拡張で多様性を確保。

Usage:
    python experiments/ibpm/generate_data.py

Output:
    /workspace/data/ibpm_h5_small/
    ├── train.h5  # (T, 48, 2, 100, 200)
    ├── valid.h5  # (T, 8, 2, 100, 200)
    └── test.h5   # (T, 8, 2, 100, 200)
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# パス設定
SOURCE_DIR = Path("/workspace/data/ibpm_h5_wide_perturbed")
OUTPUT_DIR = Path("/workspace/data/ibpm_h5_small")

# 設定
CONFIG = {
    "target_height": 100,
    "target_width": 200,
    "n_samples": 64,  # 目標サンプル数
    "time_window": 42,  # 各サンプルの時間長
}


def resize_velocity(velocity: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """速度場をリサイズ

    Args:
        velocity: (C, H, W) または (T, C, H, W) の速度場
        target_h: 目標高さ
        target_w: 目標幅

    Returns:
        リサイズされた速度場
    """
    if velocity.ndim == 3:
        # (C, H, W)
        C, H, W = velocity.shape
        zoom_factors = (1, target_h / H, target_w / W)
        return zoom(velocity, zoom_factors, order=1)
    elif velocity.ndim == 4:
        # (T, C, H, W)
        T, C, H, W = velocity.shape
        zoom_factors = (1, 1, target_h / H, target_w / W)
        return zoom(velocity, zoom_factors, order=1)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {velocity.ndim}D")


def create_samples_from_source(
    source_file: Path,
    n_samples: int,
    time_window: int,
    target_h: int,
    target_w: int,
) -> list:
    """ソースファイルからサンプルを生成"""

    with h5py.File(source_file, "r") as f:
        data = f["x"][:]  # (T, N, C, H, W)

    T, N, C, H, W = data.shape
    print(f"  Source shape: {data.shape}")

    samples = []

    # 各サンプルからスライディングウィンドウで複数のサンプルを生成
    for n in tqdm(range(N), desc="Processing samples", leave=False):
        sample_data = data[:, n, :, :, :]  # (T, C, H, W)

        # リサイズ
        resized = resize_velocity(sample_data, target_h, target_w)

        # スライディングウィンドウ
        stride = max(1, (T - time_window) // 4)  # 4つのウィンドウを取得
        for start in range(0, T - time_window + 1, stride):
            window = resized[start : start + time_window]  # (time_window, C, H, W)
            samples.append(window)

            if len(samples) >= n_samples:
                return samples

    return samples


def create_dataset(
    all_samples: list,
    output_dir: Path,
    train_ratio: float = 0.75,
    valid_ratio: float = 0.125,
):
    """HDF5データセットを作成"""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(all_samples)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    # シャッフル
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    splits = {
        "train": [all_samples[i] for i in indices[:n_train]],
        "valid": [all_samples[i] for i in indices[n_train : n_train + n_valid]],
        "test": [all_samples[i] for i in indices[n_train + n_valid :]],
    }

    for split_name, samples in splits.items():
        if len(samples) == 0:
            print(f"Warning: {split_name} split is empty")
            continue

        # HDF5には (T, N, C, H, W) で保存（IBPMDatasetが期待する形式）
        data = np.stack(samples, axis=1)  # (T, N, C, H, W)

        output_file = output_dir / f"{split_name}.h5"
        with h5py.File(output_file, "w") as f:
            dset = f.create_dataset(
                "x",
                data=data,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["description"] = "IBPM velocity field (u, v) - resized"
            dset.attrs["shape"] = "(T, N, C, H, W)"
            dset.attrs["original_resolution"] = "199x399"
            dset.attrs["target_resolution"] = f"{CONFIG['target_height']}x{CONFIG['target_width']}"

        print(f"{split_name}: {data.shape} -> {output_file}")


def main():
    print("=" * 60)
    print("IBPM Data Generation (Resize Mode)")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target resolution: {CONFIG['target_height']}×{CONFIG['target_width']} (H×W)")
    print(f"Samples: {CONFIG['n_samples']}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ソースファイルを確認
    train_source = SOURCE_DIR / "train.h5"
    if not train_source.exists():
        print(f"ERROR: Source file not found: {train_source}")
        sys.exit(1)

    # サンプル生成
    print("\nGenerating samples from source data...")
    all_samples = create_samples_from_source(
        train_source,
        CONFIG["n_samples"],
        CONFIG["time_window"],
        CONFIG["target_height"],
        CONFIG["target_width"],
    )

    print(f"\nCollected {len(all_samples)} samples")
    if all_samples:
        print(f"Sample shape: {all_samples[0].shape}")

    if len(all_samples) == 0:
        print("ERROR: No samples collected!")
        sys.exit(1)

    # データセット作成
    print("\nCreating HDF5 dataset...")
    create_dataset(all_samples, OUTPUT_DIR)

    # Normalizer用の統計量を計算
    print("\nComputing normalization statistics...")
    all_data = np.stack(all_samples, axis=1)  # (T, N, C, H, W)
    mean_u = all_data[:, :, 0, :, :].mean()
    mean_v = all_data[:, :, 1, :, :].mean()
    std_u = all_data[:, :, 0, :, :].std()
    std_v = all_data[:, :, 1, :, :].std()

    print("\nNormalizer statistics (update ibpm_dataset.py):")
    print(f"  DEFAULT_MEAN = torch.tensor([{mean_u:.6f}, {mean_v:.6f}])")
    print(f"  DEFAULT_STD = torch.tensor([{std_u:.6f}, {std_v:.6f}])")

    print("\n✓ Data generation completed!")


if __name__ == "__main__":
    main()
