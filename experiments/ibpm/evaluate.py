#!/usr/bin/env python3
"""IBPM Flow 実験の評価スクリプト

Usage:
    # Auto-detect latest run:
    python experiments/ibpm/evaluate.py --mode sample

    # Specify run ID:
    python experiments/ibpm/evaluate.py --run-id ibpm_vpsde_xxx --mode all

    # Specify full path (legacy):
    python experiments/ibpm/evaluate.py --run-dir /path/to/run --mode all
"""

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.ibpm.utils import (
    compute_vorticity,
    load_ibpm_data,
    load_trained_model,
    plot_velocity_and_vorticity,
)
from sda.data.ibpm_dataset import IBPMDataset, IBPMNormalizer, build_cylinder_mask, build_inflow_profile
from sda.paths import get_results_dir, get_run_results_dir, get_runs_dir, get_data_dir
from sda.score import VPSDE


def find_latest_ibpm_run() -> Path:
    """Find the latest IBPM run.

    Returns:
        Path to the latest run directory
    """
    runs_dir = get_runs_dir() / "ibpm"
    if not runs_dir.exists():
        raise ValueError(f"No IBPM runs found: {runs_dir}")

    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not runs:
        raise ValueError(f"No runs found in {runs_dir}")

    # Sort by modification time (newest first)
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

# ============================================================================
# 条件テンソルの円柱位置設定
# ============================================================================
# 学習時の設定:
#   - データ内の実際の円柱位置: (99, 49) ピクセル
#   - 条件テンソルの円柱位置: (63.5, 63.5) ピクセル (IBPMDataset デフォルト)
#   - オフセット: データ - 条件 = (35.5, -14.5)
#
# 評価時も同じオフセットを維持することで、学習時と一貫した入力を与える。
# 詳細: docs/ibpm_generalization_experiment.md
# ============================================================================
TRAIN_COND_CENTER = (63.5, 63.5)  # 学習時の条件テンソル円柱位置
TRAIN_COND_RADIUS = 15.875        # 学習時の条件テンソル円柱半径 (127x127用デフォルト)
TRAIN_DATA_CENTER = (99.0, 49.0)  # 学習データ内の実際の円柱位置
COND_OFFSET = (35.5, -14.5)       # データ位置 - 条件位置


def get_condition_center_for_data(data_cylinder_y_pixel: float) -> tuple:
    """データ内の円柱y位置から条件テンソルの中心を計算

    学習時のオフセットを維持する。

    Args:
        data_cylinder_y_pixel: データ内の円柱のyピクセル位置

    Returns:
        (cx, cy): 条件テンソルの円柱中心位置
    """
    # 条件 = データ位置 - オフセット
    return (TRAIN_COND_CENTER[0], data_cylinder_y_pixel + COND_OFFSET[1])


# ============================================================================
# 物理座標 ↔ ピクセル座標変換
# ============================================================================
# IBPM wide_centered シミュレーション設定:
#   - nx=400, ny=200, length=16, xoffset=-4, yoffset=-4
#   - 物理領域: x ∈ [-4, 12], y ∈ [-4, 4]
#   - 出力解像度: 399 x 199 (境界を除く)
# ============================================================================

def physical_to_pixel_y(
    y_physical: float,
    H: int = 199,
    yoffset: float = -4.0,
    ylength: float = 8.0
) -> float:
    """物理座標yをピクセル座標に変換

    Args:
        y_physical: 物理座標でのy位置
        H: 画像の高さ（ピクセル）
        yoffset: シミュレーション領域のy下端
        ylength: シミュレーション領域のy方向長さ

    Returns:
        ピクセル座標でのy位置
    """
    dy = ylength / H
    return (y_physical - yoffset) / dy


def physical_to_pixel_radius(
    r_physical: float,
    H: int = 199,
    ylength: float = 8.0
) -> float:
    """物理半径をピクセル半径に変換

    Args:
        r_physical: 物理座標での半径
        H: 画像の高さ（ピクセル）
        ylength: シミュレーション領域のy方向長さ

    Returns:
        ピクセル座標での半径
    """
    dy = ylength / H
    return r_physical / dy


# ============================================================================
# 汎化テスト用データパスとパラメータのマッピング
# ============================================================================
# 各ジオメトリ設定に対応するシミュレーションデータディレクトリ
GEOMETRY_DATA_PATHS = {
    "baseline": "ibpm_h5_400x200",  # 学習データ（perturbed）
    "y_m01": "ibpm_h5_gen_v2_y_m01",
    "y_m02": "ibpm_h5_gen_v2_y_m02",
    "y_p02": "ibpm_h5_gen_v2_y_p02",
    "r_04": "ibpm_h5_gen_v2_r04",
    "r_06": "ibpm_h5_gen_v2_r06",
    # 大変動ジオメトリ（派手な変化テスト）
    "r_025": "ibpm_h5_gen_r025",  # 半径 50%
    "r_075": "ibpm_h5_gen_r075",  # 半径 150%
    "y_p10": "ibpm_h5_gen_y_p10",  # y=+1.0
    "y_m10": "ibpm_h5_gen_y_m10",  # y=-1.0
    # 極端条件ジオメトリ（極端な変化テスト）
    "r_015": "ibpm_h5_gen_r015",  # 半径 30% (-70%)
    "r_100": "ibpm_h5_gen_r100",  # 半径 200% (+100%)
    "y_p20": "ibpm_h5_gen_y_p20",  # y=+2.0
    "y_m20": "ibpm_h5_gen_y_m20",  # y=-2.0
}

# 各ジオメトリ設定の物理パラメータ（y=0ベース）
GEOMETRY_PARAMS = {
    "baseline": {"y": 0.0, "radius": 0.5},
    "y_m01": {"y": -0.1, "radius": 0.5},
    "y_m02": {"y": -0.2, "radius": 0.5},
    "y_p02": {"y": 0.2, "radius": 0.5},
    "r_04": {"y": 0.05, "radius": 0.4},  # y=0.05 for vortex shedding
    "r_06": {"y": 0.05, "radius": 0.6},  # y=0.05 for vortex shedding
    # 大変動ジオメトリ（派手な変化テスト）
    "r_025": {"y": 0.05, "radius": 0.25},  # 半径 50%
    "r_075": {"y": 0.05, "radius": 0.75},  # 半径 150%
    "y_p10": {"y": 1.0, "radius": 0.5},   # y=+1.0
    "y_m10": {"y": -1.0, "radius": 0.5},  # y=-1.0
    # 極端条件ジオメトリ（極端な変化テスト）
    "r_015": {"y": 0.05, "radius": 0.15},  # 半径 30% (-70%)
    "r_100": {"y": 0.05, "radius": 1.0},   # 半径 200% (+100%)
    "y_p20": {"y": 2.0, "radius": 0.5},    # y=+2.0
    "y_m20": {"y": -2.0, "radius": 0.5},   # y=-2.0
}

# Reynolds数テスト用データパス（Phase 2で生成予定）
REYNOLDS_DATA_PATHS = {
    "Re_080": "ibpm_h5_gen_Re80",
    "Re_100": "ibpm_h5_wide_centered",  # baseline
    "Re_120": "ibpm_h5_gen_Re120",
}


def visualize_data(data_path: Path, output_dir: Path) -> None:
    """Train/Testデータの可視化"""
    print("=" * 60)
    print("DATA VISUALIZATION")
    print("=" * 60)

    # Train data
    train_data = load_ibpm_data(data_path, split="train")
    print(f"Train data shape: {train_data.shape}")
    print(f"  Samples: {train_data.shape[0]}, Timesteps: {train_data.shape[1]}")
    print(f"  Resolution: {train_data.shape[3]}x{train_data.shape[4]}")
    print(f"  Range: [{train_data.min():.3f}, {train_data.max():.3f}]")

    # 複数サンプルのt=0を可視化（u, v, 渦度の3行）
    sample_indices = [0, 10, 20, 30, 40, min(41, train_data.shape[0] - 1)]
    frames = [train_data[i, 0] for i in sample_indices if i < train_data.shape[0]]
    x_train = torch.stack(frames)

    fig = plot_velocity_and_vorticity(
        x_train,
        title="Train Data: Different samples at t=0",
        figsize=(20, 9),
        save_path=output_dir / "data_train_samples.png",
    )
    plt.close(fig)
    print(f"  Saved: {output_dir / 'data_train_samples.png'}")

    # 速度分布
    u_train = train_data[:, :, 0].flatten()
    v_train = train_data[:, :, 1].flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(u_train.numpy(), bins=100, alpha=0.7, color="blue")
    axes[0].set_xlabel("u velocity")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"u distribution (mean={u_train.mean():.3f}, std={u_train.std():.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].hist(v_train.numpy(), bins=100, alpha=0.7, color="red")
    axes[1].set_xlabel("v velocity")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"v distribution (mean={v_train.mean():.3f}, std={v_train.std():.3f})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "data_velocity_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'data_velocity_stats.png'}")

    # Test data
    test_path = data_path / "test.h5"
    if test_path.exists():
        test_data = load_ibpm_data(data_path, split="test")
        print(f"\nTest data shape: {test_data.shape}")

        for idx in range(min(3, test_data.shape[0])):
            x_test = test_data[idx, :8]
            fig = plot_velocity_and_vorticity(
                x_test,
                title=f"Test Sample {idx}: Velocity and Vorticity Evolution",
                figsize=(20, 9),
                save_path=output_dir / f"data_test_sample_{idx}.png",
            )
            plt.close(fig)
            print(f"  Saved: {output_dir / f'data_test_sample_{idx}.png'}")


def unconditional_sample(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    n_samples: int = 4,
) -> None:
    """無条件サンプリング"""
    print("\n" + "=" * 60)
    print("UNCONDITIONAL SAMPLING")
    print("=" * 60)

    # 正規化用のNormalizer
    normalizer = IBPMNormalizer()

    # データセットから条件と形状を取得（正規化済み）
    window = config.get("window", 16)
    ds = IBPMDataset(str(data_path / "train.h5"), time_window=window, normalize=True)
    x_sample, c_sample, _ = ds[0]

    H, W = x_sample.shape[-2], x_sample.shape[-1]
    T = x_sample.shape[0]
    C = x_sample.shape[1]

    c_batch = c_sample.unsqueeze(0).cuda()
    shape_flat = torch.Size((T * C, H, W))

    print(f"Sampling shape: {shape_flat}")
    print(f"Condition shape: {c_batch.shape}")

    # VPSDE でサンプリング（eta=0.01で数値安定性向上）
    sde = VPSDE(score.kernel, shape=shape_flat, eta=0.01).cuda()

    # 訓練データの統計（比較用）
    train_data = load_ibpm_data(data_path, split="train")
    train_ref = train_data[0, :T]  # 参照用

    for i in range(n_samples):
        print(f"  Generating sample {i + 1}/{n_samples}...")
        x_sampled = sde.sample(torch.Size([1]), c=c_batch, steps=256, corrections=1).cpu()
        x_vis = x_sampled[0].unflatten(0, (T, C))

        # 逆正規化して元のスケールに戻す
        x_vis_denorm = normalizer.denormalize(x_vis)

        # u, v, 渦度の3行でプロット
        fig = plot_velocity_and_vorticity(
            x_vis_denorm,
            title=f"Unconditional Sample {i + 1}",
            figsize=(20, 9),
            save_path=output_dir / f"sample_uncond_{i + 1}.png",
        )
        plt.close(fig)
        print(f"    Saved: {output_dir / f'sample_uncond_{i + 1}.png'}")

        # 速度の統計表示
        u_mean, u_std = x_vis_denorm[:, 0].mean().item(), x_vis_denorm[:, 0].std().item()
        v_mean, v_std = x_vis_denorm[:, 1].mean().item(), x_vis_denorm[:, 1].std().item()
        print(f"    Sample {i + 1} stats: u(mean={u_mean:.4f}, std={u_std:.4f}), v(mean={v_mean:.4f}, std={v_std:.4f})")

    # 訓練データとの統計比較
    print("\n  Train data stats (reference):")
    print(f"    u: mean={train_ref[:, 0].mean():.4f}, std={train_ref[:, 0].std():.4f}")
    print(f"    v: mean={train_ref[:, 1].mean():.4f}, std={train_ref[:, 1].std():.4f}")


def sparse_reconstruction(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    subsample_rates: list = [2, 4, 8, 16],
) -> None:
    """スパース観測からの再構成

    学習時と同じ方式でscore.kernelを使用し、flattened shape (T*C, H, W) で処理
    """
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("SPARSE OBSERVATION RECONSTRUCTION")
    print("=" * 60)

    # 正規化用のNormalizer
    normalizer = IBPMNormalizer()

    # windowサイズ（学習時と同じtimesteps数を使用）
    window = config.get("window", 16)

    # テストデータをロード（生データ）
    test_data = load_ibpm_data(data_path, split="test")
    n_timesteps = min(window, test_data.shape[1])
    x_star_raw = test_data[0, :n_timesteps]  # (T, C, H, W) = (16, 2, H, W)
    T, C, H, W = x_star_raw.shape
    print(f"Ground truth shape: {x_star_raw.shape}")

    # 正規化（モデルは正規化空間で動作）
    x_star_norm = normalizer.normalize(x_star_raw)

    # Flatten（学習時と同じ形式: (T*C, H, W) = (32, H, W)）
    x_star_flat = x_star_norm.flatten(0, 1)
    print(f"Flattened shape: {x_star_flat.shape}")
    print(
        f"Ground truth 値範囲: raw=[{x_star_raw.min():.2f}, {x_star_raw.max():.2f}], norm=[{x_star_norm.min():.2f}, {x_star_norm.max():.2f}]"
    )

    # 幾何条件を生成
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

    # Ground truth 可視化（生データで、u, v, 渦度の3行）
    fig = plot_velocity_and_vorticity(
        x_star_raw,
        title="Ground Truth: Velocity and Vorticity",
        figsize=(20, 9),
        save_path=output_dir / "sparse_ground_truth.png",
    )
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sparse_ground_truth.png'}")

    # 各subsampleレートで再構成（score.kernel + flattened shape）
    print(f"\nReconstructing with subsample rates: {subsample_rates}")
    steps = 256

    # === Clamped Linear Scaling ===
    # 詳細: docs/ibpm/gaussian_score_scaling_issue.md section 13

    # ベースパラメータ（Energy Ratio 1.0 を目指して調整）
    # BASE_STD=0.15でsub=8,16が爆発傾向 → 0.2に戻す
    BASE_STD = 0.2
    BASE_GAMMA = 0.04

    # 安全装置（Floor）: これ以下には絶対に下げない
    # std < 0.15 で危険域、0.10 で爆発確定
    MIN_STD = 0.15
    MIN_GAMMA = 0.02

    # 基準: sub=4
    n_obs_ref = (H // 4) * (W // 4) * T * C

    for sub in subsample_rates:
        n_obs = (H // sub) * (W // sub) * T * C
        ratio = n_obs / n_obs_ref

        # Clamped Linear Scaling: Linear Scalingに最小値フロアを設定
        raw_std = BASE_STD * ratio
        raw_gamma = BASE_GAMMA * (ratio**2)
        std_scaled = max(raw_std, MIN_STD)
        gamma_scaled = max(raw_gamma, MIN_GAMMA)

        clamped_std = " (clamped)" if raw_std < MIN_STD else ""
        clamped_gamma = " (clamped)" if raw_gamma < MIN_GAMMA else ""
        print(
            f"  subsample={sub:2d} (std={std_scaled:.4f}{clamped_std}, gamma={gamma_scaled:.6f}{clamped_gamma})...",
            end=" ",
            flush=True,
        )

        # 空間サブサンプリング演算子
        def A(x, s=sub):
            return x[..., ::s, ::s]

        # 観測（flattened形式）- ノイズはベースラインstdで生成
        y_star = torch.normal(A(x_star_flat), BASE_STD)

        # score.kernelを使用（学習時と同じ）
        # Clamped Linear Scaling: std と gamma をスケーリング（最小値フロア付き）
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=std_scaled,
                gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape,  # (32, H, W) flattened
            eta=0.01,
        ).cuda()

        # サンプリング
        x_recon_flat = sde.sample(
            torch.Size([1]),
            c=cond,
            steps=steps,
            corrections=1,
            tau=0.5,
        ).cpu()[0]

        # Unflatten して (T, C, H, W) に戻す
        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # 逆正規化して可視化
        x_recon = normalizer.denormalize(x_recon_norm)

        # === 診断指標（正規化空間で比較） ===
        # 1. 値の範囲 (Range)
        recon_min = x_recon_norm.min().item()
        recon_max = x_recon_norm.max().item()
        gt_min = x_star_norm.min().item()
        gt_max = x_star_norm.max().item()

        # 2. 標準偏差の比率 (Energy Ratio) ← 最重要
        recon_std = x_recon_norm.std().item()
        gt_std = x_star_norm.std().item()
        energy_ratio = recon_std / gt_std

        # 3. RMSE
        rmse = torch.sqrt(torch.mean((x_recon_norm - x_star_norm) ** 2)).item()

        print(f"\n    [診断] Range: [{recon_min:.2f}, {recon_max:.2f}] (GT: [{gt_min:.2f}, {gt_max:.2f}])")
        print(f"    [診断] Energy Ratio: {energy_ratio:.3f} (recon_std={recon_std:.3f}, gt_std={gt_std:.3f})")
        print(f"    [診断] RMSE: {rmse:.4f}")

        # u, v, 渦度の3行でプロット
        fig = plot_velocity_and_vorticity(
            x_recon,
            title=f"Reconstructed (subsample={sub})",
            figsize=(20, 9),
            save_path=output_dir / f"sparse_sub{sub}_reconstructed.png",
        )
        plt.close(fig)
        print("Saved")


def sparse_generalization_test(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    subsample_rates: list = [2, 4, 8, 16],
) -> dict:
    """全ジオメトリ × 全サブサンプル率のスパース復元テスト

    Args:
        score: 学習済みスコアモデル
        config: モデル設定
        data_path: ベースラインデータのパス
        output_dir: 出力ディレクトリ
        subsample_rates: テストするサブサンプリングレート

    Returns:
        {
            "baseline": {"sub2": metrics, "sub4": metrics, ...},
            "y_m01": {"sub2": metrics, "sub4": metrics, ...},
            ...
        }
    """
    from sda.score import GaussianScore
    import numpy as np

    print("\n" + "=" * 60)
    print("SPARSE GENERALIZATION TEST")
    print(f"Geometries: {list(GEOMETRY_PARAMS.keys())}")
    print(f"Subsample rates: {subsample_rates}")
    print("=" * 60)

    data_root = data_path.parent
    normalizer = IBPMNormalizer()
    window = config.get("window", 16)

    # Clamped Linear Scaling パラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02

    results = {}

    for geom_name, geom_params in GEOMETRY_PARAMS.items():
        # 1. データロード
        gen_data_dir = GEOMETRY_DATA_PATHS[geom_name]
        gen_data_path = data_root / gen_data_dir

        if not gen_data_path.exists():
            print(f"\n  Skipping {geom_name}: data not found ({gen_data_path})")
            continue

        try:
            test_data = load_ibpm_data(gen_data_path, split="test")
        except Exception as e:
            print(f"\n  Skipping {geom_name}: load failed ({e})")
            continue

        n_timesteps = min(window, test_data.shape[1])
        x_star_raw = test_data[0, :n_timesteps]
        T, C, H, W = x_star_raw.shape

        x_star_norm = normalizer.normalize(x_star_raw)
        x_star_flat = x_star_norm.flatten(0, 1)

        # 2. 物理座標 → ピクセル座標
        y_physical = geom_params["y"]
        r_physical = geom_params["radius"]
        data_y_pixel = physical_to_pixel_y(y_physical, H)
        cond_radius = physical_to_pixel_radius(r_physical, H)
        cond_center = get_condition_center_for_data(data_y_pixel)

        # 3. 条件テンソル生成
        cylinder_mask = build_cylinder_mask(H, W, center=cond_center, radius=cond_radius)
        inflow_profile = build_inflow_profile(H, W, U=1.0)
        cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

        print(f"\n  {geom_name} (y={y_physical}, r={r_physical}):")

        geom_results = {}
        reconstructions = {}  # 比較プロット用に復元結果を保存
        n_obs_ref = (H // 4) * (W // 4) * T * C

        for sub in subsample_rates:
            # 4. スケーリングパラメータ
            n_obs = (H // sub) * (W // sub) * T * C
            ratio = n_obs / n_obs_ref
            std_scaled = max(BASE_STD * ratio, MIN_STD)
            gamma_scaled = max(BASE_GAMMA * (ratio**2), MIN_GAMMA)

            print(f"    sub={sub:2d}...", end=" ", flush=True)

            # 5. サブサンプリング演算子
            def A(x, s=sub):
                return x[..., ::s, ::s]

            # 6. 観測生成
            y_star = torch.normal(A(x_star_flat), BASE_STD)

            # 7. GaussianScore で復元
            sde = VPSDE(
                GaussianScore(
                    y_star,
                    A=A,
                    std=std_scaled,
                    gamma=gamma_scaled,
                    sde=VPSDE(score.kernel, shape=(), eta=0.01),
                ),
                shape=x_star_flat.shape,
                eta=0.01,
            ).cuda()

            x_recon_flat = sde.sample(
                torch.Size([1]),
                c=cond,
                steps=256,
                corrections=1,
                tau=0.5,
            ).cpu()[0]

            x_recon_norm = x_recon_flat.unflatten(0, (T, C))

            # 8. メトリクス計算
            metrics = compute_reconstruction_metrics(x_recon_norm, x_star_norm, normalizer)
            geom_results[f"sub{sub}"] = metrics

            print(f"RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

            # 9. 可視化
            x_recon = normalizer.denormalize(x_recon_norm)
            reconstructions[sub] = x_recon  # 比較プロット用に保存

            fig = plot_velocity_and_vorticity(
                x_recon,
                title=f"{geom_name} sub={sub}: RMSE={metrics['rmse']:.4f}",
                figsize=(20, 9),
                save_path=output_dir / f"sparse_gen_{geom_name}_sub{sub}.png",
            )
            plt.close(fig)

        # 10. GT比較プロットを生成
        x_gt = normalizer.denormalize(x_star_norm)
        _plot_sparse_gen_comparison(
            x_gt, reconstructions, subsample_rates, geom_name, output_dir,
            y_physical=y_physical, r_physical=r_physical
        )

        # ジオメトリのメタ情報を追加
        geom_results["y_physical"] = y_physical
        geom_results["r_physical"] = r_physical
        results[geom_name] = geom_results

    # サマリーヒートマップの生成
    _plot_sparse_gen_summary(results, subsample_rates, output_dir)

    return results


def _plot_sparse_gen_summary(results: dict, subsample_rates: list, output_dir: Path) -> None:
    """Generate summary heatmap for sparse generalization test"""
    import numpy as np
    import matplotlib.gridspec as gridspec

    geom_names = [name for name in GEOMETRY_PARAMS.keys() if name in results]
    if not geom_names:
        return

    # Human-readable geometry labels
    geom_labels = {
        "baseline": "Baseline (y=0, r=0.5)",
        "y_m01": "Y-shift -0.1",
        "y_m02": "Y-shift -0.2",
        "y_p02": "Y-shift +0.2",
        "r_04": "Small (r=0.4)",
        "r_06": "Large (r=0.6)",
        # 大変動ジオメトリ
        "r_025": "Tiny (r=0.25)",
        "r_075": "Huge (r=0.75)",
        "y_p10": "Y-shift +1.0",
        "y_m10": "Y-shift -1.0",
    }

    # Build RMSE matrix
    rmse_matrix = []
    for geom in geom_names:
        row = []
        for sub in subsample_rates:
            key = f"sub{sub}"
            if key in results[geom]:
                row.append(results[geom][key]["rmse"])
            else:
                row.append(np.nan)
        rmse_matrix.append(row)

    rmse_matrix = np.array(rmse_matrix)

    # Create figure with GridSpec for separate colorbar space
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = ax.imshow(rmse_matrix, cmap="YlOrRd", aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(subsample_rates)))
    ax.set_xticklabels([f"1/{s}\n({100.0/(s**2):.1f}%)" for s in subsample_rates], fontsize=11)
    ax.set_yticks(range(len(geom_names)))
    ax.set_yticklabels([geom_labels.get(g, g) for g in geom_names], fontsize=11)
    ax.set_xlabel("Subsample Rate (Observation %)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Geometry Configuration", fontsize=12, fontweight="bold")
    ax.set_title("Sparse Observation Reconstruction: RMSE by Geometry and Subsample Rate",
                 fontsize=14, fontweight="bold", pad=15)

    # Display values in cells
    for i in range(len(geom_names)):
        for j in range(len(subsample_rates)):
            val = rmse_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")

    # Colorbar in dedicated axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("RMSE", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(output_dir / "sparse_gen_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary heatmap saved: {output_dir / 'sparse_gen_summary.png'}")


def _plot_sparse_gen_comparison(
    x_gt: torch.Tensor,
    reconstructions: dict,
    subsample_rates: list,
    geom_name: str,
    output_dir: Path,
    y_physical: float = 0.0,
    r_physical: float = 0.5,
    timestep: int = 8,
) -> None:
    """Generate comparison plot of GT vs reconstructions at different subsample rates

    Args:
        x_gt: Ground truth data (T, C, H, W)
        reconstructions: {sub_rate: x_recon} dict (each x_recon is (T, C, H, W))
        subsample_rates: List of subsample rates
        geom_name: Geometry name
        output_dir: Output directory
        y_physical: Cylinder y-position in physical coordinates
        r_physical: Cylinder radius in physical coordinates
        timestep: Timestep to visualize
    """
    import numpy as np
    from matplotlib.patches import Circle

    T, C, H, W = x_gt.shape
    timestep = min(timestep, T - 1)

    # Convert physical coordinates to pixel coordinates
    # Physical domain: x ∈ [-4, 12], y ∈ [-4, 4]
    XOFFSET, XLENGTH = -4.0, 16.0
    YOFFSET, YLENGTH = -4.0, 8.0

    dx = XLENGTH / W
    dy = YLENGTH / H
    cyl_x_pixel = (0.0 - XOFFSET) / dx  # Cylinder at x=0
    cyl_y_pixel = (y_physical - YOFFSET) / dy  # y position
    cyl_r_pixel = r_physical / dy  # Radius in pixels

    # GT + subsample rates + colorbar = 6 columns
    import matplotlib.gridspec as gridspec
    n_cols = 1 + len(subsample_rates)
    fig = plt.figure(figsize=(3.5 * n_cols + 1, 7))
    gs = gridspec.GridSpec(2, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.3)
    axes = [[fig.add_subplot(gs[row, col]) for col in range(n_cols)] for row in range(2)]
    cbar_axes = [fig.add_subplot(gs[row, n_cols]) for row in range(2)]

    component_names = ["u (horizontal)", "v (vertical)"]

    # Determine colormap range from GT
    u_gt = x_gt[timestep, 0].numpy()
    v_gt = x_gt[timestep, 1].numpy()
    u_vmin, u_vmax = u_gt.min(), u_gt.max()
    v_vmin, v_vmax = v_gt.min(), v_gt.max()

    for row, (comp_idx, comp_name) in enumerate(zip([0, 1], component_names)):
        vmin = u_vmin if comp_idx == 0 else v_vmin
        vmax = u_vmax if comp_idx == 0 else v_vmax

        # GT column
        ax = axes[row][0]
        gt_data = x_gt[timestep, comp_idx].numpy()
        im = ax.imshow(gt_data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

        # Add cylinder
        cylinder = Circle((cyl_x_pixel, cyl_y_pixel), cyl_r_pixel,
                          fill=False, edgecolor="black", linewidth=1.5, linestyle="--")
        ax.add_patch(cylinder)

        if row == 0:
            ax.set_title("Ground Truth", fontsize=11, fontweight="bold")
        ax.set_ylabel(comp_name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Each subsample rate
        for col, sub in enumerate(subsample_rates, start=1):
            ax = axes[row][col]
            if sub in reconstructions:
                recon_data = reconstructions[sub][timestep, comp_idx].numpy()
                rmse = np.sqrt(np.mean((recon_data - gt_data) ** 2))
                im = ax.imshow(recon_data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

                # Add cylinder
                cylinder = Circle((cyl_x_pixel, cyl_y_pixel), cyl_r_pixel,
                                  fill=False, edgecolor="black", linewidth=1.5, linestyle="--")
                ax.add_patch(cylinder)

                if row == 0:
                    obs_ratio = 100.0 / (sub ** 2)
                    ax.set_title(f"1/{sub} Subsample\n({obs_ratio:.1f}% obs)\nRMSE={rmse:.3f}",
                                 fontsize=10)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"1/{sub} Subsample", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar in dedicated axes
    for row in range(2):
        vmin = u_vmin if row == 0 else v_vmin
        vmax = u_vmax if row == 0 else v_vmax
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_axes[row])
        cbar.set_label("velocity [m/s]", fontsize=10)

    # Title with geometry info
    geom_desc = {
        "baseline": "Baseline (y=0, r=0.5)",
        "y_m01": "Y-shift -0.1 (y=-0.1, r=0.5)",
        "y_m02": "Y-shift -0.2 (y=-0.2, r=0.5)",
        "y_p02": "Y-shift +0.2 (y=+0.2, r=0.5)",
        "r_04": "Small cylinder (y=0.05, r=0.4)",
        "r_06": "Large cylinder (y=0.05, r=0.6)",
    }
    title = geom_desc.get(geom_name, geom_name)
    fig.suptitle(f"Sparse Observation Reconstruction: {title}\nTimestep t={timestep}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = output_dir / f"sparse_gen_{geom_name}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Comparison plot saved: {save_path.name}")


def diffusion_trajectory(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
) -> None:
    """拡散過程の中間状態を可視化"""
    import numpy as np
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("DIFFUSION TRAJECTORY VISUALIZATION")
    print("=" * 60)

    train_data = load_ibpm_data(data_path, split="train")
    H, W = train_data.shape[-2], train_data.shape[-1]
    window = config.get("window", 16)
    shape_flat = torch.Size((window * 2, H, W))

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).cuda()

    # VPSDE でサンプリング（中間状態を記録、eta=0.01で数値安定性向上）
    sde = VPSDE(score.kernel, shape=shape_flat, eta=0.01).cuda()
    n_steps = 256
    record_steps = [0, 16, 32, 64, 128, 192, 255]

    print(f"Recording trajectory at steps: {record_steps}")

    # 初期ノイズ
    x_t = torch.randn(1, *shape_flat).cuda()
    trajectory = [(0, x_t.cpu().clone())]

    t_values = torch.linspace(1.0, 0.0, n_steps + 1).cuda()

    with torch.no_grad():
        for i in tqdm(range(n_steps), desc="Sampling"):
            t_curr = t_values[i]
            t_next = t_values[i + 1]
            dt = t_next - t_curr

            eps = score.kernel(x_t, t_curr.unsqueeze(0), cond)
            alpha_t = sde.alpha(t_curr)

            # Euler-Maruyama step (simplified)
            beta_t = 1 - alpha_t**2
            drift = -0.5 * beta_t / (1 - t_curr + 1e-5) * (x_t + eps)
            diffusion = torch.sqrt(beta_t / (1 - t_curr + 1e-5))

            if t_next > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + drift * (-dt) + diffusion * torch.sqrt(-dt) * noise
            else:
                x_t = x_t + drift * (-dt)

            if i + 1 in record_steps:
                trajectory.append((i + 1, x_t.cpu().clone()))

    # 統計表示
    print("\nTrajectory statistics:")
    for step, x in trajectory:
        t_val = 1.0 - step / n_steps
        print(f"  Step {step:3d} (t={t_val:.3f}): mean={x.mean():.4f}, std={x.std():.4f}")

    # 可視化（u, v, 渦度の3行）
    fig, axes = plt.subplots(3, len(trajectory), figsize=(20, 9))

    for col, (step, x_flat) in enumerate(trajectory):
        t_val = 1.0 - step / n_steps
        x_vis = x_flat[0].unflatten(0, (window, 2))
        w = compute_vorticity(x_vis)[0]
        w_np = w.numpy()
        u_np = x_vis[0, 0].numpy()
        v_np = x_vis[0, 1].numpy()

        # 範囲決定
        if step == 0:
            u_vmin, u_vmax = -3, 3
            v_vmin, v_vmax = -3, 3
            w_vmin, w_vmax = -3, 3
        else:
            u_low, u_high = np.percentile(u_np, [2, 98])
            u_vmax = max(abs(u_low), abs(u_high))
            u_vmin = -u_vmax

            v_low, v_high = np.percentile(v_np, [2, 98])
            v_vmax = max(abs(v_low), abs(v_high))
            v_vmin = -v_vmax

            w_low, w_high = np.percentile(w_np, [2, 98])
            w_vmax = max(abs(w_low), abs(w_high))
            w_vmin = -w_vmax

        # u velocity
        im0 = axes[0, col].imshow(u_np, cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax, origin="lower")
        axes[0, col].set_title(f"Step {step}\nt={t_val:.2f}")
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        # v velocity
        im1 = axes[1, col].imshow(v_np, cmap="RdBu_r", vmin=v_vmin, vmax=v_vmax, origin="lower")
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

        # vorticity
        im2 = axes[2, col].imshow(w_np, cmap="RdBu_r", vmin=w_vmin, vmax=w_vmax, origin="lower")
        axes[2, col].axis("off")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046)

    axes[0, 0].set_ylabel("u velocity", fontsize=12)
    axes[1, 0].set_ylabel("v velocity", fontsize=12)
    axes[2, 0].set_ylabel("vorticity", fontsize=12)

    fig.suptitle("Diffusion Trajectory: Noise → Sample", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'trajectory.png'}")


def compare_observations(
    data_path: Path,
    output_dir: Path,
    subsample_rates: list = [2, 4, 8, 16],
) -> None:
    """GT vs 各subsampleレートの観測を比較表示（u, v, 渦度の3行）"""
    print("\n" + "=" * 60)
    print("OBSERVATION COMPARISON")
    print("=" * 60)

    test_data = load_ibpm_data(data_path, split="test")
    x_star = test_data[0]
    t_show = 4  # 渦が見えやすいフレーム

    print(f"Ground truth shape: {x_star.shape}")
    print(f"Showing timestep: {t_show}")

    fig, axes = plt.subplots(3, len(subsample_rates) + 1, figsize=(20, 10))

    x_gt = x_star[t_show]
    u_gt = x_gt[0]
    v_gt = x_gt[1]
    w_gt = compute_vorticity(x_star)[t_show]

    # 対称な範囲
    u_vmax = max(abs(u_gt.min().item()), abs(u_gt.max().item()))
    v_vmax = max(abs(v_gt.min().item()), abs(v_gt.max().item()))
    w_vmax = max(abs(w_gt.min().item()), abs(w_gt.max().item()))

    # Ground truth
    im0 = axes[0, 0].imshow(u_gt.numpy(), cmap="RdBu_r", vmin=-u_vmax, vmax=u_vmax, origin="lower")
    axes[0, 0].set_title(f"Ground Truth\n{x_star.shape[2]}x{x_star.shape[3]}")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label="u")

    im1 = axes[1, 0].imshow(v_gt.numpy(), cmap="RdBu_r", vmin=-v_vmax, vmax=v_vmax, origin="lower")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, label="v")

    im2 = axes[2, 0].imshow(w_gt.numpy(), cmap="RdBu_r", vmin=-w_vmax, vmax=w_vmax, origin="lower")
    axes[2, 0].axis("off")
    plt.colorbar(im2, ax=axes[2, 0], fraction=0.046, label="ω")

    for i, sub in enumerate(subsample_rates):

        def A(x, s=sub):
            return x[..., ::s, ::s]

        y_obs = torch.normal(A(x_star), 0.1)
        y_t = y_obs[t_show]

        u_obs = y_t[0]
        v_obs = y_t[1]
        w_obs = compute_vorticity(y_obs)[t_show]

        H_sub, W_sub = y_t.shape[1], y_t.shape[2]
        n_obs = H_sub * W_sub
        pct = 100 * n_obs / (x_star.shape[2] * x_star.shape[3])

        print(f"  subsample={sub:2d}: {H_sub}x{W_sub} = {n_obs:,} pts ({pct:.1f}%)")

        im_u = axes[0, i + 1].imshow(u_obs.numpy(), cmap="RdBu_r", vmin=-u_vmax, vmax=u_vmax, origin="lower")
        axes[0, i + 1].set_title(f"sub={sub}\n{H_sub}x{W_sub} ({pct:.1f}%)")
        axes[0, i + 1].axis("off")
        plt.colorbar(im_u, ax=axes[0, i + 1], fraction=0.046, label="u")

        im_v = axes[1, i + 1].imshow(v_obs.numpy(), cmap="RdBu_r", vmin=-v_vmax, vmax=v_vmax, origin="lower")
        axes[1, i + 1].axis("off")
        plt.colorbar(im_v, ax=axes[1, i + 1], fraction=0.046, label="v")

        im_w = axes[2, i + 1].imshow(w_obs.numpy(), cmap="RdBu_r", vmin=-w_vmax, vmax=w_vmax, origin="lower")
        axes[2, i + 1].axis("off")
        plt.colorbar(im_w, ax=axes[2, i + 1], fraction=0.046, label="ω")

    axes[0, 0].set_ylabel("u velocity", fontsize=12)
    axes[1, 0].set_ylabel("v velocity", fontsize=12)
    axes[2, 0].set_ylabel("vorticity", fontsize=12)

    fig.suptitle(f"Sparse Observation Comparison (t={t_show}, noise std=0.1)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "observation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'observation_comparison.png'}")


def kolmogorov_comparison(
    data_path: Path,
    output_dir: Path,
    kolmo_run_dir: Path,
) -> None:
    """IBPMデータにKolmogorovモデルを適用して比較"""
    import torch.nn.functional as F

    from experiments.kolmogorov.utils import make_score as make_kolmo_score
    from sda.score import GaussianScore
    from sda.utils import load_config

    print("\n" + "=" * 60)
    print("KOLMOGOROV MODEL COMPARISON")
    print("=" * 60)

    # Kolmogorovモデルをロード
    if not kolmo_run_dir.exists():
        print(f"  [ERROR] Kolmogorov run not found: {kolmo_run_dir}")
        return

    kolmo_config = load_config(kolmo_run_dir)
    kolmo_score = make_kolmo_score(**kolmo_config).cuda()

    state_path = kolmo_run_dir / "state.pth"
    if not state_path.exists():
        state_path = kolmo_run_dir / "state_final.pth"

    kolmo_score.load_state_dict(torch.load(state_path, map_location="cuda"))
    kolmo_score.eval()
    print(f"  Loaded Kolmogorov model from: {kolmo_run_dir}")

    # IBPMデータをロード
    test_data = load_ibpm_data(data_path, split="test")
    x_star = test_data[0, :8]

    # 64x64にリサイズ（Kolmogorovモデルの解像度）
    x_star_64 = F.interpolate(x_star, size=(64, 64), mode="bilinear", align_corners=False)
    print(f"  Original: {x_star.shape} -> Resized: {x_star_64.shape}")

    # スパース再構成（subsample=4）
    subsample = 4

    def A_64(x):
        return x[..., ::subsample, ::subsample]

    y_star_64 = torch.normal(A_64(x_star_64), 0.1)

    # Kolmogorovモデルで再構成
    print("  Reconstructing with Kolmogorov model...")
    sde_kolmo = VPSDE(
        GaussianScore(
            y_star_64,
            A=A_64,
            std=0.1,
            sde=VPSDE(kolmo_score, shape=()),
        ),
        shape=x_star_64.shape,
    ).cuda()

    x_kolmo = sde_kolmo.sample(steps=256, corrections=1, tau=0.5).cpu()

    # 可視化
    w_gt_64 = compute_vorticity(x_star_64)
    w_kolmo = compute_vorticity(x_kolmo)

    vmax = max(abs(w_gt_64.min()), abs(w_gt_64.max()))

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))

    for t in range(5):
        im = axes[0, t].imshow(w_gt_64[t].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[0, t].set_title(f"t={t}")
        axes[0, t].axis("off")

        axes[1, t].imshow(w_kolmo[t].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("Ground Truth\n(64x64)", fontsize=12)
    axes[1, 0].set_ylabel("Kolmogorov\nReconstruction", fontsize=12)

    plt.colorbar(im, ax=axes[:, :], shrink=0.6, label="Vorticity")
    fig.suptitle(f"Kolmogorov Model on IBPM Data (sub={subsample})", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "kolmogorov_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    error = (A_64(x_kolmo) - y_star_64).std()
    print(f"  Reconstruction error: {error:.4f}")
    print(f"  Saved: {output_dir / 'kolmogorov_comparison.png'}")


def debug_model(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
) -> None:
    """モデル出力の診断"""
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)

    train_data = load_ibpm_data(data_path, split="train")
    H, W = train_data.shape[-2], train_data.shape[-1]
    window = config.get("window", 16)

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond_kernel = torch.stack([cylinder_mask, inflow_profile], dim=0).cuda()

    # Test 1: ランダムノイズ入力
    print("\n[Test 1] Random noise input:")
    x_noise = torch.randn(1, window * 2, H, W).cuda()
    t_test = torch.tensor([0.5]).cuda()
    with torch.no_grad():
        eps_out = score.kernel(x_noise, t_test, cond_kernel)
    print(f"  Input:  mean={x_noise.mean():.4f}, std={x_noise.std():.4f}")
    print(f"  Output: mean={eps_out.mean():.4f}, std={eps_out.std():.4f}")
    print(f"  Output: min={eps_out.min():.4f}, max={eps_out.max():.4f}")
    print(f"  NaN: {eps_out.isnan().any()}, Inf: {eps_out.isinf().any()}")

    # Test 2: 実データ入力
    print("\n[Test 2] Real train data input:")
    x_real = train_data[0, :window].flatten(0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        eps_real = score.kernel(x_real, t_test, cond_kernel)
    print(f"  Input:  mean={x_real.mean():.4f}, std={x_real.std():.4f}")
    print(f"  Output: mean={eps_real.mean():.4f}, std={eps_real.std():.4f}")

    # Test 3: 複数timestep
    print("\n[Test 3] Output at different timesteps:")
    for t_val in [0.01, 0.1, 0.5, 0.9, 0.99]:
        t_i = torch.tensor([t_val]).cuda()
        with torch.no_grad():
            eps_i = score.kernel(x_noise, t_i, cond_kernel)
        print(f"  t={t_val:.2f}: mean={eps_i.mean():.4f}, std={eps_i.std():.4f}")

    # 診断結果
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    if eps_out.std() < 0.1:
        print("  [WARNING] Output std is very low - model may have collapsed!")
        print("  -> Recommendation: Retrain with data normalization")
    elif eps_out.std() > 10:
        print("  [WARNING] Output std is very high - model may be unstable!")
    else:
        print("  [OK] Output statistics look reasonable")
    print("=" * 60)


# =============================================================================
# 汎化性能テスト (Generalization Tests)
# =============================================================================


def add_perturbation(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """発達済み流れ場に微小擾乱を追加

    Args:
        x: 入力テンソル
        noise_std: 追加するノイズの標準偏差

    Returns:
        擾乱を加えたテンソル
    """
    return x + torch.randn_like(x) * noise_std


def compute_reconstruction_metrics(
    x_recon_norm: torch.Tensor,
    x_gt_norm: torch.Tensor,
    normalizer: IBPMNormalizer,
) -> dict:
    """再構成結果の評価指標を計算

    Args:
        x_recon_norm: 再構成結果（正規化空間）
        x_gt_norm: Ground Truth（正規化空間）
        normalizer: 正規化用オブジェクト

    Returns:
        評価指標の辞書
    """
    # RMSE（正規化空間）
    rmse = torch.sqrt(torch.mean((x_recon_norm - x_gt_norm) ** 2)).item()

    # Energy Ratio（標準偏差の比率）
    recon_std = x_recon_norm.std().item()
    gt_std = x_gt_norm.std().item()
    energy_ratio = recon_std / gt_std if gt_std > 0 else 0.0

    # チャネル別RMSE
    u_rmse = torch.sqrt(torch.mean((x_recon_norm[:, 0] - x_gt_norm[:, 0]) ** 2)).item()
    v_rmse = torch.sqrt(torch.mean((x_recon_norm[:, 1] - x_gt_norm[:, 1]) ** 2)).item()

    # 渦度RMSE（物理空間）
    x_recon_phys = normalizer.denormalize(x_recon_norm)
    x_gt_phys = normalizer.denormalize(x_gt_norm)
    w_recon = compute_vorticity(x_recon_phys)
    w_gt = compute_vorticity(x_gt_phys)
    vorticity_rmse = torch.sqrt(torch.mean((w_recon - w_gt) ** 2)).item()

    return {
        "rmse": rmse,
        "energy_ratio": energy_ratio,
        "u_rmse": u_rmse,
        "v_rmse": v_rmse,
        "vorticity_rmse": vorticity_rmse,
        "recon_std": recon_std,
        "gt_std": gt_std,
    }


def generalization_grid_offset(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    subsample_rate: int = 4,
    offsets: list[tuple[int, int]] = None,
) -> dict:
    """グリッドオフセットによる汎化テスト

    観測グリッドの開始位置をずらして、モデルの汎化性能を評価

    Args:
        score: 学習済みスコアモデル
        config: モデル設定
        data_path: データパス
        output_dir: 出力ディレクトリ
        subsample_rate: サブサンプリングレート
        offsets: テストするオフセットのリスト [(h_offset, w_offset), ...]

    Returns:
        テスト結果の辞書
    """
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("GENERALIZATION TEST: GRID OFFSET")
    print("=" * 60)

    if offsets is None:
        # デフォルトオフセット（subsample_rate=4の場合）
        offsets = [(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)]

    normalizer = IBPMNormalizer()
    window = config.get("window", 16)

    # テストデータをロード
    test_data = load_ibpm_data(data_path, split="test")
    n_timesteps = min(window, test_data.shape[1])
    x_star_raw = test_data[0, :n_timesteps]
    T, C, H, W = x_star_raw.shape

    x_star_norm = normalizer.normalize(x_star_raw)
    x_star_flat = x_star_norm.flatten(0, 1)

    print(f"Test data shape: {x_star_raw.shape}")
    print(f"Subsample rate: {subsample_rate}")
    print(f"Testing offsets: {offsets}")

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

    # Clamped Linear Scalingパラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02
    n_obs_ref = (H // 4) * (W // 4) * T * C

    sub = subsample_rate
    n_obs = (H // sub) * (W // sub) * T * C
    ratio = n_obs / n_obs_ref
    std_scaled = max(BASE_STD * ratio, MIN_STD)
    gamma_scaled = max(BASE_GAMMA * (ratio**2), MIN_GAMMA)

    results = {}

    for offset_h, offset_w in offsets:
        key = f"offset_{offset_h}_{offset_w}"
        print(f"\n  Testing {key}...", end=" ", flush=True)

        # オフセット付きサブサンプリング演算子
        def A(x, s=sub, oh=offset_h, ow=offset_w):
            return x[..., oh::s, ow::s]

        # 観測生成
        y_star = torch.normal(A(x_star_flat), BASE_STD)

        # GaussianScoreで再構成
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=std_scaled,
                gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape,
            eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]),
            c=cond,
            steps=256,
            corrections=1,
            tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # 評価指標計算
        metrics = compute_reconstruction_metrics(x_recon_norm, x_star_norm, normalizer)
        results[key] = metrics

        print(f"RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

        # GT を保存
        fig_gt = plot_velocity_and_vorticity(
            x_star_raw,
            title=f"Ground Truth",
            figsize=(20, 9),
            save_path=output_dir / f"grid_offset_{offset_h}_{offset_w}_gt.png",
        )
        plt.close(fig_gt)

        # 再構成結果を保存
        x_recon = normalizer.denormalize(x_recon_norm)
        fig = plot_velocity_and_vorticity(
            x_recon,
            title=f"Grid Offset ({offset_h}, {offset_w}): RMSE={metrics['rmse']:.4f}",
            figsize=(20, 9),
            save_path=output_dir / f"grid_offset_{offset_h}_{offset_w}.png",
        )
        plt.close(fig)

    return results


def generalization_perturbation(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    noise_stds: list[float] = None,
    subsample_rate: int = 4,
) -> dict:
    """微小擾乱による汎化テスト

    テストデータに擾乱を加えて、モデルの頑健性を評価

    Args:
        score: 学習済みスコアモデル
        config: モデル設定
        data_path: データパス
        output_dir: 出力ディレクトリ
        noise_stds: テストするノイズ標準偏差のリスト
        subsample_rate: サブサンプリングレート

    Returns:
        テスト結果の辞書
    """
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("GENERALIZATION TEST: PERTURBATION")
    print("=" * 60)

    if noise_stds is None:
        noise_stds = [0.0, 0.01, 0.02, 0.05]

    normalizer = IBPMNormalizer()
    window = config.get("window", 16)

    # テストデータをロード
    test_data = load_ibpm_data(data_path, split="test")
    n_timesteps = min(window, test_data.shape[1])
    x_star_raw = test_data[0, :n_timesteps]
    T, C, H, W = x_star_raw.shape

    print(f"Test data shape: {x_star_raw.shape}")
    print(f"Testing noise levels: {noise_stds}")

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

    # Clamped Linear Scalingパラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02
    n_obs_ref = (H // 4) * (W // 4) * T * C

    sub = subsample_rate
    n_obs = (H // sub) * (W // sub) * T * C
    ratio = n_obs / n_obs_ref
    std_scaled = max(BASE_STD * ratio, MIN_STD)
    gamma_scaled = max(BASE_GAMMA * (ratio**2), MIN_GAMMA)

    def A(x, s=sub):
        return x[..., ::s, ::s]

    results = {}

    for noise_std in noise_stds:
        key = f"noise_{noise_std:.3f}"
        print(f"\n  Testing {key}...", end=" ", flush=True)

        # 擾乱を加える（正規化前の物理空間で）
        if noise_std > 0:
            x_perturbed_raw = add_perturbation(x_star_raw, noise_std)
        else:
            x_perturbed_raw = x_star_raw

        # 正規化してflatten
        x_perturbed_norm = normalizer.normalize(x_perturbed_raw)
        x_perturbed_flat = x_perturbed_norm.flatten(0, 1)

        # 元のGT（擾乱なし）も正規化
        x_star_norm = normalizer.normalize(x_star_raw)

        # 観測生成（擾乱を加えたデータから）
        y_star = torch.normal(A(x_perturbed_flat), BASE_STD)

        # GaussianScoreで再構成
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=std_scaled,
                gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_perturbed_flat.shape,
            eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]),
            c=cond,
            steps=256,
            corrections=1,
            tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # 評価指標計算（元のGTと比較）
        metrics = compute_reconstruction_metrics(x_recon_norm, x_star_norm, normalizer)
        results[key] = metrics

        print(f"RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

        # GT を保存
        fig_gt = plot_velocity_and_vorticity(
            x_star_raw,
            title=f"Ground Truth",
            figsize=(20, 9),
            save_path=output_dir / f"perturbation_noise_{noise_std:.3f}_gt.png",
        )
        plt.close(fig_gt)

        # 再構成結果を保存
        x_recon = normalizer.denormalize(x_recon_norm)
        fig = plot_velocity_and_vorticity(
            x_recon,
            title=f"Perturbation (noise={noise_std:.3f}): RMSE={metrics['rmse']:.4f}",
            figsize=(20, 9),
            save_path=output_dir / f"perturbation_noise_{noise_std:.3f}.png",
        )
        plt.close(fig)

    return results


def generalization_geometry(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    subsample_rate: int = 4,
) -> dict:
    """幾何条件変更による汎化テスト

    異なるジオメトリ（円柱位置・サイズ）でシミュレーションしたデータに対する
    再構成性能を評価。GEOMETRY_DATA_PATHSに定義されたデータディレクトリから
    実際のシミュレーションデータをロードする。

    Args:
        score: 学習済みスコアモデル
        config: モデル設定
        data_path: ベースラインデータのパス（親ディレクトリから汎化データを探索）
        output_dir: 出力ディレクトリ
        subsample_rate: サブサンプリングレート

    Returns:
        テスト結果の辞書
    """
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("GENERALIZATION TEST: GEOMETRY")
    print("=" * 60)

    # データディレクトリのルートを取得
    # data_path = .../ibpm_h5_wide_centered -> data_root = .../
    data_root = data_path.parent

    normalizer = IBPMNormalizer()
    window = config.get("window", 16)

    # Clamped Linear Scalingパラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02

    sub = subsample_rate

    def A(x, s=sub):
        return x[..., ::s, ::s]

    results = {}

    print(f"Testing geometry configs: {list(GEOMETRY_PARAMS.keys())}")

    for name, params in GEOMETRY_PARAMS.items():
        # 1. 対応するデータをロード
        gen_data_dir = GEOMETRY_DATA_PATHS[name]
        gen_data_path = data_root / gen_data_dir

        if not gen_data_path.exists():
            print(f"\n  Skipping {name}: data directory not found ({gen_data_path})")
            continue

        try:
            test_data = load_ibpm_data(gen_data_path, split="test")
        except Exception as e:
            print(f"\n  Skipping {name}: failed to load data ({e})")
            continue

        n_timesteps = min(window, test_data.shape[1])
        x_star_raw = test_data[0, :n_timesteps]
        T, C, H, W = x_star_raw.shape

        x_star_norm = normalizer.normalize(x_star_raw)
        x_star_flat = x_star_norm.flatten(0, 1)

        # 2. 物理座標からピクセル座標へ変換
        y_physical = params["y"]
        r_physical = params["radius"]
        data_y_pixel = physical_to_pixel_y(y_physical, H)
        cond_radius = physical_to_pixel_radius(r_physical, H)

        # 条件テンソルの中心を計算（学習時のオフセットを維持）
        cond_center = get_condition_center_for_data(data_y_pixel)

        print(f"\n  Testing {name}:")
        print(f"    Data: {gen_data_dir}, shape={x_star_raw.shape}")
        print(f"    Physical: y={y_physical}, r={r_physical}")
        print(f"    Pixel: data_y={data_y_pixel:.1f}, cond_y={cond_center[1]:.1f}, r={cond_radius:.1f}")

        # 3. スケーリングパラメータ計算
        n_obs_ref = (H // 4) * (W // 4) * T * C
        n_obs = (H // sub) * (W // sub) * T * C
        ratio = n_obs / n_obs_ref
        std_scaled = max(BASE_STD * ratio, MIN_STD)
        gamma_scaled = max(BASE_GAMMA * (ratio**2), MIN_GAMMA)

        # 4. 観測生成（このジオメトリのデータから）
        y_star = torch.normal(A(x_star_flat), BASE_STD)

        # 5. 条件テンソルを生成（実際のジオメトリパラメータを使用）
        cylinder_mask = build_cylinder_mask(H, W, center=cond_center, radius=cond_radius)
        inflow_profile = build_inflow_profile(H, W, U=1.0)
        cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

        # 6. GaussianScoreで再構成
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=std_scaled,
                gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape,
            eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]),
            c=cond,
            steps=256,
            corrections=1,
            tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # 7. 評価指標計算
        metrics = compute_reconstruction_metrics(x_recon_norm, x_star_norm, normalizer)
        metrics["data_dir"] = gen_data_dir
        metrics["y_physical"] = y_physical
        metrics["r_physical"] = r_physical
        metrics["data_y_pixel"] = data_y_pixel
        metrics["cond_center"] = cond_center
        metrics["cond_radius"] = cond_radius
        results[name] = metrics

        print(f"    RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

        # 8. GT を保存
        fig_gt = plot_velocity_and_vorticity(
            x_star_raw,
            title=f"Ground Truth ({name}: y={y_physical}, r={r_physical})",
            figsize=(20, 9),
            save_path=output_dir / f"geometry_{name}_gt.png",
        )
        plt.close(fig_gt)

        # 9. 再構成結果を保存
        x_recon = normalizer.denormalize(x_recon_norm)
        fig = plot_velocity_and_vorticity(
            x_recon,
            title=f"Geometry {name}: RMSE={metrics['rmse']:.4f}",
            figsize=(20, 9),
            save_path=output_dir / f"geometry_{name}.png",
        )
        plt.close(fig)

    return results


def generalization_reynolds(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    subsample_rate: int = 4,
) -> dict:
    """レイノルズ数変更による汎化テスト

    異なるReynolds数でシミュレーションしたデータに対する再構成性能を評価。

    注意: 条件テンソルにReynolds数情報は含まれていないため、
    モデルはRe=100で学習したデータの分布を持つ。異なるReのデータに対する
    再構成は、データ分布のずれに対するロバスト性をテストする。

    Args:
        score: 学習済みスコアモデル
        config: モデル設定
        data_path: データパス
        output_dir: 出力ディレクトリ
        subsample_rate: サブサンプリングレート

    Returns:
        テスト結果の辞書
    """
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("GENERALIZATION TEST: REYNOLDS NUMBER")
    print("=" * 60)

    # データディレクトリのルートを取得
    data_root = data_path.parent

    normalizer = IBPMNormalizer()
    window = config.get("window", 16)

    # Clamped Linear Scalingパラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02

    sub = subsample_rate

    def A(x, s=sub):
        return x[..., ::s, ::s]

    results = {}

    print(f"Testing Reynolds numbers: {list(REYNOLDS_DATA_PATHS.keys())}")
    print("  Note: Model trained on Re=100. Testing distribution shift robustness.")

    for name, data_dir in REYNOLDS_DATA_PATHS.items():
        # 1. 対応するデータをロード
        re_data_path = data_root / data_dir

        if not re_data_path.exists():
            print(f"\n  Skipping {name}: data directory not found ({re_data_path})")
            continue

        try:
            test_data = load_ibpm_data(re_data_path, split="test")
        except Exception as e:
            print(f"\n  Skipping {name}: failed to load data ({e})")
            continue

        n_timesteps = min(window, test_data.shape[1])
        x_star_raw = test_data[0, :n_timesteps]
        T, C, H, W = x_star_raw.shape

        x_star_norm = normalizer.normalize(x_star_raw)
        x_star_flat = x_star_norm.flatten(0, 1)

        # Re値を名前から抽出
        re_value = int(name.split("_")[1])

        print(f"\n  Testing {name}:")
        print(f"    Data: {data_dir}, shape={x_star_raw.shape}")
        print(f"    Reynolds number: {re_value}")

        # 2. スケーリングパラメータ計算
        n_obs_ref = (H // 4) * (W // 4) * T * C
        n_obs = (H // sub) * (W // sub) * T * C
        ratio = n_obs / n_obs_ref
        std_scaled = max(BASE_STD * ratio, MIN_STD)
        gamma_scaled = max(BASE_GAMMA * (ratio**2), MIN_GAMMA)

        # 3. 観測生成（このRe設定のデータから）
        y_star = torch.normal(A(x_star_flat), BASE_STD)

        # 4. 条件テンソルを生成（学習時のデフォルトを使用）
        cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
        inflow_profile = build_inflow_profile(H, W, U=1.0)
        cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

        # 5. GaussianScoreで再構成
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=std_scaled,
                gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape,
            eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]),
            c=cond,
            steps=256,
            corrections=1,
            tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # 6. 評価指標計算
        metrics = compute_reconstruction_metrics(x_recon_norm, x_star_norm, normalizer)
        metrics["data_dir"] = data_dir
        metrics["Re"] = re_value
        results[name] = metrics

        print(f"    RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

        # 7. GT を保存
        fig_gt = plot_velocity_and_vorticity(
            x_star_raw,
            title=f"Ground Truth (Re={re_value})",
            figsize=(20, 9),
            save_path=output_dir / f"reynolds_{name}_gt.png",
        )
        plt.close(fig_gt)

        # 8. 再構成結果を保存
        x_recon = normalizer.denormalize(x_recon_norm)
        fig = plot_velocity_and_vorticity(
            x_recon,
            title=f"Reynolds {name}: RMSE={metrics['rmse']:.4f}",
            figsize=(20, 9),
            save_path=output_dir / f"reynolds_{name}.png",
        )
        plt.close(fig)

    return results


def generalization_test(
    score: torch.nn.Module,
    config: dict,
    data_path: Path,
    output_dir: Path,
    run_dir: Path,
) -> None:
    """汎化性能テストのメインエントリポイント

    グリッドオフセット、擾乱、幾何条件、流入速度テストを実行し、結果をJSON出力
    """
    print("\n" + "=" * 60)
    print("GENERALIZATION PERFORMANCE TESTS")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_output_dir = output_dir / f"generalization_{timestamp}"
    gen_output_dir.mkdir(parents=True, exist_ok=True)

    # サブディレクトリ作成
    grid_offset_dir = gen_output_dir / "grid_offset"
    perturbation_dir = gen_output_dir / "perturbation"
    geometry_dir = gen_output_dir / "geometry"
    grid_offset_dir.mkdir(exist_ok=True)
    perturbation_dir.mkdir(exist_ok=True)
    geometry_dir.mkdir(exist_ok=True)

    all_results = {
        "model": str(run_dir.name),
        "timestamp": timestamp,
        "tests": {},
    }

    # グリッドオフセットテスト
    print("\n--- Grid Offset Tests ---")
    grid_results = generalization_grid_offset(
        score,
        config,
        data_path,
        grid_offset_dir,
        subsample_rate=4,
        offsets=[(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)],
    )
    all_results["tests"]["grid_offset"] = grid_results

    # 擾乱テスト
    print("\n--- Perturbation Tests ---")
    perturb_results = generalization_perturbation(
        score,
        config,
        data_path,
        perturbation_dir,
        noise_stds=[0.0, 0.01, 0.02, 0.05],
        subsample_rate=4,
    )
    all_results["tests"]["perturbation"] = perturb_results

    # 幾何条件テスト（異なるジオメトリのシミュレーションデータを使用）
    print("\n--- Geometry Tests (Real Generalization) ---")
    geometry_results = generalization_geometry(
        score,
        config,
        data_path,
        geometry_dir,
        subsample_rate=4,
    )
    all_results["tests"]["geometry"] = geometry_results

    # レイノルズ数テスト（異なるReのシミュレーションデータを使用）
    print("\n--- Reynolds Number Tests ---")
    reynolds_dir = gen_output_dir / "reynolds"
    reynolds_dir.mkdir(exist_ok=True)
    reynolds_results = generalization_reynolds(
        score,
        config,
        data_path,
        reynolds_dir,
        subsample_rate=4,
    )
    all_results["tests"]["reynolds"] = reynolds_results

    # 結果をJSON出力
    report_path = gen_output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {report_path}")

    # サマリー表示
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST SUMMARY")
    print("=" * 60)

    print("\nGrid Offset Results:")
    for key, metrics in grid_results.items():
        print(f"  {key}: RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

    print("\nPerturbation Results:")
    for key, metrics in perturb_results.items():
        print(f"  {key}: RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

    print("\nGeometry Results:")
    for key, metrics in geometry_results.items():
        print(f"  {key}: RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

    print("\nReynolds Number Results:")
    for key, metrics in reynolds_results.items():
        print(f"  {key}: RMSE={metrics['rmse']:.4f}, Energy={metrics['energy_ratio']:.3f}")

    print(f"\nOutput directory: {gen_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="IBPM Flow 実験の評価")
    parser.add_argument("--run-dir", type=Path, default=None, help="学習済みモデルのディレクトリ（フルパス）")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID（最新を自動検出する場合は省略）")
    parser.add_argument(
        "--data-path", type=Path, default=None, help="IBPMデータのパス（自動検出する場合は省略）"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="出力ディレクトリ（未指定時はrun-idから自動生成）")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "data", "sample", "sparse", "sparse-gen", "debug", "trajectory", "compare", "kolmogorov", "generalization"],
        help="実行モード (sparse-gen: 汎化データに対するスパース復元テスト)",
    )
    parser.add_argument(
        "--kolmo-run-dir", type=Path, default=None, help="Kolmogorovモデルのディレクトリ（kolmogorovモードで必要）"
    )
    args = parser.parse_args()

    # Resolve run directory
    if args.run_dir:
        # Full path specified
        run_dir = args.run_dir
    elif args.run_id:
        # Run ID specified
        run_dir = get_runs_dir() / "ibpm" / args.run_id
    else:
        # Auto-detect latest run
        run_dir = find_latest_ibpm_run()
        print(f"Auto-detected latest run: {run_dir.name}")

    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = get_data_dir("ibpm_h5_400x200")
        if not data_path.exists():
            # Fallback to legacy path
            data_path = Path("/home/devuser/fluid-sbi/data/ibpm_h5_400x200")

    # Update args for compatibility
    args.run_dir = run_dir
    args.data_path = data_path

    # run_idを抽出（run-dirの名前がrun_id）
    run_id = run_dir.name

    # 出力ディレクトリを決定
    # 指定されていない場合はrun_idに紐付いたディレクトリを使用
    if args.output_dir:
        base_dir = args.output_dir
    else:
        # run_id based output: results/ibpm/{run_id}/evaluate_{timestamp}/
        base_dir = get_run_results_dir("ibpm", run_id, "evaluate")
        print(f"Using run-linked output directory: {base_dir}")

    output_dirs = {
        "data": base_dir / "data",
        "sample": base_dir / "sample",
        "sparse": base_dir / "sparse",
        "sparse-gen": base_dir / "sparse_gen",
        "trajectory": base_dir / "trajectory",
        "compare": base_dir / "compare",
        "kolmogorov": base_dir / "kolmogorov",
    }

    # 必要なディレクトリを作成
    for mode_dir in output_dirs.values():
        mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_dir}")
    print(f"Run ID: {run_id}")

    # モデルロード（data/compare/kolmogorovモード以外で必要）
    score, config = None, None
    modes_without_model = ["data", "compare", "kolmogorov"]
    if args.mode not in modes_without_model:
        print(f"\nLoading model from: {args.run_dir}")
        score, config = load_trained_model(args.run_dir, device="cuda")
        print(f"Config: {config}")
        print("Model loaded successfully")

    # 実行
    if args.mode == "all" or args.mode == "data":
        visualize_data(args.data_path, output_dirs["data"])

    if args.mode == "all" or args.mode == "debug":
        if score is not None and config is not None:
            debug_model(score, config, args.data_path)

    if args.mode == "all" or args.mode == "sample":
        if score is not None and config is not None:
            unconditional_sample(score, config, args.data_path, output_dirs["sample"], n_samples=4)

    if args.mode == "all" or args.mode == "sparse":
        if score is not None and config is not None:
            sparse_reconstruction(score, config, args.data_path, output_dirs["sparse"])

    if args.mode == "sparse-gen":
        if score is not None and config is not None:
            results = sparse_generalization_test(
                score, config, args.data_path, output_dirs["sparse-gen"]
            )
            # JSON出力
            report = {
                "model": run_id,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "subsample_rates": [2, 4, 8, 16],
                "results": results,
            }
            report_path = output_dirs["sparse-gen"] / "report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=float)
            print(f"\n  Report saved: {report_path}")

    if args.mode == "all" or args.mode == "trajectory":
        if score is not None and config is not None:
            diffusion_trajectory(score, config, args.data_path, output_dirs["trajectory"])

    if args.mode == "all" or args.mode == "compare":
        compare_observations(args.data_path, output_dirs["compare"])

    if args.mode == "kolmogorov":
        if args.kolmo_run_dir is None:
            print("[ERROR] --kolmo-run-dir is required for kolmogorov mode")
        else:
            kolmogorov_comparison(args.data_path, output_dirs["kolmogorov"], args.kolmo_run_dir)

    if args.mode == "generalization":
        if score is not None and config is not None:
            generalization_test(score, config, args.data_path, base_dir, args.run_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
