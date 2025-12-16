#!/usr/bin/env python3
"""IBPM Flow 実験の評価スクリプト

Usage:
    python experiments/ibpm/evaluate.py --run-dir runs/ibpm_vpsde_xxx --mode all
    python experiments/ibpm/evaluate.py --run-dir runs/ibpm_vpsde_xxx --mode data
    python experiments/ibpm/evaluate.py --run-dir runs/ibpm_vpsde_xxx --mode sample
    python experiments/ibpm/evaluate.py --run-dir runs/ibpm_vpsde_xxx --mode sparse
    python experiments/ibpm/evaluate.py --run-dir runs/ibpm_vpsde_xxx --mode debug
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import torch

from pathlib import Path

from sda.data.ibpm_dataset import build_cylinder_mask, build_inflow_profile, IBPMDataset, IBPMNormalizer
from sda.score import VPSDE

from experiments.ibpm.utils import (
    compute_vorticity,
    load_ibpm_data,
    load_trained_model,
    plot_vorticity,
    reconstruct_sparse,
)


def visualize_data(data_path: Path, output_dir: Path) -> None:
    """Train/Testデータの可視化"""
    print("=" * 60)
    print("DATA VISUALIZATION")
    print("=" * 60)

    # Train data
    train_data = load_ibpm_data(data_path, split='train')
    print(f"Train data shape: {train_data.shape}")
    print(f"  Samples: {train_data.shape[0]}, Timesteps: {train_data.shape[1]}")
    print(f"  Resolution: {train_data.shape[3]}x{train_data.shape[4]}")
    print(f"  Range: [{train_data.min():.3f}, {train_data.max():.3f}]")

    # 複数サンプルのt=0を可視化
    sample_indices = [0, 10, 20, 30, 40, min(41, train_data.shape[0]-1)]
    frames = [train_data[i, 0] for i in sample_indices if i < train_data.shape[0]]
    x_train = torch.stack(frames)
    w_train = compute_vorticity(x_train)

    fig = plot_vorticity(
        w_train,
        title='Train Data: Different samples at t=0',
        figsize=(20, 3),
        save_path=output_dir / 'data_train_samples.png',
    )
    plt.close(fig)
    print(f"  Saved: {output_dir / 'data_train_samples.png'}")

    # 速度分布
    u_train = train_data[:, :, 0].flatten()
    v_train = train_data[:, :, 1].flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(u_train.numpy(), bins=100, alpha=0.7, color='blue')
    axes[0].set_xlabel('u velocity')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'u distribution (mean={u_train.mean():.3f}, std={u_train.std():.3f})')
    axes[0].grid(alpha=0.3)

    axes[1].hist(v_train.numpy(), bins=100, alpha=0.7, color='red')
    axes[1].set_xlabel('v velocity')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'v distribution (mean={v_train.mean():.3f}, std={v_train.std():.3f})')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'data_velocity_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'data_velocity_stats.png'}")

    # Test data
    test_path = data_path / 'test.h5'
    if test_path.exists():
        test_data = load_ibpm_data(data_path, split='test')
        print(f"\nTest data shape: {test_data.shape}")

        for idx in range(min(3, test_data.shape[0])):
            x_test = test_data[idx, :8]
            w_test = compute_vorticity(x_test)
            fig = plot_vorticity(
                w_test,
                title=f'Test Sample {idx}: Vorticity Evolution',
                figsize=(20, 3),
                save_path=output_dir / f'data_test_sample_{idx}.png',
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
    window = config.get('window', 5)
    ds = IBPMDataset(str(data_path / 'train.h5'), time_window=window, normalize=True)
    x_sample, c_sample, _ = ds[0]

    H, W = x_sample.shape[-2], x_sample.shape[-1]
    T = x_sample.shape[0]
    C = x_sample.shape[1]

    c_batch = c_sample.unsqueeze(0).cuda()
    shape_flat = torch.Size((T * C, H, W))

    print(f"Sampling shape: {shape_flat}")
    print(f"Condition shape: {c_batch.shape}")

    # VPSDE でサンプリング
    sde = VPSDE(score.kernel, shape=shape_flat).cuda()

    for i in range(n_samples):
        print(f"  Generating sample {i+1}/{n_samples}...")
        x_sampled = sde.sample(torch.Size([1]), c=c_batch, steps=256, corrections=1).cpu()
        x_vis = x_sampled[0].unflatten(0, (T, C))

        # 逆正規化して元のスケールに戻す
        x_vis_denorm = normalizer.denormalize(x_vis)
        w_vis = compute_vorticity(x_vis_denorm)

        fig = plot_vorticity(
            w_vis,
            title=f'Unconditional Sample {i+1}',
            figsize=(20, 3),
            save_path=output_dir / f'sample_uncond_{i+1}.png',
        )
        plt.close(fig)
        print(f"    Saved: {output_dir / f'sample_uncond_{i+1}.png'}")

    # 統計比較（正規化後）
    train_flat = x_sample.flatten(0, 1)
    print(f"\n  Train stats (normalized):  mean={train_flat.mean():.4f}, std={train_flat.std():.4f}")
    print(f"  Sample stats (normalized): mean={x_sampled.mean():.4f}, std={x_sampled.std():.4f}")


def sparse_reconstruction(
    score: torch.nn.Module,
    data_path: Path,
    output_dir: Path,
    subsample_rates: list = [2, 4, 8, 16],
) -> None:
    """スパース観測からの再構成"""
    print("\n" + "=" * 60)
    print("SPARSE OBSERVATION RECONSTRUCTION")
    print("=" * 60)

    # 正規化用のNormalizer
    normalizer = IBPMNormalizer()

    # テストデータをロード（生データ）
    test_data = load_ibpm_data(data_path, split='test')
    n_timesteps = min(8, test_data.shape[1])
    x_star_raw = test_data[0, :n_timesteps]
    print(f"Ground truth shape: {x_star_raw.shape}")

    # 正規化（モデルは正規化空間で動作）
    x_star_norm = normalizer.normalize(x_star_raw)

    # 幾何条件を生成
    H, W = x_star_raw.shape[-2], x_star_raw.shape[-1]
    cylinder_mask = build_cylinder_mask(H, W, center=(100.0, 100.0), radius=12.5)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

    # Ground truth 可視化（生データで）
    w_star = compute_vorticity(x_star_raw)
    fig = plot_vorticity(
        w_star,
        title='Ground Truth Vorticity',
        figsize=(20, 3),
        save_path=output_dir / 'sparse_ground_truth.png',
    )
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sparse_ground_truth.png'}")

    # 各subsampleレートで再構成（正規化空間で）
    print(f"\nReconstructing with subsample rates: {subsample_rates}")
    results = reconstruct_sparse(
        x_star_norm,  # 正規化済みデータを渡す
        score,
        cond,
        subsample_rates=subsample_rates,
        noise_std=0.1,
        steps=256,
        corrections=1,
        tau=0.5,
    )

    for sub, x_recon_norm in results.items():
        # 逆正規化して可視化
        x_recon = normalizer.denormalize(x_recon_norm)
        w_recon = compute_vorticity(x_recon)
        fig = plot_vorticity(
            w_recon,
            title=f'Reconstructed (subsample={sub})',
            figsize=(20, 3),
            save_path=output_dir / f'sparse_sub{sub}_reconstructed.png',
        )
        plt.close(fig)

        def A(x, s=sub):
            return x[..., ::s, ::s]

        # エラー計算は正規化空間で
        y_obs = torch.normal(A(x_star_norm), 0.1)
        error = (A(x_recon_norm) - y_obs).std()
        print(f"  subsample={sub:2d}: error={error:.4f} (should be ~0.1)")
        print(f"    Saved: {output_dir / f'sparse_sub{sub}_reconstructed.png'}")


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

    train_data = load_ibpm_data(data_path, split='train')
    H, W = train_data.shape[-2], train_data.shape[-1]
    window = config.get('window', 5)
    shape_flat = torch.Size((window * 2, H, W))

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=(100.0, 100.0), radius=12.5)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).cuda()

    # VPSDE でサンプリング（中間状態を記録）
    sde = VPSDE(score.kernel, shape=shape_flat).cuda()
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
            beta_t = 1 - alpha_t ** 2
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

    # 可視化
    fig, axes = plt.subplots(2, len(trajectory), figsize=(20, 6))

    for col, (step, x_flat) in enumerate(trajectory):
        t_val = 1.0 - step / n_steps
        x_vis = x_flat[0].unflatten(0, (window, 2))
        w = compute_vorticity(x_vis)[0]
        w_np = w.numpy()

        if step == 0:
            vmin, vmax = -3, 3
        else:
            low, high = np.percentile(w_np, [2, 98])
            vmax = max(abs(low), abs(high))
            vmin = -vmax

        im0 = axes[0, col].imshow(w_np, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, col].set_title(f'Step {step}\nt={t_val:.2f}')
        axes[0, col].axis('off')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        u = x_vis[0, 0].numpy()
        if step == 0:
            u_vmin, u_vmax = -3, 3
        else:
            u_low, u_high = np.percentile(u, [2, 98])
            u_vmax = max(abs(u_low), abs(u_high))
            u_vmin = -u_vmax

        im1 = axes[1, col].imshow(u, cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax, origin='lower')
        axes[1, col].axis('off')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel('Vorticity', fontsize=12)
    axes[1, 0].set_ylabel('u velocity', fontsize=12)

    fig.suptitle('Diffusion Trajectory: Noise → Sample', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'trajectory.png'}")


def compare_observations(
    data_path: Path,
    output_dir: Path,
    subsample_rates: list = [2, 4, 8, 16],
) -> None:
    """GT vs 各subsampleレートの観測を比較表示"""
    print("\n" + "=" * 60)
    print("OBSERVATION COMPARISON")
    print("=" * 60)

    test_data = load_ibpm_data(data_path, split='test')
    x_star = test_data[0]
    t_show = 4  # 渦が見えやすいフレーム

    print(f"Ground truth shape: {x_star.shape}")
    print(f"Showing timestep: {t_show}")

    fig, axes = plt.subplots(2, len(subsample_rates) + 1, figsize=(20, 8))

    x_gt = x_star[t_show]
    vel_mag_gt = torch.sqrt(x_gt[0]**2 + x_gt[1]**2)
    w_gt = compute_vorticity(x_star)[t_show]

    vel_vmax = vel_mag_gt.max().item()
    w_vmax = max(abs(w_gt.min().item()), abs(w_gt.max().item()))

    # Ground truth
    im0 = axes[0, 0].imshow(vel_mag_gt.numpy(), cmap='viridis', vmin=0, vmax=vel_vmax, origin='lower')
    axes[0, 0].set_title(f'Ground Truth\n{x_star.shape[2]}x{x_star.shape[3]}')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label='|v|')

    im1 = axes[1, 0].imshow(w_gt.numpy(), cmap='RdBu_r', vmin=-w_vmax, vmax=w_vmax, origin='lower')
    axes[1, 0].set_title('Vorticity')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, label='ω')

    for i, sub in enumerate(subsample_rates):
        def A(x, s=sub):
            return x[..., ::s, ::s]

        y_obs = torch.normal(A(x_star), 0.1)
        y_t = y_obs[t_show]

        vel_mag_obs = torch.sqrt(y_t[0]**2 + y_t[1]**2)
        w_obs = compute_vorticity(y_obs)[t_show]

        H_sub, W_sub = y_t.shape[1], y_t.shape[2]
        n_obs = H_sub * W_sub
        pct = 100 * n_obs / (x_star.shape[2] * x_star.shape[3])

        print(f"  subsample={sub:2d}: {H_sub}x{W_sub} = {n_obs:,} pts ({pct:.1f}%)")

        im_vel = axes[0, i+1].imshow(vel_mag_obs.numpy(), cmap='viridis', vmin=0, vmax=vel_vmax, origin='lower')
        axes[0, i+1].set_title(f'sub={sub}\n{H_sub}x{W_sub} ({pct:.1f}%)')
        axes[0, i+1].axis('off')
        plt.colorbar(im_vel, ax=axes[0, i+1], fraction=0.046, label='|v|')

        im_w = axes[1, i+1].imshow(w_obs.numpy(), cmap='RdBu_r', vmin=-w_vmax, vmax=w_vmax, origin='lower')
        axes[1, i+1].axis('off')
        plt.colorbar(im_w, ax=axes[1, i+1], fraction=0.046, label='ω')

    axes[0, 0].set_ylabel('Velocity Magnitude', fontsize=12)
    axes[1, 0].set_ylabel('Vorticity', fontsize=12)

    fig.suptitle(f'Sparse Observation Comparison (t={t_show}, noise std=0.1)', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'observation_comparison.png', dpi=150, bbox_inches='tight')
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
    from sda.utils import load_config
    from sda.score import GaussianScore

    print("\n" + "=" * 60)
    print("KOLMOGOROV MODEL COMPARISON")
    print("=" * 60)

    # Kolmogorovモデルをロード
    if not kolmo_run_dir.exists():
        print(f"  [ERROR] Kolmogorov run not found: {kolmo_run_dir}")
        return

    kolmo_config = load_config(kolmo_run_dir)
    kolmo_score = make_kolmo_score(**kolmo_config).cuda()

    state_path = kolmo_run_dir / 'state.pth'
    if not state_path.exists():
        state_path = kolmo_run_dir / 'state_final.pth'

    kolmo_score.load_state_dict(torch.load(state_path, map_location='cuda'))
    kolmo_score.eval()
    print(f"  Loaded Kolmogorov model from: {kolmo_run_dir}")

    # IBPMデータをロード
    test_data = load_ibpm_data(data_path, split='test')
    x_star = test_data[0, :8]

    # 64x64にリサイズ（Kolmogorovモデルの解像度）
    x_star_64 = F.interpolate(x_star, size=(64, 64), mode='bilinear', align_corners=False)
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
        im = axes[0, t].imshow(w_gt_64[t].numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(w_kolmo[t].numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('Ground Truth\n(64x64)', fontsize=12)
    axes[1, 0].set_ylabel('Kolmogorov\nReconstruction', fontsize=12)

    plt.colorbar(im, ax=axes[:, :], shrink=0.6, label='Vorticity')
    fig.suptitle(f'Kolmogorov Model on IBPM Data (sub={subsample})', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'kolmogorov_comparison.png', dpi=150, bbox_inches='tight')
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

    train_data = load_ibpm_data(data_path, split='train')
    H, W = train_data.shape[-2], train_data.shape[-1]
    window = config.get('window', 5)

    # 幾何条件
    cylinder_mask = build_cylinder_mask(H, W, center=(100.0, 100.0), radius=12.5)
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


def main():
    # デフォルト出力先: プロジェクト内 results/ibpm/evaluate/
    default_output = Path(__file__).parent.parent.parent / 'results' / 'ibpm' / 'evaluate'

    parser = argparse.ArgumentParser(description='IBPM Flow 実験の評価')
    parser.add_argument('--run-dir', type=Path, required=True, help='学習済みモデルのディレクトリ')
    parser.add_argument('--data-path', type=Path, default=Path('/workspace/data/ibpm_h5_wide_perturbed'),
                        help='IBPMデータのパス')
    parser.add_argument('--output-dir', type=Path, default=default_output,
                        help='出力ディレクトリ')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'data', 'sample', 'sparse', 'debug', 'trajectory', 'compare', 'kolmogorov'],
                        help='実行モード')
    parser.add_argument('--kolmo-run-dir', type=Path, default=None,
                        help='Kolmogorovモデルのディレクトリ（kolmogorovモードで必要）')
    args = parser.parse_args()

    # モード別出力ディレクトリを設定
    base_dir = args.output_dir
    output_dirs = {
        'data': base_dir / 'data',
        'sample': base_dir / 'sample',
        'sparse': base_dir / 'sparse',
        'trajectory': base_dir / 'trajectory',
        'compare': base_dir / 'compare',
        'kolmogorov': base_dir / 'kolmogorov',
    }

    # 必要なディレクトリを作成
    for mode_dir in output_dirs.values():
        mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_dir}")

    # モデルロード（data/compare/kolmogorovモード以外で必要）
    score, config = None, None
    modes_without_model = ['data', 'compare', 'kolmogorov']
    if args.mode not in modes_without_model:
        print(f"\nLoading model from: {args.run_dir}")
        score, config = load_trained_model(args.run_dir, device='cuda')
        print(f"Config: {config}")
        print("Model loaded successfully")

    # 実行
    if args.mode == 'all' or args.mode == 'data':
        visualize_data(args.data_path, output_dirs['data'])

    if args.mode == 'all' or args.mode == 'debug':
        if score is not None and config is not None:
            debug_model(score, config, args.data_path)

    if args.mode == 'all' or args.mode == 'sample':
        if score is not None and config is not None:
            unconditional_sample(score, config, args.data_path, output_dirs['sample'], n_samples=4)

    if args.mode == 'all' or args.mode == 'sparse':
        if score is not None:
            sparse_reconstruction(score, args.data_path, output_dirs['sparse'])

    if args.mode == 'all' or args.mode == 'trajectory':
        if score is not None and config is not None:
            diffusion_trajectory(score, config, args.data_path, output_dirs['trajectory'])

    if args.mode == 'all' or args.mode == 'compare':
        compare_observations(args.data_path, output_dirs['compare'])

    if args.mode == 'kolmogorov':
        if args.kolmo_run_dir is None:
            print("[ERROR] --kolmo-run-dir is required for kolmogorov mode")
        else:
            kolmogorov_comparison(args.data_path, output_dirs['kolmogorov'], args.kolmo_run_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
