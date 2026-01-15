#!/usr/bin/env python
"""Evaluation script for Latent SDA - comparing methods.

Usage:
    # Auto-detect latest run:
    uv run python sda/experiments/latent_sda/evaluate.py

    # Specify run ID:
    uv run python sda/experiments/latent_sda/evaluate.py --run-id dim64_20260115_140100

    # Specify full checkpoint path (legacy):
    uv run python sda/experiments/latent_sda/evaluate.py --checkpoint /path/to/checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

# Add sda/ to path for both direct execution and module execution
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import csv

import numpy as np
import torch

from experiments.kolmogorov.utils import draw
from experiments.latent_sda.latent_sda import LatentSDA
from experiments.latent_sda.train import load_kolmogorov_data
from sda.mcs import KolmogorovFlow
from sda.paths import get_runs_dir, get_results_dir, get_data_dir
from sda.utils import save_eval_config


def find_latest_run(method: str) -> Path:
    """Find the latest run for a given method.

    Args:
        method: Encoder method ('pca' or 'vae')

    Returns:
        Path to the latest run directory
    """
    runs_dir = get_runs_dir() / "latent_sda" / method
    if not runs_dir.exists():
        raise ValueError(f"No runs found: {runs_dir}")

    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not runs:
        raise ValueError(f"No runs found in {runs_dir}")

    # Sort by modification time (newest first)
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def subsample_observation(x: torch.Tensor, rate: int = 4) -> tuple:
    """Create sparse observation by subsampling.

    Args:
        x: Full data of shape (C, H, W)
        rate: Subsampling rate

    Returns:
        (observation, observation_operator, mask)
    """
    C, H, W = x.shape

    # Create subsampling mask
    mask = torch.zeros(H, W, dtype=torch.bool, device=x.device)
    mask[::rate, ::rate] = True

    # Observation: subsampled + noise
    y = x[:, mask].clone()

    def A(x_hat):
        """Observation operator: subsample the field."""
        if x_hat.ndim == 4:
            # Batch: (B, C, H, W)
            return x_hat[:, :, mask]
        else:
            # Single: (C, H, W)
            return x_hat[:, mask]

    return y, A, mask


def coarsen_observation(x: torch.Tensor, factor: int = 4) -> tuple:
    """Create coarse observation by spatial averaging.

    Args:
        x: Full data of shape (C, H, W)
        factor: Coarsening factor

    Returns:
        (observation, observation_operator, mask)
    """
    C, H, W = x.shape
    H_c, W_c = H // factor, W // factor

    # Coarsen using KolmogorovFlow
    y = KolmogorovFlow.coarsen(x, factor)  # (C, H//factor, W//factor)

    def A(x_hat):
        """Observation operator: coarsen the field."""
        return KolmogorovFlow.coarsen(x_hat, factor)

    # Mask: all points in coarsened grid (for visualization)
    mask = torch.ones(H_c, W_c, dtype=torch.bool, device=x.device)

    return y, A, mask


# Observation scenarios (Kolmogorov-style)
OBS_SCENARIOS = {
    "sub": {"type": "subsample", "rate": 2, "noise": 0.1},     # スパース観測
    "coarse": {"type": "coarsen", "factor": 4, "noise": 0.1},  # 荒い観測
}


def compute_metrics(x_true: torch.Tensor, x_pred: torch.Tensor) -> dict:
    """Compute evaluation metrics.

    Args:
        x_true: Ground truth (C, H, W) or (B, C, H, W)
        x_pred: Prediction (C, H, W) or (B, C, H, W)

    Returns:
        Dict of metrics
    """
    # Handle batch dimension
    if x_true.ndim == 3:
        x_true = x_true.unsqueeze(0)
    if x_pred.ndim == 3:
        x_pred = x_pred.unsqueeze(0)

    # MSE
    mse = ((x_true - x_pred) ** 2).mean().item()

    # RMSE
    rmse = np.sqrt(mse)

    # Relative error
    rel_error = (torch.norm(x_true - x_pred) / torch.norm(x_true)).item()

    # Per-channel metrics
    channel_mse = ((x_true - x_pred) ** 2).mean(dim=(0, 2, 3)).tolist()

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_error": rel_error,
        "channel_mse": channel_mse,
    }


def compute_vorticity(x: torch.Tensor) -> torch.Tensor:
    """Compute vorticity from velocity field using KolmogorovFlow.

    Args:
        x: Velocity field (C=2, H, W) or (B, C, H, W)

    Returns:
        Vorticity field (H, W) or (B, H, W)
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        return KolmogorovFlow.vorticity(x).squeeze(0)
    return KolmogorovFlow.vorticity(x)


def plot_comparison(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    save_path: Path,
    title: str = "",
    obs_type: str = "subsample",
    obs_param: int = 2,
):
    """Plot comparison using Kolmogorov-style visualization.

    Creates a 1x3 grid:
    - Col 1: Ground truth vorticity
    - Col 2: Observation (masked for subsample, upsampled for coarse)
    - Col 3: Reconstruction vorticity

    Uses the same colormap (icefire) as kolmogorov/utils.py
    """
    from experiments.kolmogorov.utils import vorticity2rgb
    from PIL import Image

    # Compute vorticity using KolmogorovFlow
    w_true = compute_vorticity(x_true).cpu().numpy()
    w_pred = compute_vorticity(x_pred).cpu().numpy()

    # Auto-adjust color scale based on data range
    vmax = max(np.abs(w_true).max(), np.abs(w_pred).max())
    vmax = max(vmax, 0.5)  # Minimum range to avoid division issues

    # Create observation visualization
    H, W = w_true.shape
    if obs_type == "subsample":
        # Subsample: show observed points, mask others with gray
        w_obs = w_true.copy()
        mask_np = mask.cpu().numpy()
        non_obs_mask = ~mask_np  # True where no observation
    else:
        # Coarse: upsample the coarse observation for display
        # y_obs is (C, H//factor, W//factor), compute vorticity and upsample
        y_obs_full = y_obs.unsqueeze(0)  # (1, C, H_c, W_c)
        w_obs_coarse = KolmogorovFlow.vorticity(y_obs_full).squeeze(0).cpu().numpy()
        # Upsample using nearest neighbor
        from scipy.ndimage import zoom as scipy_zoom
        w_obs = scipy_zoom(w_obs_coarse, obs_param, order=0)  # nearest neighbor
        non_obs_mask = None

    # Convert to RGB images
    rgb_true = vorticity2rgb(w_true, vmin=-vmax, vmax=vmax)
    rgb_obs = vorticity2rgb(w_obs, vmin=-vmax, vmax=vmax)
    rgb_pred = vorticity2rgb(w_pred, vmin=-vmax, vmax=vmax)

    # Apply gray mask to observation image for non-observed points
    if non_obs_mask is not None:
        gray_value = 240  # Light gray
        rgb_obs[non_obs_mask] = gray_value

    # Create grid with padding
    pad = 4
    zoom = 2
    n_cols = 3
    canvas_w = n_cols * (W * zoom + pad) + pad
    canvas_h = H * zoom + 2 * pad

    img = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # Paste each image
    for i, rgb in enumerate([rgb_true, rgb_obs, rgb_pred]):
        # Zoom
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((W * zoom, H * zoom), Image.NEAREST)
        # Paste
        offset = (i * (W * zoom + pad) + pad, pad)
        img.paste(pil_img, offset)

    # Save as PNG
    img.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent SDA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Full path to checkpoint (optional, uses latest if not specified)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (uses latest if not specified)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "vae"],
        help="Encoder method for auto-detection",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of posterior samples for DA",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Diffusion steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint:
        # Full path specified
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            checkpoint_dir = project_root / checkpoint_dir
    elif args.run_id:
        # Run ID specified
        checkpoint_dir = get_runs_dir() / "latent_sda" / args.method / args.run_id
    else:
        # Auto-detect latest run
        checkpoint_dir = find_latest_run(args.method)
        print(f"Auto-detected latest run: {checkpoint_dir.name}")

    # Data directory
    data_dir = get_data_dir("kolmogorov_flow")

    # Output to results/ (include method in path)
    output_dir = get_results_dir("latent_sda") / args.method / checkpoint_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {checkpoint_dir}")
    print(f"Device: {args.device}")

    # Load model
    latent_sda = LatentSDA.load(checkpoint_dir, device=args.device)
    latent_sda.encoder.eval()

    # Save evaluation config
    eval_config = {
        "run_id": checkpoint_dir.name,
        "method": args.method,
        "checkpoint_dir": str(checkpoint_dir),
        "n_samples": args.n_samples,
        "steps": args.steps,
        "device": args.device,
        "latent_dim": latent_sda.encoder.latent_dim,
        "observation_scenarios": OBS_SCENARIOS,
    }
    save_eval_config(eval_config, output_dir)

    # Load test data
    print("\nLoading test data...")
    X_test = load_kolmogorov_data(data_dir, "test")
    print(f"Test data shape: {X_test.shape}")

    # Flatten if needed
    if X_test.ndim == 5:
        N, T, C, H, W = X_test.shape
        X_test_flat = X_test.reshape(N * T, C, H, W)
    else:
        X_test_flat = X_test

    # Select test samples
    n_eval = min(10, len(X_test_flat))
    X_eval = X_test_flat[:n_eval].to(args.device)

    # Evaluation metrics storage
    results = {
        "reconstruction": [],
        "assimilation": [],
        "timing": {},
    }

    # 1. Reconstruction quality
    print("\n--- Reconstruction Evaluation ---")
    t0 = time.time()
    with torch.no_grad():
        X_recon = latent_sda.encoder.reconstruct(X_eval)
    results["timing"]["reconstruction"] = time.time() - t0

    for i in range(n_eval):
        metrics = compute_metrics(X_eval[i], X_recon[i])
        results["reconstruction"].append(metrics)

    avg_recon_mse = np.mean([r["mse"] for r in results["reconstruction"]])
    avg_recon_rmse = np.mean([r["rmse"] for r in results["reconstruction"]])
    print(f"Avg Reconstruction MSE: {avg_recon_mse:.6f}")
    print(f"Avg Reconstruction RMSE: {avg_recon_rmse:.6f}")

    # 2. Data assimilation (multiple observation scenarios)
    print("\n--- Data Assimilation Evaluation ---")

    # Store results per scenario
    results["assimilation"] = {scenario: [] for scenario in OBS_SCENARIOS}

    t0 = time.time()
    n_da_samples = min(3, n_eval)  # Limit DA evaluation (expensive)

    for scenario_name, params in OBS_SCENARIOS.items():
        print(f"\n=== {scenario_name.upper()} Observation ===")
        if params["type"] == "subsample":
            print(f"  Type: subsample, Rate: {params['rate']}x, Noise: {params['noise']}")
        else:
            print(f"  Type: coarsen, Factor: {params['factor']}x, Noise: {params['noise']}")

        for i in range(n_da_samples):
            print(f"\n  Sample {i+1}:")
            x_true = X_eval[i]

            # Create observation based on scenario
            if params["type"] == "subsample":
                y_obs, A, mask = subsample_observation(x_true, rate=params["rate"])
            else:  # coarsen
                y_obs, A, mask = coarsen_observation(x_true, factor=params["factor"])

            # Add noise
            y_obs = y_obs + params["noise"] * torch.randn_like(y_obs)

            # Run data assimilation
            t_da = time.time()
            x_pred = latent_sda.assimilate(
                y=y_obs,
                A=A,
                std=params["noise"],
                steps=args.steps,
                n_samples=args.n_samples,
            )
            da_time = time.time() - t_da

            # Use mean of posterior samples
            x_pred_mean = x_pred.mean(dim=0)

            # Compute metrics
            metrics = compute_metrics(x_true, x_pred_mean)
            metrics["da_time"] = da_time
            metrics["scenario"] = scenario_name
            metrics["type"] = params["type"]
            metrics["param"] = params.get("rate", params.get("factor"))
            metrics["noise"] = params["noise"]
            results["assimilation"][scenario_name].append(metrics)

            print(f"    MSE: {metrics['mse']:.6f}, RMSE: {metrics['rmse']:.6f}")
            print(f"    DA time: {da_time:.2f}s")

            # Plot comparison
            plot_comparison(
                x_true,
                x_pred_mean,
                y_obs,
                mask,
                output_dir / f"da_{scenario_name}_{i}.png",
                title=f"DA ({scenario_name}) - Sample {i+1}",
                obs_type=params["type"],
                obs_param=params.get("rate", params.get("factor")),
            )

        # Print scenario summary
        scenario_results = results["assimilation"][scenario_name]
        if scenario_results:
            avg_mse = np.mean([r["mse"] for r in scenario_results])
            avg_rmse = np.mean([r["rmse"] for r in scenario_results])
            print(f"\n  {scenario_name} Avg MSE: {avg_mse:.6f}, RMSE: {avg_rmse:.6f}")

    results["timing"]["total_da"] = time.time() - t0

    # Save results (PyTorch format)
    torch.save(results, output_dir / "results.pt")

    # Save results (CSV format)
    csv_path = output_dir / "stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "sample", "method", "scenario", "type", "param", "noise",
            "mse", "rmse", "relative_error", "da_time"
        ])
        # Data assimilation results (all scenarios)
        method_name = checkpoint_dir.name
        for scenario_name, scenario_results in results["assimilation"].items():
            for i, r in enumerate(scenario_results):
                writer.writerow([
                    i, method_name, scenario_name, r["type"], r["param"], r["noise"],
                    f"{r['mse']:.6f}", f"{r['rmse']:.6f}",
                    f"{r['relative_error']:.6f}", f"{r['da_time']:.2f}"
                ])

    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.pt (PyTorch format)")
    print(f"  - stats.csv (CSV format)")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Method: {checkpoint_dir.name}")
    print(f"Latent dim: {latent_sda.encoder.latent_dim}")
    print(f"Reconstruction MSE: {avg_recon_mse:.6f}")
    for scenario_name, scenario_results in results["assimilation"].items():
        if scenario_results:
            avg_mse = np.mean([r["mse"] for r in scenario_results])
            print(f"DA MSE ({scenario_name}): {avg_mse:.6f}")


if __name__ == "__main__":
    main()
