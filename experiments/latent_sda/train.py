#!/usr/bin/env python
"""Training script for Latent Space SDA.

Usage:
    # From fluid-sbi root:
    uv run python sda/experiments/latent_sda/train.py --method pca --latent-dim 64

    # Or from sda/:
    uv run python -m experiments.latent_sda.train --method pca --latent-dim 64
"""

import argparse
import sys
import time
from pathlib import Path

# Add sda/ to path for both direct execution and module execution
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import h5py
import numpy as np
import torch

from sda.paths import get_runs_dir, get_data_dir

from experiments.latent_sda.encoders.pca import PCAEncoder
from experiments.latent_sda.encoders.vae import VAEEncoder
from experiments.latent_sda.latent_sda import LatentSDA


def load_kolmogorov_data(data_dir: Path, split: str = "train") -> torch.Tensor:
    """Load Kolmogorov flow data from HDF5.

    Args:
        data_dir: Directory containing train.h5, valid.h5, test.h5
        split: 'train', 'valid', or 'test'

    Returns:
        Data tensor of shape (N, T, C, H, W)
    """
    path = data_dir / f"{split}.h5"
    with h5py.File(path, "r") as f:
        # Assume data is stored under 'x' key
        # Shape: (N, T, C, H, W) = (819, 64, 2, 64, 64)
        if "x" in f:
            data = torch.from_numpy(f["x"][:]).float()
        else:
            # Try first key
            key = list(f.keys())[0]
            data = torch.from_numpy(f[key][:]).float()

    # Ensure shape is (N, T, C, H, W)
    # Data is already in correct format: (819, 64, 2, 64, 64)
    return data


def main():
    parser = argparse.ArgumentParser(description="Train Latent SDA models")
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "vae"],
        help="Encoder method",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent space dimension",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=256,
        help="Number of training epochs for score model",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=100,
        help="Number of training epochs for VAE",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with minimal epochs",
    )
    parser.add_argument(
        "--score-hidden",
        type=str,
        default="256,256,256",
        help="Score network hidden layers (comma-separated, e.g., '512,512,512')",
    )

    args = parser.parse_args()

    # Parse score_hidden
    args.score_hidden = tuple(int(x) for x in args.score_hidden.split(","))

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Paths (auto-managed)
    from datetime import datetime

    data_dir = get_data_dir("kolmogorov_flow")
    runs_dir = get_runs_dir() / "latent_sda" / args.method

    # Run name: auto-generate if not specified
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"dim{args.latent_dim}_{timestamp}"
    output_dir = runs_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Score hidden: {args.score_hidden}")
    print(f"Epochs: {args.epochs}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")

    # Dry run settings
    if args.dry_run:
        args.epochs = 2
        args.vae_epochs = 2
        print("DRY RUN MODE")

    # Load data
    print("\nLoading data...")
    t0 = time.time()
    X_train = load_kolmogorov_data(data_dir, "train")
    X_valid = load_kolmogorov_data(data_dir, "valid")
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
    print(f"Loaded in {time.time() - t0:.2f}s")

    # Flatten temporal dimension for encoder fitting
    # (N, T, C, H, W) -> (N*T, C, H, W)
    if X_train.ndim == 5:
        N, T, C, H, W = X_train.shape
        X_train_flat = X_train.reshape(N * T, C, H, W)
    else:
        X_train_flat = X_train
        C, H, W = X_train.shape[1:]

    print(f"Flattened shape: {X_train_flat.shape}")

    # Create encoder
    print(f"\nInitializing {args.method.upper()} encoder...")
    if args.method == "pca":
        encoder = PCAEncoder(n_components=args.latent_dim, whiten=True)
    else:
        encoder = VAEEncoder(
            in_channels=C,
            latent_dim=args.latent_dim,
            hidden_channels=(32, 64, 128),
            image_size=H,
            kl_weight=1e-4,
        )

    # Create LatentSDA
    latent_sda = LatentSDA(
        encoder=encoder,
        score_hidden=args.score_hidden,
        score_embedding=64,
        sde_alpha="cos",
        device=args.device,
    )

    # Fit encoder
    print("\nFitting encoder...")
    t0 = time.time()
    if args.method == "pca":
        latent_sda.fit_encoder(X_train_flat)
        print(f"PCA explained variance: {encoder.total_explained_variance:.4f}")
    else:
        latent_sda.fit_encoder(
            X_train_flat,
            epochs=args.vae_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    print(f"Encoder fitted in {time.time() - t0:.2f}s")

    # Test reconstruction
    print("\nTesting reconstruction...")
    with torch.no_grad():
        X_test = X_train_flat[:10].to(args.device)
        X_recon = latent_sda.encoder.reconstruct(X_test)
        recon_mse = ((X_test - X_recon) ** 2).mean().item()
        print(f"Reconstruction MSE: {recon_mse:.6f}")

    # Train score model
    print("\nTraining score model...")
    t0 = time.time()
    history = latent_sda.train_score(
        X_train_flat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        scheduler="cosine",
    )
    print(f"Score training completed in {time.time() - t0:.2f}s")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    # Test sampling
    print("\nTesting sampling...")
    t0 = time.time()
    samples = latent_sda.sample(n=4, steps=64)
    print(f"Generated {samples.shape} samples in {time.time() - t0:.2f}s")

    # Save model
    print(f"\nSaving model to {output_dir}...")
    latent_sda.save(output_dir)

    # Save training history
    torch.save(history, output_dir / "history.pt")

    # Save reconstruction metrics
    metrics = {
        "reconstruction_mse": recon_mse,
        "final_loss": history["loss"][-1],
        "latent_dim": args.latent_dim,
        "method": args.method,
    }
    if args.method == "pca":
        metrics["explained_variance"] = encoder.total_explained_variance

    torch.save(metrics, output_dir / "metrics.pt")

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
