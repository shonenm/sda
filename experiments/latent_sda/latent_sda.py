"""Unified interface for Latent Space Score-based Data Assimilation."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Tuple, Union

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sda.score import VPSDE, ScoreNet

from experiments.latent_sda.encoders.base import LatentEncoder
from experiments.latent_sda.encoders.pca import PCAEncoder
from experiments.latent_sda.encoders.vae import VAEEncoder
from experiments.latent_sda.latent_score import LatentGaussianScore


class LatentSDA:
    """Unified interface for latent space Score-based Data Assimilation.

    Combines:
    - Encoder: Maps high-dimensional data to latent space (PCA or VAE)
    - Score Model: Learns the score function in latent space
    - SDE: Provides forward/reverse diffusion processes
    - Data Assimilation: Guides sampling with observations

    Example:
        >>> encoder = PCAEncoder(n_components=64)
        >>> latent_sda = LatentSDA(encoder)
        >>> latent_sda.fit_encoder(X_train)
        >>> latent_sda.train_score(X_train, epochs=256)
        >>> samples = latent_sda.sample(n=10)
        >>> reconstructed = latent_sda.assimilate(y_obs, A_operator)
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        score_hidden: Tuple[int, ...] = (256, 256, 256),
        score_embedding: int = 64,
        sde_alpha: str = "cos",
        device: str = "cuda",
    ):
        """Initialize Latent SDA.

        Args:
            encoder: Latent encoder (PCAEncoder or VAEEncoder)
            score_hidden: Hidden layer sizes for ScoreNet
            score_embedding: Time embedding dimension
            sde_alpha: Noise schedule ('lin', 'cos', 'exp')
            device: Device to use
        """
        self.encoder = encoder
        self.score_hidden = score_hidden
        self.score_embedding = score_embedding
        self.sde_alpha = sde_alpha
        self.device = device

        # Will be initialized after encoder is fitted
        self._score: Optional[ScoreNet] = None
        self._sde: Optional[VPSDE] = None
        self._fitted = False

    def fit_encoder(self, X: Tensor, **kwargs) -> "LatentSDA":
        """Fit the encoder to training data.

        Args:
            X: Training data of shape (N, C, H, W) or (N, T, C, H, W)
            **kwargs: Additional arguments for encoder.fit()

        Returns:
            self
        """
        self.encoder.fit(X, **kwargs)

        # Initialize score network with correct latent dimension
        self._score = ScoreNet(
            features=self.encoder.latent_dim,
            embedding=self.score_embedding,
            hidden_features=self.score_hidden,
        )

        # Initialize SDE
        self._sde = VPSDE(
            self._score,
            shape=Size([self.encoder.latent_dim]),
            alpha=self.sde_alpha,
        )

        self._fitted = True
        return self

    def train_score(
        self,
        X: Tensor,
        epochs: int = 256,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        verbose: bool = True,
    ) -> dict:
        """Train the score model in latent space.

        Args:
            X: Training data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler: LR scheduler type
            verbose: Show progress bar

        Returns:
            Training history dict
        """
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit_encoder() first.")

        # Encode data to latent space (batch-wise to avoid OOM)
        self.encoder.eval()
        with torch.no_grad():
            # Handle temporal dimension
            if X.ndim == 5:
                N, T, C, H, W = X.shape
                X_flat = X.reshape(N * T, C, H, W)
            else:
                X_flat = X

            # Encode in batches to avoid OOM
            encode_batch_size = min(batch_size * 4, 512)
            Z_list = []
            for i in range(0, len(X_flat), encode_batch_size):
                batch = X_flat[i:i + encode_batch_size].to(self.device)
                Z_list.append(self.encoder.encode(batch).cpu())
            Z = torch.cat(Z_list, dim=0).to(self.device)

        # Move to device
        self._sde = self._sde.to(self.device)

        # DataLoader
        dataset = TensorDataset(Z)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self._score.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        if scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler == "linear":
            sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
            )
        else:
            sched = None

        # Training loop
        history = {"loss": []}
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training Score Model")

        for epoch in iterator:
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = self._sde.loss(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["loss"].append(avg_loss)

            if sched is not None:
                sched.step()

            if verbose:
                iterator.set_postfix(loss=f"{avg_loss:.4f}")

        return history

    def sample(
        self,
        n: int,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        """Generate samples via unconditional sampling.

        Args:
            n: Number of samples
            steps: Number of diffusion steps
            corrections: Langevin correction steps
            tau: Langevin step size

        Returns:
            Samples in full space of shape (n, C, H, W)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit_encoder() and train_score() first.")

        # Sample in latent space
        with torch.no_grad():
            z = self._sde.sample(
                Size([n]),
                steps=steps,
                corrections=corrections,
                tau=tau,
            )

            # Decode to full space
            x = self.encoder.decode(z)

        return x

    def assimilate(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: float = 0.1,
        gamma: float = 1e-2,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
        n_samples: int = 1,
    ) -> Tensor:
        """Data assimilation with observation guidance.

        Args:
            y: Observation in full space
            A: Observation operator
            std: Observation noise std
            gamma: Noise regularization
            steps: Diffusion steps
            corrections: Langevin corrections
            tau: Langevin step size
            n_samples: Number of posterior samples

        Returns:
            Reconstructed samples of shape (n_samples, C, H, W)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        # Create guided score module
        guided_score = LatentGaussianScore(
            y=y.to(self.device),
            A=A,
            encoder=self.encoder,
            std=std,
            sde=self._sde,
            gamma=gamma,
        ).to(self.device)

        # Create guided SDE
        guided_sde = VPSDE(
            guided_score,
            shape=Size([self.encoder.latent_dim]),
            alpha=self.sde_alpha,
        ).to(self.device)

        # Sample from posterior
        with torch.no_grad():
            z = guided_sde.sample(
                Size([n_samples]),
                steps=steps,
                corrections=corrections,
                tau=tau,
            )

            # Decode to full space
            x = self.encoder.decode(z)

        return x

    def encode(self, x: Tensor) -> Tensor:
        """Encode data to latent space."""
        return self.encoder.encode(x.to(self.device))

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to full space."""
        return self.encoder.decode(z)

    def to(self, device: str) -> "LatentSDA":
        """Move model to device."""
        self.device = device
        self.encoder = self.encoder.to(device)
        if self._sde is not None:
            self._sde = self._sde.to(device)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save model state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save encoder
        encoder_type = type(self.encoder).__name__
        self.encoder.save(path / "encoder.pt")

        # Save score model
        if self._score is not None:
            torch.save(self._score.state_dict(), path / "score.pt")

        # Save config
        config = {
            "encoder_type": encoder_type,
            "score_hidden": self.score_hidden,
            "score_embedding": self.score_embedding,
            "sde_alpha": self.sde_alpha,
            "latent_dim": self.encoder.latent_dim,
        }
        torch.save(config, path / "config.pt")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cuda") -> "LatentSDA":
        """Load model from path."""
        path = Path(path)

        # Load config
        config = torch.load(path / "config.pt", weights_only=False)

        # Load encoder
        encoder_type = config["encoder_type"]
        if encoder_type == "PCAEncoder":
            encoder = PCAEncoder.load(path / "encoder.pt")
        elif encoder_type == "VAEEncoder":
            encoder = VAEEncoder.load(path / "encoder.pt", device=device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Create instance
        latent_sda = cls(
            encoder=encoder,
            score_hidden=config["score_hidden"],
            score_embedding=config["score_embedding"],
            sde_alpha=config["sde_alpha"],
            device=device,
        )

        # Initialize and load score
        latent_sda._score = ScoreNet(
            features=config["latent_dim"],
            embedding=config["score_embedding"],
            hidden_features=config["score_hidden"],
        )
        latent_sda._score.load_state_dict(
            torch.load(path / "score.pt", map_location=device, weights_only=True)
        )

        latent_sda._sde = VPSDE(
            latent_sda._score,
            shape=Size([config["latent_dim"]]),
            alpha=config["sde_alpha"],
        ).to(device)

        latent_sda._fitted = True
        latent_sda.encoder = encoder.to(device)

        return latent_sda
