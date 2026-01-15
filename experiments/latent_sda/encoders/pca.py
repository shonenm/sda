"""PCA-based encoder for latent space compression."""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from experiments.latent_sda.encoders.base import LatentEncoder


class PCAEncoder(LatentEncoder):
    """PCA-based dimensionality reduction for velocity fields.

    Uses sklearn PCA for efficient SVD-based compression.
    Supports whitening for normalized latent space (recommended for score matching).

    Attributes:
        n_components: Number of principal components (latent dimension)
        whiten: If True, normalize latent to unit variance
    """

    def __init__(
        self,
        n_components: int = 64,
        whiten: bool = True,
        random_state: int = 42,
    ):
        """Initialize PCA encoder.

        Args:
            n_components: Number of principal components
            whiten: Whether to normalize latent space to unit variance
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self._pca: Optional[PCA] = None
        self._data_shape: Optional[Tuple[int, ...]] = None
        self._device: torch.device = torch.device("cpu")

        # PyTorch tensors for GPU-accelerated encode/decode
        self._mean: Optional[Tensor] = None
        self._components: Optional[Tensor] = None
        self._explained_variance: Optional[Tensor] = None

    def fit(self, X: Tensor, **kwargs) -> "PCAEncoder":
        """Fit PCA to training data.

        Args:
            X: Training data of shape (N, C, H, W) or (N, T, C, H, W)
               If 5D, time dimension is flattened into batch.

        Returns:
            self
        """
        # Handle temporal data by flattening time into batch
        if X.ndim == 5:
            N, T, C, H, W = X.shape
            X = X.reshape(N * T, C, H, W)
            self._data_shape = (C, H, W)
        elif X.ndim == 4:
            self._data_shape = X.shape[1:]  # (C, H, W)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {X.ndim}D")

        # Flatten spatial dimensions: (N, C*H*W)
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()

        # Fit sklearn PCA
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self._pca.fit(X_flat)

        # Store as PyTorch tensors for GPU support
        self._mean = torch.from_numpy(self._pca.mean_).float()
        self._components = torch.from_numpy(self._pca.components_).float()
        self._explained_variance = torch.from_numpy(
            self._pca.explained_variance_
        ).float()

        return self

    def encode(self, x: Tensor) -> Tensor:
        """Encode data to latent space using PCA projection.

        Args:
            x: Input data of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Latent representation of shape (B, latent_dim) or (B, T, latent_dim)
        """
        if self._pca is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        # Handle temporal dimension
        temporal = x.ndim == 5
        if temporal:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        # Flatten spatial: (B, C*H*W)
        x_flat = x.reshape(x.shape[0], -1)

        # Move parameters to same device
        mean = self._mean.to(x.device)
        components = self._components.to(x.device)

        # PCA transform: z = (x - mean) @ components.T
        z = (x_flat - mean) @ components.T

        # Apply whitening scaling if enabled
        if self.whiten:
            explained_var = self._explained_variance.to(x.device)
            z = z / torch.sqrt(explained_var + 1e-8)

        # Restore temporal dimension
        if temporal:
            z = z.reshape(B, T, -1)

        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation back to data space.

        Args:
            z: Latent of shape (B, latent_dim) or (B, T, latent_dim)

        Returns:
            Reconstructed data of shape (B, C, H, W) or (B, T, C, H, W)
        """
        if self._pca is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        # Handle temporal dimension
        temporal = z.ndim == 3
        if temporal:
            B, T, D = z.shape
            z = z.reshape(B * T, D)

        # Move parameters to same device
        mean = self._mean.to(z.device)
        components = self._components.to(z.device)

        # Undo whitening if applied
        if self.whiten:
            explained_var = self._explained_variance.to(z.device)
            z = z * torch.sqrt(explained_var + 1e-8)

        # PCA inverse transform: x = z @ components + mean
        x_flat = z @ components + mean

        # Reshape to spatial: (B, C, H, W)
        x = x_flat.reshape(-1, *self._data_shape)

        # Restore temporal dimension
        if temporal:
            x = x.reshape(B, T, *self._data_shape)

        return x

    @property
    def latent_dim(self) -> int:
        """Return latent dimensionality."""
        return self.n_components

    @property
    def data_shape(self) -> Tuple[int, ...]:
        """Return original data shape (C, H, W)."""
        if self._data_shape is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        return self._data_shape

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        if self._pca is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        return self._pca.explained_variance_ratio_

    @property
    def total_explained_variance(self) -> float:
        """Return total explained variance (sum of ratios)."""
        return float(self.explained_variance_ratio.sum())

    def to(self, device: torch.device) -> "PCAEncoder":
        """Move encoder parameters to device."""
        self._device = device
        if self._mean is not None:
            self._mean = self._mean.to(device)
            self._components = self._components.to(device)
            self._explained_variance = self._explained_variance.to(device)
        return self

    def save(self, path: str) -> None:
        """Save encoder state to file."""
        state = {
            "n_components": self.n_components,
            "whiten": self.whiten,
            "random_state": self.random_state,
            "data_shape": self._data_shape,
            "mean": self._mean,
            "components": self._components,
            "explained_variance": self._explained_variance,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "PCAEncoder":
        """Load encoder state from file."""
        state = torch.load(path, weights_only=False)
        encoder = cls(
            n_components=state["n_components"],
            whiten=state["whiten"],
            random_state=state["random_state"],
        )
        encoder._data_shape = state["data_shape"]
        encoder._mean = state["mean"]
        encoder._components = state["components"]
        encoder._explained_variance = state["explained_variance"]
        # Create a dummy PCA object for compatibility
        encoder._pca = True  # Flag that encoder is fitted
        return encoder
