"""Abstract base class for latent space encoders."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor


class LatentEncoder(ABC):
    """Abstract base class for latent space encoders.

    Defines the interface for encoding high-dimensional data (e.g., velocity fields)
    to a low-dimensional latent space and decoding back.

    Subclasses must implement:
        - fit(): Learn the encoding from data
        - encode(): Map from data space to latent space
        - decode(): Map from latent space back to data space
        - latent_dim: Return the latent dimensionality
    """

    @abstractmethod
    def fit(self, X: Tensor, **kwargs) -> "LatentEncoder":
        """Fit the encoder to training data.

        Args:
            X: Training data of shape (N, C, H, W) or (N, T, C, H, W)
            **kwargs: Additional arguments for specific encoders

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode data to latent space.

        Args:
            x: Input data of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Latent representation of shape (B, latent_dim) or (B, T, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation back to data space.

        Args:
            z: Latent representation of shape (B, latent_dim) or (B, T, latent_dim)

        Returns:
            Reconstructed data of shape (B, C, H, W) or (B, T, C, H, W)
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the latent space dimensionality."""
        pass

    @property
    @abstractmethod
    def data_shape(self) -> Tuple[int, ...]:
        """Return the original data shape (C, H, W)."""
        pass

    def reconstruct(self, x: Tensor) -> Tensor:
        """Encode and decode (reconstruction).

        Args:
            x: Input data

        Returns:
            Reconstructed data
        """
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: Tensor) -> Tensor:
        """Compute reconstruction MSE.

        Args:
            x: Input data

        Returns:
            Mean squared error (scalar)
        """
        x_hat = self.reconstruct(x)
        return ((x - x_hat) ** 2).mean()

    def to(self, device: torch.device) -> "LatentEncoder":
        """Move encoder to device (for subclasses with parameters)."""
        return self

    def eval(self) -> "LatentEncoder":
        """Set to evaluation mode (for subclasses with dropout etc.)."""
        return self

    def train(self, mode: bool = True) -> "LatentEncoder":
        """Set training mode (for subclasses)."""
        return self
