"""Convolutional VAE encoder for latent space compression."""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.latent_sda.encoders.base import LatentEncoder


class ConvEncoder(nn.Module):
    """Convolutional encoder network for VAE."""

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        image_size: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.image_size = image_size

        # Encoder layers
        layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            ch_in = ch_out

        self.encoder = nn.Sequential(*layers)

        # Calculate flattened size after convolutions
        # Each conv with stride=2 halves spatial dimensions
        n_downsamples = len(hidden_channels)
        self.flat_size = hidden_channels[-1] * (image_size // (2 ** n_downsamples)) ** 2

        # Latent projections
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode to mu and logvar."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Convolutional decoder network for VAE."""

    def __init__(
        self,
        out_channels: int = 2,
        latent_dim: int = 64,
        hidden_channels: Tuple[int, ...] = (128, 64, 32),
        image_size: int = 64,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.image_size = image_size

        # Calculate initial spatial size
        n_upsamples = len(hidden_channels)
        self.init_size = image_size // (2 ** n_upsamples)
        self.flat_size = hidden_channels[0] * self.init_size ** 2

        # Project from latent
        self.fc = nn.Linear(latent_dim, self.flat_size)

        # Decoder layers
        layers = []
        for i, (ch_in, ch_out) in enumerate(zip(hidden_channels[:-1], hidden_channels[1:])):
            layers.extend([
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final layer (no BatchNorm, no activation)
        layers.append(
            nn.ConvTranspose2d(hidden_channels[-1], out_channels, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """Decode from latent."""
        h = self.fc(z)
        h = h.view(h.size(0), self.hidden_channels[0], self.init_size, self.init_size)
        return self.decoder(h)


class VAEEncoder(LatentEncoder, nn.Module):
    """Convolutional VAE for velocity field compression.

    Implements a standard VAE with:
    - Convolutional encoder: (B, C, H, W) -> (mu, logvar)
    - Reparameterization trick for sampling
    - Convolutional decoder: z -> (B, C, H, W)

    Training uses ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        image_size: int = 64,
        kl_weight: float = 1e-4,
    ):
        """Initialize VAE encoder.

        Args:
            in_channels: Number of input channels (2 for velocity u,v)
            latent_dim: Latent space dimensionality
            hidden_channels: Channel sizes for encoder (reversed for decoder)
            image_size: Spatial size of input (assumes square)
            kl_weight: Weight for KL divergence term (beta in beta-VAE)
        """
        nn.Module.__init__(self)

        self._latent_dim = latent_dim
        self._data_shape: Tuple[int, ...] = (in_channels, image_size, image_size)
        self.kl_weight = kl_weight

        # Encoder and decoder networks
        self.encoder_net = ConvEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            image_size=image_size,
        )
        self.decoder_net = ConvDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=tuple(reversed(hidden_channels)),
            image_size=image_size,
        )

        # Latent normalization stats (computed after training)
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_std", torch.ones(latent_dim))
        self._normalize_latent = False

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu  # Deterministic during inference

    def encode(self, x: Tensor) -> Tensor:
        """Encode data to latent space.

        Args:
            x: Input of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Latent of shape (B, latent_dim) or (B, T, latent_dim)
        """
        # Handle temporal dimension
        temporal = x.ndim == 5
        if temporal:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)

        # Normalize latent if enabled
        if self._normalize_latent:
            z = (z - self.z_mean) / (self.z_std + 1e-8)

        # Restore temporal dimension
        if temporal:
            z = z.reshape(B, T, -1)

        return z

    def encode_with_stats(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode with mu and logvar (for training)."""
        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to data space.

        Args:
            z: Latent of shape (B, latent_dim) or (B, T, latent_dim)

        Returns:
            Reconstruction of shape (B, C, H, W) or (B, T, C, H, W)
        """
        # Handle temporal dimension
        temporal = z.ndim == 3
        if temporal:
            B, T, D = z.shape
            z = z.reshape(B * T, D)

        # Denormalize if needed
        if self._normalize_latent:
            z = z * (self.z_std + 1e-8) + self.z_mean

        x = self.decoder_net(z)

        # Restore temporal dimension
        if temporal:
            x = x.reshape(B, T, *self._data_shape)

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning reconstruction and latent stats."""
        z, mu, logvar = self.encode_with_stats(x)
        x_recon = self.decoder_net(z)
        return x_recon, mu, logvar

    def loss(self, x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, dict]:
        """Compute VAE loss (ELBO).

        Args:
            x: Original input
            x_recon: Reconstruction
            mu: Latent mean
            logvar: Latent log-variance

        Returns:
            Total loss and dict of individual losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total ELBO loss
        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

    def fit(
        self,
        X: Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cuda",
        verbose: bool = True,
        **kwargs,
    ) -> "VAEEncoder":
        """Train VAE on data.

        Args:
            X: Training data of shape (N, C, H, W) or (N, T, C, H, W)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for Adam
            device: Device to train on
            verbose: Whether to show progress bar

        Returns:
            self
        """
        # Handle temporal data
        if X.ndim == 5:
            N, T, C, H, W = X.shape
            X = X.reshape(N * T, C, H, W)

        self._data_shape = X.shape[1:]
        self.to(device)
        self.train()

        # DataLoader
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training VAE")

        for epoch in iterator:
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                x_recon, mu, logvar = self(batch)
                loss, losses = self.loss(batch, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / len(loader)
                iterator.set_postfix(loss=f"{avg_loss:.4f}")

        # Compute latent normalization stats
        self.eval()
        with torch.no_grad():
            all_z = []
            for (batch,) in loader:
                batch = batch.to(device)
                mu, _ = self.encoder_net(batch)
                all_z.append(mu)
            all_z = torch.cat(all_z, dim=0)
            self.z_mean = all_z.mean(dim=0)
            self.z_std = all_z.std(dim=0)

        return self

    def enable_latent_normalization(self, enable: bool = True) -> "VAEEncoder":
        """Enable/disable latent space normalization."""
        self._normalize_latent = enable
        return self

    @property
    def latent_dim(self) -> int:
        """Return latent dimensionality."""
        return self._latent_dim

    @property
    def data_shape(self) -> Tuple[int, ...]:
        """Return original data shape (C, H, W)."""
        return self._data_shape

    def to(self, device) -> "VAEEncoder":
        """Move to device."""
        nn.Module.to(self, device)
        return self

    def eval(self) -> "VAEEncoder":
        """Set evaluation mode."""
        nn.Module.eval(self)
        return self

    def train(self, mode: bool = True) -> "VAEEncoder":
        """Set training mode."""
        nn.Module.train(self, mode)
        return self

    def save(self, path: str) -> None:
        """Save encoder state."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "in_channels": self._data_shape[0],
                "latent_dim": self._latent_dim,
                "hidden_channels": self.encoder_net.hidden_channels,
                "image_size": self._data_shape[1],
                "kl_weight": self.kl_weight,
            },
            "data_shape": self._data_shape,
            "normalize_latent": self._normalize_latent,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "VAEEncoder":
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(**checkpoint["config"])
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder._data_shape = checkpoint["data_shape"]
        encoder._normalize_latent = checkpoint["normalize_latent"]
        return encoder.to(device)
