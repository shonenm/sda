"""Score module for data assimilation in latent space."""

import sys
from collections.abc import Callable
from pathlib import Path

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import torch
import torch.nn as nn
from torch import Tensor

from sda.score import VPSDE

from experiments.latent_sda.encoders.base import LatentEncoder


class LatentGaussianScore(nn.Module):
    """Score module for Gaussian inverse problems in latent space.

    Performs data assimilation by combining:
    - Prior score: Learned score function in latent space
    - Likelihood score: Gradient of log p(y | z) where y = A(decode(z)) + noise

    The observation operator A operates in full space, with gradients flowing
    through the decoder for latent-space guidance.

    This module returns -sigma(t) * s(z(t), t | y), following the VPSDE convention.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        encoder: LatentEncoder,
        std: float,
        sde: VPSDE,
        gamma: float = 1e-2,
        normalize: bool = True,
    ):
        """Initialize latent Gaussian score module.

        Args:
            y: Observation in full space (the target to match)
            A: Observation operator in full space (e.g., subsampling)
            encoder: Latent encoder with encode/decode methods
            std: Observation noise standard deviation
            sde: VPSDE with latent score network
            gamma: Noise regularization coefficient
            normalize: If True, normalize by observation count
        """
        super().__init__()

        self.register_buffer("y", y)
        self.register_buffer("std", torch.as_tensor(std))
        self.register_buffer("gamma", torch.as_tensor(gamma))

        self.A = A
        self.encoder = encoder
        self.sde = sde
        self.normalize = normalize

    def forward(self, z: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        """Compute guided score in latent space.

        Args:
            z: Noisy latent state z(t) of shape (B, latent_dim)
            t: Diffusion time in [0, 1]
            c: Optional context (not typically used)

        Returns:
            -sigma(t) * s(z(t), t | y) for reverse diffusion
        """
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            z = z.detach().requires_grad_(True)

            # Get prior score (noise prediction)
            eps = self.sde.eps(z, t, c)

            # Tweedie estimate: denoise z to get z_hat
            z_hat = (z - sigma * eps) / mu

            # Decode to full space
            x_hat = self.encoder.decode(z_hat)

            # Compute observation error in full space
            err = self.y - self.A(x_hat)

            # Time-dependent variance (observation noise + diffusion noise)
            var = self.std**2 + self.gamma * (sigma / mu) ** 2

            # Log-likelihood
            if self.normalize:
                log_p = -(err**2 / var).mean() / 2
            else:
                log_p = -(err**2 / var).sum() / 2

        # Gradient of log-likelihood w.r.t. z
        (s,) = torch.autograd.grad(log_p, z)

        # Combined score: prior + likelihood
        return eps - sigma * s


class LatentDPSScore(nn.Module):
    """DPS-style score module for latent space (normalized gradient).

    Similar to LatentGaussianScore but uses normalized gradients as in the
    original DPS paper (Chung et al., 2022).
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        encoder: LatentEncoder,
        sde: VPSDE,
        zeta: float = 1.0,
    ):
        """Initialize latent DPS score module.

        Args:
            y: Observation in full space
            A: Observation operator in full space
            encoder: Latent encoder
            sde: VPSDE with latent score network
            zeta: Guidance strength
        """
        super().__init__()

        self.register_buffer("y", y)

        self.A = A
        self.encoder = encoder
        self.sde = sde
        self.zeta = zeta

    def forward(self, z: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        """Compute DPS-guided score in latent space."""
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            z = z.detach().requires_grad_(True)

            eps = self.sde.eps(z, t, c)
            z_hat = (z - sigma * eps) / mu
            x_hat = self.encoder.decode(z_hat)

            err = (self.y - self.A(x_hat)).square().sum()

        (s,) = torch.autograd.grad(err, z)
        s = -s * self.zeta / (err.sqrt() + 1e-8)

        return eps - sigma * s
