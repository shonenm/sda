"""Latent Space Score-based Data Assimilation

This module implements SDA in a compressed latent space using PCA or VAE encoders.
"""

import sys
from pathlib import Path

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

from experiments.latent_sda.encoders import LatentEncoder, PCAEncoder, VAEEncoder
from experiments.latent_sda.latent_sda import LatentSDA
from experiments.latent_sda.latent_score import LatentGaussianScore

__all__ = [
    "LatentEncoder",
    "PCAEncoder",
    "VAEEncoder",
    "LatentSDA",
    "LatentGaussianScore",
]
