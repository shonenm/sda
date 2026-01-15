"""Latent space encoders for dimensionality reduction."""

import sys
from pathlib import Path

# Add sda/ to path
_sda_root = Path(__file__).parent.parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

from experiments.latent_sda.encoders.base import LatentEncoder
from experiments.latent_sda.encoders.pca import PCAEncoder
from experiments.latent_sda.encoders.vae import VAEEncoder

__all__ = ["LatentEncoder", "PCAEncoder", "VAEEncoder"]
