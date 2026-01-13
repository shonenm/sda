r"""Data loading and management utilities for SDA.

Provides:
- Dataset loading (IBPMDataset)
- Dataset versioning and registry (DataRegistry, DatasetVersion)
"""

from .ibpm_dataset import *
from .registry import DataRegistry, DatasetVersion

__all__ = [
    "IBPMDataset",
    "build_cylinder_mask",
    "build_inflow_profile",
    "build_sdf",
    "DataRegistry",
    "DatasetVersion",
]
