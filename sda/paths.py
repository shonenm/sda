"""Centralized path configuration for SDA experiments.

All output directories (runs, results, wandb, etc.) are defined here.
Experiments should import from this module instead of defining their own paths.

Directory Structure:
    {PROJECT_ROOT}/
    ├── runs/           # Training checkpoints (state.pth, config.yaml, metadata.yaml)
    │   └── {experiment}/
    │       └── {run_id}/
    ├── results/        # Evaluation outputs (images, stats)
    │   └── {experiment}/
    │       └── {run_id}/           # Linked to training run
    │           └── {mode}_{timestamp}/
    ├── data/           # Data storage
    │   ├── .registry/              # Dataset registry
    │   │   └── datasets.yaml
    │   ├── raw/                    # Raw simulation outputs
    │   │   └── {experiment}_{date}/
    │   └── processed/              # Versioned HDF5 datasets
    │       └── {experiment}_{resolution}_v{N}/
    │           ├── train.h5
    │           ├── valid.h5
    │           ├── test.h5
    │           └── metadata.yaml
    ├── docs/           # Documentation and reports
    ├── wandb/          # WandB logs (auto-managed)
    └── .dawgz/         # DAWGZ job files (auto-managed)
"""

import os
from pathlib import Path

__all__ = [
    "get_data_dir",
    "get_docs_dir",
    "get_processed_data_dir",
    "get_project_root",
    "get_raw_data_dir",
    "get_registry_dir",
    "get_results_dir",
    "get_run_results_dir",
    "get_runs_dir",
]


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root where runs/, results/ directories are located.
        - If SCRATCH env var is set: {SCRATCH}/sda
        - Otherwise: parent of sda package (this file's parent.parent)
    """
    if "SCRATCH" in os.environ:
        root = Path(os.environ["SCRATCH"]) / "sda"
    else:
        # sda/paths.py -> sda/ -> project root
        root = Path(__file__).parent.parent

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_runs_dir() -> Path:
    """Get training runs directory.

    Returns:
        Path to runs/ directory for storing training checkpoints.
    """
    runs_dir = get_project_root() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_results_dir(experiment: str | None = None) -> Path:
    """Get results directory for an experiment.

    Args:
        experiment: Experiment name (e.g., 'ibpm', 'lorenz', 'kolmogorov').
                   If None, returns the base results directory.

    Returns:
        Path to results/{experiment}/ directory.
    """
    results_dir = get_project_root() / "results"
    if experiment:
        results_dir = results_dir / experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_data_dir(experiment: str) -> Path:
    """Get data directory for an experiment.

    Note: Data directories may be in different locations depending on the environment.
    This function checks multiple locations in order:
    1. Parent directory (fluid-sbi/data/) - for submodule usage
    2. Project root (sda/data/) - for standalone usage

    Args:
        experiment: Experiment name (e.g., 'ibpm', 'lorenz', 'kolmogorov').

    Returns:
        Path to data/{experiment}/ directory.
    """
    project_root = get_project_root()

    # Check parent directory first (fluid-sbi/data/)
    parent_data = project_root.parent / "data" / experiment
    if parent_data.exists():
        return parent_data

    # Fall back to project root (sda/data/)
    data_dir = project_root / "data" / experiment
    return data_dir


def get_docs_dir(experiment: str | None = None) -> Path:
    """Get documentation directory.

    Args:
        experiment: Experiment name (e.g., 'ibpm', 'lorenz', 'kolmogorov').
                   If None, returns the base docs directory.

    Returns:
        Path to docs/{experiment}/ directory.
    """
    docs_dir = get_project_root() / "docs"
    if experiment:
        docs_dir = docs_dir / experiment
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


# =============================================================================
# Data Management Paths (New)
# =============================================================================


def get_registry_dir() -> Path:
    """Get data registry directory.

    Returns:
        Path to data/.registry/ directory for dataset manifest.
    """
    registry_dir = get_project_root() / "data" / ".registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    return registry_dir


def get_raw_data_dir(name: str | None = None) -> Path:
    """Get raw simulation data directory.

    Args:
        name: Dataset name (e.g., 'ibpm_128x128_20260112').
              If None, returns the base raw/ directory.

    Returns:
        Path to data/raw/{name}/ directory.
    """
    raw_dir = get_project_root() / "data" / "raw"
    if name:
        raw_dir = raw_dir / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def get_processed_data_dir(name: str | None = None) -> Path:
    """Get processed (versioned) data directory.

    Args:
        name: Dataset name with version (e.g., 'ibpm_128x128_v1').
              If None, returns the base processed/ directory.

    Returns:
        Path to data/processed/{name}/ directory.
    """
    processed_dir = get_project_root() / "data" / "processed"
    if name:
        processed_dir = processed_dir / name
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def get_run_results_dir(experiment: str, run_id: str, mode: str | None = None) -> Path:
    """Get results directory for a specific run.

    Args:
        experiment: Experiment name (e.g., 'ibpm').
        run_id: Training run ID (e.g., 'ibpm_vpsde_20260112_143022_abc123').
        mode: Evaluation mode (e.g., 'sparse', 'sample').
              If None, returns the run's results directory.

    Returns:
        Path to results/{experiment}/{run_id}/{mode}/ directory.
    """
    from datetime import datetime

    results_dir = get_project_root() / "results" / experiment / run_id
    if mode:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = results_dir / f"{mode}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
