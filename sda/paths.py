"""Centralized path configuration for SDA experiments.

All output directories (runs, results, wandb, etc.) are defined here.
Experiments should import from this module instead of defining their own paths.

Directory Structure:
    {PROJECT_ROOT}/
    ├── runs/           # Training checkpoints (state.pth, config.yaml)
    ├── results/        # Evaluation outputs (images, stats)
    │   └── {experiment}/
    │       └── evaluate/
    ├── data/           # Generated data (train.h5, valid.h5, test.h5)
    │   └── {experiment}/
    ├── docs/           # Documentation and reports
    │   └── {experiment}/
    ├── wandb/          # WandB logs (auto-managed)
    └── .dawgz/         # DAWGZ job files (auto-managed)
"""

import os
from pathlib import Path

__all__ = ["get_data_dir", "get_docs_dir", "get_project_root", "get_results_dir", "get_runs_dir"]


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
    This function provides a default location, but experiments may override it.

    Args:
        experiment: Experiment name (e.g., 'ibpm', 'lorenz', 'kolmogorov').

    Returns:
        Path to data/{experiment}/ directory.
    """
    data_dir = get_project_root() / "data" / experiment
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
