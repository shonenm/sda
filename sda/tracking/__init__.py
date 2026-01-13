"""Experiment tracking module for SDA.

Provides unified run management with full traceability:
- Automatic metadata collection (git, environment, data version)
- WandB integration with artifacts
- Reproducibility tracking
"""

from .run_manager import RunContext, RunManager, collect_environment_info, collect_git_info

__all__ = [
    "RunContext",
    "RunManager",
    "collect_environment_info",
    "collect_git_info",
]
