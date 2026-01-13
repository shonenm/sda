"""Run management with full traceability.

Provides unified experiment tracking with automatic metadata collection,
WandB integration, and reproducibility features.

Example:
    >>> from sda.tracking import RunManager
    >>> from sda.data import DataRegistry
    >>>
    >>> registry = DataRegistry()
    >>> manager = RunManager(
    ...     experiment="ibpm",
    ...     name="vpsde_baseline",
    ...     config={"epochs": 1000, "lr": 1e-4},
    ...     dataset_name="ibpm_128x128",
    ...     registry=registry,
    ... )
    >>>
    >>> with manager.create_run() as run:
    ...     for epoch in range(config["epochs"]):
    ...         # training...
    ...         if epoch % 50 == 0:
    ...             run.save_checkpoint(epoch, model.state_dict())
"""

from __future__ import annotations

import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from sda.paths import get_run_results_dir, get_runs_dir

if TYPE_CHECKING:
    from sda.data.registry import DataRegistry, DatasetVersion


def collect_git_info() -> dict[str, Any]:
    """Collect git repository information.

    Returns:
        Dictionary with commit hash, branch, and dirty status.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": True,
        }


def collect_environment_info() -> dict[str, Any]:
    """Collect environment and system information.

    Returns:
        Dictionary with Python, PyTorch, CUDA, and system info.
    """
    env_info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }

    # CUDA info
    if torch.cuda.is_available():
        env_info["cuda"] = torch.version.cuda or "unknown"
        env_info["gpu"] = torch.cuda.get_device_name(0)
        env_info["gpu_count"] = torch.cuda.device_count()
    else:
        env_info["cuda"] = None
        env_info["gpu"] = None

    return env_info


@dataclass
class RunContext:
    """Context for a training run.

    Provides methods for saving checkpoints and logging during training.

    Attributes:
        run_id: Unique identifier for this run
        run_dir: Directory for saving checkpoints and metadata
        metadata: Collected metadata (git, environment, data version)
        wandb_run: WandB run object (if initialized)
    """

    run_id: str
    run_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    wandb_run: Any = None
    _closed: bool = field(default=False, repr=False)

    def save_checkpoint(
        self,
        epoch: int,
        state_dict: dict[str, Any],
        is_final: bool = False,
    ) -> Path:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number
            state_dict: Model state dictionary
            is_final: Whether this is the final checkpoint

        Returns:
            Path to saved checkpoint file
        """
        if is_final:
            filename = "state_final.pth"
        else:
            filename = f"state_epoch{epoch}.pth"

        path = self.run_dir / filename
        torch.save(state_dict, path)

        # Log to WandB as artifact if available
        if self.wandb_run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(
                    f"model-{self.run_id}-{'final' if is_final else f'epoch{epoch}'}",
                    type="model",
                    metadata={"epoch": epoch, "is_final": is_final},
                )
                artifact.add_file(str(path))
                self.wandb_run.log_artifact(artifact)
            except Exception:
                pass  # Don't fail training if artifact logging fails

        return path

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def get_results_dir(self, mode: str) -> Path:
        """Get results directory for this run.

        Args:
            mode: Evaluation mode (e.g., 'sparse', 'sample')

        Returns:
            Path to timestamped results directory
        """
        # Extract experiment from run_id (e.g., 'ibpm_vpsde_...' -> 'ibpm')
        experiment = self.run_id.split("_")[0]
        return get_run_results_dir(experiment, self.run_id, mode)

    def close(self) -> None:
        """Close the run and finalize WandB."""
        if self._closed:
            return
        self._closed = True

        if self.wandb_run is not None:
            self.wandb_run.finish()


class RunManager:
    """Manages experiment runs with full traceability.

    Creates runs with automatic metadata collection, directory setup,
    and optional WandB integration.

    Example:
        >>> manager = RunManager(
        ...     experiment="ibpm",
        ...     name="vpsde_baseline",
        ...     config={"epochs": 1000},
        ...     dataset_name="ibpm_128x128",
        ... )
        >>> with manager.create_run() as run:
        ...     # Training loop
        ...     pass
    """

    def __init__(
        self,
        experiment: str,
        name: str,
        config: dict[str, Any],
        dataset_name: str | None = None,
        registry: DataRegistry | None = None,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        wandb_tags: list[str] | None = None,
        use_wandb: bool = True,
    ):
        """Initialize the run manager.

        Args:
            experiment: Experiment name (e.g., 'ibpm', 'lorenz')
            name: Run name (e.g., 'vpsde_baseline')
            config: Training configuration dictionary
            dataset_name: Name of dataset in registry (optional)
            registry: DataRegistry instance (optional)
            wandb_project: WandB project name (default: sda-{experiment})
            wandb_group: WandB group name (optional)
            wandb_tags: WandB tags (optional)
            use_wandb: Whether to use WandB (default: True)
        """
        self.experiment = experiment
        self.name = name
        self.config = config
        self.dataset_name = dataset_name
        self.registry = registry
        self.wandb_project = wandb_project or f"sda-{experiment}"
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags or []
        self.use_wandb = use_wandb

    def _generate_run_id(self, wandb_id: str | None = None) -> str:
        """Generate a unique run ID.

        Format: {experiment}_{name}_{timestamp}_{wandb_id}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.experiment, self.name, timestamp]
        if wandb_id:
            parts.append(wandb_id[:8])  # Use first 8 chars of WandB ID
        return "_".join(parts)

    def _collect_metadata(
        self,
        run_id: str,
        wandb_run: Any = None,
    ) -> dict[str, Any]:
        """Collect comprehensive metadata for reproducibility."""
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "experiment": self.experiment,
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "git": collect_git_info(),
            "environment": collect_environment_info(),
            "config": self.config,
        }

        # WandB info
        if wandb_run is not None:
            metadata["wandb"] = {
                "run_id": wandb_run.id,
                "project": wandb_run.project,
                "url": wandb_run.url,
            }

        # Dataset info from registry
        if self.dataset_name and self.registry:
            try:
                dataset = self.registry.get_latest(self.dataset_name)
                metadata["data"] = {
                    "dataset": dataset.name,
                    "version": dataset.version,
                    "path": str(dataset.path),
                    "checksums": dataset.checksums,
                }
            except KeyError:
                metadata["data"] = {
                    "dataset": self.dataset_name,
                    "version": None,
                    "note": "Dataset not found in registry",
                }

        return metadata

    @contextmanager
    def create_run(self):
        """Create a new training run with full traceability.

        Yields:
            RunContext for the training run

        Example:
            >>> with manager.create_run() as run:
            ...     for epoch in range(epochs):
            ...         # training...
            ...         run.save_checkpoint(epoch, model.state_dict())
        """
        wandb_run = None

        # Initialize WandB
        if self.use_wandb:
            try:
                import wandb

                wandb_run = wandb.init(
                    project=self.wandb_project,
                    group=self.wandb_group,
                    tags=self.wandb_tags,
                    config=self.config,
                    reinit=True,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                wandb_run = None

        # Generate run ID
        wandb_id = wandb_run.id if wandb_run else None
        run_id = self._generate_run_id(wandb_id)

        # Create run directory
        run_dir = get_runs_dir() / self.experiment / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Collect metadata
        metadata = self._collect_metadata(run_id, wandb_run)

        # Save config and metadata
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        with open(run_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

        # Log artifacts to WandB
        if wandb_run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(f"config-{run_id}", type="config")
                artifact.add_file(str(run_dir / "config.yaml"))
                artifact.add_file(str(run_dir / "metadata.yaml"))
                wandb_run.log_artifact(artifact)
            except Exception:
                pass

        # Create context
        context = RunContext(
            run_id=run_id,
            run_dir=run_dir,
            metadata=metadata,
            wandb_run=wandb_run,
        )

        try:
            yield context
        finally:
            context.close()

    @staticmethod
    def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load config and metadata from a run directory.

        Args:
            run_dir: Path to run directory

        Returns:
            Tuple of (config, metadata)
        """
        config_path = run_dir / "config.yaml"
        metadata_path = run_dir / "metadata.yaml"

        config = {}
        metadata = {}

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = yaml.safe_load(f) or {}

        return config, metadata
