"""Rich console utilities for SDA.

CLIの視認性向上:
- 美しい進捗バー
- 実験結果のテーブル表示
- スタイル付きコンソール出力
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# グローバルコンソールインスタンス
console = Console()


def create_progress(
    description: str = "Processing",
    transient: bool = False,
) -> Progress:
    """学習・処理用の進捗バーを作成

    Args:
        description: タスクの説明
        transient: 完了後に消すかどうか

    Returns:
        Progress: Rich進捗バーインスタンス

    Example:
        >>> from sda.console import create_progress
        >>> with create_progress("Training") as progress:
        ...     task = progress.add_task("Epochs", total=100)
        ...     for epoch in range(100):
        ...         # training...
        ...         progress.update(task, advance=1)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def track(
    iterable: Iterable,
    description: str = "Processing...",
    total: int | None = None,
    transient: bool = False,
) -> Iterable:
    """tqdm風の進捗トラッキング

    Args:
        iterable: イテレート対象
        description: 説明文
        total: 総数 (Noneの場合は自動検出)
        transient: 完了後に消すかどうか

    Yields:
        イテレートされた要素

    Example:
        >>> from sda.console import track
        >>> for batch in track(dataloader, description="Training..."):
        ...     # process batch
        ...     pass
    """
    from rich.progress import track as rich_track

    return rich_track(
        iterable,
        description=description,
        total=total,
        transient=transient,
        console=console,
    )


def print_metrics_table(
    metrics: dict[str, float],
    title: str = "Metrics",
    precision: int = 4,
) -> None:
    """メトリクスをテーブル形式で表示

    Args:
        metrics: メトリクス名と値の辞書
        title: テーブルタイトル
        precision: 小数点以下の桁数

    Example:
        >>> from sda.console import print_metrics_table
        >>> metrics = {"loss": 0.0234, "accuracy": 0.9567, "lr": 1e-4}
        >>> print_metrics_table(metrics, title="Epoch 10")
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.{precision}f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def print_config_table(
    config: dict,
    title: str = "Configuration",
    max_depth: int = 2,
) -> None:
    """設定をテーブル形式で表示

    Args:
        config: 設定辞書
        title: テーブルタイトル
        max_depth: ネストの最大深さ

    Example:
        >>> from sda.console import print_config_table
        >>> config = {"epochs": 100, "model": {"hidden": 256}}
        >>> print_config_table(config)
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    def add_rows(d: dict, prefix: str = "", depth: int = 0) -> None:
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and depth < max_depth:
                add_rows(value, full_key, depth + 1)
            else:
                table.add_row(full_key, str(value))

    add_rows(config)
    console.print(table)


def print_experiment_results(
    results: Sequence[dict[str, float]],
    metric_names: Sequence[str],
    row_names: Sequence[str] | None = None,
    title: str = "Experiment Results",
) -> None:
    """複数実験の結果をテーブルで比較表示

    Args:
        results: 各実験のメトリクス辞書のリスト
        metric_names: 表示するメトリクス名
        row_names: 各実験の名前 (Noneの場合は番号)
        title: テーブルタイトル

    Example:
        >>> from sda.console import print_experiment_results
        >>> results = [
        ...     {"loss": 0.05, "accuracy": 0.95},
        ...     {"loss": 0.03, "accuracy": 0.97},
        ... ]
        >>> print_experiment_results(results, ["loss", "accuracy"], ["baseline", "improved"])
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Experiment", style="cyan")

    for name in metric_names:
        table.add_column(name, style="green", justify="right")

    for i, result in enumerate(results):
        row_name = row_names[i] if row_names else f"Exp {i + 1}"
        values = [
            f"{result.get(name, 'N/A'):.4f}" if isinstance(result.get(name), float) else str(result.get(name, "N/A"))
            for name in metric_names
        ]
        table.add_row(row_name, *values)

    console.print(table)


def print_success(message: str) -> None:
    """成功メッセージを表示"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """エラーメッセージを表示"""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """警告メッセージを表示"""
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    """情報メッセージを表示"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")
