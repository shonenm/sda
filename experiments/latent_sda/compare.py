#!/usr/bin/env python
"""Compare PCA vs VAE results for Latent SDA.

Usage:
    # Compare all available results:
    uv run python sda/experiments/latent_sda/compare.py

    # Compare specific dimensions:
    uv run python sda/experiments/latent_sda/compare.py --dim 64

    # Output to CSV:
    uv run python sda/experiments/latent_sda/compare.py --output comparison.csv
"""

import argparse
import sys
from pathlib import Path

# Add sda/ to path for both direct execution and module execution
_sda_root = Path(__file__).parent.parent.parent
if str(_sda_root) not in sys.path:
    sys.path.insert(0, str(_sda_root))

import csv
from collections import defaultdict

import numpy as np

from sda.paths import get_results_dir


def load_stats(stats_path: Path) -> list[dict]:
    """Load stats from CSV file."""
    results = []
    with open(stats_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["mse"] = float(row["mse"])
            row["rmse"] = float(row["rmse"])
            row["relative_error"] = float(row["relative_error"])
            row["da_time"] = float(row["da_time"])
            row["sample"] = int(row["sample"])
            results.append(row)
    return results


def collect_results(results_dir: Path) -> dict:
    """Collect all results from results directory.

    Returns:
        Dict with structure: {method: {run_id: {scenario: [metrics]}}}
    """
    all_results = {}

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name  # 'pca' or 'vae'
        all_results[method] = {}

        for run_dir in method_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            stats_path = run_dir / "stats.csv"

            if not stats_path.exists():
                continue

            # Parse latent dim from run_id (e.g., "dim64_20260115_140100")
            try:
                latent_dim = int(run_id.split("_")[0].replace("dim", ""))
            except (ValueError, IndexError):
                latent_dim = None

            stats = load_stats(stats_path)
            all_results[method][run_id] = {
                "latent_dim": latent_dim,
                "stats": stats,
            }

    return all_results


def aggregate_by_scenario(stats: list[dict]) -> dict:
    """Aggregate metrics by scenario."""
    by_scenario = defaultdict(list)
    for row in stats:
        scenario = row["scenario"]
        by_scenario[scenario].append(row)

    aggregated = {}
    for scenario, rows in by_scenario.items():
        mse_values = [r["mse"] for r in rows]
        rmse_values = [r["rmse"] for r in rows]
        time_values = [r["da_time"] for r in rows]

        aggregated[scenario] = {
            "mse_mean": np.mean(mse_values),
            "mse_std": np.std(mse_values),
            "rmse_mean": np.mean(rmse_values),
            "rmse_std": np.std(rmse_values),
            "time_mean": np.mean(time_values),
            "n_samples": len(rows),
        }

    return aggregated


def print_comparison_table(all_results: dict, dim_filter: int | None = None):
    """Print comparison table to console."""
    print("\n" + "=" * 70)
    print("LATENT SDA: PCA vs VAE COMPARISON")
    print("=" * 70)

    # Collect data for table
    table_data = []

    for method in ["pca", "vae"]:
        if method not in all_results:
            continue

        for run_id, run_data in all_results[method].items():
            latent_dim = run_data["latent_dim"]

            # Apply dimension filter
            if dim_filter is not None and latent_dim != dim_filter:
                continue

            aggregated = aggregate_by_scenario(run_data["stats"])

            for scenario, metrics in aggregated.items():
                table_data.append({
                    "method": method.upper(),
                    "dim": latent_dim,
                    "scenario": scenario,
                    "mse": metrics["mse_mean"],
                    "mse_std": metrics["mse_std"],
                    "rmse": metrics["rmse_mean"],
                    "time": metrics["time_mean"],
                    "n": metrics["n_samples"],
                })

    if not table_data:
        print("\nNo results found.")
        return

    # Sort by dim, method, scenario
    table_data.sort(key=lambda x: (x["dim"] or 0, x["method"], x["scenario"]))

    # Print table header
    print(f"\n{'Method':<8} {'Dim':<6} {'Scenario':<10} {'MSE':<12} {'RMSE':<10} {'Time(s)':<8} {'N':<4}")
    print("-" * 70)

    current_dim = None
    for row in table_data:
        # Add separator between dimensions
        if current_dim is not None and row["dim"] != current_dim:
            print("-" * 70)
        current_dim = row["dim"]

        mse_str = f"{row['mse']:.4f}±{row['mse_std']:.2f}"
        print(
            f"{row['method']:<8} {row['dim'] or '?':<6} {row['scenario']:<10} "
            f"{mse_str:<12} {row['rmse']:.4f}     {row['time']:.2f}     {row['n']}"
        )

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY BY DIMENSION")
    print("=" * 70)

    # Group by dimension
    dims = sorted(set(r["dim"] for r in table_data if r["dim"] is not None))

    for dim in dims:
        print(f"\n--- Latent Dim: {dim} ---")
        dim_data = [r for r in table_data if r["dim"] == dim]

        for scenario in ["sub", "coarse"]:
            scenario_data = [r for r in dim_data if r["scenario"] == scenario]
            if not scenario_data:
                continue

            print(f"\n  {scenario.upper()}:")
            pca_data = [r for r in scenario_data if r["method"] == "PCA"]
            vae_data = [r for r in scenario_data if r["method"] == "VAE"]

            if pca_data:
                print(f"    PCA: MSE={pca_data[0]['mse']:.4f}")
            if vae_data:
                print(f"    VAE: MSE={vae_data[0]['mse']:.4f}")

            if pca_data and vae_data:
                ratio = pca_data[0]["mse"] / vae_data[0]["mse"]
                winner = "VAE" if ratio > 1 else "PCA"
                print(f"    → {winner} is {max(ratio, 1/ratio):.1f}x better")


def save_comparison_csv(all_results: dict, output_path: Path):
    """Save comparison results to CSV."""
    rows = []

    for method in all_results:
        for run_id, run_data in all_results[method].items():
            latent_dim = run_data["latent_dim"]
            aggregated = aggregate_by_scenario(run_data["stats"])

            for scenario, metrics in aggregated.items():
                rows.append({
                    "method": method,
                    "latent_dim": latent_dim,
                    "run_id": run_id,
                    "scenario": scenario,
                    "mse_mean": f"{metrics['mse_mean']:.6f}",
                    "mse_std": f"{metrics['mse_std']:.6f}",
                    "rmse_mean": f"{metrics['rmse_mean']:.6f}",
                    "time_mean": f"{metrics['time_mean']:.2f}",
                    "n_samples": metrics["n_samples"],
                })

    # Sort
    rows.sort(key=lambda x: (x["latent_dim"] or 0, x["method"], x["scenario"]))

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare PCA vs VAE results")
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Filter by latent dimension",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )

    args = parser.parse_args()

    # Collect results
    results_dir = get_results_dir("latent_sda")
    if not results_dir.exists():
        print(f"No results found at: {results_dir}")
        return

    all_results = collect_results(results_dir)

    if not all_results:
        print("No results found.")
        return

    # Print comparison table
    print_comparison_table(all_results, dim_filter=args.dim)

    # Save to CSV if requested
    if args.output:
        save_comparison_csv(all_results, Path(args.output))


if __name__ == "__main__":
    main()
