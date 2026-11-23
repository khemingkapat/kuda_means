#!/usr/bin/env python3
"""Plot K-Means runtime and speedup curves from benchmark CSV files.

Usage:
    uv run python scripts/plot_results.py results/benchmarks.csv --out results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for version, grp in df.groupby("version"):
        grp = grp.sort_values("N")
        ax.plot(grp["N"], grp["kernel_ms"], marker="o", label=str(version))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Points (N)")
    ax.set_ylabel("Kernel Execution Time (ms)")
    ax.set_title("K-Means Scaling Across Implementations")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_scaling.pdf")
    fig.savefig(out_dir / "runtime_scaling.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE-ready benchmark plots.")
    parser.add_argument("csv_path", type=Path, help="Path to benchmark CSV results")
    parser.add_argument("--out", type=Path, default=Path("results/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    plot_scaling(df, args.out)
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()
