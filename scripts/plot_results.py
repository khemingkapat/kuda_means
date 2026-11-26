#!/usr/bin/env python3
"""Plot K-Means benchmarks from verified JSON outputs with strict hardware provenance.

Usage:
    uv run python scripts/plot_results.py results/ --out results/figures/
    uv run python scripts/plot_results.py results/ --plot scaling --d 16 --k 8
    uv run python scripts/plot_results.py results/ \\
        --gpu "NVIDIA GeForce RTX 4090" --n 1000000 --d 16 --k 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REFERENCE_VERSIONS = frozenset({"cpu", "v0"})
STAGE_ORDER = ("v0", "v1", "v2", "v3", "v4", "v5")
STAGE_LABELS = {
    "v0": "V0 (CPU)",
    "v1": "V1 (Naive)",
    "v2": "V2 (Memory)",
    "v3": "V3 (Privatized)",
    "v4": "V4 (MatMul)",
    "v5": "V5 (Streaming)",
}
# Colorblind-safe Okabe-Ito palette
STAGE_COLORS = {
    "v0": "#000000",
    "v1": "#D55E00",
    "v2": "#E69F00",
    "v3": "#009E73",
    "v4": "#0072B2",
    "v5": "#CC79A7",
}
STAGE_MARKERS = {
    "v0": "o",
    "v1": "s",
    "v2": "^",
    "v3": "D",
    "v4": "v",
    "v5": "P",
}
PLOT_CHOICES = ("all", "scaling", "speedup", "breakdown", "crossover")
BASELINE_MATCH_COLS = ("n_points", "dim", "k_clusters", "seed", "max_iters", "epsilon")
TIMER_INVERSION_TOL_MS = 1e-6


class PlotValidationError(ValueError):
    """Raised when benchmark JSON inputs fail schema, uniqueness, or slice checks."""


class ProvenanceError(PlotValidationError):
    """Raised when verification or hardware provenance constraints are violated."""


@dataclass(frozen=True)
class ProvenanceContext:
    gpu_names: tuple[str, ...]
    cpu_names: tuple[str, ...]
    compute_caps: tuple[str, ...]
    commits: tuple[str, ...]
    has_unverified: bool
    is_mixed_hw: bool

    @property
    def diagnostic_banner(self) -> str | None:
        tags: list[str] = []
        if self.has_unverified:
            tags.append("UNVERIFIED")
        if self.is_mixed_hw:
            tags.append("MIXED HARDWARE")
        if not tags:
            return None
        return f"[DIAGNOSTIC: {' + '.join(tags)}]"

    @property
    def subtitle(self) -> str:
        parts: list[str] = []
        if self.gpu_names:
            gpu_label = ", ".join(self.gpu_names)
            if self.compute_caps:
                gpu_label = f"{gpu_label} ({', '.join(self.compute_caps)})"
            parts.append(gpu_label)
        elif self.cpu_names:
            parts.append(", ".join(self.cpu_names))
        if self.commits:
            commit_str = (
                self.commits[0] if len(self.commits) == 1 else f"{len(self.commits)} commits"
            )
            parts.append(f"git:{commit_str}")
        return " | ".join(parts)


def _require_bool(val: Any, field: str, path: Path) -> bool:
    if type(val) is not bool:
        raise PlotValidationError(f"{path}: '{field}' must be a JSON boolean, got {val!r}.")
    return val


def _require_int(val: Any, field: str, path: Path, *, min_value: int) -> int:
    if not isinstance(val, int) or isinstance(val, bool) or val < min_value:
        raise PlotValidationError(
            f"{path}: '{field}' must be an integer >= {min_value}, got {val!r}."
        )
    return val


def _require_float(
    val: Any,
    field: str,
    path: Path,
    *,
    min_value: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise PlotValidationError(f"{path}: '{field}' must be a finite number, got {val!r}.")
    fval = float(val)
    if not math.isfinite(fval):
        raise PlotValidationError(f"{path}: '{field}' must be finite, got {val!r}.")
    if strictly_positive and fval <= min_value:
        raise PlotValidationError(f"{path}: '{field}' must be > {min_value}, got {fval}.")
    if not strictly_positive and fval < min_value:
        raise PlotValidationError(f"{path}: '{field}' must be >= {min_value}, got {fval}.")
    return fval


def _require_str(val: Any, field: str, path: Path) -> str:
    if not isinstance(val, str) or not val.strip():
        raise PlotValidationError(f"{path}: '{field}' must be a non-empty string, got {val!r}.")
    return val.strip()


def _require_dict(val: Any, field: str, path: Path) -> dict[str, Any]:
    if not isinstance(val, dict):
        raise PlotValidationError(f"{path}: '{field}' must be a JSON object.")
    return val


def discover_json_files(inputs: Sequence[Path]) -> list[Path]:
    """Resolve input files and flat directories (*.json) into a sorted, unique list."""
    if not inputs:
        raise PlotValidationError("At least one JSON file or directory must be provided.")

    found: dict[Path, Path] = {}
    for raw_path in inputs:
        if not raw_path.exists():
            raise PlotValidationError(f"Input path does not exist: {raw_path}")
        if raw_path.is_dir():
            for child in sorted(raw_path.glob("*.json")):
                if child.is_file():
                    found.setdefault(child.resolve(), child)
        elif raw_path.is_file():
            if raw_path.suffix.lower() != ".json":
                raise PlotValidationError(
                    f"Unsupported input file '{raw_path}' (expected .json benchmark output)."
                )
            found.setdefault(raw_path.resolve(), raw_path)
        else:
            raise PlotValidationError(f"Input path is neither a file nor directory: {raw_path}")

    if not found:
        joined = ", ".join(str(p) for p in inputs)
        raise PlotValidationError(f"No .json benchmark files found in: {joined}")

    return [found[key] for key in sorted(found)]


def _parse_json_record(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotValidationError(f"{path}: failed to parse JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlotValidationError(f"{path}: root JSON value must be an object.")

    raw_version = _require_str(raw.get("version"), "version", path).lower()
    config = _require_dict(raw.get("config"), "config", path)
    hardware = _require_dict(raw.get("hardware"), "hardware", path)
    metrics = _require_dict(raw.get("metrics"), "metrics", path)
    verified = _require_bool(raw.get("verified"), "verified", path)

    n_points = _require_int(config.get("n_points"), "config.n_points", path, min_value=1)
    dim = _require_int(config.get("dim"), "config.dim", path, min_value=1)
    k_clusters = _require_int(config.get("k_clusters"), "config.k_clusters", path, min_value=1)
    max_iters = _require_int(config.get("max_iters"), "config.max_iters", path, min_value=1)
    epsilon = _require_float(config.get("epsilon"), "config.epsilon", path, min_value=0.0)
    seed = _require_int(config.get("seed"), "config.seed", path, min_value=0)
    warmup_runs = _require_int(config.get("warmup_runs"), "config.warmup_runs", path, min_value=0)
    bench_runs = _require_int(config.get("bench_runs"), "config.bench_runs", path, min_value=1)

    gpu_name = _require_str(hardware.get("gpu_name"), "hardware.gpu_name", path)
    compute_cap = _require_str(hardware.get("compute_cap"), "hardware.compute_cap", path)
    sm_count = _require_int(hardware.get("sm_count"), "hardware.sm_count", path, min_value=0)
    global_mem_mb = _require_float(
        hardware.get("global_mem_mb"), "hardware.global_mem_mb", path, min_value=0.0
    )
    cuda_version = _require_str(hardware.get("cuda_version"), "hardware.cuda_version", path)
    git_commit = _require_str(hardware.get("git_commit"), "hardware.git_commit", path)

    iterations = _require_int(metrics.get("iterations"), "metrics.iterations", path, min_value=0)
    sse = _require_float(metrics.get("sse"), "metrics.sse", path, min_value=0.0)
    kernel_ms = _require_float(
        metrics.get("kernel_ms"), "metrics.kernel_ms", path, strictly_positive=True
    )
    e2e_ms = _require_float(metrics.get("e2e_ms"), "metrics.e2e_ms", path, strictly_positive=True)
    ms_per_iter = _require_float(
        metrics.get("ms_per_iter"), "metrics.ms_per_iter", path, min_value=0.0
    )

    if e2e_ms + TIMER_INVERSION_TOL_MS < kernel_ms:
        raise PlotValidationError(
            f"{path}: invalid timing (e2e_ms={e2e_ms:.6f} < kernel_ms={kernel_ms:.6f})."
        )

    is_cpu = raw_version in REFERENCE_VERSIONS
    if is_cpu:
        if compute_cap.lower() != "none":
            raise PlotValidationError(
                f"{path}: CPU reference version '{raw_version}' must have compute_cap='none', "
                f"got '{compute_cap}'."
            )
        canonical_version = "v0"
    else:
        if compute_cap.lower() == "none" or cuda_version.lower() == "none":
            raise PlotValidationError(
                f"{path}: GPU version '{raw_version}' has missing GPU provenance "
                f"(compute_cap='{compute_cap}', cuda_version='{cuda_version}')."
            )
        canonical_version = raw_version

    verification_passed: bool | None = None
    if "verification" in raw and raw["verification"] is not None:
        verification = _require_dict(raw["verification"], "verification", path)
        verification_passed = _require_bool(verification.get("passed"), "verification.passed", path)
        _require_int(
            verification.get("assignment_mismatches"),
            "verification.assignment_mismatches",
            path,
            min_value=0,
        )
        _require_float(
            verification.get("max_centroid_abs_diff"),
            "verification.max_centroid_abs_diff",
            path,
            min_value=0.0,
        )
        _require_float(
            verification.get("sse_rel_diff"),
            "verification.sse_rel_diff",
            path,
            min_value=0.0,
        )
        if verified and not verification_passed:
            raise PlotValidationError(
                f"{path}: contradictory verification fields "
                "(verified=true but verification.passed=false)."
            )

    return {
        "source_file": str(path),
        "version": canonical_version,
        "raw_version": raw_version,
        "is_cpu": is_cpu,
        "n_points": n_points,
        "dim": dim,
        "k_clusters": k_clusters,
        "max_iters": max_iters,
        "epsilon": epsilon,
        "seed": seed,
        "warmup_runs": warmup_runs,
        "bench_runs": bench_runs,
        "gpu_name": gpu_name,
        "compute_cap": compute_cap,
        "sm_count": sm_count,
        "global_mem_mb": global_mem_mb,
        "cuda_version": cuda_version,
        "git_commit": git_commit,
        "iterations": iterations,
        "sse": sse,
        "kernel_ms": kernel_ms,
        "e2e_ms": e2e_ms,
        "ms_per_iter": ms_per_iter,
        "verified": verified,
        "verification_passed": verification_passed,
    }


def load_results(inputs: Sequence[Path]) -> pd.DataFrame:
    """Load and schema-validate benchmark JSON files into a DataFrame."""
    json_files = discover_json_files(inputs)
    records = [_parse_json_record(path) for path in json_files]
    return pd.DataFrame(records)


def validate_provenance(
    df: pd.DataFrame,
    *,
    gpu_filter: str | None = None,
    allow_unverified: bool = False,
    allow_mixed_gpu: bool = False,
) -> tuple[pd.DataFrame, ProvenanceContext]:
    """Enforce verified==true, single-GPU/single-CPU provenance, and record uniqueness."""
    if df.empty:
        raise PlotValidationError("No benchmark records loaded.")

    filtered = df.copy()
    if gpu_filter is not None:
        gpu_rows = filtered[~filtered["is_cpu"]]
        available_gpus = sorted(gpu_rows["gpu_name"].unique().tolist())
        matched_gpu = gpu_rows[gpu_rows["gpu_name"] == gpu_filter]
        if matched_gpu.empty:
            raise ProvenanceError(
                f"No GPU records found for --gpu {gpu_filter!r}. "
                f"Available GPUs: {available_gpus or ['(none)']}"
            )
        filtered = filtered[filtered["is_cpu"] | (filtered["gpu_name"] == gpu_filter)].copy()

    unverified_mask = (~filtered["verified"]) | filtered["verification_passed"].eq(False)
    unverified_rows = filtered[unverified_mask]
    has_unverified = not unverified_rows.empty
    if has_unverified and not allow_unverified:
        files = ", ".join(sorted(unverified_rows["source_file"].tolist()))
        raise ProvenanceError(
            f"Refusing to plot unverified benchmark run(s): {files}. "
            "Pass --allow-unverified for diagnostic plots only."
        )

    gpu_df = filtered[~filtered["is_cpu"]]
    cpu_df = filtered[filtered["is_cpu"]]
    gpu_names = tuple(sorted(gpu_df["gpu_name"].unique().tolist()))
    cpu_names = tuple(sorted(cpu_df["gpu_name"].unique().tolist()))
    compute_caps = tuple(sorted(gpu_df["compute_cap"].unique().tolist()))
    commits = tuple(sorted(filtered["git_commit"].unique().tolist()))

    is_mixed_hw = len(gpu_names) > 1 or len(cpu_names) > 1
    if len(gpu_names) > 1 and not allow_mixed_gpu:
        raise ProvenanceError(
            f"Refusing to mix multiple GPUs in the same comparison: {list(gpu_names)}. "
            "Select one with --gpu <name>, or pass --allow-mixed-gpu for diagnostic plots."
        )
    if len(cpu_names) > 1 and not allow_mixed_gpu:
        raise ProvenanceError(
            f"Refusing to mix multiple CPU baseline hosts: {list(cpu_names)}. "
            "Provide baseline JSONs from a single CPU host, or pass --allow-mixed-gpu "
            "for diagnostic plots."
        )

    dup_cols = ["version", "n_points", "dim", "k_clusters", "gpu_name"]
    dup_mask = filtered.duplicated(subset=dup_cols, keep=False)
    if dup_mask.any():
        dup_rows = filtered[dup_mask]
        details: list[str] = []
        for key, grp in dup_rows.groupby(dup_cols, sort=True):
            files = ", ".join(sorted(grp["source_file"].tolist()))
            details.append(f"{key}: [{files}]")
        raise PlotValidationError(
            "Duplicate benchmark records found for identical (version, N, D, K, gpu_name): "
            + "; ".join(details)
        )

    ctx = ProvenanceContext(
        gpu_names=gpu_names,
        cpu_names=cpu_names,
        compute_caps=compute_caps,
        commits=commits,
        has_unverified=has_unverified,
        is_mixed_hw=is_mixed_hw,
    )
    return filtered, ctx


def _version_sort_key(version: str) -> tuple[int, str]:
    try:
        return (STAGE_ORDER.index(version), version)
    except ValueError:
        return (len(STAGE_ORDER), version)


def _stage_label(version: str, gpu_name: str | None = None, *, include_hw: bool = False) -> str:
    base = STAGE_LABELS.get(version, version.upper())
    if include_hw and gpu_name:
        return f"{base} [{gpu_name}]"
    return base


def _stage_color(version: str) -> str:
    return STAGE_COLORS.get(version, "#555555")


def _stage_marker(version: str) -> str:
    return STAGE_MARKERS.get(version, "o")


def _apply_optional_filters(
    df: pd.DataFrame,
    *,
    n_filter: int | None,
    d_filter: int | None,
    k_filter: int | None,
) -> pd.DataFrame:
    out = df
    if n_filter is not None:
        available = sorted(out["n_points"].unique().tolist())
        out = out[out["n_points"] == n_filter]
        if out.empty:
            raise PlotValidationError(
                f"No records match --n {n_filter}. Available N values: {available}"
            )
    if d_filter is not None:
        available = sorted(out["dim"].unique().tolist())
        out = out[out["dim"] == d_filter]
        if out.empty:
            raise PlotValidationError(
                f"No records match --d {d_filter}. Available D values: {available}"
            )
    if k_filter is not None:
        available = sorted(out["k_clusters"].unique().tolist())
        out = out[out["k_clusters"] == k_filter]
        if out.empty:
            raise PlotValidationError(
                f"No records match --k {k_filter}. Available K values: {available}"
            )
    return out.copy()


def _require_unique_dk(df: pd.DataFrame, plot_name: str) -> tuple[int, int]:
    pairs = sorted({(int(r.dim), int(r.k_clusters)) for r in df.itertuples()})
    if len(pairs) > 1:
        raise PlotValidationError(
            f"Ambiguous (D, K) slice for '{plot_name}': found {pairs}. "
            "Specify --d and --k explicitly."
        )
    return pairs[0]


def _require_unique_n(df: pd.DataFrame, plot_name: str) -> int:
    ns = sorted(int(v) for v in df["n_points"].unique().tolist())
    if len(ns) > 1:
        raise PlotValidationError(
            f"Ambiguous N slice for '{plot_name}': found {ns}. Specify --n explicitly."
        )
    return ns[0]


def resolve_plot_plan(
    df: pd.DataFrame,
    ctx: ProvenanceContext,
    *,
    selected_plots: Sequence[str],
    n_filter: int | None = None,
    d_filter: int | None = None,
    k_filter: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate all requested plot slices up front (atomic fail-fast before writing files)."""
    scoped = _apply_optional_filters(df, n_filter=n_filter, d_filter=d_filter, k_filter=k_filter)
    gpu_scoped = scoped[~scoped["is_cpu"]]
    cpu_scoped = scoped[scoped["is_cpu"]]

    plan: dict[str, dict[str, Any]] = {}
    for plot_name in selected_plots:
        if plot_name == "scaling":
            dim, k_val = _require_unique_dk(scoped, "scaling")
            plan["scaling"] = {"df": scoped, "dim": dim, "k_clusters": k_val}
        elif plot_name == "speedup":
            if gpu_scoped.empty or cpu_scoped.empty:
                plan["speedup"] = {"skip": "requires both CPU (v0) and GPU benchmark records"}
                continue
            if len(ctx.cpu_names) > 1:
                raise ProvenanceError(
                    "Refusing to compute 'speedup' against multiple CPU baseline hosts "
                    f"{list(ctx.cpu_names)}. Provide V0 baselines from a single CPU host."
                )
            dim, k_val = _require_unique_dk(scoped, "speedup")
            plan["speedup"] = {"df": scoped, "dim": dim, "k_clusters": k_val}
        elif plot_name == "breakdown":
            if gpu_scoped.empty:
                plan["breakdown"] = {"skip": "requires at least one GPU benchmark record"}
                continue
            n_val = _require_unique_n(gpu_scoped, "breakdown")
            dim, k_val = _require_unique_dk(gpu_scoped, "breakdown")
            plan["breakdown"] = {
                "df": gpu_scoped,
                "n_points": n_val,
                "dim": dim,
                "k_clusters": k_val,
            }
        elif plot_name == "crossover":
            if gpu_scoped.empty:
                plan["crossover"] = {"skip": "requires at least one GPU benchmark record"}
                continue
            if ctx.is_mixed_hw:
                raise ProvenanceError(
                    "Refusing to generate 'crossover' winner map across mixed hardware "
                    f"(GPUs={list(ctx.gpu_names)}, CPUs={list(ctx.cpu_names)}). "
                    "Filter to a single GPU and single CPU host."
                )
            n_val = _require_unique_n(gpu_scoped, "crossover")
            plan["crossover"] = {
                "gpu_df": gpu_scoped,
                "cpu_df": cpu_scoped[cpu_scoped["n_points"] == n_val],
                "n_points": n_val,
            }
        else:
            raise PlotValidationError(f"Unknown plot target: {plot_name!r}")

    return plan


def _configure_ieee_style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
        }
    )


def _apply_diagnostic_watermark(fig: plt.Figure, ctx: ProvenanceContext) -> None:
    if ctx.diagnostic_banner:
        fig.text(
            0.5,
            0.995,
            ctx.diagnostic_banner,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#B00020",
        )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def _iter_series_groups(
    df: pd.DataFrame, *, include_hw: bool
) -> list[tuple[str, str | None, pd.DataFrame]]:
    if include_hw:
        raw_groups = [
            (str(key[0]), str(key[1]), grp)
            for key, grp in df.groupby(["version", "gpu_name"], sort=False)
        ]
    else:
        raw_groups = [(str(key), None, grp) for key, grp in df.groupby("version", sort=False)]
    return sorted(raw_groups, key=lambda item: (_version_sort_key(item[0]), item[1] or ""))


def plot_runtime_scaling(
    df: pd.DataFrame,
    *,
    dim: int,
    k_clusters: int,
    ctx: ProvenanceContext,
    out_dir: Path,
) -> list[Path]:
    """Generate IEEE single-column log-log runtime scaling plot."""
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    for version, gpu_name, grp in _iter_series_groups(df, include_hw=ctx.is_mixed_hw):
        grp = grp.sort_values("n_points")
        ax.plot(
            grp["n_points"],
            grp["kernel_ms"],
            marker=_stage_marker(version),
            color=_stage_color(version),
            label=_stage_label(version, gpu_name, include_hw=ctx.is_mixed_hw),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Points ($N$)")
    ax.set_ylabel("Kernel Time (ms)")
    ax.set_title(f"Runtime Scaling ($D={dim}, K={k_clusters}$)\n{ctx.subtitle}", fontsize=8.0)
    ax.legend(frameon=True, framealpha=0.9)
    _apply_diagnostic_watermark(fig, ctx)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if ctx.diagnostic_banner else 1.0))
    return _save_figure(fig, out_dir, "runtime_scaling")


def plot_speedup_vs_v0(
    df: pd.DataFrame,
    *,
    dim: int,
    k_clusters: int,
    ctx: ProvenanceContext,
    out_dir: Path,
) -> tuple[list[Path], str | None]:
    """Generate IEEE double-column speedup plot (kernel and E2E speedup vs. V0)."""
    cpu_df = df[df["is_cpu"]][list(BASELINE_MATCH_COLS) + ["kernel_ms", "e2e_ms"]].rename(
        columns={"kernel_ms": "cpu_kernel_ms", "e2e_ms": "cpu_e2e_ms"}
    )
    gpu_df = df[~df["is_cpu"]]

    merged = gpu_df.merge(cpu_df, on=list(BASELINE_MATCH_COLS), how="inner")
    unmatched = len(gpu_df) - len(merged)
    warning: str | None = None
    if unmatched > 0:
        warning = (
            f"speedup_vs_v0: skipped {unmatched} GPU record(s) without a matching V0 baseline "
            f"on {list(BASELINE_MATCH_COLS)}."
        )
    if merged.empty:
        return [], warning or "speedup_vs_v0: no GPU records matched a V0 baseline."

    merged["kernel_speedup"] = merged["cpu_kernel_ms"] / merged["kernel_ms"]
    merged["e2e_speedup"] = merged["cpu_e2e_ms"] / merged["e2e_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True)
    groups = _iter_series_groups(merged, include_hw=ctx.is_mixed_hw)

    for ax, metric_col, panel_title in (
        (axes[0], "kernel_speedup", "Kernel-Only Speedup"),
        (axes[1], "e2e_speedup", "End-to-End Speedup (incl. H2D/D2H)"),
    ):
        for version, gpu_name, grp in groups:
            grp = grp.sort_values("n_points")
            ax.plot(
                grp["n_points"],
                grp[metric_col],
                marker=_stage_marker(version),
                color=_stage_color(version),
                label=_stage_label(version, gpu_name, include_hw=ctx.is_mixed_hw),
            )
        ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0, label="1× (V0 CPU)")
        ax.set_xscale("log")
        ax.set_xlabel("Number of Points ($N$)")
        ax.set_title(panel_title, fontsize=8.5)

    axes[0].set_ylabel("Speedup vs. V0 ($\\times$)")
    axes[1].legend(frameon=True, framealpha=0.9, loc="best")
    fig.suptitle(f"Speedup vs. V0 ($D={dim}, K={k_clusters}$) — {ctx.subtitle}", fontsize=8.5)
    _apply_diagnostic_watermark(fig, ctx)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if ctx.diagnostic_banner else 0.94))
    return _save_figure(fig, out_dir, "speedup_vs_v0"), warning


def plot_kernel_vs_e2e(
    gpu_df: pd.DataFrame,
    *,
    n_points: int,
    dim: int,
    k_clusters: int,
    ctx: ProvenanceContext,
    out_dir: Path,
) -> list[Path]:
    """Generate IEEE single-column stacked bar of GPU kernel vs. transfer/host overhead."""
    ordered = gpu_df.copy()
    ordered["_sort_key"] = ordered["version"].map(_version_sort_key)
    ordered = ordered.sort_values(["_sort_key", "gpu_name"])

    labels = [
        _stage_label(r.version, r.gpu_name, include_hw=ctx.is_mixed_hw)
        for r in ordered.itertuples()
    ]
    kernel_vals = ordered["kernel_ms"].to_numpy(dtype=float)
    e2e_vals = ordered["e2e_ms"].to_numpy(dtype=float)
    overhead_vals = np.maximum(0.0, e2e_vals - kernel_vals)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    ax.bar(x, kernel_vals, width=0.58, color="#0072B2", label="Kernel Time")
    ax.bar(
        x,
        overhead_vals,
        width=0.58,
        bottom=kernel_vals,
        color="#E69F00",
        label="Host / PCIe Overhead",
    )

    for idx, (k_ms, total_ms) in enumerate(zip(kernel_vals, e2e_vals, strict=True)):
        pct = 100.0 * k_ms / total_ms if total_ms > 0 else 0.0
        ax.text(
            idx,
            total_ms * 1.02,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=20 if ctx.is_mixed_hw else 0, ha="right" if ctx.is_mixed_hw else "center"
    )
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0.0, float(np.max(e2e_vals)) * 1.18)
    ax.set_title(
        f"GPU Kernel vs. E2E ($N={n_points:,}, D={dim}, K={k_clusters}$)\n{ctx.subtitle}",
        fontsize=8.0,
    )
    ax.legend(frameon=True, framealpha=0.9)
    _apply_diagnostic_watermark(fig, ctx)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if ctx.diagnostic_banner else 1.0))
    return _save_figure(fig, out_dir, "kernel_vs_e2e")


def plot_regime_crossover(
    gpu_df: pd.DataFrame,
    cpu_df: pd.DataFrame,
    *,
    n_points: int,
    ctx: ProvenanceContext,
    out_dir: Path,
) -> list[Path]:
    """Generate IEEE 2D (D, K) crossover matrix showing fastest verified GPU version."""
    dims = sorted(int(v) for v in gpu_df["dim"].unique().tolist())
    ks = sorted(int(v) for v in gpu_df["k_clusters"].unique().tolist())

    cpu_lookup: dict[tuple[int, int, int, int, int, float], float] = {
        (
            int(r.n_points),
            int(r.dim),
            int(r.k_clusters),
            int(r.seed),
            int(r.max_iters),
            float(r.epsilon),
        ): float(r.kernel_ms)
        for r in cpu_df.itertuples()
    }

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    present_winners: list[str] = []

    for row_idx, d_val in enumerate(dims):
        for col_idx, k_val in enumerate(ks):
            cell_df = gpu_df[(gpu_df["dim"] == d_val) & (gpu_df["k_clusters"] == k_val)]
            if cell_df.empty:
                rect = mpatches.Rectangle(
                    (col_idx, row_idx), 1.0, 1.0, facecolor="#F0F0F0", edgecolor="#CCCCCC"
                )
                ax.add_patch(rect)
                ax.text(
                    col_idx + 0.5, row_idx + 0.5, "—", ha="center", va="center", color="#888888"
                )
                continue

            best = cell_df.sort_values("kernel_ms").iloc[0]
            winner = str(best["version"])
            if winner not in present_winners:
                present_winners.append(winner)

            color = _stage_color(winner)
            rect = mpatches.Rectangle(
                (col_idx, row_idx),
                1.0,
                1.0,
                facecolor=color,
                alpha=0.82,
                edgecolor="#FFFFFF",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            match_key = (
                int(best["n_points"]),
                int(best["dim"]),
                int(best["k_clusters"]),
                int(best["seed"]),
                int(best["max_iters"]),
                float(best["epsilon"]),
            )
            label_lines = [winner.upper(), f"{float(best['kernel_ms']):.2f} ms"]
            if match_key in cpu_lookup:
                speedup = cpu_lookup[match_key] / float(best["kernel_ms"])
                label_lines.append(f"({speedup:.1f}×)")

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                "\n".join(label_lines),
                ha="center",
                va="center",
                color="#FFFFFF",
                fontsize=7.2,
                fontweight="bold",
            )

    ax.set_xlim(0, len(ks))
    ax.set_ylim(0, len(dims))
    ax.set_xticks(np.arange(len(ks)) + 0.5)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(np.arange(len(dims)) + 0.5)
    ax.set_yticklabels([str(d) for d in dims])
    ax.set_xlabel("Clusters ($K$)")
    ax.set_ylabel("Dimensions ($D$)")
    ax.grid(False)
    ax.set_title(f"Fastest GPU Stage by $(D, K)$ ($N={n_points:,}$)\n{ctx.subtitle}", fontsize=8.2)

    present_winners.sort(key=_version_sort_key)
    handles = [
        mpatches.Patch(color=_stage_color(v), alpha=0.82, label=_stage_label(v))
        for v in present_winners
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(handles)),
            frameon=True,
        )

    _apply_diagnostic_watermark(fig, ctx)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if ctx.diagnostic_banner else 1.0))
    return _save_figure(fig, out_dir, "regime_crossover")


def run_plotting(
    inputs: Sequence[Path],
    *,
    out_dir: Path,
    plot_target: str = "all",
    gpu_filter: str | None = None,
    n_filter: int | None = None,
    d_filter: int | None = None,
    k_filter: int | None = None,
    allow_unverified: bool = False,
    allow_mixed_gpu: bool = False,
) -> tuple[list[Path], list[str]]:
    """Load, validate, resolve slices atomically, and render requested IEEE figures."""
    df = load_results(inputs)
    validated_df, ctx = validate_provenance(
        df,
        gpu_filter=gpu_filter,
        allow_unverified=allow_unverified,
        allow_mixed_gpu=allow_mixed_gpu,
    )

    selected_plots = (
        ("scaling", "speedup", "breakdown", "crossover") if plot_target == "all" else (plot_target,)
    )
    plan = resolve_plot_plan(
        validated_df,
        ctx,
        selected_plots=selected_plots,
        n_filter=n_filter,
        d_filter=d_filter,
        k_filter=k_filter,
    )

    _configure_ieee_style()
    saved_paths: list[Path] = []
    notices: list[str] = []

    if "scaling" in plan:
        entry = plan["scaling"]
        saved_paths.extend(
            plot_runtime_scaling(
                entry["df"],
                dim=entry["dim"],
                k_clusters=entry["k_clusters"],
                ctx=ctx,
                out_dir=out_dir,
            )
        )

    if "speedup" in plan:
        entry = plan["speedup"]
        if "skip" in entry:
            notices.append(f"Skipped speedup_vs_v0: {entry['skip']}.")
        else:
            paths, warn = plot_speedup_vs_v0(
                entry["df"],
                dim=entry["dim"],
                k_clusters=entry["k_clusters"],
                ctx=ctx,
                out_dir=out_dir,
            )
            saved_paths.extend(paths)
            if warn:
                notices.append(warn)

    if "breakdown" in plan:
        entry = plan["breakdown"]
        if "skip" in entry:
            notices.append(f"Skipped kernel_vs_e2e: {entry['skip']}.")
        else:
            saved_paths.extend(
                plot_kernel_vs_e2e(
                    entry["df"],
                    n_points=entry["n_points"],
                    dim=entry["dim"],
                    k_clusters=entry["k_clusters"],
                    ctx=ctx,
                    out_dir=out_dir,
                )
            )

    if "crossover" in plan:
        entry = plan["crossover"]
        if "skip" in entry:
            notices.append(f"Skipped regime_crossover: {entry['skip']}.")
        else:
            saved_paths.extend(
                plot_regime_crossover(
                    entry["gpu_df"],
                    entry["cpu_df"],
                    n_points=entry["n_points"],
                    ctx=ctx,
                    out_dir=out_dir,
                )
            )

    return saved_paths, notices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IEEE 2-column benchmark figures from verified kuda_means JSON results."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more benchmark .json files or directories containing *.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for generated .pdf and .png figures (default: results/figures)",
    )
    parser.add_argument(
        "--plot",
        choices=PLOT_CHOICES,
        default="all",
        help="Which figure(s) to generate (default: all)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Filter GPU records to a single hardware.gpu_name",
    )
    parser.add_argument("--n", type=int, default=None, help="Filter to a single N (n_points)")
    parser.add_argument("--d", type=int, default=None, help="Filter to a single D (dim)")
    parser.add_argument("--k", type=int, default=None, help="Filter to a single K (k_clusters)")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Diagnostic mode: allow verified=false records (watermarked on output)",
    )
    parser.add_argument(
        "--allow-mixed-gpu",
        action="store_true",
        help="Diagnostic mode: allow mixed GPU/CPU hardware (labeled per device & watermarked)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        saved, notices = run_plotting(
            args.inputs,
            out_dir=args.out,
            plot_target=args.plot,
            gpu_filter=args.gpu,
            n_filter=args.n,
            d_filter=args.d,
            k_filter=args.k,
            allow_unverified=args.allow_unverified,
            allow_mixed_gpu=args.allow_mixed_gpu,
        )
    except PlotValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for msg in notices:
        print(f"Note: {msg}")
    for path in saved:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
