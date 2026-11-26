"""Unit tests for scripts/plot_results.py JSON ingestion and provenance checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import plot_results  # noqa: E402


def _make_record(
    *,
    version: str = "v1",
    n_points: int = 100000,
    dim: int = 16,
    k_clusters: int = 8,
    kernel_ms: float = 10.0,
    e2e_ms: float = 12.5,
    gpu_name: str = "NVIDIA GeForce RTX 4090",
    compute_cap: str = "sm_89",
    cuda_version: str = "12.2",
    verified: bool = True,
    seed: int = 42,
    max_iters: int = 50,
    epsilon: float = 1e-4,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_cpu = version.lower() in {"cpu", "v0"}
    record: dict[str, Any] = {
        "version": version,
        "config": {
            "n_points": n_points,
            "dim": dim,
            "k_clusters": k_clusters,
            "max_iters": max_iters,
            "epsilon": epsilon,
            "seed": seed,
            "warmup_runs": 1,
            "bench_runs": 5,
        },
        "hardware": {
            "gpu_name": "CPU (AMD EPYC 7313 16-Core Processor)" if is_cpu else gpu_name,
            "compute_cap": "none" if is_cpu else compute_cap,
            "sm_count": 0 if is_cpu else 128,
            "global_mem_mb": 0.0 if is_cpu else 24564.0,
            "cuda_version": "none" if is_cpu else cuda_version,
            "git_commit": "abc1234",
        },
        "metrics": {
            "iterations": max_iters,
            "sse": 8388364.26,
            "kernel_ms": kernel_ms,
            "e2e_ms": kernel_ms if is_cpu else e2e_ms,
            "ms_per_iter": kernel_ms / max_iters,
        },
        "verified": verified,
    }
    if verification is not None:
        record["verification"] = verification
    return record


def _write_json(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class PlotResultsTests(unittest.TestCase):
    def test_real_committed_cpu_json_runs_cleanly(self) -> None:
        real_json = REPO_ROOT / "results" / "v0_cpu_n100k_d16_k8.json"
        self.assertTrue(real_json.exists(), "Expected committed V0 CPU benchmark JSON")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "figs"
            saved, notices = plot_results.run_plotting([real_json], out_dir=out_dir)
            self.assertIn(out_dir / "runtime_scaling.pdf", saved)
            self.assertIn(out_dir / "runtime_scaling.png", saved)
            self.assertTrue((out_dir / "runtime_scaling.pdf").exists())
            self.assertTrue((out_dir / "runtime_scaling.png").exists())
            # GPU-only plots should skip gracefully with notices
            self.assertEqual(len(notices), 3)

    def test_rejects_unverified_by_default_and_allows_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_json(tmp_dir, "v0.json", _make_record(version="cpu", kernel_ms=400.0))
            _write_json(
                tmp_dir,
                "v1_bad.json",
                _make_record(version="v1", kernel_ms=20.0, e2e_ms=25.0, verified=False),
            )

            out_dir = tmp_dir / "figs"
            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting([tmp_dir], out_dir=out_dir)
            self.assertFalse(out_dir.exists())

            saved, _ = plot_results.run_plotting([tmp_dir], out_dir=out_dir, allow_unverified=True)
            self.assertTrue((out_dir / "speedup_vs_v0.pdf").exists())
            self.assertGreaterEqual(len(saved), 6)

    def test_rejects_mixed_gpu_and_supports_gpu_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_json(tmp_dir, "v0.json", _make_record(version="cpu", kernel_ms=400.0))
            _write_json(
                tmp_dir,
                "v1_4090.json",
                _make_record(
                    version="v1",
                    gpu_name="NVIDIA GeForce RTX 4090",
                    compute_cap="sm_89",
                    kernel_ms=10.0,
                    e2e_ms=14.0,
                ),
            )
            _write_json(
                tmp_dir,
                "v1_t4.json",
                _make_record(
                    version="v1",
                    gpu_name="Tesla T4",
                    compute_cap="sm_75",
                    kernel_ms=45.0,
                    e2e_ms=52.0,
                ),
            )

            out_dir = tmp_dir / "figs"
            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting([tmp_dir], out_dir=out_dir)

            # Filtering to a single GPU succeeds
            saved, _ = plot_results.run_plotting(
                [tmp_dir], out_dir=out_dir, gpu_filter="NVIDIA GeForce RTX 4090"
            )
            self.assertTrue((out_dir / "speedup_vs_v0.pdf").exists())
            self.assertTrue((out_dir / "kernel_vs_e2e.pdf").exists())
            self.assertTrue((out_dir / "regime_crossover.pdf").exists())
            self.assertEqual(len(saved), 8)

            # Diagnostic mixed-GPU mode still refuses unsegregated crossover winner map
            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting(
                    [tmp_dir],
                    out_dir=tmp_dir / "diag",
                    plot_target="crossover",
                    allow_mixed_gpu=True,
                )

            # Diagnostic mixed-GPU mode works for scaling
            diag_saved, _ = plot_results.run_plotting(
                [tmp_dir], out_dir=tmp_dir / "diag", plot_target="scaling", allow_mixed_gpu=True
            )
            self.assertEqual(len(diag_saved), 2)

    def test_rejects_mixed_cpu_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rec1 = _make_record(version="cpu", n_points=10000, kernel_ms=40.0)
            rec2 = _make_record(version="cpu", n_points=100000, kernel_ms=400.0)
            rec2["hardware"]["gpu_name"] = "CPU (Intel Xeon)"
            _write_json(tmp_dir, "cpu1.json", rec1)
            _write_json(tmp_dir, "cpu2.json", rec2)
            _write_json(tmp_dir, "gpu1.json", _make_record(version="v1", n_points=10000))

            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting([tmp_dir], out_dir=tmp_dir / "figs")

            # Even with --allow-mixed-gpu, speedup/crossover reject mixed CPU baselines
            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting(
                    [tmp_dir],
                    out_dir=tmp_dir / "figs",
                    plot_target="speedup",
                    allow_mixed_gpu=True,
                )
            with self.assertRaises(plot_results.ProvenanceError):
                plot_results.run_plotting(
                    [tmp_dir],
                    out_dir=tmp_dir / "figs",
                    plot_target="crossover",
                    allow_mixed_gpu=True,
                )

    def test_rejects_malformed_gpu_provenance_and_timer_inversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bad_gpu = _make_record(version="v1", compute_cap="none", cuda_version="none")
            bad_gpu_path = _write_json(tmp_dir, "bad_gpu.json", bad_gpu)
            with self.assertRaises(plot_results.PlotValidationError):
                plot_results.load_results([bad_gpu_path])

            inverted = _make_record(version="v1", kernel_ms=20.0, e2e_ms=10.0)
            inv_path = _write_json(tmp_dir, "inverted.json", inverted)
            with self.assertRaises(plot_results.PlotValidationError):
                plot_results.load_results([inv_path])

            null_metric = _make_record(version="v1")
            null_metric["metrics"]["kernel_ms"] = None
            null_path = _write_json(tmp_dir, "null_metric.json", null_metric)
            with self.assertRaises(plot_results.PlotValidationError):
                plot_results.load_results([null_path])

    def test_rejects_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_json(tmp_dir, "a.json", _make_record(version="v1", kernel_ms=10.0))
            _write_json(tmp_dir, "b.json", _make_record(version="v1", kernel_ms=11.0))

            with self.assertRaises(plot_results.PlotValidationError):
                plot_results.run_plotting([tmp_dir], out_dir=tmp_dir / "figs")

    def test_strict_fail_fast_on_ambiguous_multi_regime_slices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for n_val in (10000, 100000):
                for d_val, k_val in ((16, 8), (64, 32)):
                    _write_json(
                        tmp_dir,
                        f"v0_n{n_val}_d{d_val}_k{k_val}.json",
                        _make_record(
                            version="cpu",
                            n_points=n_val,
                            dim=d_val,
                            k_clusters=k_val,
                            kernel_ms=0.004 * n_val,
                        ),
                    )
                    _write_json(
                        tmp_dir,
                        f"v1_n{n_val}_d{d_val}_k{k_val}.json",
                        _make_record(
                            version="v1",
                            n_points=n_val,
                            dim=d_val,
                            k_clusters=k_val,
                            kernel_ms=0.0002 * n_val,
                            e2e_ms=0.00025 * n_val,
                        ),
                    )
                    _write_json(
                        tmp_dir,
                        f"v3_n{n_val}_d{d_val}_k{k_val}.json",
                        _make_record(
                            version="v3",
                            n_points=n_val,
                            dim=d_val,
                            k_clusters=k_val,
                            kernel_ms=0.0001 * n_val if d_val == 16 else 0.00015 * n_val,
                            e2e_ms=0.00014 * n_val if d_val == 16 else 0.00019 * n_val,
                        ),
                    )

            out_dir = tmp_dir / "figs"
            # Without explicit flags, fail fast and write nothing
            with self.assertRaises(plot_results.PlotValidationError):
                plot_results.run_plotting([tmp_dir], out_dir=out_dir)
            self.assertFalse(out_dir.exists())

            # Scaling + speedup succeed with explicit --d and --k
            saved_scaling, _ = plot_results.run_plotting(
                [tmp_dir], out_dir=out_dir, plot_target="scaling", d_filter=16, k_filter=8
            )
            self.assertEqual(len(saved_scaling), 2)

            saved_speedup, _ = plot_results.run_plotting(
                [tmp_dir], out_dir=out_dir, plot_target="speedup", d_filter=16, k_filter=8
            )
            self.assertEqual(len(saved_speedup), 2)

            # Breakdown succeeds with explicit --n, --d, --k
            saved_breakdown, _ = plot_results.run_plotting(
                [tmp_dir],
                out_dir=out_dir,
                plot_target="breakdown",
                n_filter=100000,
                d_filter=16,
                k_filter=8,
            )
            self.assertEqual(len(saved_breakdown), 2)

            # Crossover succeeds with explicit --n
            saved_crossover, _ = plot_results.run_plotting(
                [tmp_dir], out_dir=out_dir, plot_target="crossover", n_filter=100000
            )
            self.assertEqual(len(saved_crossover), 2)

    def test_cli_exit_code_on_error_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bad_csv = tmp_dir / "benchmarks.csv"
            bad_csv.write_text("version,N,kernel_ms\n", encoding="utf-8")
            self.assertEqual(plot_results.main([str(bad_csv)]), 1)

            real_json = REPO_ROOT / "results" / "v0_cpu_n100k_d16_k8.json"
            self.assertEqual(plot_results.main([str(real_json), "--out", str(tmp_dir / "out")]), 0)


if __name__ == "__main__":
    unittest.main()
