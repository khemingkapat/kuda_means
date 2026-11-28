"""Gate C verification tests for unit v4-matmul.

Tests cover the Acceptance Criteria from docs/aidlc/units/v4-matmul.md:
- AC1: Source existence, "v4" registration, shared-memory GEMM tiles, centroid norms,
       full-expansion (cancellation) mode, empty-cluster safety, formatting / CPU build.
- AC2: Small regime vs. V0 CPU baseline (N=100000, D=16, K=8).
- AC3: Large regime (D=128, K=64) in both shifted and full-expansion modes.
- AC4: Tile edges (N=99999, D=20, K=100).
- AC5: compute-sanitizer memcheck / racecheck / synccheck report 0 errors.
- AC6: results/v4_n100k_d16_k8.json schema, verified: true, full HardwareProvenance.

GPU-dependent tests are skipped when bin/kuda_means_gpu (or compute-sanitizer) is absent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = REPO_ROOT / "src" / "v4_matmul" / "kmeans_v4.cu"
GPU_BIN = REPO_ROOT / "bin" / "kuda_means_gpu"
BENCHMARK_JSON = REPO_ROOT / "results" / "v4_n100k_d16_k8.json"

VERIFY_RE = re.compile(
    r"\[verify\]\s+status=(PASS|FAIL)\s+assignment_mismatches=(\d+)\s+"
    r"max_centroid_abs_diff=([0-9.eE+-]+)\s+sse_rel_diff=([0-9.eE+-]+)"
)
MODE_RE = re.compile(r"\[v4\]\s+distance:\s+gemm.*expansion=(shifted|full)")


def _read_v4_source() -> str:
    if not V4_SRC.exists():
        raise AssertionError(f"Source file does not exist: {V4_SRC}")
    return V4_SRC.read_text(encoding="utf-8")


def _run_v4(
    n: int, d: int, k: int, max_iters: int, full_expansion: bool = False
) -> subprocess.CompletedProcess[str]:
    args = [
        str(GPU_BIN),
        "--version",
        "v4",
        "-n",
        str(n),
        "-d",
        str(d),
        "-k",
        str(k),
        "--max-iters",
        str(max_iters),
        "--warmup",
        "0",
        "--runs",
        "1",
    ]
    env = dict(os.environ)
    env.pop("KUDA_V4_FULL_EXPANSION", None)
    if full_expansion:
        env["KUDA_V4_FULL_EXPANSION"] = "1"
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def _parse_verify(stdout: str) -> dict[str, Any]:
    match = VERIFY_RE.search(stdout)
    if not match:
        return {}
    return {
        "status": match.group(1),
        "assignment_mismatches": int(match.group(2)),
        "max_centroid_abs_diff": float(match.group(3)),
        "sse_rel_diff": float(match.group(4)),
    }


def _expansion_mode(stderr: str) -> str | None:
    match = MODE_RE.search(stderr)
    return match.group(1) if match else None


class TestV4MatMul(unittest.TestCase):
    # =========================================================================
    # Helpers
    # =========================================================================

    def _require_gpu(self) -> None:
        self.assertTrue(V4_SRC.exists(), f"Source file does not exist: {V4_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

    def _assert_verified(self, res: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        self.assertEqual(
            res.returncode,
            0,
            f"V4 execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )
        metrics = _parse_verify(res.stdout)
        self.assertEqual(metrics.get("status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertLessEqual(metrics.get("max_centroid_abs_diff", 1.0), 1e-3)
        self.assertLessEqual(metrics.get("sse_rel_diff", 1.0), 1e-3)
        return metrics

    # =========================================================================
    # AC1: Structure
    # =========================================================================

    def test_ac1_source_existence_and_registration(self) -> None:
        """AC1: src/v4_matmul/kmeans_v4.cu exists and registers the 'v4' runner."""
        src = _read_v4_source()
        self.assertRegex(src, r'KUDA_REGISTER_GPU_RUNNER\s*\(\s*"v4"\s*,')

    def test_ac1_tiled_gemm_structure(self) -> None:
        """AC1: Shared-memory GEMM tiles with block barriers and centroid norms."""
        src = _read_v4_source()
        self.assertRegex(src, r"__shared__[^;]*\[\s*kTileK\s*\]", "Missing shared GEMM tiles")
        self.assertIn("__syncthreads()", src, "Missing __syncthreads() barrier")
        self.assertIn("centroid_norms_v4_kernel", src, "Missing ||c||^2 precompute kernel")
        self.assertIn("point_norms_v4_kernel", src, "Missing ||x||^2 precompute kernel")
        self.assertIn("KUDA_V4_FULL_EXPANSION", src, "Missing full-expansion (cancellation) mode")

    def test_ac1_empty_cluster_safety(self) -> None:
        """AC1: Centroid update guards empty clusters."""
        src = _read_v4_source()
        self.assertRegex(src, r"(count\s*==\s*0|counts\[\w+\]\s*==\s*0)")

    def test_ac1_formatting_and_cpu_build(self) -> None:
        """AC1: clang-format clean (when available) and `make cpu` still builds."""
        self.assertTrue(V4_SRC.exists(), f"Source file does not exist: {V4_SRC}")
        clang_format = shutil.which("clang-format")
        if clang_format:
            fmt_res = subprocess.run(
                [clang_format, "--dry-run", "--Werror", str(V4_SRC)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(fmt_res.returncode, 0, f"clang-format failed:\n{fmt_res.stderr}")

        if not shutil.which("make"):
            self.skipTest("make not available on this host")
        build_res = subprocess.run(["make", "cpu"], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(build_res.returncode, 0, f"'make cpu' failed:\n{build_res.stderr}")

    # =========================================================================
    # AC2: Small regime
    # =========================================================================

    def test_ac2_small_regime_cpu_verification(self) -> None:
        """AC2: N=100000, D=16, K=8 matches the V0 CPU baseline (shifted distance)."""
        self._require_gpu()
        res = _run_v4(100000, 16, 8, 50)
        self._assert_verified(res)
        self.assertEqual(_expansion_mode(res.stderr), "shifted", res.stderr)

    # =========================================================================
    # AC3: Large regime + cancellation check
    # =========================================================================

    def test_ac3_large_regime_shifted(self) -> None:
        """AC3: D=128, K=64 verifies with the shifted distance."""
        self._require_gpu()
        self._assert_verified(_run_v4(20000, 128, 64, 10))

    def test_ac3_large_regime_full_expansion_reports_mismatches(self) -> None:
        """AC3: Full expansion runs and reports its mismatch count (cancellation check)."""
        self._require_gpu()
        res = _run_v4(20000, 128, 64, 10, full_expansion=True)
        self.assertEqual(_expansion_mode(res.stderr), "full", res.stderr)
        metrics = _parse_verify(res.stdout)
        self.assertIn("assignment_mismatches", metrics, f"No [verify] line:\n{res.stdout}")

    # =========================================================================
    # AC4: Tile edges
    # =========================================================================

    def test_ac4_tile_edges(self) -> None:
        """AC4: N, D, K not multiples of the 64x64x16 tile verify."""
        self._require_gpu()
        self._assert_verified(_run_v4(99999, 20, 100, 5))

    # =========================================================================
    # AC5: Race freedom (compute-sanitizer)
    # =========================================================================

    def test_ac5_compute_sanitizer_clean(self) -> None:
        """AC5: memcheck, racecheck, synccheck report 0 errors on a small dataset."""
        self._require_gpu()
        sanitizer = shutil.which("compute-sanitizer")
        if not sanitizer:
            self.skipTest("compute-sanitizer not available on this host")

        for tool in ("memcheck", "racecheck", "synccheck"):
            with self.subTest(tool=tool):
                res = subprocess.run(
                    [
                        sanitizer,
                        "--tool",
                        tool,
                        "--error-exitcode",
                        "1",
                        str(GPU_BIN),
                        "--version",
                        "v4",
                        "-n",
                        "4099",
                        "-d",
                        "20",
                        "-k",
                        "70",
                        "--max-iters",
                        "3",
                        "--warmup",
                        "0",
                        "--runs",
                        "1",
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    res.returncode, 0, f"{tool} reported errors:\n{res.stdout}\n{res.stderr}"
                )
                self.assertRegex(
                    res.stdout + res.stderr,
                    r"(ERROR SUMMARY:\s+0 errors|RACECHECK SUMMARY:\s+0 hazards)",
                )

    # =========================================================================
    # AC6: Benchmark artifact & provenance
    # =========================================================================

    def test_ac6_benchmark_result_json_schema_and_provenance(self) -> None:
        """AC6: results/v4_n100k_d16_k8.json exists, verified, with full provenance."""
        self.assertTrue(BENCHMARK_JSON.exists(), f"Missing benchmark result: {BENCHMARK_JSON}")
        data: dict[str, Any] = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))

        for key in ("version", "config", "hardware", "metrics", "verified", "verification"):
            self.assertIn(key, data, f"Missing top-level key '{key}'")

        self.assertEqual(data["version"], "v4")
        self.assertIs(data["verified"], True)

        config = data["config"]
        self.assertEqual(config.get("n_points"), 100000)
        self.assertEqual(config.get("dim"), 16)
        self.assertEqual(config.get("k_clusters"), 8)

        verification = data["verification"]
        self.assertIs(verification.get("passed"), True)
        self.assertLessEqual(float(verification.get("max_centroid_abs_diff", 1.0)), 1e-3)
        self.assertLessEqual(float(verification.get("sse_rel_diff", 1.0)), 1e-3)

        metrics = data["metrics"]
        self.assertGreater(metrics.get("iterations", 0), 0)
        self.assertGreater(metrics.get("kernel_ms", 0.0), 0.0)
        self.assertGreaterEqual(metrics.get("e2e_ms", 0.0), metrics.get("kernel_ms", 0.0))

        hardware = data["hardware"]
        self.assertTrue(str(hardware.get("gpu_name", "")).strip())
        self.assertRegex(str(hardware.get("compute_cap", "")), r"^sm_\d+$")
        self.assertGreater(hardware.get("sm_count", 0), 0)
        self.assertNotIn(str(hardware.get("cuda_version", "")).strip(), ("none", ""))
        self.assertNotIn(str(hardware.get("git_commit", "")).strip(), ("unknown", ""))


if __name__ == "__main__":
    unittest.main()
