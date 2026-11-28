"""Gate C verification tests for unit v3-privatized.

Tests cover the Acceptance Criteria from docs/aidlc/units/v3-privatized.md:
- AC1: Source existence, "v3" registration, shared-memory accumulators, warp primitives,
       shared-memory capacity check, empty-cluster safety, formatting / CPU build.
- AC2: Happy path vs. V0 CPU baseline (N=100000, D=16, K=8).
- AC3: Shared-memory path (K=64, D=64) and capacity-wall fallback (K=1024, D=32).
- AC4: Unaligned N=99999 (partial last warp).
- AC5: compute-sanitizer memcheck / racecheck / synccheck report 0 errors.
- AC6: results/v3_n100k_d16_k8.json schema, verified: true, full HardwareProvenance.

GPU-dependent tests are skipped when bin/kuda_means_gpu (or compute-sanitizer) is absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_SRC = REPO_ROOT / "src" / "v3_privatized" / "kmeans_v3.cu"
GPU_BIN = REPO_ROOT / "bin" / "kuda_means_gpu"
BENCHMARK_JSON = REPO_ROOT / "results" / "v3_n100k_d16_k8.json"

VERIFY_RE = re.compile(
    r"\[verify\]\s+status=(PASS|FAIL)\s+assignment_mismatches=(\d+)\s+"
    r"max_centroid_abs_diff=([0-9.eE+-]+)\s+sse_rel_diff=([0-9.eE+-]+)"
)
PATH_RE = re.compile(r"\[v3\]\s+reduction path:\s+(shared|global)")


def _read_v3_source() -> str:
    if not V3_SRC.exists():
        raise AssertionError(f"Source file does not exist: {V3_SRC}")
    return V3_SRC.read_text(encoding="utf-8")


def _run_v3(n: int, d: int, k: int, max_iters: int) -> subprocess.CompletedProcess[str]:
    args = [
        str(GPU_BIN),
        "--version",
        "v3",
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
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)


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


def _reduction_path(stderr: str) -> str | None:
    match = PATH_RE.search(stderr)
    return match.group(1) if match else None


class TestV3Privatized(unittest.TestCase):
    # =========================================================================
    # Helpers
    # =========================================================================

    def _require_gpu(self) -> None:
        self.assertTrue(V3_SRC.exists(), f"Source file does not exist: {V3_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

    def _assert_verified(self, res: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        self.assertEqual(
            res.returncode,
            0,
            f"V3 execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )
        metrics = _parse_verify(res.stdout)
        self.assertEqual(metrics.get("status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertEqual(metrics.get("assignment_mismatches"), 0)
        self.assertLessEqual(metrics.get("max_centroid_abs_diff", 1.0), 1e-3)
        self.assertLessEqual(metrics.get("sse_rel_diff", 1.0), 1e-3)
        return metrics

    # =========================================================================
    # AC1: Structure
    # =========================================================================

    def test_ac1_source_existence_and_registration(self) -> None:
        """AC1: src/v3_privatized/kmeans_v3.cu exists and registers the 'v3' runner."""
        src = _read_v3_source()
        self.assertRegex(src, r'KUDA_REGISTER_GPU_RUNNER\s*\(\s*"v3"\s*,')

    def test_ac1_shared_memory_privatization(self) -> None:
        """AC1: Per-block dynamic shared-memory accumulators with block barriers."""
        src = _read_v3_source()
        self.assertRegex(src, r"extern\s+__shared__", "Missing dynamic shared-memory buffer")
        self.assertIn("__syncthreads()", src, "Missing __syncthreads() barrier")

    def test_ac1_warp_primitives(self) -> None:
        """AC1: Warp-level aggregation via match/shuffle intrinsics."""
        src = _read_v3_source()
        self.assertIn("__match_any_sync", src, "Missing __match_any_sync peer grouping")
        self.assertRegex(src, r"__shfl(_down|_xor|_up)?_sync", "Missing warp shuffle reduction")

    def test_ac1_shared_memory_capacity_check(self) -> None:
        """AC1: Shared-memory capacity is queried and large requests opt in or fall back."""
        src = _read_v3_source()
        self.assertIn(
            "cudaDevAttrMaxSharedMemoryPerBlock",
            src,
            "Must query cudaDevAttrMaxSharedMemoryPerBlock[Optin] before launching",
        )
        self.assertIn(
            "cudaFuncAttributeMaxDynamicSharedMemorySize",
            src,
            "Must opt in to >48 KB dynamic shared memory via cudaFuncSetAttribute",
        )

    def test_ac1_empty_cluster_safety(self) -> None:
        """AC1: Centroid update guards empty clusters."""
        src = _read_v3_source()
        self.assertRegex(src, r"(count\s*==\s*0|counts\[\w+\]\s*==\s*0)")

    def test_ac1_formatting_and_cpu_build(self) -> None:
        """AC1: clang-format clean (when available) and `make cpu` still builds."""
        self.assertTrue(V3_SRC.exists(), f"Source file does not exist: {V3_SRC}")
        clang_format = shutil.which("clang-format")
        if clang_format:
            fmt_res = subprocess.run(
                [clang_format, "--dry-run", "--Werror", str(V3_SRC)],
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
    # AC2: Happy path
    # =========================================================================

    def test_ac2_happy_path_cpu_verification(self) -> None:
        """AC2: N=100000, D=16, K=8 matches the V0 CPU baseline via the shared path."""
        self._require_gpu()
        res = _run_v3(100000, 16, 8, 50)
        self._assert_verified(res)
        self.assertEqual(_reduction_path(res.stderr), "shared", res.stderr)

    # =========================================================================
    # AC3: Shared-memory capacity wall
    # =========================================================================

    def test_ac3_shared_path_mid_size(self) -> None:
        """AC3: K=64, D=64 (32.25 KB < 48 KB default) runs privatized and verifies."""
        self._require_gpu()
        res = _run_v3(20000, 64, 64, 5)
        self._assert_verified(res)
        self.assertEqual(_reduction_path(res.stderr), "shared", res.stderr)

    def test_ac3_capacity_wall_fallback(self) -> None:
        """AC3: K=1024, D=32 (~260 KB) exceeds shared memory, falls back, still verifies."""
        self._require_gpu()
        res = _run_v3(2000, 32, 1024, 5)
        self._assert_verified(res)
        self.assertEqual(_reduction_path(res.stderr), "global", res.stderr)

    # =========================================================================
    # AC4: Unaligned warp safety
    # =========================================================================

    def test_ac4_unaligned_points_warp_safety(self) -> None:
        """AC4: N=99999 (partial last warp) verifies."""
        self._require_gpu()
        self._assert_verified(_run_v3(99999, 16, 8, 5))

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
                        "v3",
                        "-n",
                        "4099",
                        "-d",
                        "16",
                        "-k",
                        "8",
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
        """AC6: results/v3_n100k_d16_k8.json exists, verified, with full provenance."""
        self.assertTrue(BENCHMARK_JSON.exists(), f"Missing benchmark result: {BENCHMARK_JSON}")
        data: dict[str, Any] = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))

        for key in ("version", "config", "hardware", "metrics", "verified", "verification"):
            self.assertIn(key, data, f"Missing top-level key '{key}'")

        self.assertEqual(data["version"], "v3")
        self.assertIs(data["verified"], True)

        config = data["config"]
        self.assertEqual(config.get("n_points"), 100000)
        self.assertEqual(config.get("dim"), 16)
        self.assertEqual(config.get("k_clusters"), 8)

        verification = data["verification"]
        self.assertIs(verification.get("passed"), True)
        self.assertEqual(verification.get("assignment_mismatches"), 0)
        self.assertLessEqual(float(verification.get("max_centroid_abs_diff", 1.0)), 1e-3)
        self.assertLessEqual(float(verification.get("sse_rel_diff", 1.0)), 1e-3)

        metrics = data["metrics"]
        self.assertGreater(metrics.get("iterations", 0), 0)
        self.assertGreater(metrics.get("sse", 0.0), 0.0)
        self.assertGreater(metrics.get("kernel_ms", 0.0), 0.0)
        self.assertGreaterEqual(metrics.get("e2e_ms", 0.0), metrics.get("kernel_ms", 0.0))
        self.assertGreater(metrics.get("ms_per_iter", 0.0), 0.0)

        hardware = data["hardware"]
        gpu_name = str(hardware.get("gpu_name", "")).strip()
        self.assertTrue(gpu_name)
        self.assertNotEqual(gpu_name, "CPU")
        self.assertRegex(str(hardware.get("compute_cap", "")), r"^sm_\d+$")
        self.assertGreater(hardware.get("sm_count", 0), 0)
        self.assertGreater(hardware.get("global_mem_mb", 0.0), 0.0)
        self.assertNotIn(str(hardware.get("cuda_version", "")).strip(), ("none", ""))
        self.assertNotIn(str(hardware.get("git_commit", "")).strip(), ("unknown", ""))


if __name__ == "__main__":
    unittest.main()
