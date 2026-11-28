"""Gate C verification tests for unit v2-memory-tuning.

Tests cover all Acceptance Criteria from docs/aidlc/units/v2-memory-tuning.md:
- AC1: Implementation existence, registration, constant memory capacity (<= 16384 floats),
       template fallback structure (UseConstant), empty-cluster safety, formatting/syntax.
- AC2: Happy path execution & verification vs V0 CPU baseline (N=100000, D=16, K=8).
- AC3: Exact boundary (K*D = 16384) and overflow fallback (K*D > 16384) execution & correctness.
- AC4: Unaligned points warp safety (N=99999, not divisible by 32).
- AC5: Target benchmark result verification: results/v2_n100k_d16_k8.json existence,
       schema compliance with v1 baseline, version == "v2", verified: true, kernel_ms > 0,
       and complete HardwareProvenance.
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
V2_SRC = REPO_ROOT / "src" / "v2_memory" / "kmeans_v2.cu"
GPU_BIN = REPO_ROOT / "bin" / "kuda_means_gpu"
BENCHMARK_JSON = REPO_ROOT / "results" / "v2_n100k_d16_k8.json"
V1_BASELINE_JSON = REPO_ROOT / "results" / "v1_n100k_d16_k8.json"

MAX_CONSTANT_FLOATS = 16384  # 64 KB / sizeof(float) = 16384


def _read_v2_source() -> str:
    if not V2_SRC.exists():
        raise AssertionError(f"Source file does not exist: {V2_SRC}")
    return V2_SRC.read_text(encoding="utf-8")


def _run_gpu_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(GPU_BIN)] + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _parse_output_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    verify_match = re.search(
        r"\[verify\]\s+status=(PASS|FAIL)\s+assignment_mismatches=(\d+)\s+"
        r"max_centroid_abs_diff=([0-9.eE+-]+)\s+sse_rel_diff=([0-9.eE+-]+)",
        stdout,
    )
    if verify_match:
        metrics["verify_status"] = verify_match.group(1)
        metrics["assignment_mismatches"] = int(verify_match.group(2))
        metrics["max_centroid_abs_diff"] = float(verify_match.group(3))
        metrics["sse_rel_diff"] = float(verify_match.group(4))

    run_match = re.search(
        r"\[kuda_means\]\s+version=(\S+)\s+hw=\"([^\"]+)\"\s+commit=(\S+)\s+"
        r"N=(\d+)\s+D=(\d+)\s+K=(\d+)\s+iters=(\d+)\s+sse=([0-9.eE+-]+)\s+"
        r"kernel_ms=([0-9.eE+-]+)\s+e2e_ms=([0-9.eE+-]+)\s+ms_per_iter=([0-9.eE+-]+)",
        stdout,
    )
    if run_match:
        metrics["version"] = run_match.group(1)
        metrics["gpu_name"] = run_match.group(2)
        metrics["git_commit"] = run_match.group(3)
        metrics["n_points"] = int(run_match.group(4))
        metrics["dim"] = int(run_match.group(5))
        metrics["k_clusters"] = int(run_match.group(6))
        metrics["iterations"] = int(run_match.group(7))
        metrics["sse"] = float(run_match.group(8))
        metrics["kernel_ms"] = float(run_match.group(9))
        metrics["e2e_ms"] = float(run_match.group(10))
        metrics["ms_per_iter"] = float(run_match.group(11))

    return metrics


class TestV2MemoryTuning(unittest.TestCase):
    # =========================================================================
    # AC1: Implementation Existence, Contract & Structural Integrity
    # =========================================================================

    def test_ac1_source_existence_and_registration(self) -> None:
        """AC1: src/v2_memory/kmeans_v2.cu exists and registers 'v2' runner."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        src = _read_v2_source()
        self.assertGreater(len(src.strip()), 0, "Source file must not be empty")

        # Registration macro KUDA_REGISTER_GPU_RUNNER("v2", ...)
        reg_pattern = r'KUDA_REGISTER_GPU_RUNNER\s*\(\s*"v2"\s*,'
        self.assertRegex(
            src,
            reg_pattern,
            'Missing GPU runner registration: KUDA_REGISTER_GPU_RUNNER("v2", ...)',
        )

    def test_ac1_constant_memory_capacity(self) -> None:
        """AC1: Centroids staged into __constant__ memory <= 16384 floats (64 KB)."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        src = _read_v2_source()

        # Check for __constant__ declaration
        self.assertIn("__constant__", src, "Missing __constant__ memory declaration")

        # Capacity check: c_centroids[16384] or c_centroids[kMaxConstantCentroids] <= 16384
        const_match = re.search(
            r"__constant__\s+(?:float\s+)?([A-Za-z0-9_]+)\s*\[([^\]]+)\]",
            src,
        )
        self.assertIsNotNone(
            const_match,
            "Could not find __constant__ array declaration matching '__constant__ float name[size]'",
        )
        assert const_match is not None
        size_expr = const_match.group(2).strip()

        # Determine numeric capacity
        if size_expr.isdigit():
            capacity = int(size_expr)
        else:
            # Look up constexpr / const / #define definition in file
            def_match = re.search(
                rf"(?:constexpr\s+(?:int|size_t)|#define)\s+{re.escape(size_expr)}\s+([0-9]+)",
                src,
            )
            self.assertIsNotNone(
                def_match,
                f"Constant memory size identifier '{size_expr}' not resolved to an integer literal",
            )
            assert def_match is not None
            capacity = int(def_match.group(1))

        self.assertLessEqual(
            capacity,
            MAX_CONSTANT_FLOATS,
            f"Constant centroid buffer size ({capacity}) exceeds 64 KB limit ({MAX_CONSTANT_FLOATS} floats)",
        )
        self.assertGreater(capacity, 0, "Constant centroid buffer size must be > 0")

        # Staging mechanism: cudaMemcpyToSymbol
        self.assertIn(
            "cudaMemcpyToSymbol",
            src,
            "Missing cudaMemcpyToSymbol for staging centroids into constant memory",
        )

    def test_ac1_template_fallback_and_soa_structure(self) -> None:
        """AC1: Zero-branch UseConstant template fallback and Coalesced SoA layout."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        src = _read_v2_source()

        # Template fallback structure
        self.assertRegex(
            src,
            r"template\s*<[^>]*UseConstant[^>]*>",
            "Kernel must define template parameter 'UseConstant' for zero-branch fallback",
        )
        self.assertRegex(
            src,
            r"if\s+constexpr\s*\(\s*UseConstant\s*\)",
            "Kernel must branch at compile time via 'if constexpr (UseConstant)'",
        )

        # Global memory fallback path reading from d_centroids (via __ldg or __restrict__)
        self.assertRegex(
            src,
            r"(__ldg|d_centroids)",
            "Fallback path must access centroids from global memory (d_centroids / __ldg)",
        )

        # SoA layout indexing pattern (e.g. j * n + idx or D x N indexing)
        self.assertRegex(
            src,
            r"(\b[jJ]\s*\*\s*[nN]\s*\+\s*idx|\b[nN]\s*\*\s*[jJ]\s*\+\s*idx|SoA|Structure-of-Arrays)",
            "Implementation must implement or document Coalesced SoA layout (D x N)",
        )

    def test_ac1_empty_cluster_safety(self) -> None:
        """AC1: Empty cluster safety avoids division by zero and preserves prior centroid."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        src = _read_v2_source()

        # Check that empty clusters (count == 0) are handled safely
        self.assertRegex(
            src,
            r"(count\s*==\s*0|counts\[\w+\]\s*==\s*0)",
            "Centroid update must check for count == 0 to prevent division by zero on empty clusters",
        )

    def test_ac1_formatting_and_cpu_build(self) -> None:
        """AC1: Formatting and syntax pass clang-format and clean make cpu build."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")

        # clang-format dry-run check
        clang_format = shutil.which("clang-format")
        if clang_format:
            fmt_res = subprocess.run(
                [clang_format, "--dry-run", "--Werror", str(V2_SRC)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                fmt_res.returncode,
                0,
                f"clang-format style check failed on {V2_SRC}:\n{fmt_res.stderr}",
            )

        # Build CPU baseline to confirm no headers or common files are broken
        build_res = subprocess.run(
            ["make", "cpu"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            build_res.returncode,
            0,
            f"'make cpu' failed:\n{build_res.stderr}",
        )

    # =========================================================================
    # AC2: Happy Path Execution & Verification vs V0 CPU Baseline
    # =========================================================================

    def test_ac2_happy_path_cpu_verification(self) -> None:
        """AC2: Happy path execution (N=100000, D=16, K=8) matches V0 CPU baseline."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

        cmd = [
            "--version", "v2",
            "-n", "100000",
            "-d", "16",
            "-k", "8",
            "--max-iters", "50",
            "--warmup", "0",
            "--runs", "1",
        ]
        res = _run_gpu_command(cmd)
        self.assertEqual(
            res.returncode,
            0,
            f"V2 happy path execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )

        metrics = _parse_output_metrics(res.stdout)
        self.assertEqual(metrics.get("verify_status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertEqual(metrics.get("assignment_mismatches"), 0)
        self.assertLessEqual(metrics.get("max_centroid_abs_diff", 1.0), 1e-3)
        self.assertLessEqual(metrics.get("sse_rel_diff", 1.0), 1e-3)
        self.assertGreater(metrics.get("iterations", 0), 0)

    # =========================================================================
    # AC3: Constant Boundary & Overflow Fallback Execution
    # =========================================================================

    def test_ac3_exact_boundary_constant_memory(self) -> None:
        """AC3: Exact boundary K*D = 16384 (K=512, D=32) executes via constant memory."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

        cmd = [
            "--version", "v2",
            "-n", "2000",
            "-d", "32",
            "-k", "512",
            "--max-iters", "5",
            "--warmup", "0",
            "--runs", "1",
        ]
        res = _run_gpu_command(cmd)
        self.assertEqual(
            res.returncode,
            0,
            f"V2 exact boundary execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )

        metrics = _parse_output_metrics(res.stdout)
        self.assertEqual(metrics.get("verify_status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertEqual(metrics.get("assignment_mismatches"), 0)
        self.assertLessEqual(metrics.get("max_centroid_abs_diff", 1.0), 1e-3)

    def test_ac3_overflow_fallback_global_memory(self) -> None:
        """AC3: Overflow K*D = 32768 > 16384 (K=1024, D=32) falls back to global memory."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

        cmd = [
            "--version", "v2",
            "-n", "2000",
            "-d", "32",
            "-k", "1024",
            "--max-iters", "5",
            "--warmup", "0",
            "--runs", "1",
        ]
        res = _run_gpu_command(cmd)
        self.assertEqual(
            res.returncode,
            0,
            f"V2 overflow fallback execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )

        metrics = _parse_output_metrics(res.stdout)
        self.assertEqual(metrics.get("verify_status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertEqual(metrics.get("assignment_mismatches"), 0)
        self.assertLessEqual(metrics.get("max_centroid_abs_diff", 1.0), 1e-3)

    # =========================================================================
    # AC4: Unaligned Warp Boundary Safety
    # =========================================================================

    def test_ac4_unaligned_points_warp_safety(self) -> None:
        """AC4: Unaligned points N=99999 (not divisible by warp size 32) executes safely."""
        self.assertTrue(V2_SRC.exists(), f"Source file does not exist: {V2_SRC}")
        if not GPU_BIN.exists():
            self.skipTest(f"GPU binary {GPU_BIN} not available on this host")

        cmd = [
            "--version", "v2",
            "-n", "99999",
            "-d", "16",
            "-k", "8",
            "--max-iters", "5",
            "--warmup", "0",
            "--runs", "1",
        ]
        res = _run_gpu_command(cmd)
        self.assertEqual(
            res.returncode,
            0,
            f"V2 unaligned N execution failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}",
        )

        metrics = _parse_output_metrics(res.stdout)
        self.assertEqual(metrics.get("verify_status"), "PASS", f"Verification failed:\n{res.stdout}")
        self.assertEqual(metrics.get("assignment_mismatches"), 0)

    # =========================================================================
    # AC5: Target Benchmark Result Verification & Hardware Provenance
    # =========================================================================

    def test_ac5_benchmark_result_json_schema_and_provenance(self) -> None:
        """AC5: results/v2_n100k_d16_k8.json exists, verified, and has full provenance."""
        self.assertTrue(
            BENCHMARK_JSON.exists(),
            f"Target benchmark result not found: {BENCHMARK_JSON}",
        )

        data: dict[str, Any] = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))

        # Version & schema top-level keys
        for key in ("version", "config", "hardware", "metrics", "verified", "verification"):
            self.assertIn(key, data, f"Missing required top-level key '{key}' in {BENCHMARK_JSON}")

        self.assertEqual(data.get("version"), "v2", "version field must be 'v2'")
        self.assertIs(data.get("verified"), True, "verified field must be true")

        # Config validation
        config = data.get("config", {})
        self.assertEqual(config.get("n_points"), 100000)
        self.assertEqual(config.get("dim"), 16)
        self.assertEqual(config.get("k_clusters"), 8)

        # Verification block validation
        verification = data.get("verification", {})
        self.assertIs(verification.get("passed"), True)
        self.assertEqual(verification.get("assignment_mismatches"), 0)
        self.assertLessEqual(float(verification.get("max_centroid_abs_diff", 1.0)), 1e-3)
        self.assertLessEqual(float(verification.get("sse_rel_diff", 1.0)), 1e-3)

        # Metrics validation
        metrics = data.get("metrics", {})
        self.assertGreater(metrics.get("iterations", 0), 0)
        self.assertGreater(metrics.get("sse", 0.0), 0.0)
        self.assertGreater(metrics.get("kernel_ms", 0.0), 0.0)
        self.assertGreaterEqual(metrics.get("e2e_ms", 0.0), metrics.get("kernel_ms", 0.0))
        self.assertGreater(metrics.get("ms_per_iter", 0.0), 0.0)

        # Complete HardwareProvenance validation
        hardware = data.get("hardware", {})
        gpu_name = str(hardware.get("gpu_name", "")).strip()
        self.assertTrue(gpu_name, "hardware.gpu_name must be non-empty")
        self.assertNotEqual(gpu_name, "CPU", "hardware.gpu_name must be a GPU, not CPU")

        compute_cap = str(hardware.get("compute_cap", "")).strip()
        self.assertRegex(
            compute_cap,
            r"^sm_\d+$",
            f"hardware.compute_cap must be formatted as 'sm_XX', got '{compute_cap}'",
        )

        self.assertGreater(hardware.get("sm_count", 0), 0, "hardware.sm_count must be > 0")
        self.assertGreater(hardware.get("global_mem_mb", 0.0), 0.0, "hardware.global_mem_mb must be > 0")

        cuda_version = str(hardware.get("cuda_version", "")).strip()
        self.assertTrue(cuda_version, "hardware.cuda_version must be non-empty")
        self.assertNotIn(cuda_version, ("none", ""), "hardware.cuda_version must be specified")

        git_commit = str(hardware.get("git_commit", "")).strip()
        self.assertTrue(git_commit, "hardware.git_commit must be non-empty")
        self.assertNotIn(git_commit, ("unknown", ""), "hardware.git_commit must be valid commit SHA")


if __name__ == "__main__":
    unittest.main()
