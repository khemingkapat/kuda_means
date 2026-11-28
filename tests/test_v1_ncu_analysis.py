"""Gate C verification tests for unit v1-ncu-analysis."""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = REPO_ROOT / "profiles"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
PROFILE_SBATCH = REPO_ROOT / "scripts" / "slurm" / "profile.sbatch"
NCU_REPORT = PROFILES_DIR / "v1_ncu.txt"
ANALYSIS_DOC = DOCS_DIR / "v1_analysis.md"
REFERENCES_BIB = DOCS_DIR / "references.bib"
BASELINE_V1_JSON = "v1_n100k_d16_k8.json"


def _find_new_v1_result_jsons() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(path for path in RESULTS_DIR.glob("v1_*.json") if path.name != BASELINE_V1_JSON)


def _load_bib_keys(bib_path: Path) -> set[str]:
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", text))


def _strip_markdown_code(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    return re.sub(r"`[^`]*`", "", text)


class V1NcuAnalysisTests(unittest.TestCase):
    def test_ac1_ncu_report_exists_and_contains_required_sections(self) -> None:
        self.assertTrue(NCU_REPORT.exists(), f"Missing ncu text report: {NCU_REPORT}")
        raw = NCU_REPORT.read_bytes()
        self.assertNotIn(b"\x00", raw, "profiles/v1_ncu.txt must not contain NUL bytes")
        text = raw.decode("utf-8")

        for forbidden in ("ERR_NVGPUCTRPERM", "==ERROR==", "No kernels were profiled"):
            self.assertNotIn(forbidden, text, f"Forbidden error marker found: {forbidden}")

        for required in (
            "assign_v1_kernel",
            "update_centroids_v1_kernel",
            "3907",
            "256",
            "Memory Workload Analysis",
            "Warp State Statistics",
        ):
            self.assertIn(required, text, f"Missing expected ncu content: {required}")

    def test_ac2_new_v1_json_schema_and_provenance(self) -> None:
        self.assertTrue(NCU_REPORT.exists(), f"Missing ncu text report: {NCU_REPORT}")
        ncu_text = NCU_REPORT.read_text(encoding="utf-8")

        candidates = _find_new_v1_result_jsons()
        self.assertEqual(
            len(candidates),
            1,
            f"Expected exactly 1 new results/v1_*.json (excluding {BASELINE_V1_JSON}), "
            f"found {len(candidates)}: {[p.name for p in candidates]}",
        )
        data: dict[str, Any] = json.loads(candidates[0].read_text(encoding="utf-8"))

        self.assertEqual(data.get("version"), "v1")

        config = data.get("config", {})
        self.assertEqual(config.get("n_points"), 1000000)
        self.assertEqual(config.get("dim"), 32)
        self.assertEqual(config.get("k_clusters"), 16)

        verification = data.get("verification", {})
        self.assertIs(verification.get("passed"), True)

        metrics = data.get("metrics", {})
        self.assertGreater(metrics.get("iterations", 0), 0)
        self.assertGreater(metrics.get("kernel_ms", 0), 0)
        self.assertGreaterEqual(metrics.get("e2e_ms", 0), metrics["kernel_ms"])
        self.assertGreater(metrics.get("ms_per_iter", 0), 0)

        hardware = data.get("hardware", {})
        gpu_name = str(hardware.get("gpu_name", "")).strip()
        self.assertTrue(gpu_name, "hardware.gpu_name must be non-empty")
        self.assertIn(
            gpu_name,
            ncu_text,
            f"hardware.gpu_name {gpu_name!r} must match GPU reported in profiles/v1_ncu.txt",
        )
        self.assertNotIn(hardware.get("compute_cap"), (None, "", "none"))
        self.assertNotIn(hardware.get("cuda_version"), (None, "", "none"))
        self.assertNotIn(hardware.get("git_commit"), (None, "", "unknown"))

    def test_ac3_analysis_doc_cites_provenance_counters_v2_v3_and_bib(self) -> None:
        self.assertTrue(ANALYSIS_DOC.exists(), f"Missing analysis doc: {ANALYSIS_DOC}")
        self.assertTrue(NCU_REPORT.exists(), f"Missing ncu text report: {NCU_REPORT}")
        self.assertTrue(REFERENCES_BIB.exists(), f"Missing BibTeX file: {REFERENCES_BIB}")

        candidates = _find_new_v1_result_jsons()
        self.assertEqual(len(candidates), 1, "Expected exactly 1 new results/v1_*.json")
        data: dict[str, Any] = json.loads(candidates[0].read_text(encoding="utf-8"))

        doc_text = ANALYSIS_DOC.read_text(encoding="utf-8")
        ncu_text = NCU_REPORT.read_text(encoding="utf-8")

        hardware = data["hardware"]
        metrics = data["metrics"]
        self.assertIn(str(hardware["gpu_name"]), doc_text)
        self.assertIn(str(hardware["compute_cap"]), doc_text)
        self.assertIn(str(hardware["cuda_version"]), doc_text)

        for metric_key in ("kernel_ms", "e2e_ms"):
            val = metrics[metric_key]
            exact_strs = {str(val), f"{val:.6f}"}
            self.assertTrue(
                any(s in doc_text for s in exact_strs),
                f"docs/v1_analysis.md must cite exact {metric_key} ({exact_strs})",
            )

        self.assertIn("Tesla T4", doc_text)
        self.assertIn(BASELINE_V1_JSON, doc_text)

        ncu_decimals = set(re.findall(r"\b\d+\.\d+\b", ncu_text))
        quoted_decimals = [num for num in ncu_decimals if num in doc_text]
        self.assertGreaterEqual(
            len(quoted_decimals),
            2,
            "docs/v1_analysis.md must quote numeric counter values present in profiles/v1_ncu.txt",
        )

        self.assertRegex(doc_text, r"\bV2\b")
        self.assertRegex(doc_text, r"(?i)(uncoalesced|aos|constant memory)")
        self.assertRegex(doc_text, r"\bV3\b")
        self.assertRegex(doc_text, r"(?i)(atomicadd|atomic contention)")

        bib_keys = _load_bib_keys(REFERENCES_BIB)
        self.assertTrue(bib_keys, "docs/references.bib must contain BibTeX keys")
        stripped_doc = _strip_markdown_code(doc_text)
        cited_keys = re.findall(r"(?<!\!)\[([a-z][a-z0-9_:-]{2,})\](?!\()", stripped_doc)
        for key in cited_keys:
            self.assertIn(
                key,
                bib_keys,
                f"Citation key [{key}] in docs/v1_analysis.md not found in docs/references.bib",
            )

    def test_ac4_repo_hygiene_and_sbatch_shell_syntax(self) -> None:
        ls_proc = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = ls_proc.stdout.splitlines()
        forbidden_patterns = (
            "*.ncu-rep",
            "*.nsys-rep",
            "*.sqlite",
            "*.qdrep",
            "profiles/*_job*.txt",
            "logs/slurm-*.log",
        )
        violations = [
            f for f in tracked_files if any(fnmatch.fnmatch(f, pat) for pat in forbidden_patterns)
        ]
        self.assertEqual(violations, [], f"Forbidden tracked profiler/log files: {violations}")

        syntax_proc = subprocess.run(
            ["bash", "-n", str(PROFILE_SBATCH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            syntax_proc.returncode,
            0,
            f"bash -n failed on {PROFILE_SBATCH}: {syntax_proc.stderr}",
        )

    def test_ac5_profile_sbatch_ncu_contract(self) -> None:
        self.assertTrue(PROFILE_SBATCH.exists(), f"Missing {PROFILE_SBATCH}")
        sbatch_text = PROFILE_SBATCH.read_text(encoding="utf-8")

        self.assertIn("--set full", sbatch_text)
        self.assertIn("--import", sbatch_text)
        self.assertIn("--page details", sbatch_text)

        logical_lines = re.sub(r"\\\n\s*", " ", sbatch_text).splitlines()
        ncu_profile_lines = [
            line.strip()
            for line in logical_lines
            if re.match(r"^\s*ncu\b", line) and "--import" not in line
        ]
        self.assertTrue(ncu_profile_lines, "Expected an ncu profiling command in profile.sbatch")
        for line in ncu_profile_lines:
            self.assertNotIn(
                "|| true",
                line,
                f"ncu profiling command must not swallow failure with '|| true': {line}",
            )


if __name__ == "__main__":
    unittest.main()
