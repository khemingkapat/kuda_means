# Team Collaboration, Branching & CI Rules (`kuda_means`)

This repository is developed collaboratively by a 4-person team (Khem, Thanchoti, Patrik, Kritipoom). Follow these conventions so everyone's CUDA kernels, Python verification scripts, and Markdown/JSON experiment logs integrate cleanly.

---

## 1. Pull-Based Issue Queue, Branching & Pull Request Policy

- **Pull-Based Single-Function Issues:** Work is tracked as atomic, single-function **GitHub Sub-Issues** grouped under **Parent Epics** (`epic`) and **Milestones** (`M1`–`M3`). Do not split work into rigid personal silos; any available teammate assigns themselves to the next unblocked sub-issue, completes its exact acceptance criteria, and closes it via a linked PR (`Closes #<issue>`).
- **Honesty & Traceability:** Every literature claim must cite a verified BibTeX entry (`docs/references.bib`), and every reported benchmark/profiler number must map verbatim to a committed `results/*.json` or `profiles/*.txt` artifact with `git_commit` and `HardwareProvenance`.
- **Never push directly to `main`.** Keep `main` always buildable and passing CI.
- **Branch Naming:**
  - Feature / kernel stage: `feat/v0-cpu-baseline`, `feat/v1-naive`, `feat/v2-memory`, `feat/v3-privatized`, `feat/v4-matmul`, `feat/v5-streaming`
  - Benchmarks & profiling experiments: `exp/<short-topic>` (e.g. `exp/v3-blocksize-sweep`)
  - Bug fixes: `fix/<short-topic>`
  - Documentation & report figures: `docs/<short-topic>`
- **Commit Messages:** Use short conventional prefixes:
  - `feat(v1): implement naive GPU assignment and atomic update kernels`
  - `perf(v3): privatize centroid accumulators in shared memory`
  - `test(verify): add Python cross-check script for SSE and centroids`
  - `docs(results): log RTX 4090 scaling benchmarks for V0-V3`
- **PR Approval:** Do not auto-push or auto-merge PRs without explicit human review. After merge, delete the feature branch and pull latest `main`.

---

## 2. File & Data Conventions

| Artifact Type | Location | Format & Rules |
| :--- | :--- | :--- |
| **CUDA / C++ Source** | `include/`, `src/` | Formatted with `.clang-format` (LLVM, 4 spaces, 100 col). |
| **Python Scripting** | `scripts/`, `tests/` | Managed via `uv run` (verification, JSON parsing, IEEE plotting). |
| **Benchmark Data** | `results/` | Structured **JSON** (`*.json`) containing `HardwareProvenance`, `KMeansConfig`, `kernel_ms`, `e2e_ms`, `iterations`, `sse`, and `verified`. |
| **Experiment Notes & Docs** | `docs/` | Clean **Markdown** (`*.md`) design notes, optimization analyses, and profiling write-ups. Never commit raw course `.pdf` files here (keep them locally in `.references/`, which is gitignored). |
| **Profiling Summaries** | `profiles/` | Small text/CSV summaries (`*.txt`, `*.csv`) committed as evidence; large binary `.ncu-rep` / `.nsys-rep` files stay gitignored. |

---

## 3. CI & Verification Rules

### Automated CI (GitHub Actions — `.github/workflows/ci.yml`)
Runs automatically on every Pull Request and push to `main` (CPU-only runner, enforced by GitHub Ruleset `protect-main`):
1. **CPU Build:** `make cpu`
2. **C++/CUDA Static Checks:** `clang-format --dry-run --Werror` + `cppcheck` on C++ sources
3. **Python Linting & Formatting:** `ruff check .` and `ruff format --check .` (replaces `black`, `flake8`, `isort`)
4. **Shell & Slurm Scripts:** `bash -n` + `shellcheck -S warning`
5. **JSON Integrity:** Validate all `results/*.json` files parse cleanly.

### GPU Dynamic Analysis & Profiling Gate (Required in PR Template `.github/PULL_REQUEST_TEMPLATE.md`)
Because GitHub Actions has no NVIDIA GPU, GPU verification & profiling run on KMUTT Slurm (`slurm-ctl profile ...`) and must be reported in the PR checklist for `src/v*` changes:
- **Hardware Provenance:** GPU model, compute capability, CUDA version (e.g., `RTX 4090 / sm_89 / CUDA 12.2` or `Tesla T4 / sm_75`).
- **Correctness vs. CPU Baseline:** `assignment_mismatches`, `max_centroid_abs_diff`, `sse_rel_diff` (all passing tolerance).
- **Dynamic Sanitizers:** `compute-sanitizer` (`memcheck`, plus `racecheck` & `synccheck` for V3/V4 shared-memory kernels) with 0 errors.
- **Profilers (`nsys` & `ncu`):**
  - `nsys` (**Nsight Systems**): timeline for PCIe transfer overhead and V5 stream/compute overlap.
  - `ncu` (**Nsight Compute**): kernel metrics (DRAM throughput, L2 hit rate, global atomic contention in V1→V3, compute/memory balance in V4).
