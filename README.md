# kuda_means — High-Performance GPU K-Means Clustering

**CPE 393 | GPU Programming Final Project (KMUTT)**

An honest, profiling-guided investigation of GPU K-Means clustering—exposing the bottlenecks of Lloyd's original algorithm, cataloging existing algorithmic and GPU optimizations alongside their failure modes, and measuring progressive CUDA kernels (V0–V5) across $(N, D, K)$ regimes to show **where each optimization wins and where it breaks down**.

## Core Standards (Honesty & Traceability)
- **Strict Citations:** Every algorithmic and literature claim maps to a verified entry in `docs/references.bib`.
- **Verified Benchmarks:** Every reported runtime or speedup comes from a committed `results/*.json` file with `verified: true`, `git_commit`, and `HardwareProvenance` (results from different GPUs are never mixed).
- **Automated Quality Gates:** Every PR must pass CI (`make cpu`, `make test`, `clang-format`, `cppcheck`, `ruff`, `shellcheck`, JSON validation) plus GPU dynamic checks (`compute-sanitizer`, `ncu`/`nsys`) for `src/v*` kernels.

## Team Members
| Name | Student ID | Contributions *(tracked via merged PRs/Issues)* |
| :--- | :--- | :--- |
| Khem Ingkapat | 66070503408 | *(Updated from merged PRs at M3 audit)* |
| Thanchoti Chanpetch | 66070503423 | *(Updated from merged PRs at M3 audit)* |
| Patrik Stamm | 66070503440 | *(Updated from merged PRs at M3 audit)* |
| Kritipoom Chanla | 66070503488 | *(Updated from merged PRs at M3 audit)* |

## Optimization Stages
1. **V0 — CPU Baseline (`src/cpu/`):** Single-threaded C++ reference implementation on a flat $N \times D$ float array.
2. **V1 — Naive GPU (`src/v1_naive/`):** One thread per point, global memory, global `atomicAdd` for centroid accumulation.
3. **V2 — Memory Tuning (`src/v2_memory/`):** Coalesced Structure-of-Arrays (SoA) point layout + centroids in `__constant__` memory.
4. **V3 — Privatized Reduction (`src/v3_privatized/`):** Per-block shared-memory accumulators + warp shuffles (one global write per block).
5. **V4 — Distance as MatMul (`src/v4_matmul/`):** Expands $\|\mathbf{x}-\mathbf{c}\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x}\cdot\mathbf{c} + \|\mathbf{c}\|^2$ using tiled shared-memory GEMM for large $D, K$.
6. **V5 (Stretch) — Streaming (`src/v5_streaming/`):** Splits datasets larger than GPU memory into chunks, overlapping PCIe H2D/D2H transfers with compute via CUDA streams and pinned host memory.

## Hardware & Software Environment
- **Primary Cluster:** KMUTT CPE Slurm (any available GPU partition; `.sbatch` auto-detects `ARCH=sm_XX`, CUDA 12.2, `g++-12`).
- **Manual Fallback GPU:** Google Colab (`Tesla T4`, `sm_75`) when Slurm GPU nodes are unavailable.
- **Local DevShell:** Nix Flakes (`nix develop`) providing `gcc`, `make`, `cmake`, `clangd`, `clang-format`, `uv`, `python3`, Slurm & Colab helpers.
- **Provenance Rule:** Every benchmark JSON record logs `gpu_name`, `compute_cap`, `sm_count`, `global_mem_mb`, and `cuda_version`. Results from different GPUs are never mixed in the same speedup comparison.

## Quick Start

### 1. Enter Development Environment
```bash
nix develop   # or automatic via direnv (.envrc)
```

### 2. Local CPU Build (No GPU Required)
```bash
make cpu
```

### 3. Run on KMUTT Slurm Cluster
One-time SSH gatekeeper setup (safe alongside `kv_bench`):
```bash
./.agents/scripts/setup_slurm_ssh.sh
```
Daily workflow:
```bash
slurm-ctl nodes             # Check sinfo / squeue first
slurm-ctl sync              # Sync code & scripts to ~/kuda_means/
slurm-ctl submit            # Submit scripts/slurm/build_and_bench.sbatch
slurm-ctl status            # Check your active jobs
slurm-ctl logs <job_id>     # Tail Slurm job log
slurm-ctl fetch <job_id>    # Pull results/, profiles/*.txt, logs/
```
Profiling with `compute-sanitizer`, Nsight Systems (`nsys`), and Nsight Compute (`ncu`):
```bash
slurm-ctl profile v1 1000000 32 16
```

### 4. Manual Fallback on Google Colab (Tesla T4, `sm_75`)
When `slurm-ctl nodes` shows no available cluster GPUs:
```bash
colab-start T4 kuda-means
colab-run-project -- --version v1 -n 100000 -d 32 -k 16
colab-stop kuda-means
```

### 5. Verification & Plotting (Python via `uv`)
```bash
uv run python scripts/plot_results.py results/benchmarks.csv
```

## Team Workflow & Pull-Based Issue Queue
We use a **pull-based GitHub Issue workflow** instead of rigid 4-way silos so no single teammate's schedule blocks the team:
1. **Single-Function Sub-Issues:** Every unit of work is a focused GitHub Sub-Issue serving **one clear function** (e.g., implement one kernel, capture one `ncu` profile, write one citation file), grouped under **Parent Issues** (`epic`) and **3 Milestones** (`M1: Foundation & Literature`, `M2: GPU Optimizations & Tooling`, `M3: Synthesis, Report & Audit`).
2. **Claim & Work:** Whenever you have time, pick an open, unblocked sub-issue from the current milestone, **assign yourself**, and create a feature branch (`feat/...`, `exp/...`, `docs/...`, `fix/...`). Never push directly to `main`.
3. **Verify & PR:** Satisfy every checkbox in the issue's Acceptance Criteria, open a Pull Request referencing `Closes #<issue_number>`, and wait for CI + human review.
- **Automated CI (`.github/workflows/ci.yml`):** Checks `make cpu`, `make test`, `clang-format`, `cppcheck`, `ruff`, `shellcheck`, and `results/*.json` validity on every PR.
- **GPU PR Checklist:** PRs modifying `src/v*` must report in the PR description: GPU model & `sm_XX`, correctness vs. CPU baseline (assignments, centroids, SSE), `compute-sanitizer` pass (`memcheck`, plus `racecheck`/`synccheck` for shared-memory kernels), and measured `kernel_ms` / `e2e_ms`.
- **Data & Docs:** Store structured benchmark outputs in `results/*.json`, profiler text summaries in `profiles/*.txt`, and cited Markdown write-ups in `docs/*.md`. Keep course/reference PDFs out of git (stored locally in `.references/`).

## Repository Layout
```
├── .agents/              # Agent guidelines, rules, and Slurm automation
├── .github/workflows/    # GitHub Actions CI (CPU build, formatting, JSON checks)
├── docs/                 # Project Markdown design notes and experiment analyses
├── include/              # Shared CUDA utilities and K-Means interfaces
├── src/                  # V0 (cpu) through V5 (streaming) implementations
├── scripts/              # Slurm sbatch jobs, Colab runner, Python verification & plotting
├── tests/                # Correctness & unit tests
├── results/              # Benchmark JSON/CSV outputs and generated figures
└── profiles/             # Nsight Compute / Nsight Systems text summaries
```
