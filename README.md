# kuda_means — High-Performance GPU K-Means Clustering

**CPE 393 | GPU Programming Final Project (KMUTT)**

Progressive, profiling-guided CUDA optimization of K-Means clustering—from a single-threaded C++ baseline to coalesced constant memory, shared-memory/warp-shuffle privatized reductions, tiled MatMul distance expansion, and multi-stream chunked execution.

## Team Members
| Name | Student ID | Contributions |
| :--- | :--- | :--- |
| Khem Ingkapat | 66070503408 | *(TBD)* |
| Thanchoti Chanpetch | 66070503423 | *(TBD)* |
| Patrik Stamm | 66070503440 | *(TBD)* |
| Kritipoom Chanla | 66070503488 | *(TBD)* |

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

## Team Workflow & CI Rules
- **Branching:** Never push directly to `main`. Use feature/experiment branches (`feat/v1-naive`, `exp/v3-blocksize`, `fix/...`, `docs/...`) and open a Pull Request.
- **Automated CI (`.github/workflows/ci.yml`):** Every PR automatically checks `make cpu`, `clang-format`, shell/sbatch syntax, Python syntax, and `results/*.json` validity.
- **GPU PR Checklist:** PRs modifying `src/v*` must report in the PR description: GPU model & `sm_XX`, correctness vs. CPU baseline (assignments, centroids, SSE), `compute-sanitizer --tool memcheck` pass, and measured `kernel_ms` / `e2e_ms`.
- **Data & Docs:** Store structured benchmark outputs in `results/*.json` and Markdown write-ups in `docs/*.md`. Keep course PDFs out of git (stored locally in `.references/`).

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
