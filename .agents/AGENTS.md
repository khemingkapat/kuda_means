# Guidelines for Agents: kuda_means

> **Course:** CPE 393 — GPU Programming (KMUTT)  
> **Topic:** High-Performance GPU K-Means Clustering  
> **Deadline:** Wednesday, 2 December 2026  
> **Team:** Khem Ingkapat (`66070503408`), Thanchoti Chanpetch (`66070503423`), Patrik Stamm (`66070503440`), Kritipoom Chanla (`66070503488`)

---

## 1. Mission & Project Architecture

We build and benchmark K-Means clustering on GPUs in progressive, profiling-guided stages. Every version uses the **same synthetic Gaussian-blob input data (fixed seed), identical initial centroids, and identical convergence stopping rule** ($\max_k \|\mathbf{c}_k^{(t)} - \mathbf{c}_k^{(t-1)}\|^2 < \varepsilon^2$ or `max_iters`) so results are directly comparable.

### Implementation Roadmap
| Stage | Directory | Description |
| :--- | :--- | :--- |
| **V0 (Baseline)** | `src/cpu/` | Single-threaded C++ baseline, flat $N \times D$ float array (golden reference) |
| **V1 (Naive GPU)** | `src/v1_naive/` | 1 thread per point, global memory, global `atomicAdd` for centroid sums & counts |
| **V2 (Memory Tuning)** | `src/v2_memory/` | Coalesced SoA point layout ($D \times N$) + centroids in `__constant__` memory |
| **V3 (Privatized Reduction)** | `src/v3_privatized/` | Per-block shared-memory accumulators + warp shuffles (one global atomic write per block) |
| **V4 (Distance as MatMul)** | `src/v4_matmul/` | Expand $\|\mathbf{x}-\mathbf{c}\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x}\cdot\mathbf{c} + \|\mathbf{c}\|^2$ with tiled shared-memory GEMM (targets large $D, K$) |
| **V5 (Stretch: Streaming)** | `src/v5_streaming/` | Chunked execution overlapping PCIe transfers & compute via CUDA streams + pinned host memory |

### Shared Infrastructure
- `include/cuda_utils.cuh` — `CHECK_CUDA`, `CHECK_LAST_CUDA_ERROR`, `CudaTimer` (`cudaEvent_t`)
- `include/kmeans_common.h` — Shared structs (`KMeansConfig`, `KMeansResult`), deterministic Gaussian-blob generator, SSE calculator, and correctness checker
- `src/common/` — Implementations of dataset generation, verification, CLI argument parsing, and CSV/JSON result logging
- `scripts/plot_results.py` — Generates publication-ready runtime and speedup plots (via `uv run`) for the IEEE 2-column report

---

## 2. Build & Python Tooling

- **C++ / CUDA Build:** Use `Makefile`:
  - Local CPU build (no GPU required): `make cpu`
  - Slurm RTX 4090 build: `make gpu ARCH=sm_89 NVCC_CCBIN=/usr/bin/g++-12`
  - Google Colab T4 build: `make gpu ARCH=sm_75`
- **Python (Plotting & Analysis):** Always use `uv run <command>` and `uv add <pkg>` (never `pip install`).

---

## 3. Mandatory Verification Pipeline

Before claiming any kernel stage (V1–V5) is complete:
1. **Correctness vs. CPU Baseline:**
   - Verify cluster assignments match the CPU baseline (report any tie-breaking mismatches).
   - Verify final centroids agree within floating-point tolerance (`1e-4` relative / `1e-3` absolute depending on $N$).
   - Verify final Within-Cluster Sum of Squares (**SSE**) matches within tolerance.
2. **Sanitizer Check (on GPU):**
   - Run `compute-sanitizer --tool memcheck` (and `racecheck` for shared-memory kernels V3/V4) on a small/medium dataset.
3. **Timing & Scaling:**
   - Measure both **kernel-only time** and **end-to-end time** (including H2D/D2H transfers) with `cudaEvent_t`, warm-up runs, and repeated averages.
   - Sweep $N \in [10^4, 10^7]$ across varying $K$ and $D$.
4. **Profiler Evidence:**
   - Capture Nsight Compute (`ncu`) metrics (DRAM throughput, L2 hit rate, warp stall reasons / atomic contention) and Nsight Systems (`nsys`) timeline (for V5 transfer-compute overlap) to justify the next optimization in the IEEE technical report.

---

## 4. Deliverables Alignment (CPE 393 Grading)

Keep the repository clean and reproducible for grading:
- Complete source code, `Makefile`, Slurm/Colab scripts, and verification tests.
- `README.md` with goal, team contributions, hardware/software specs, build/run/reproduce commands.
- Curated benchmark CSVs in `results/` and profiling summaries in `profiles/` to back every claim in the 4–6 page IEEE two-column report.

---

## 5. Living Workflow & Preferences (`.agents/`)

As we work together, record confirmed workflow conventions and hardware findings in `.agents/` so everything stays unified:
- **Cluster-first, manual Colab fallback:** Always run `./.agents/scripts/slurm_ctl.sh nodes` (`sinfo`) first. If no Slurm GPU nodes are available, report back to the user and wait for explicit approval before using Google Colab.
- **Flexible hardware, strict explainability:** Any available GPU partition/node (with auto-detected `ARCH=sm_XX`) is fine for experimentation, as long as every run logs `HardwareProvenance` (`gpu_name`, `compute_cap`, `sm_count`, `global_mem_mb`, `cuda_version`) and results from different GPUs are never mixed in the same speedup comparison.
- **Teamwork, CI & Data Hygiene:** Follow `.agents/rules/team-workflow.md` — feature branches only (never push directly to `main`), CI checks CPU build + `clang-format` + script/JSON validation, structured benchmark results in `results/*.json`, clean Markdown notes in `docs/*.md`, and untracked course PDFs kept locally in `.references/`.
