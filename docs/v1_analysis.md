# V1 Naive GPU K-Means (`assign_v1_kernel`) — Nsight Compute (`ncu`) Bottleneck Analysis

This report documents the empirical hardware-counter profile of **V1 (`src/v1_naive/kmeans_v1.cu`)** captured with NVIDIA Nsight Compute (`ncu --set full`) and explains how the measured bottlenecks directly motivate **V2 (Memory Tuning)** and **V3 (Privatized Reduction)**.

---

## 1. Hardware & Benchmark Provenance

All primary profiling and `N=1,000,000` timing measurements in this analysis were collected on the **same GPU**:

| Property | Value | Source |
| :--- | :--- | :--- |
| **GPU** | `Tesla T4` (`sm_75`, 40 SMs, `14912.687500` MB) | `results/v1_t4_n1m_d32_k16.json` |
| **CUDA Version** | `13.0` | `results/v1_t4_n1m_d32_k16.json` |
| **Git Commit** | `754c451` | `results/v1_t4_n1m_d32_k16.json` |
| **Workload** | `N=1000000`, `D=32`, `K=16`, `max_iters=50`, `epsilon=1e-4`, `seed=42` | `results/v1_t4_n1m_d32_k16.json` |
| **Timing (5-run avg, 1 warmup)** | `kernel_ms = 983.940527`, `e2e_ms = 1014.485620`, `ms_per_iter = 19.678811` | `results/v1_t4_n1m_d32_k16.json` |
| **Verification** | `passed = true` (`assignment_mismatches = 0`, `sse = 142835632.253586`) | `results/v1_t4_n1m_d32_k16.json` |
| **Profile Artifact** | `profiles/v1_ncu.txt` (`ncu --set full --launch-skip 0 --launch-count 2`) | `profiles/v1_ncu.txt` |

> **Backend Note:** KMUTT Slurm `gpu4090` (`gpu06`, job `30850`) blocks non-admin hardware performance counters (`ERR_NVGPUCTRPERM`), so `ncu` profiling and the matching `1M` V1 timing run were executed on Google Colab `Tesla T4` (`sm_75`).

### Separate Provenance: Smaller-Scale `N=100,000` T4 vs. CPU Baseline
For reference on the `N=100,000, D=16, K=8` workload (separate run on `Tesla T4`, `sm_75`, `cuda_version = 13.0`, `git_commit = b8d0b74`):
- **V1 (`results/v1_n100k_d16_k8.json`):** `kernel_ms = 38.163155`, `e2e_ms = 40.607999`, `ms_per_iter = 0.763263` (`50` iters, `sse = 8388364.263866`, `verified = true`).
- **V0 CPU (`results/v0_cpu_n100k_d16_k8.json`):** `AMD EPYC 7313 16-Core Processor`, `kernel_ms = 398.097308`, `ms_per_iter = 7.961946` (`10.43x` kernel speedup over V0 at `N=100k`).
- No `N=1,000,000` CPU baseline JSON is recorded in `results/`, so no V0-vs-V1 speedup is claimed for `N=1M, D=32, K=16`.

---

## 2. Nsight Compute (`ncu`) Kernel Breakdown

`profiles/v1_ncu.txt` captures the first two kernel launches of a single Lloyd iteration (`--warmup 0 --iters 1 --runs 1 --no-verify`) [lloyd1982least]:

1. **`assign_v1_kernel<1>`** — Grid `(3907, 1, 1)`, Block `(256, 1, 1)` (`1,000,192` threads, `54` registers/thread, `0` bytes shared memory, `24.42` waves/SM):
   - **Duration:** `51.39 ms` (`99.96%` of iteration kernel time).
   - **Achieved Occupancy:** `98.35 %` (`31.47` active warps/SM out of `32` theoretical), yet **Compute (SM) Throughput** is only `5.67 %` and **Executed IPC Active** is `0.08` inst/cycle.
2. **`update_centroids_v1_kernel`** — Grid `(1, 1, 1)`, Block `(256, 1, 1)` (`256` threads, `38` registers/thread, `0.01` waves/SM):
   - **Duration:** `20.54 us` (`0.04%` of iteration kernel time).
   - Negligible in total runtime despite `4.34 %` achieved occupancy (`1` block across `40` SMs).

---

## 3. Root-Cause Bottleneck Diagnosis (`assign_v1_kernel<1>`)

### Bottleneck A: Uncoalesced AoS Global Loads & Global Centroid Reads (-> Motivates V2)
| Metric (`profiles/v1_ncu.txt`) | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **Memory Throughput** | `49.37 %` (`34.22 Gbyte/s`) | Moderate bus traffic, heavily amplified by wasted sector bytes |
| **DRAM Throughput** | `10.71 %` | DRAM is not saturated; latency & L1/TEX serialization dominate |
| **L1/TEX Cache Throughput** | `98.75 %` | L1/TEX pipeline is **saturated** |
| **L1/TEX Hit Rate / L2 Hit Rate** | `85.05 %` / `77.75 %` | Centroid reuse hits cache, but AoS point reads thrash sectors |
| **Bytes Utilized per 32-Byte Sector** | `4.0` of `32` bytes (`12.5%`) | Strided AoS layout (`pt = points + idx * d`, stride `4*D = 128 B`) |
| **Excessive Global Sectors** | `481,058,144` of `573,120,612` (`84%`) | `ncu` estimates `50.41%` speedup (`86.41%` in Memory Tables) from coalescing |

In `src/v1_naive/kmeans_v1.cu`, `points` is stored in row-major **Array-of-Structures (AoS)** layout (`N x D`). Adjacent threads `idx` and `idx + 1` in a warp read `pt[j]` `128` bytes apart (`D = 32` floats), so each warp load utilizes only `4.0` of `32` bytes per sector (`84%` excessive sectors). Meanwhile, all threads read `centroids[c * d + j]` through the generic global/L1TEX path instead of `__constant__` memory, adding further pressure to the `98.75 %`-saturated L1/TEX pipeline [krulis2020detailed].

### Bottleneck B: Global `atomicAdd` Contention & LG Queue Stalls (-> Motivates V3)
| Metric (`profiles/v1_ncu.txt`) | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **Warp Cycles Per Issued Instruction** | `398.26` cycles | Extreme instruction-issue latency despite `98.35 %` occupancy |
| **LG (Local/Global) Queue Stall** | `380.9` cycles (`95.7%` of total) | Warps stalled waiting on full L1 local/global instruction queue |
| **Local Memory Spilling Requests** | `0` (`0 %` overhead) | Confirms LG stalls are **not** register spills |
| **No Eligible Warps** | `98.02 %` (`0.07` eligible / `7.87` active) | Schedulers issue only `0.02` warps/cycle (`Issue Slots Busy = 1.96 %`) |
| **L2 Slice Active Cycle Imbalance** | `+16.47%` max / `-8.40%` min vs avg | Hot-spotting across L2 slices serving `K=16` centroid accumulators |

Every thread in `assign_v1_kernel<1>` executes `1` integer `atomicAdd(&counts[best_k], 1)` plus `D = 32` `double` `atomicAdd(&sum_row[j], ...)` operations directly into global memory (`33` global atomics per point, `33,000,000` global atomics per iteration across only `K * (D + 1) = 528` addresses). This floods the L1TEX local/global queue (`380.9` stall cycles, `95.7%` of warp cycles) and serializes updates in L2, leaving SM compute pipelines at `5.67 %` [krulis2020detailed, johnson2019billion].

---

## 4. Optimization Roadmap (V2 & V3)

1. **V2 — Memory Tuning (`src/v2_memory/`):**
   - Transpose `points` from AoS (`N x D`) to **Structure-of-Arrays (SoA, `D x N`)** so adjacent threads in a warp read contiguous `4`-byte `float` elements (`32/32` bytes per sector, eliminating the `84%` excessive sectors).
   - Place `centroids` (`K * D * 4 = 2048` bytes for `K=16, D=32`) in **`__constant__` memory** so uniform warp reads broadcast via the constant cache without competing with L1/TEX global traffic.
2. **V3 — Privatized Reduction (`src/v3_privatized/`):**
   - Accumulate per-cluster `sums` and `counts` in **per-block shared memory** (with warp-level `__shfl_down_sync` reduction) and emit at most one global `atomicAdd` per block per cluster, reducing global atomic traffic by `~256x` (`3907` blocks vs `1,000,000` threads) to eliminate the `380.9`-cycle LG queue stall.
