# Unit Specification: V3 Privatized Shared-Memory Reduction + Warp Shuffles

**Unit Name:** `v3-privatized`  
**GitHub Issue:** [#13 - [4.1] Implement V3 kernel (src/v3_privatized/) with shared-memory accumulators and warp shuffles](https://github.com/khemingkapat/kuda_means/issues/13)  
**Parent Epic:** [#12 - [P4] V3 Privatized Shared-Memory Reduction + Warp Shuffles](https://github.com/khemingkapat/kuda_means/issues/12)  
**Status:** `in-progress`  

---

## 1. Decision & Architecture
Implement V3 K-Means in `src/v3_privatized/kmeans_v3.cu` registering `"v3"` via `KUDA_REGISTER_GPU_RUNNER`. V3 keeps everything V2 introduced (coalesced SoA $D \times N$ points, centroids in `__constant__` memory with `__ldg` fallback) and replaces only the **accumulation step**, the remaining bottleneck identified in `docs/v1_analysis.md` (Bottleneck B: global `atomicAdd` contention).

1. **Warp-level pre-aggregation (warp shuffles):** after the nearest-centroid search, lanes of a warp that picked the same cluster are grouped with `__match_any_sync`. For each coordinate $j$, a segmented tree reduction over that peer group (`__shfl_sync` + `__ballot_sync`, $\lceil \log_2 32 \rceil = 5$ steps max) leaves the group sum in the group leader (lowest lane). Only the leader issues a shared-memory atomic. The count uses `__popc(peers)` directly.
2. **Per-block privatized accumulators (shared memory):** each block owns `double s_sums[K * D]` and `int s_counts[K]` in dynamic shared memory, zeroed at block start.
3. **One global write per coordinate per block:** after a `__syncthreads()`, the block flushes its private accumulators with **at most one** global `atomicAdd` per `(k, j)` coordinate (and per count). Zero entries are skipped.
4. **Persistent grid-stride blocks:** the accumulation grid is sized from `cudaOccupancyMaxActiveBlocksPerMultiprocessor × SM count` (capped at $\lceil N / 256 \rceil$), so each block processes many points and amortizes its flush. The loop base is block-uniform so every warp executes `*_sync` intrinsics with a full mask.
5. **Shared-memory capacity wall + clean fallback:** required bytes $= K \cdot D \cdot 8 + K \cdot 4$. If this exceeds `cudaDevAttrMaxSharedMemoryPerBlockOptin`, V3 falls back to V2-style global `atomicAdd` accumulation (logged on stderr). Above 48 KB, the kernel opts in via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
6. **Numerical & Contract Fidelity (unchanged from V0–V2):**
   - `double` accumulation for sums (shared and global), `int` counts.
   - Exact Lloyd stopping rule: $\max_k \|\mathbf{c}_k^{(t)} - \mathbf{c}_k^{(t-1)}\|^2 < \varepsilon^2$ in double precision.
   - Empty cluster safety: retain prior centroid if `counts[k] == 0`.
   - Post-convergence assignment pass inside the timed scope.
   - Host-side SoA transpose outside the timed scope (identical to V2).

**Hardware requirement:** `__match_any_sync` needs `sm_70+` (T4 `sm_75`, RTX 4090 `sm_89` both qualify).

---

## 2. Scope & Boundaries
- **In Scope:**
  - `src/v3_privatized/kmeans_v3.cu` implementing and registering `run_kmeans_v3`.
  - Gate C test probes in `tests/test_v3_privatized.py`.
  - Benchmark result artifact `results/v3_n100k_d16_k8.json` (same GPU as the V2 artifact for comparison: Tesla T4).
- **Explicitly Out of Scope:**
  - Tiling over $K$ when the private accumulators exceed shared memory (fallback is documented instead; motivates V4).
  - `ncu` profiling write-up (separate issue).
  - Any modifications to V0/V1/V2 sources or shared infrastructure.

### Files Allowed to Touch
| File | Action | Purpose / Specific Changes |
|---|---|---|
| `src/v3_privatized/kmeans_v3.cu` | Create | Privatized shared-memory + warp-shuffle accumulation, capacity check & fallback, Lloyd loop, registers `"v3"`. |
| `tests/test_v3_privatized.py` | Create | Gate C black-box verification probes. |
| `results/v3_n100k_d16_k8.json` | Create | Benchmark receipt with full `HardwareProvenance` and `verified: true`. |
| `docs/aidlc/units/v3-privatized.md` | Create | This unit specification. |

### Files NOT to Touch
| File | Reason |
|---|---|
| `src/cpu/*`, `src/v1_naive/*`, `src/v2_memory/*` | Earlier stages are locked references. |
| `src/gpu/gpu_dispatch.cu`, `include/*` | Shared interfaces are locked. |
| `Makefile` | Already includes `src/v3_privatized/*.cu` in `GPU_SRCS`. |

---

## 3. Acceptance Criteria (Given / When / Then)
1. **Structure:** Given `src/v3_privatized/kmeans_v3.cu`, it registers `"v3"`, uses `extern __shared__` accumulators with `__syncthreads()`, uses warp primitives (`__match_any_sync` / `__shfl_sync`), checks `cudaDevAttrMaxSharedMemoryPerBlock*` before launch, and guards empty clusters (`count == 0`).
2. **Happy Path:** Given $N=100{,}000, D=16, K=8$, when running V3, then verification vs. V0 passes (`assignment_mismatches == 0`, `max_centroid_abs_diff <= 1e-3`, `sse_rel_diff <= 1e-3`).
3. **Capacity Wall:**
   - Given $K=64, D=64$ (32.25 KB, fits default 48 KB), V3 runs the shared-memory path and verifies.
   - Given $K=1024, D=32$ (260 KB, exceeds every supported GPU), V3 falls back to global atomics and still verifies.
4. **Unaligned Warp Safety:** Given $N=99{,}999$, V3 verifies (partial last warp handled without early return before `*_sync`).
5. **Race Freedom:** `compute-sanitizer --tool memcheck`, `--tool racecheck`, and `--tool synccheck` report 0 errors on a small dataset.
6. **Benchmark Provenance:** `results/v3_n100k_d16_k8.json` contains full `HardwareProvenance` and `verified: true`.
