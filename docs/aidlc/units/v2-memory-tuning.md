# Unit Specification: V2 Memory Tuning

**Unit Name:** `v2-memory-tuning`  
**GitHub Issue:** [#10 - [3.1] Implement V2 kernel (src/v2_memory/) with SoA layout, __constant__ centroids, and 64 KB limit handling](https://github.com/khemingkapat/kuda_means/issues/10)  
**Status:** `done`  

---

## 1. Decision & Architecture
Implement V2 K-Means in `src/v2_memory/kmeans_v2.cu` registering `"v2"` via `KUDA_REGISTER_GPU_RUNNER`:
1. **Coalesced SoA Layout ($D \times N$):** Transpose input row-major AoS points into Structure-of-Arrays ($D \times N$, index `j * n + idx`) prior to H2D `cudaMemcpy`. Adjacent threads in a warp access consecutive addresses in dimension $j$, achieving 100% bus utilization (128-byte coalesced transactions).
2. **`__constant__` Centroid Broadcast:** Centroids are staged into `__constant__ float c_centroids[16384]` per iteration using `cudaMemcpyToSymbol` when $K \cdot D \le 16{,}384$.
3. **Zero-Branch Fallback:** `template <bool Accumulate, bool UseConstant> __global__ void assign_v2_kernel(...)` dispatches with `UseConstant=true` when $K \cdot D \le 16{,}384$ and falls back to `UseConstant=false` (reading from `const float* __restrict__ d_centroids` via `__ldg()`) when $K \cdot D > 16{,}384$.
4. **Numerical & Contract Fidelity:**
   - Atomic accumulation into `double` cluster sums (`d_sums`) and `int` counts (`d_counts`), matching V1 precision.
   - Exact Lloyd stopping rule: $\max_k \|\mathbf{c}_k^{(t)} - \mathbf{c}_k^{(t-1)}\|^2 < \varepsilon^2$ in double precision.
   - Empty cluster safety: retain prior iteration's centroid if `counts[k] == 0`.
   - Post-convergence assignment pass inside timed scope to compute final `assignments` and `sse`.

---

## 2. Scope & Boundaries
- **In Scope:**
  - `src/v2_memory/kmeans_v2.cu` implementing and registering `run_kmeans_v2`.
  - Gate C test probes in `tests/test_v2_memory.py`.
  - Benchmark result artifact `results/v2_n100k_d16_k8.json` on Tesla T4.
- **Explicitly Out of Scope:**
  - Shared-memory privatized reductions (reserved for V3 / Issue #13).
  - Custom GPU device transposition kernels.
  - Any modifications to CPU baseline (`src/cpu/`), V1 naive kernel (`src/v1_naive/`), or dispatch infrastructure (`src/gpu/gpu_dispatch.cu`, `include/kmeans_gpu.cuh`).

### Files Allowed to Touch
| File | Action | Purpose / Specific Changes |
|---|---|---|
| `src/v2_memory/kmeans_v2.cu` | Create | Implements SoA transposition, constant memory centroid staging, template fallback, Lloyd iteration loop, and registers `"v2"`. |
| `tests/test_v2_memory.py` | Create | Gate C black-box verification probe testing happy path, constant limit boundary ($16{,}384$), overflow fallback ($>16{,}384$), and unaligned $N$. |
| `results/v2_n100k_d16_k8.json` | Create | Final benchmark receipt with full `HardwareProvenance` and `verified: true`. |

### Files NOT to Touch
| File | Reason |
|---|---|
| `src/cpu/*` | Golden V0 reference; must remain unmodified. |
| `src/v1_naive/*` | V1 baseline reference; must remain unmodified. |
| `src/gpu/gpu_dispatch.cu` | GPU runner registry map; already supports static registration. |
| `include/kmeans_common.h` | Shared configuration and verification contracts; locked. |
| `include/kmeans_gpu.cuh` | GPU interface and registration macros; locked. |
| `include/cuda_utils.cuh` | Error checking and timing utilities; locked. |
| `Makefile` | Already includes `src/v2_memory/*.cu` in `GPU_SRCS`. |

---

## 3. Acceptance Criteria (Given / When / Then)
1. **Happy Path:** Given $N=100{,}000, D=16, K=8$ ($K \cdot D = 128 \le 16{,}384$), when running V2 with `--verify`, then it passes all verification checks against V0 CPU (`assignment_mismatches == 0`, `max_centroid_abs_diff <= 1e-3`, `sse_rel_diff <= 1e-3`), matches V0 convergence iterations, and reports 0 CUDA errors.
2. **Exact Boundary & Overflow Rejection (Negative Proof):**
   - Given $K \cdot D = 16{,}384$ (e.g. $K=512, D=32$), when running V2, then it executes via `c_centroids` without buffer overrun.
   - Given $K \cdot D > 16{,}384$ (e.g. $K=1024, D=32 \implies 32{,}768$), when running V2, then it routes to global memory fallback (`UseConstant=false`), produces verified results against V0, and exhibits zero memory violations.
3. **Unaligned Warp Safety:** Given $N=99{,}999$ (not divisible by warp size 32), when running V2 under `compute-sanitizer --tool memcheck`, then it reports 0 errors and zero illegal memory accesses.
4. **Integration & Build Integrity:** Given `src/v2_memory/kmeans_v2.cu`, when checked via `make cpu` and `clang-format -dry-run`, then it builds cleanly with zero warnings/errors.
5. **Benchmark Provenance:** Given benchmark run on Tesla T4 with $N=100{,}000, D=16, K=8$, when results are saved, `results/v2_n100k_d16_k8.json` contains full `HardwareProvenance` and `verified: true`.
