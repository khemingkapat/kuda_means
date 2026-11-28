# Unit Specification: V4 Tiled GEMM Distance Expansion

**Unit Name:** `v4-matmul`  
**GitHub Issue:** [#16 - [5.1] Implement V4 tiled GEMM distance kernel (src/v4_matmul/) with V0 verification and cancellation check](https://github.com/khemingkapat/kuda_means/issues/16)  
**Parent Epic:** [#15 - [P5] V4 Tiled GEMM Distance Expansion (FAISS-Style)](https://github.com/khemingkapat/kuda_means/issues/15)  
**Status:** `in-progress`  

---

## 1. Decision & Architecture
Implement V4 K-Means in `src/v4_matmul/kmeans_v4.cu` registering `"v4"`. V3 profiling (`docs/v3_analysis.md`) showed that, once global atomics are privatized, the kernel is **compute-bound in the $O(N \cdot K \cdot D)$ distance loop**. V4 replaces that loop with the matrix-multiplication expansion [johnson2019billion]:

$$\|\mathbf{x}-\mathbf{c}\|^2 = \|\mathbf{x}\|^2 - 2\,\mathbf{x}\cdot\mathbf{c} + \|\mathbf{c}\|^2$$

1. **Tiled shared-memory GEMM fused with argmin** (`gemm_argmin_v4_kernel`): each block of $16 \times 16$ threads owns a tile of $64$ points and sweeps all centroids in tiles of $64$, staging $16$-dimension slices of $X$ (SoA, $D \times N$) and $C$ ($K \times D$) in shared memory. Each thread accumulates a $4 \times 4$ register micro-tile of dot products with `float4` shared-memory reads. The $N \times K$ distance matrix is **never materialized**: the argmin is folded into the epilogue and reduced across the 16 threads sharing a point row with `__shfl_xor_sync`.
2. **Norms:** $\|\mathbf{c}\|^2$ is computed every iteration (`centroid_norms_v4_kernel`). Because $\|\mathbf{x}\|^2$ is constant per point, it does not change the argmin; by default V4 ranks centroids by the **shifted distance** $\|\mathbf{c}\|^2 - 2\,\mathbf{x}\cdot\mathbf{c}$, which avoids adding and cancelling the largest term.
3. **Cancellation check:** setting `KUDA_V4_FULL_EXPANSION=1` precomputes $\|\mathbf{x}\|^2$ once per run (`point_norms_v4_kernel`) and ranks by the **full expansion** $\|\mathbf{x}\|^2 + \|\mathbf{c}\|^2 - 2\,\mathbf{x}\cdot\mathbf{c}$ in `float`. Comparing mismatches vs. V0 between the two modes measures floating-point cancellation directly. The chosen mode is logged on stderr.
4. **Tie-breaking:** equal distances resolve to the lowest centroid index, matching V0's first-minimum scan.
4a. **Exact top-2 re-rank (default):** the first Colab run (commit `0a5d1d1`, T4) showed that the GEMM ranking alone flips a few near-tie assignments vs. V0 in **both** expansions (e.g. `N=100k, D=16, K=8`, 50 iterations: 393 mismatches, centroid diff `3.6e-2`; `D=128, K=64`: identical 135 mismatches shifted vs. full). The expansion's terms are large relative to $\|\mathbf{x}-\mathbf{c}\|^2$, so `float` cancellation decides close calls. V4 therefore keeps each point's **two best GEMM candidates** (merged across lanes with an xor butterfly) and re-scores them with the exact $(\mathbf{x}-\mathbf{c})^2$ form used by V0–V3 (`2·D` extra operations per point vs. `K·D` for the GEMM). `KUDA_V4_EXACT_RERANK=0` disables it to measure the raw cancellation error.
4b. **Close-call filter (default):** re-scoring every point made V4 slower than V3 at small $K$ (second T4 run, commit `22c8673`: `0.48–0.89x` of V3 for $K \le 64$ except $D=128,K=64$), because only 4 of 16 lanes do the scattered exact reads while the rest wait. Since at most a few hundred of 100,000 points were ever wrong, V4 now re-scores a point only when its two best GEMM distances differ by less than $8\,(D+2)\,\varepsilon_{\text{float}}\,(\|\mathbf{x}\|^2 + \max\|\mathbf{c}\|^2)$, a conservative bound on the expansion's rounding error, so any pair whose order could differ from V0 is still checked. $\|\mathbf{x}\|^2$ is now always computed once per run for this tolerance. `KUDA_V4_EXACT_RERANK=all` keeps the re-score-everything behaviour as a cost reference, and the number of re-scored points in the final pass is logged on stderr.
5. **Accumulation reuses V3:** warp-aggregated (`__match_any_sync` + `__shfl_sync`) per-block shared-memory accumulators with at most one global atomic per coordinate per block, falling back to global atomics when `K·D·8 + K·4` exceeds `cudaDevAttrMaxSharedMemoryPerBlockOptin`. The accumulation kernel reads `assignments[]` written by the GEMM kernel.
6. **Centroids are read with `__ldg`**, not `__constant__` (`docs/v2_analysis.md` showed `__ldg` is ~2.3x cheaper on T4).
7. **Contract (unchanged from V0–V3):** `double` sums, exact stopping rule, empty-cluster retention, final assignment pass inside the timed scope, host SoA transpose outside it.

---

## 2. Scope & Boundaries
- **In Scope:** `src/v4_matmul/kmeans_v4.cu`, `tests/test_v4_matmul.py`, `results/v4_n100k_d16_k8.json`, this spec.
- **Out of Scope:** Tensor Cores / cuBLAS, half precision, `ncu` analysis (Issue #17), changes to V0–V3 or shared infrastructure.

### Files Allowed to Touch
| File | Action |
|---|---|
| `src/v4_matmul/kmeans_v4.cu` | Create |
| `tests/test_v4_matmul.py` | Create |
| `results/v4_n100k_d16_k8.json` | Create |
| `docs/aidlc/units/v4-matmul.md` | Create |

---

## 3. Acceptance Criteria (Given / When / Then)
1. **Structure:** registers `"v4"`; uses shared-memory GEMM tiles with `__syncthreads()`; computes centroid norms; supports the full-expansion mode; guards empty clusters.
2. **Small regime:** $N=100{,}000, D=16, K=8$ verifies vs. V0 (`PASS`).
3. **Large regime:** $D=128, K=64$ verifies vs. V0 in both expansions with the re-rank; with `KUDA_V4_EXACT_RERANK=0` the raw GEMM mismatches are reported, not hidden.
4. **Tile edges:** $N$, $D$ and $K$ that are not multiples of the tile sizes ($N=99{,}999$, $D=20$, $K=100$) verify.
5. **Race freedom:** `compute-sanitizer` memcheck / racecheck / synccheck report 0 errors.
6. **Benchmark provenance:** `results/v4_n100k_d16_k8.json` with full `HardwareProvenance` and `verified: true`.
