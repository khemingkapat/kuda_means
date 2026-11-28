# V4 Tiled GEMM Distance (`gemm_argmin_v4_kernel`) — Nsight Compute (`ncu`) Analysis: Large-(D, K) Gains, Small-K Overhead & Floating-Point Cancellation

This report compares **V4 (`src/v4_matmul/kmeans_v4.cu`)** with **V3 (`src/v3_privatized/kmeans_v3.cu`)** on the same GPU. V4 computes distances with the matrix-multiplication expansion $\|\mathbf{x}-\mathbf{c}\|^2 = \|\mathbf{x}\|^2 - 2\,\mathbf{x}\cdot\mathbf{c} + \|\mathbf{c}\|^2$ popularised by FAISS [johnson2019billion], using a register-tiled shared-memory GEMM with a fused top-2 argmin and an exact re-rank of close calls.

---

## 1. Hardware & Benchmark Provenance

| Property | Value | Source |
| :--- | :--- | :--- |
| **GPU** | `Tesla T4` (`sm_75`, 40 SMs, `14912.687500` MB), Google Colab | `results/v4_n100k_d16_k8.json` |
| **CUDA Version** | `13.0` | `results/v4_n100k_d16_k8.json` |
| **Code** | commit `65222e1` (V4 final, V3 unchanged since `f9578a1`) | all `profiles/v4_*` files |
| **Regime sweep** | V3 vs. V4 (3 re-rank modes), `N=10^6`, $D \in \{16,128\}$, $K \in \{8,64,256\}$, `--iters 10 --runs 3 --no-verify` | `profiles/v4_regime_sweep_t4.txt` |
| **Cancellation check** | V4, `N=100000`, 50 iterations, re-rank off / close calls / all × shifted / full expansion, verified vs. V0 | `profiles/v4_cancellation_t4.txt` |
| **`ncu` profiles** | `--set full`, one iteration; V4 at $(D,K)=(32,16)$, `N=10^6` and $(128,256)$, `N=10^5`; V3 at $(128,256)$, `N=10^5` | `profiles/v4_ncu.txt`, `profiles/v4_ncu_d128_k256.txt`, `profiles/v3_ncu_d128_k256.txt` |
| **V3 reference profile** | $(32,16)$, `N=10^6` | `profiles/v3_ncu.txt` (`docs/v3_analysis.md`) |

> `ncu` locks the SM clock at `585 MHz`; its durations are used to diagnose bottlenecks. Speed comparisons use the `cudaEvent` timings from the sweep, where V3 and V4 were measured **in the same Colab session** (absolute times vary between sessions on Colab, e.g. V3 at $D=16, K=8$ measured `28.16 ms` and `39.99 ms` in two earlier sessions).

---

## 2. Correctness: Floating-Point Cancellation and the Exact Re-Rank

The expansion's terms ($\|\mathbf{c}\|^2$ and $2\,\mathbf{x}\cdot\mathbf{c}$, or additionally $\|\mathbf{x}\|^2$) are large compared with $\|\mathbf{x}-\mathbf{c}\|^2$ when points lie far from the origin, so `float` rounding can swap two nearly equidistant centroids. V4 therefore keeps each point's **two best GEMM candidates** and, when their gap is below a conservative rounding bound $8\,(D+2)\,\varepsilon_{\text{float}}\,(\|\mathbf{x}\|^2 + \max\|\mathbf{c}\|^2)$, re-scores both with the exact $(\mathbf{x}-\mathbf{c})^2$ form used by V0–V3.

`profiles/v4_cancellation_t4.txt` (`N=100000`, 50 iterations, verified vs. V0):

| $D, K$ | No re-rank, shifted $\|\mathbf{c}\|^2 - 2\mathbf{x}\cdot\mathbf{c}$ | No re-rank, full expansion | **Close-call re-rank** (either form) | Points re-scored | Re-rank all |
| :--- | :--- | :--- | :--- | ---: | :--- |
| 16, 8 | FAIL: `393` mismatches, centroid diff `3.65e-2` | FAIL: `73`, `9.43e-3` | **PASS, `0` mismatches** | `41` (`0.04%`) | PASS, `0` |
| 128, 64 | FAIL: `557`, `1.24e-1` | FAIL: `578`, `1.24e-1` | **PASS, `0`** | `1,446` (`1.45%`) | PASS, `0` |
| 128, 256 | FAIL: `62`, `1.04e-1` | FAIL: `54`, `8.01e-2` | **PASS, `0`** | `521` (`0.52%`) | PASS, `0` |

**Findings:**
1. **Cancellation is real and affects both forms.** Without re-ranking, `54–578` of `100,000` assignments differ from V0 and the centroid error exceeds the `1e-3` tolerance by up to `124x`. SSE still agrees to `1e-6` relative, so the clustering is nearly the same but not bit-faithful.
2. **Dropping $\|\mathbf{x}\|^2$ does not reliably help.** The shifted form was worse at $(16, 8)$ (`393` vs. `73`) and slightly better at $(128, 64)$ (`557` vs. `578`). Both forms share the same large-term cancellation; only the exact re-score removes it.
3. **The close-call filter is exact on every tested shape** while re-scoring `0.04–1.45%` of points; at `N=10^6` it re-scored `0.06%` ($D=16$) to `6.35%` ($D=128, K=8$) because the tolerance grows with $D$ (`profiles/v4_regime_sweep_t4.txt`).
4. Sanitizers: memcheck / racecheck / synccheck `0` errors, plus a memcheck run that entered the re-rank path (`212` points re-scored) with `0` errors (`profiles/v4_sanitizer_t4.txt`).

---

## 3. Regime Sweep: Where V4 Wins and Where It Loses

`profiles/v4_regime_sweep_t4.txt` (`N=10^6`, 10 iterations, 3-run average `kernel_ms`, same session):

| $D, K$ | V3 | V4 close calls (default) | V4 re-rank all | V4 no re-rank | **V3 / V4 (default)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 16, 8 | `30.7583` | `31.4921` | `49.5159` | `26.8744` | **`0.98x`** |
| 16, 64 | `57.6645` | `48.5130` | `67.3910` | `43.9274` | **`1.19x`** |
| 16, 256 | `151.8591` | `95.7565` | `112.4238` | `85.9721` | **`1.59x`** |
| 128, 8 | `206.6667` | `253.4171` | `434.8224` | `213.6237` | **`0.82x`** |
| 128, 64 | `1730.5963` | `177.4254` | `385.4197` | `150.9823` | **`9.75x`** |
| 128, 256 | `6956.8026` | `413.1185` | `629.2939` | `381.1892` | **`16.84x`** |

- **Large $(D, K)$: big gains.** V4 is `9.75x` faster at $(128, 64)$ and `16.84x` at $(128, 256)$. Part of this gap comes from V3 hitting **both** of its capacity walls there: its shared-memory accumulators do not fit (`65,792 B` > `65,536 B` at $K=64$), and at $K=256$ its centroids exceed `__constant__` too (`docs/v3_analysis.md`, `docs/v2_analysis.md`), so V3 runs its slowest path. V4 also falls back to global atomics for accumulation at these sizes, but its distance step no longer depends on either limit.
- **Small $K$: overhead.** At $K=8$ V4 ties ($D=16$: `0.98x`) or loses ($D=128$: `0.82x`); even with no re-rank it does not win at $(128, 8)$ (`0.97x`). Reasons: (i) a centroid tile is 64 wide, so with $K=8$ only `12.5%` of each tile's FMAs are useful; (ii) V4 needs a **separate accumulation pass** that re-reads every point, whereas V3 fuses distance and accumulation in one pass; (iii) at $D=128$ the close-call tolerance re-scores `6.35%` of points.
- **Cost of exactness:** the close-call filter adds `8–19%` over no re-rank, versus `+31–155%` for re-scoring every point.
- **Crossover on T4:** V4 overtakes V3 between $K=8$ and $K=64$ at both $D=16$ and $D=128$.

---

## 4. `ncu`: Compute/Memory Utilization, V3 vs. V4

### 4.1 Large regime $(D, K) = (128, 256)$, `N=10^5`

| Metric | V3 `accumulate_global_v3_kernel<0>` (fused distance + global atomics) | V4 `gemm_argmin_v4_kernel<0,1>` | V4 `accumulate_global_v4_kernel` |
| :--- | ---: | ---: | ---: |
| Duration (SM @ `585 MHz`) | **`62.82 ms`** | **`4.69 ms`** | **`1.24 ms`** |
| DRAM Throughput | **`94.55%`** (`303.35 GB/s`) | `12.87%` | `19.46%` |
| Compute (SM) Throughput | `27.80%` | **`64.97%`** (FMA pipe `65.0%`) | `22.05%` |
| Executed IPC Active | `0.40` | **`2.18`** | `0.09` |
| Executed Instructions | `588,846,893` | `233,580,046` | `2,403,143` |
| L1/TEX Throughput / Hit | `29.68%` / `18.92%` | `93.30%` / `14.08%` | `93.71%` / `0%` |
| L2 Hit Rate | `5.12%` | `70.13%` | `88.82%` |
| Top stall | LG queue `48.7%`, L1TEX scoreboard `44.0%` | — (not memory-stalled) | LG queue `96.0%` |
| Registers / Static Shared | `62` / `0` | `64` / `8.45 KB` | `58` / `0` |

V3's fused kernel re-reads each point coordinate and centroid coordinate for every (point, centroid) pair and is **DRAM-bound at `94.55%`**. V4's GEMM stages a `64`-point × `64`-centroid × `16`-dimension tile in shared memory and reuses every staged value **64 times** from registers, turning the same work into a **compute-bound** kernel (FMA pipe `65%`, IPC `2.18`) with `2.5x` fewer instructions. Distance + accumulation take `5.93 ms` in V4 vs. `62.82 ms` in V3 (`10.6x`) under `ncu` clocks, consistent with the `16.84x` wall-clock gain at `N=10^6`.

### 4.2 Small regime $(D, K) = (32, 16)$, `N=10^6`

| Metric | V3 `accumulate_privatized_v3_kernel<1>` (fused) | V4 `gemm_argmin_v4_kernel<0,1>` | V4 `accumulate_privatized_v4_kernel` |
| :--- | ---: | ---: | ---: |
| Duration (SM @ `585 MHz`) | `12.21 ms` | `4.02 ms` | **`11.95 ms`** |
| Compute (SM) Throughput | `76.86%` | `72.35%` (FMA pipe `72.3%`) | `78.69%` (**FP64 pipe `78.7%`**) |
| Executed IPC Active | `1.00` | `2.61` | `0.59` |
| Shared-memory bank conflicts | `48,948,228` | `1,107,826` (stores) | `49,109,106` |
| Top stall | scoreboard (shared) `54.9%` | — | scoreboard (shared) `59.4%` |

At small $K$ the GEMM is fast (`4.02 ms`, IPC `2.61`), but V4's accumulation pass alone (`11.95 ms`) costs as much as V3's **entire** fused kernel (`12.21 ms`). Both are dominated by the same FP64 shared-memory atomics identified in `docs/v3_analysis.md` (FP64 pipe `78.7%`, `~49 M` bank conflicts). The distance step was never V3's bottleneck at this size, so replacing it with a GEMM cannot pay off; the extra pass is pure overhead.

---

## 5. Summary

| Question (Issue #17) | Answer |
| :--- | :--- |
| Arithmetic-intensity gain at large $(D, K)$? | V4's GEMM reuses each staged value 64x; at $(128,256)$ it is compute-bound (FMA `65%`, IPC `2.18`) where V3 is DRAM-bound (`94.55%`); `10.6x` faster under `ncu`, **`16.84x`** wall-clock at `N=10^6` |
| Overhead at small $(D, K)$? | Ties or loses at $K=8$ (`0.98x`, `0.82x`): padded 64-wide tiles, a separate accumulation pass, and at $D=128$ more close-call re-scores. Crossover between $K=8$ and $K=64$ on T4 |
| Floating-point cancellation? | Raw GEMM ranking: `54–578` mismatches per 100k points in both forms. Close-call exact re-rank: `0` mismatches on all shapes, re-scoring `0.04–6.35%` of points, `+8–19%` time |
| Compute / memory vs. V3 (same T4)? | Section 4: V3 large-regime DRAM `94.55%` / SM `27.80%`; V4 GEMM DRAM `12.87%` / SM `64.97%` |

**Next bottleneck:** at small $K$, the privatized FP64 shared-memory accumulation (shared by V3 and V4) dominates; at large $K$, the global-atomic fallback once accumulators exceed shared memory. Fusing accumulation into the GEMM epilogue, or `float` accumulators with a verified error bound, are the natural follow-ups.
