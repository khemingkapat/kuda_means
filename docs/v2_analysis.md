# V2 Memory Tuning (`assign_v2_kernel`) — Nsight Compute (`ncu`) Analysis, Constant-Memory Wall & Remaining Atomic Stalls

This report profiles **V2 (`src/v2_memory/kmeans_v2.cu`)** against **V1 (`src/v1_naive/kmeans_v1.cu`)** on the same GPU, measures what the coalesced Structure-of-Arrays (SoA) layout fixed, quantifies the 64 KB `__constant__` capacity threshold, and documents the global `atomicAdd` stall that V2 leaves for V3.

---

## 1. Hardware & Benchmark Provenance

| Property | Value | Source |
| :--- | :--- | :--- |
| **GPU** | `Tesla T4` (`sm_75`, 40 SMs, `14912.687500` MB), Google Colab | `results/v2_t4_n1m_d32_k16.json` |
| **CUDA Version** | `13.0` | `results/v2_t4_n1m_d32_k16.json` |
| **V1 vs. V2 profiles** | `ncu --set full --launch-skip 0 --launch-count 2`, `N=1000000, D=32, K=16`, one iteration, `--no-verify` | `profiles/v1_ncu.txt`, `profiles/v2_ncu.txt` |
| **Constant-wall sweep** | V2, `N=1000000, D=32, K ∈ {64 … 1024}`, `--iters 10 --runs 3 --no-verify`, commit `3b8202a` | `profiles/v2_constant_wall_t4.txt` |
| **Constant-wall profiles** | `ncu` (SpeedOfLight, MemoryWorkloadAnalysis, WarpStateStats), `N=100000, D=32, K=512 / 513` | `profiles/v2_ncu_d32_k512.txt`, `profiles/v2_ncu_d32_k513.txt` |

> As in `docs/v1_analysis.md`, `ncu` runs on Colab because KMUTT Slurm blocks GPU performance counters (`ERR_NVGPUCTRPERM`). `ncu` locks the SM clock at `585 MHz`, so its durations diagnose bottlenecks; wall-clock comparisons use the `cudaEvent` timings in `results/` and `profiles/v2_constant_wall_t4.txt`.

---

## 2. End-to-End Timing (`N=1M, D=32, K=16`, 50 iterations, 5-run average, same T4)

| Version | `kernel_ms` | `e2e_ms` | `ms_per_iter` | Verified | Source |
| :--- | ---: | ---: | ---: | :--- | :--- |
| V1 (AoS, global centroids) | `983.940527` | `1014.485620` | `19.678811` | `true` | `results/v1_t4_n1m_d32_k16.json` (commit `754c451`) |
| **V2 (SoA, `__constant__` centroids)** | **`320.199239`** | **`350.333600`** | **`6.403985`** | `true` | `results/v2_t4_n1m_d32_k16.json` (commit `f9578a1`) |

**V2 is `3.07x` faster than V1** (kernel; `2.90x` end-to-end), with identical `iterations = 50` and `sse = 142835632.253586`.

---

## 3. V1 vs. V2: Coalescing Efficiency & Warp Stalls (`ncu`, `N=1M, D=32, K=16`)

Both kernels launch `(3907,1,1) x (256,1,1)` and perform assignment + global accumulation for one iteration.

| Metric | V1 `assign_v1_kernel<1>` | V2 `assign_v2_kernel<1,1>` | Change |
| :--- | ---: | ---: | :--- |
| **Duration** (SM @ `585 MHz`) | `51.39 ms` (`30,062,145` cycles) | `9.57 ms` (`5,596,158` cycles) | **`5.37x` faster** |
| **Total L2 global sectors** | `573,120,612` | `81,120,612` | **`7.07x` fewer** |
| **Excessive (uncoalesced) sectors** | `481,058,144` (**`84%`**) | `5,058,144` (**`6%`**) | coalescing fixed |
| L1/TEX Cache Throughput | `98.75 %` (saturated) | `46.49 %` | L1 no longer the limiter |
| L1/TEX Hit Rate | `85.05 %` | `2.98 %` | V2 streams each sector once instead of re-touching partially used sectors |
| DRAM Throughput | `10.71 %` | `22.89 %` (`73.16 GB/s`) | useful bytes now arrive faster |
| L2 Cache Throughput | `6.32 %` | `37.15 %` | L2 now dominated by atomics (Section 5) |
| Compute (SM) Throughput | `5.67 %` | `16.16 %` | |
| Executed IPC Active | `0.08` | `0.55` | `6.9x` |
| Warp Cycles per Issued Instruction | `398.26` | `54.52` | `-86%` |
| **Top stall: LG queue full** | `380.9` of `398.3` cycles (**`95.7%`**) | `39.8` of `54.5` cycles (**`73.1%`**) | still the top stall |
| No Eligible Warps | `98.02 %` | `86.19 %` | |
| Achieved Occupancy | `98.35 %` | `93.83 %` | |

**Interpretation.** V1 reads `pt[j]` from a row-major `N x D` array, so adjacent threads are `4·D = 128` bytes apart and `84%` of transferred sectors are wasted. V2's `D x N` layout (`points[j * n + idx]`) makes every warp read 32 consecutive floats, cutting excessive sectors to `6%` and total sectors by `7x`; the L1/TEX pipeline drops from saturation (`98.75%`) to `46.49%` [krulis2020detailed].

---

## 4. The 64 KB `__constant__` Threshold

### 4.1 Calculated threshold
V2 stores centroids in `__constant__ float c_centroids[16384]` (`64 KB`) when $K \cdot D \le 16{,}384$, otherwise it reads them from global memory via `__ldg`:

$$K \cdot D \cdot 4 \text{ B} \le 65{,}536 \text{ B} \iff K \le \left\lfloor \frac{16384}{D} \right\rfloor$$

| $D$ | Largest $K$ in `__constant__` | First $K$ on `__ldg` fallback |
| ---: | ---: | ---: |
| 16 | 1024 | 1025 |
| **32** | **512** (verified) | **513** (verified) |
| 64 | 256 | 257 |
| 128 | 128 | 129 |

**Correctness on both sides** (`N=100000, D=32`, 10 iterations): $K=512$ and $K=513$ both `PASS` with `0` mismatches, `0` centroid and SSE difference (`profiles/v2_constant_wall_t4.txt`).

### 4.2 Measured: crossing the threshold makes V2 *faster*, not slower

`profiles/v2_constant_wall_t4.txt` (`N=1M, D=32`, 10 iterations, 3-run average):

| $K$ | $K \cdot D$ | Centroid path | `kernel_ms` | `kernel_ms` / $K$ |
| ---: | ---: | :--- | ---: | ---: |
| 64 | 2,048 | `__constant__` | `218.2698` | `3.41` |
| 128 | 4,096 | `__constant__` | `446.8574` | `3.49` |
| 256 | 8,192 | `__constant__` | `882.5421` | `3.45` |
| 384 | 12,288 | `__constant__` | `1288.7840` | `3.36` |
| **512** | **16,384** | **`__constant__`** | **`1700.7930`** | **`3.32`** |
| **513** | **16,416** | **`__ldg` fallback** | **`758.0021`** | **`1.48`** |
| 640 | 20,480 | `__ldg` fallback | `941.8905` | `1.47` |
| 1024 | 32,768 | `__ldg` fallback | `1533.9346` | `1.50` |

- Cost per cluster is flat on each side: **`~3.4 ms` with `__constant__`** vs. **`~1.5 ms` with `__ldg`**.
- Going from $K=512$ to $K=513$ makes V2 **`2.24x` faster** (`1700.79 → 758.00 ms`). At `N=100000` the same step is `1.48x` (`142.78 → 96.41 ms`).
- So on this T4, the `__constant__` path is **not** the fast path for any $K \ge 64$ at $D=32$. The "wall" at 64 KB is a capacity limit only; performance already favours `__ldg` well below it. The sweep did not cover $K < 64$, so whether `__constant__` wins for very small centroid sets is **not measured**.

### 4.3 What `ncu` shows at the threshold (`N=100000, D=32`)

| Metric | $K=512$ (`assign_v2_kernel<1,1>`, `__constant__`) | $K=513$ (`assign_v2_kernel<1,0>`, `__ldg`) |
| :--- | ---: | ---: |
| Duration (SM @ `585 MHz`) | `12.74 ms` | `10.77 ms` |
| **DRAM Throughput** | **`61.10 %`** (`215.22 GB/s`) | **`5.93 %`** (`18.95 GB/s`) |
| **L1/TEX Hit Rate** | **`18.96 %`** | **`59.04 %`** |
| **L2 Hit Rate** | **`50.72 %`** | **`96.19 %`** |
| Compute (SM) Throughput | `41.57 %` | `81.76 %` |
| Bottleneck (`ncu` rule) | memory (DRAM) | compute / L1 pipes (`>80%`) |
| Top stall | L1TEX scoreboard (global loads): `9.0` of `25.8` cycles (`35.0%`) | LG queue full: `13.4` of `20.3` cycles (`66.1%`) |

The `__constant__` variant does **not** stall on constant-cache misses; its top stall is waiting on **global** (L1TEX) loads, and its point data misses L1 and L2 far more often, pushing DRAM traffic up `~11x`. Both variants re-read each point's `D` coordinates from global memory once per centroid (`K·D` loads per thread); the measurements show the `__ldg` variant keeps that working set in cache much better, but the profiles collected here (no source-level counters) do not isolate *why* the instruction stream with constant-memory operands caches worse. That needs `--section SourceCounters` or a SASS comparison.

---

## 5. Remaining Weakness: Global `atomicAdd` Contention (-> V3)

V2 does not change accumulation: every point still issues `1 + D = 33` global atomics (`33,000,000` per iteration at `N=1M, D=32`) onto `K·(D+1) = 528` addresses.

- At $K=16$, the **LG-queue stall is still `73.1%`** of warp cycles (`profiles/v2_ncu.txt`), down from V1's `95.7%` only because the loads around it got cheaper.
- **L2 throughput rises from `6.32%` (V1) to `37.15%` (V2)**: with coalesced loads, the atomics become the dominant L2 traffic.
- On the `__ldg` side of the threshold ($K=513$) the LG-queue stall is again the top stall (`66.1%`), now shared between atomics and centroid loads.

This is exactly what V3 removes: per-block shared-memory accumulators cut global atomics `~390x` and eliminate the LG-queue stall (`docs/v3_analysis.md`).

---

## 6. Summary

| Question (Issue #11) | Answer |
| :--- | :--- |
| V1 vs. V2 coalescing efficiency (same T4)? | Excessive sectors `84% → 6%`; total sectors `7.07x` fewer; L1/TEX `98.75% → 46.49%` |
| V1 vs. V2 warp stalls? | LG-queue stall `95.7% → 73.1%` of cycles; cycles per issued instruction `398.26 → 54.52` |
| Speedup? | `3.07x` kernel time (`983.94 → 320.20 ms`, `N=1M, D=32, K=16`) |
| $K \cdot D$ threshold where `__constant__` overflows? | $K \cdot D > 16{,}384$ floats (`64 KB`), i.e. $K > \lfloor 16384/D \rfloor$; **$K=513$ at $D=32$**, verified |
| Effect of crossing it? | Correct on both sides, but the `__ldg` fallback is **`2.24x` faster** at $K=513$ vs. $K=512$ (`~2.3x` cheaper per cluster) |
| Remaining bottleneck? | Global `atomicAdd` contention (LG-queue stall, L2 `37%`) → addressed by V3 |

**Design implication (not implemented here):** on T4 the `UseConstant` path should be re-evaluated, either lowering the threshold or defaulting to `__ldg`. V3 inherits the same `__constant__` choice (`src/v3_privatized/kmeans_v3.cu`), so the same experiment applies there [lloyd1982least].
