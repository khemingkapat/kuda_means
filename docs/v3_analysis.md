# V3 Privatized Reduction (`accumulate_privatized_v3_kernel`) — Nsight Compute (`ncu`) Analysis & Shared-Memory Capacity Wall

This report profiles **V3 (`src/v3_privatized/kmeans_v3.cu`)** against **V2 (`src/v2_memory/kmeans_v2.cu`)** on the same GPU, quantifies how much of V1/V2's global-atomic bottleneck V3 removes, and measures where V3's shared-memory privatization breaks down as $K \cdot D$ grows.

---

## 1. Hardware & Benchmark Provenance

All numbers in this report come from the **same GPU and the same binary**:

| Property | Value | Source |
| :--- | :--- | :--- |
| **GPU** | `Tesla T4` (`sm_75`, 40 SMs, `14912.687500` MB) | `results/v3_t4_n1m_d32_k16.json` |
| **CUDA Version** | `13.0` (Google Colab) | `results/v3_t4_n1m_d32_k16.json` |
| **Git Commit** | `f9578a1` (`main`, V3 merged) | `results/v2_t4_n1m_d32_k16.json`, `results/v3_t4_n1m_d32_k16.json` |
| **Primary Workload** | `N=1000000`, `D=32`, `K=16`, `max_iters=50`, `seed=42` (same as `docs/v1_analysis.md`) | `results/v*_t4_n1m_d32_k16.json` |
| **Profile Artifacts** | `ncu --set full --launch-skip 0 --launch-count 2`, one Lloyd iteration, `--no-verify` | `profiles/v2_ncu.txt`, `profiles/v3_ncu.txt` |
| **Capacity Sweep** | `N=1000000`, `D=32`, `K ∈ {16, 64, 128, 190, 252, 253}`, `--iters 10 --runs 3` | `profiles/v3_capacity_sweep_t4.txt` |

> **Backend Note:** as in `docs/v1_analysis.md`, `ncu` was run on Colab because the KMUTT Slurm nodes block hardware performance counters (`ERR_NVGPUCTRPERM`).

---

## 2. End-to-End Timing (`N=1M, D=32, K=16`, 50 iterations, 5-run average)

| Version | `kernel_ms` | `e2e_ms` | `ms_per_iter` | Verified | Source |
| :--- | ---: | ---: | ---: | :--- | :--- |
| V1 (naive) | `983.940527` | `1014.485620` | `19.678811` | `true` | `results/v1_t4_n1m_d32_k16.json` (commit `754c451`) |
| V2 (SoA + `__constant__`) | `320.199239` | `350.333600` | `6.403985` | `true` | `results/v2_t4_n1m_d32_k16.json` |
| **V3 (privatized)** | **`270.533216`** | **`300.851245`** | **`5.410664`** | `true` | `results/v3_t4_n1m_d32_k16.json` |

- **V3 vs. V2:** `1.18x` kernel speedup (`1.16x` end-to-end). Identical `iterations = 50` and `sse = 142835632.253586`, `0` assignment mismatches.
- **V3 vs. V1:** `3.64x` kernel speedup on the same T4 (V1's JSON was produced at an earlier commit, `754c451`).

At $K=16$ the gain is modest; Section 5 shows it grows to `~1.9x` at larger $K$.

---

## 3. Nsight Compute Breakdown: V2 `assign_v2_kernel<1,1>` vs. V3 `accumulate_privatized_v3_kernel<1>`

Both kernels perform the full assignment **and** accumulation for one iteration (`update_centroids_*` is `21.47 us` in both and is ignored).

| Metric | V2 (`profiles/v2_ncu.txt`) | V3 (`profiles/v3_ncu.txt`) | Change |
| :--- | ---: | ---: | :--- |
| Launch | `(3907,1,1) x (256,1,1)` | `(160,1,1) x (256,1,1)` | V3 uses a persistent grid (4 blocks × 40 SMs) |
| Dynamic Shared Memory / Block | `0` B | `4.16` KB | `K·D·8 + K·4 = 4160` B |
| **L2 Cache Throughput** | `37.15 %` | **`3.44 %`** | **`-91%`**: global atomic traffic removed |
| **DRAM Throughput** | `22.89 %` (`73.16 GB/s`) | `8.70 %` (`27.69 GB/s`) | `-62%` |
| **Compute (SM) Throughput** | `16.16 %` | **`76.86 %`** | kernel is now **compute-bound** |
| **Executed IPC Active** | `0.55` | `1.00` | `1.8x` more work issued per cycle |
| Executed Instructions | `122,750,036` | `279,627,281` | `2.28x` (see Section 4) |
| Registers / Thread | `58` | `51` | |
| Achieved Occupancy | `93.83 %` | `96.95 %` | both at the 4-block/SM warp limit |
| **Top Stall Reason** | **LG throttle**: `39.8` of `54.5` cycles (`73.1%`) | **Scoreboard (shared-memory data)**: `17.0` of `31.0` cycles (`54.9%`) | global-queue stall gone |
| Warp Cycles / Issued Instr. | `54.52` | `31.00` | `-43%` |
| Elapsed Cycles (SM @ `585 MHz`) | `5,596,158` (`9.57 ms`) | `7,144,594` (`12.21 ms`) | see note below |

### 3.1 What V3 fixed: global `atomicAdd` contention
In V2, every point issues `1 + D = 33` global atomics, i.e. `33,000,000` per iteration onto only `K·(D+1) = 528` addresses. `ncu` attributes `73.1%` of V2's warp stall cycles to the **LG (local/global) instruction queue being full**, the same signature V1 showed (`95.7%`, `docs/v1_analysis.md`).

V3 reduces this to **at most `528` global atomics per block**, i.e. `160 × 528 = 84,480` per iteration (`~390x` fewer). The LG-throttle stall disappears from V3's top stall reasons, L2 throughput falls from `37.15 %` to `3.44 %`, and the kernel moves from **memory/atomic-bound** to **compute-bound** (`76.86 %` SM throughput) [krulis2020detailed].

### 3.2 Why `ncu` shows V3 *slower* (`12.21 ms` vs `9.57 ms`) while wall-clock shows it *faster*
`ncu` locks the SM clock at `585 MHz` during replay, while DRAM keeps running at `~4.97 GHz`. A compute-bound kernel (V3) scales directly with SM clock, while V2's atomic- and memory-bound time is not limited by it in the same way. The `cudaEvent` timings in Section 2 run at normal (boost) clocks, where V3 is `1.18x` faster per run (`5.41` vs `6.40 ms/iter`). Cycle counts under `ncu` should therefore be read as a **bottleneck diagnosis**, not as a runtime comparison.

---

## 4. Remaining Weakness Inside V3: the Shared-Memory Accumulation Itself

`ncu` flags three related costs in `accumulate_privatized_v3_kernel` (`profiles/v3_ncu.txt`):

| Finding | Measured Value |
| :--- | :--- |
| Shared-memory bank conflicts | average **`6.4`-way**, `48,948,228` conflicts across `9,728,515` shared load requests |
| Excessive shared wavefronts | `55,279,437` (`65%` of shared wavefronts; `ncu` est. speedup `64.2%`) |
| Divergent branches | `60,878.41` average (V2: `0`) |
| Instruction count | `2.28x` V2 |

These are consistent with **FP64 `atomicAdd` on shared memory being emulated on `sm_75` with a load/compare-and-swap retry loop** (extra shared loads, data-dependent branching, and conflicts when group leaders of one warp update neighbouring `double` slots). The warp-level `__match_any_sync` + `__shfl_sync` pre-aggregation already limits shared atomics to one per (warp, cluster, coordinate), but each one is expensive. This can be confirmed with `cuobjdump -sass bin/kuda_means_gpu | grep -E "ATOMS|CAS"` (not yet done).

Options for a follow-up (not implemented here):
- `float` shared accumulators (native shared `atomicAdd` on `float`, half the shared memory, so roughly **2x** the $K$ ceiling), at the cost of precision; must be re-verified against V0.
- Padding or interleaving `s_sums` to reduce bank conflicts.

---

## 5. Shared-Memory Capacity Wall (Tesla T4)

### 5.1 Calculated ceiling
V3 needs `K·D·8 + K·4` bytes of dynamic shared memory per block (`double` sums + `int` counts). The T4 opt-in limit reported by `cudaDevAttrMaxSharedMemoryPerBlockOptin` is **`65,536` B** (logged as `limit=65536 B`). Hence

$$K_{\max}(D) = \left\lfloor \frac{65536}{8D + 4} \right\rfloor$$

| $D$ | Bytes per cluster | $K_{\max}$ on T4 | Above the 48 KB default (needs opt-in) for $K >$ |
| ---: | ---: | ---: | ---: |
| 16 | 132 | **496** | 372 |
| **32** | **260** | **252** (verified) | **189** (verified) |
| 64 | 516 | **127** | 95 |
| 128 | 1028 | **63** | 47 |

> The issue text estimates `K·(D+1)·4` bytes; V3 uses `double` sums (to match V0–V2 exactly), so its real footprint is about `2x` that.

### 5.2 Verified on hardware (`profiles/v3_capacity_sweep_t4.txt`, $D=32$, $N=10^6$, 10 iterations)

| $K$ | Shared / block | V3 path | Grid (blocks/SM) | Theoretical occupancy | V3 `kernel_ms` | V2 `kernel_ms` | V3 speedup |
| ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 16 | 4,160 B | shared | 160 (4) | 100 % | `70.4833` | `74.8680` | `1.06x` |
| 64 | 16,640 B | shared | 120 (3) | 75 % | `139.1690` | `216.4830` | `1.56x` |
| 128 | 33,280 B | shared | 40 (1) | 25 % | `237.2827` | `451.0273` | `1.90x` |
| 190 | 49,400 B | shared (opt-in > 48 KB) | 40 (1) | 25 % | `325.4332` | `604.9611` | `1.86x` |
| **252** | **65,520 B** | **shared** | 40 (1) | 25 % | **`417.8795`** | `769.3374` | **`1.84x`** |
| **253** | **65,780 B** | **global atomic fallback** | 3907 | — | **`767.5942`** | `718.1240` | **`0.94x`** |

**Findings:**
1. **The calculated ceiling is exact:** $K=252$ (`65,520 B`) runs privatized; $K=253$ (`65,780 B`) exceeds `65,536 B` and falls back. $K=190$ confirms the >48 KB opt-in path (`cudaFuncSetAttribute`) works.
2. **Occupancy breaks down early:** blocks per SM fall from **4 → 3 → 1** (theoretical occupancy `100% → 75% → 25%`) as soon as two blocks no longer fit in the SM's 64 KB of shared memory ($K \ge 127$ at $D=32$). The observed grid sizes (`160 → 120 → 40`) match `cudaOccupancyMaxActiveBlocksPerMultiprocessor` exactly.
3. **Low occupancy is tolerated, the cliff is not:** V3 keeps a `~1.84–1.90x` advantage even at 25 % occupancy, because each thread's distance loop ($K \cdot D$ multiply-adds) is long enough to hide latency. But one cluster past the ceiling, V3 loses its entire gain: **`417.88 ms → 767.59 ms` (`1.84x` slower) from $K=252$ to $K=253$**, landing back at V2's speed.
4. **The V3 advantage grows with $K$** (`1.06x` → `~1.9x`): from $K=16$ to $K=252$, V2's time grows `10.3x` while V3's grows `5.9x`. The sweep alone does not isolate why (both versions run the same distance loop); an `ncu` capture at $K=128$ would be needed to attribute V2's extra time.

---

## 6. Verification Note: FMA Contraction Causes Rare Mismatches in **All** GPU Versions

At $K=128$ ($D=32$, $N=10^6$, 10 iterations), **V1, V2, and V3 all report `[verify] status=FAIL`** with *identical* numbers: `assignment_mismatches=56`, `max_centroid_abs_diff=7.808685e-03`, `sse_rel_diff=5.377790e-09` (`profiles/v3_capacity_sweep_t4.txt`).

- The three versions use different accumulation orders but share the same distance loop (`dist_sq += diff * diff`), so the cause is not V3's reduction.
- `nvcc` contracts that line into a fused multiply-add (`-fmad=true` by default); the CPU baseline (`g++ -O3`) rounds the multiply and add separately. For a few points almost exactly between two centroids, this flips the `argmin`, and the difference compounds over iterations.
- **Confirmed:** rebuilding with `NVCC="nvcc -fmad=false"` makes V3 `PASS` with `0` mismatches and the CPU's exact SSE (`181643979.2185`), at essentially the same speed (`233.94 ms` vs `237.28 ms`).

This is a property of the shared build/verification harness, not of V3. Whether to build all GPU versions with `-fmad=false` (bit-faithful to V0) or to relax the verification for near-tie points is a team decision that affects V1–V5 equally.

---

## 7. Summary & Roadmap

| Question (Issue #14) | Answer |
| :--- | :--- |
| Drop in global atomic stalls vs. V2? | LG-throttle stall (`73.1%` of V2 cycles) eliminated; L2 throughput `37.15% → 3.44%`; `~390x` fewer global atomics |
| Occupancy vs. V2? | Equal at $K=16$ (`96.95%` achieved); falls to 75 % at $K=64$ and 25 % at $K \ge 127$ ($D=32$) |
| Kernel time vs. V2 (same T4)? | `1.18x` at $K=16$ (50 iters); `~1.84–1.90x` for $128 \le K \le 252$; `0.94x` past the ceiling |
| $(K, D)$ shared-memory ceiling on T4? | $K \le \lfloor 65536 / (8D+4) \rfloor$, e.g. **$K \le 252$ at $D=32$**, verified at 252 / 253 |

**Next bottleneck → V4:** with atomics removed, V3 is compute-bound (`76.86 %` SM throughput) in the $O(N \cdot K \cdot D)$ distance loop. V4 (distance as tiled GEMM, `src/v4_matmul/`) targets exactly this, and the capacity cliff above motivates tiling over $K$ instead of a full per-block accumulator [lloyd1982least, krulis2020detailed].
