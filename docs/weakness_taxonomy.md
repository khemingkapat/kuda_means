# Weakness Taxonomy: Original K-Means Bottlenecks & Existing Work Trade-Offs

> **Scope:** Literature-grounded taxonomy of the algorithmic, numerical, and GPU-architectural bottlenecks of Lloyd's K-Means clustering, together with the specific weaknesses and failure regimes of existing improvements. Every literature claim cites a verified BibTeX key in [`docs/references.bib`](./references.bib).

---

## 1. Original Lloyd's K-Means: Formulation & Bottleneck Taxonomy

Given $N$ points $X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset \mathbb{R}^D$ and $K$ centroids $C = \{\mathbf{c}_1, \dots, \mathbf{c}_K\} \subset \mathbb{R}^D$, K-Means minimizes the Within-Cluster Sum of Squares (SSE):

$$\mathrm{SSE}(X, C) = \sum_{i=1}^{N} \min_{k \in \{1,\dots,K\}} \|\mathbf{x}_i - \mathbf{c}_k\|_2^2.$$

**Lloyd's Algorithm ("Method I")** `[lloyd1982least]` (originally formulated for 1D scalar quantization in PCM and generalized to $\mathbb{R}^D$) alternates two full-batch steps per iteration $t$:
1. **Assignment (Voronoi Partition):** $a(i) = \arg\min_{k \in \{1,\dots,K\}} \|\mathbf{x}_i - \mathbf{c}_k^{(t)}\|_2^2$ for all $i = 1, \dots, N$ (with deterministic tie-breaking, e.g., left-open/right-closed intervals assigning ties to the lower index `[lloyd1982least]`).
2. **Update (Centroid Recomputation):** $\mathbf{c}_k^{(t+1)} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_i$, where $S_k = \{i : a(i) = k\}$.

In contrast, **MacQueen's online algorithm** `[macqueen1967some]`—which coined the term *"k-means"*—processes observations sequentially one at a time, updating the winning centroid immediately via a running mean $\mu_i^{(n)} = \frac{n-1}{n}\mu_i^{(n-1)} + \frac{1}{n}\mathbf{x}$. While MacQueen's sequential update avoids full-dataset synchronization per update step, it introduces strict order dependence and lacks data-parallelism across points, whereas Lloyd's batch formulation `[lloyd1982least]` is embarrassingly parallel in the assignment step and deterministic under fixed tie-breaking.

However, generalizing full-batch Lloyd's algorithm `[lloyd1982least]` to large-scale $\mathbb{R}^D$ datasets exposes **five coupled bottlenecks**:

1. **$O(NKD)$ Per-Iteration Compute ($O(TNKD)$ Total):**
   Unlike 1D scalar quantization where Voronoi boundaries reduce to $K-1$ sorted midpoints `[lloyd1982least]`, Voronoi cells in $\mathbb{R}^D$ are convex polytopes. Exact assignment evaluates all $K$ pairwise squared distances in $\mathbb{R}^D$ for every point, costing $\sim 3NKD$ FLOPs per iteration ($O(NKD)$) plus $O(ND + KD)$ for centroid reduction across $T$ iterations.
2. **Low Arithmetic Intensity & Full-Batch DRAM Streaming at Small $(K, D)$:**
   Even if all $K$ centroids fit in on-chip registers, shared memory, or constant cache, each point $\mathbf{x}_i \in \mathbb{R}^D$ occupies $4D$ bytes in FP32 and must be streamed from global DRAM every iteration (alongside assignment writes and update reads/writes). Evaluating $K$ Euclidean distances performs $\approx 3KD$ FLOPs per $4D$ bytes of point data, yielding an arithmetic intensity of roughly $\frac{3K}{4}\text{ FLOP/byte}$ during assignment. For small $K$, this lies far below modern GPU compute-to-bandwidth ridge points, making Lloyd's algorithm strictly **memory-bandwidth bound** `[krulis2020detailed]`.
3. **Full-Batch Synchronization Barrier & PCIe Host–Device Overhead:**
   Because the centroid update requires assignments over all $N$ points before iteration $t+1$ can begin `[lloyd1982least]`, every iteration incurs a global barrier. When $N \times D$ approaches or exceeds device memory capacity—or when host-to-device (H2D) and device-to-host (D2H) transfers are included in end-to-end runtime—unoverlapped PCIe transfers become a dominant bottleneck requiring tiled streaming `[johnson2019billion]`.
4. **Initialization Sensitivity, Local Minima & Empty Clusters:**
   Lloyd's Method I guarantees monotonically non-increasing distortion/SSE, $N(\rho^{(1)}) \ge N(\rho^{(2)}) \ge \dots \ge 0$, and converges to stationary points satisfying the necessary centroid and nearest-neighbor conditions `[lloyd1982least]`. However, those conditions are **not sufficient** for global optimality: Lloyd's algorithm frequently terminates in poor local minima (or even saddle points on continuous distributions `[lloyd1982least]`), and uniform random initialization can produce arbitrarily large $\frac{\phi}{\phi_{\text{OPT}}}$ ratios even for fixed $N$ and $K$ `[arthur2007kmeanspp]`. Furthermore, if a Voronoi cell receives zero points ("vanishing mass" / dead cluster), `[lloyd1982least]` notes that explicit handling is required (in `kuda_means`, empty clusters retain their previous centroid to prevent `NaN` propagation).
5. **Naive GPU Mapping Bottlenecks (Uncoalesced AoS & Global Atomic Contention):**
   A direct 1-thread-per-point CUDA implementation using row-major Array-of-Structures ($N \times D$) point storage causes adjacent threads in a warp to stride by $D$ elements across global memory, breaking memory coalescing `[krulis2020detailed]`. Simultaneously, accumulating centroid sums and counts directly into global memory via `atomicAdd` serializes warp/block updates onto $K(D+1)$ shared global accumulator addresses ($K$ clusters' coordinate sums and counts) when $N \gg K$ `[krulis2020detailed]`.

---

## 2. Existing Algorithmic & GPU Improvements — and Their Specific Weaknesses

### 2.1 Careful Seeding: `k-means++` (`[arthur2007kmeanspp]`)
- **What It Solves:** Replaces uniform random seeding (which merges separated clusters and traps Lloyd's local search in arbitrarily bad optima) with $D^2$-weighted sampling: $c_1$ is chosen uniformly at random, and each subsequent center $c_i$ is sampled with probability proportional to $D(x)^2 / \sum_{x \in \mathcal{X}} D(x)^2$, where $D(x)$ is the distance to the nearest already-selected center `[arthur2007kmeanspp]`. Theorem 1.1 proves an expected approximation guarantee of $E[\phi] \le 8(\ln k + 2)\phi_{\text{OPT}}$ right after seeding (tight up to a constant factor via the $2(\ln k)$ lower bound in Theorem 4.1) `[arthur2007kmeanspp]`.
- **Remaining Weakness / Failure Regime:** `k-means++` requires **$K$ inherently sequential passes** over all $N$ points, costing $O(NKD)$ total seeding work `[arthur2007kmeanspp]`. Because the sampling distribution for $c_i$ depends on $D(x)$ induced by $\{c_1, \dots, c_{i-1}\}$, all $K$ seeds cannot be selected in a single parallel pass. When $K$ is large, seeding alone matches or exceeds the runtime of many accelerated clustering epochs (e.g., `[bessa2026emeans]` observes `k-means++` is roughly two orders of magnitude slower than `e-means` training).

### 2.2 Stochastic & EMA Approximations: Mini-Batch K-Means (`[sculley2010web]`) & `e-means` (`[bessa2026emeans]`)
- **What Mini-Batch K-Means Solves (`[sculley2010web]`):** Samples a random mini-batch $M$ of size $b \ll N$ per step (e.g., $b = 1{,}000$ on $n = 781{,}265$ documents), caches nearest centers, and updates each hit centroid with a decaying per-center learning rate $\eta = 1/v[c]$ via $c \leftarrow (1-\eta)c + \eta x$ (Algorithm 1) `[sculley2010web]`. This eliminates full-dataset passes per step and converges to near-optimal solutions orders of magnitude faster than full-batch Lloyd's on large redundant datasets `[sculley2010web]`.
- **What `e-means` Solves (`[bessa2026emeans]`):** Replaces full-batch passes with mini-batch Exponential Moving Average (EMA) updates on cluster counts $N_j \leftarrow \gamma(t) N_j + (1-\gamma(t)) n_j$ and sums $S_j \leftarrow \gamma(t) S_j + (1-\gamma(t)) s_j$ with $c_j = S_j / N_j$, paired with a cosine annealing schedule for $\alpha(t) = 1 - \gamma(t)$, empirical power-law scaling laws for $\alpha_0$ and batch size $B$, periodic dead-cluster revival, and a `last-epoch-mean` exact finalization pass `[bessa2026emeans]`. Across $1\text{M}$–$1\text{B}$ high-dimensional embeddings, `e-means` matches 25-epoch Lloyd MSE in $1.4$–$2.6$ epochs ($9.5\times$–$45.5\times$ GPU wall-clock speedup) `[bessa2026emeans]`.
- **Remaining Weakness / Failure Regime:**
  - **Approximate, Non-Lloyd Trajectory:** Both `[sculley2010web]` and `[bessa2026emeans]` are stochastic approximations. They alter the optimization trajectory and do **not** reproduce exact deterministic batch Lloyd per-iteration centroids, cluster assignments, or SSE (for instance, `[sculley2010web]` evaluates fractional error relative to full-batch Lloyd's converged solution rather than matching it).
  - **Calibration & Regime Sensitivity (`[bessa2026emeans]`):** The `e-means` authors explicitly report that their power-law scaling laws were fitted on high-dimensional DEEP embeddings ($10^5 \le n \le 10^8$, $k \ge 256$, $n/k \ge 8$) and are **not optimal on very low-dimensional or tiny datasets** (an explicit dimensionality exponent was unstable and dropped) `[bessa2026emeans]`. Furthermore, quality degrades rapidly if $B$ exceeds the scaling-law knee, and on synthetic Gaussian/uniform datasets `e-means` can overfit held-out test MSE in later epochs `[bessa2026emeans]`.

### 2.3 GPU Memory Hierarchy & Privatized Reductions (`[krulis2020detailed]`)
- **What It Solves:** Kruliš & Kratochvíl `[krulis2020detailed]` systematically eliminate the memory and atomic bottlenecks of naive CUDA K-Means:
  1. **Structure-of-Arrays (SoA, $D \times N$) Layout:** Storing each dimension contiguously across points ensures adjacent warp threads read adjacent addresses, achieving fully coalesced global DRAM loads `[krulis2020detailed]`.
  2. **On-Chip Centroid Caching (`__constant__` / Shared Memory):** Staging read-only centroids in CUDA `__constant__` memory (broadcast cache) or shared-memory sliding windows avoids repeated global DRAM centroid reads during assignment `[krulis2020detailed]`.
  3. **Privatized Reductions:** Accumulating per-cluster coordinate sums and counts in per-block shared memory (combined with warp-level reductions) reduces global `atomicAdd` traffic from $O(N \cdot D)$ operations down to one block-level flush ($D+1$ scalar `atomicAdd`s) per active cluster per block `[krulis2020detailed]`.
- **Remaining Weakness / Failure Regime (On-Chip Capacity Walls):**
  - **64 KB Constant-Memory Wall:** CUDA `__constant__` memory is strictly capped at $64\text{ KB}$ ($16{,}384$ FP32 floats). Whenever $K \cdot D > 16{,}384$ (e.g., $K = 256, D = 128$), the entire centroid table cannot reside in `__constant__` memory `[krulis2020detailed]`.
  - **Shared-Memory Capacity & Occupancy Wall:** Privatizing $K$ centroids of dimension $D$ plus $K$ counts in per-block shared memory requires $K(D + 1) \times 4\text{ bytes}$ in FP32. As $K \cdot D$ grows past the hardware `MaxSharedMemoryPerBlock` limit (typically $48\text{–}100\text{ KB}$ depending on GPU architecture), privatization either fails to fit, forces severe SM occupancy loss, requires multi-pass centroid tiling/sliding windows `[krulis2020detailed]`, or must fall back to global atomics.

### 2.4 GPU GEMM Distance Expansion (`[johnson2019billion]`)
- **What It Solves:** FAISS GPU `[johnson2019billion]` reformulates pairwise squared L2 distances via the algebraic expansion:

  $$\|\mathbf{x}_j - \mathbf{c}_i\|_2^2 = \|\mathbf{x}_j\|_2^2 + \|\mathbf{c}_i\|_2^2 - 2\langle \mathbf{x}_j, \mathbf{c}_i\rangle.$$

  Precomputing $\|\mathbf{x}_j\|^2$ and $\|\mathbf{c}_i\|^2$ reduces the dominant $O(NKD)$ distance loop to a tiled `cuBLAS` matrix multiplication ($XY^\top$) followed by a fused reduction/selection kernel over the partial distance tile $D'$ `[johnson2019billion]`. At large $(D, K)$, tiled GEMM achieves high arithmetic intensity and approaches peak GPU compute throughput `[johnson2019billion]`.
- **Remaining Weakness / Failure Regime:**
  - **Low GEMM Efficiency & Intermediate Buffer Overhead at Small $(K, D)$:** `[johnson2019billion]` explicitly notes that `cuBLAS` GEMM suffers low efficiency for small reduction dimensions (e.g., $d = 128$ and smaller). Moreover, because row-wise reduction/$k$-selection cannot be cleanly fused into a `cuBLAS` GEMM kernel without hurting GEMM efficiency, the approach must materialize and write/read a tiled $t_q \times K$ partial distance matrix $D'$ in global memory (2 global passes) and manage multi-stream tiling `[johnson2019billion]`—overhead that is slower than a register-fused direct distance loop `[krulis2020detailed]` when $K$ and $D$ are small.
  - **Floating-Point Catastrophic Cancellation (Numerical Property of the Expansion):** While `[johnson2019billion]` does not analyze precision loss in its `arXiv:1702.08734` text, evaluating $\|\mathbf{x}\|_2^2 - 2\langle \mathbf{x}, \mathbf{c}\rangle + \|\mathbf{c}\|_2^2$ in finite-precision FP32 is inherently vulnerable to catastrophic cancellation when $\|\mathbf{x}\|_2^2, \|\mathbf{c}\|_2^2 \gg \|\mathbf{x} - \mathbf{c}\|_2^2$ (nearly equal large norms subtracting to a small residual), which can perturb nearest-centroid tie-breaking and SSE relative to direct $(\mathbf{x}_i - \mathbf{c}_k)^2$ accumulation.

---

## 3. Master Comparison Table

| Approach | Key Reference | Bottleneck Solved | Remaining Weakness / Failure Regime | Role in `kuda_means` |
| :--- | :--- | :--- | :--- | :--- |
| **Batch Lloyd's K-Means** | `[lloyd1982least]` | Deterministic, monotonically non-increasing SSE; embarrassingly parallel Voronoi assignment | $O(NKD)$ compute; low arithmetic intensity ($\approx \frac{3K}{4}$ FLOP/B) & full DRAM pass at small $K, D$; local minima & dead clusters | **V0 Golden Baseline** (exact stopping rule, empty-cluster preservation, deterministic reference) |
| **Online Sequential K-Means** | `[macqueen1967some]` | Avoids full-dataset barrier per update via immediate $1/n_k$ running-mean update | Strictly sequential, order-dependent updates; incompatible with data-parallel batch GPU execution | Contrasted in literature background (not used in V0–V5) |
| **`k-means++` Seeding** | `[arthur2007kmeanspp]` | Eliminates arbitrarily bad local minima from uniform seeding ($O(\log K)$-competitive) | $K$ inherently sequential passes ($O(NKD)$ total work); high seeding overhead at large $K$ | Explains initialization sensitivity (V0–V5 fix initial centroids via $O(K)$ sparse Forgy sampling for exact comparability) |
| **Mini-Batch K-Means** | `[sculley2010web]` | Eliminates full-batch $O(NKD)$ passes via $b \ll N$ sampling & $\eta = 1/v[c]$ updates | Stochastic approximation; does not match exact Lloyd trajectory or centroids | Related work contrast (algorithmic approximation vs. exact GPU acceleration) |
| **`e-means` (EMA Mini-Batch)** | `[bessa2026emeans]` | Converges in $1.4$–$2.6$ epochs ($9.5\times$–$45.5\times$ GPU speedup) via EMA, cosine decay & revival | Approximate trajectory; scaling laws calibrated for large high-$D$ embeddings, suboptimal on low-$D$/tiny data; can overfit synthetic data | Related work contrast (why `kuda_means` preserves exact Lloyd semantics) |
| **CUDA SoA + Constant/Shared Privatization** | `[krulis2020detailed]` | Fixes AoS uncoalesced DRAM reads and global `atomicAdd` contention | Hard capacity walls: $64\text{ KB}$ `__constant__` limit ($K \cdot D \le 16{,}384$ floats) and per-block shared-memory limit $K(D+1)\cdot 4\text{ B}$ | **V2** (SoA + guarded `__constant__`) & **V3** (shared-mem + warp-shuffle privatized reduction) |
| **Tiled GEMM L2 Distance** | `[johnson2019billion]` | Converts $O(NKD)$ distance loop into compute-bound GEMM at large $(D, K)$ | Low GEMM efficiency & 2-pass global $D'$ tile overhead at small $(K, D)$; expansion vulnerable to FP32 catastrophic cancellation when $\|\mathbf{x}\|^2 \approx \|\mathbf{c}\|^2 \gg \|\mathbf{x}-\mathbf{c}\|^2$ | **V4** (tiled GEMM distance targeting large $D, K$, gated by V0 cancellation/SSE verification) |

---

## 4. Architectural Implications for `kuda_means` (V0–V5 Roadmap)

The taxonomy above directly dictates the progressive, profiling-guided architecture of `kuda_means`:

1. **Exact Lloyd Contract over Algorithmic Approximation:**
   Because `[sculley2010web]` and `[bessa2026emeans]` alter the optimization trajectory—and `[bessa2026emeans]` notes scaling-law suboptimality on low-$D$/tiny datasets—all `kuda_means` GPU stages (V1–V5) implement **exact batch Lloyd's algorithm** `[lloyd1982least]` with identical deterministic Gaussian-blob data, identical sparse Forgy initial centroids, identical stopping rule ($\max_k \|\mathbf{c}_k^{(t)} - \mathbf{c}_k^{(t-1)}\|^2 < \varepsilon^2$ or `max_iters`), empty-cluster preservation, and post-loop final assignment pass, verified against **V0 (`src/cpu/`)**.
2. **V1 (`src/v1_naive/`) — Exposing Baseline GPU Bottlenecks:**
   1-thread-per-point AoS global memory + global `atomicAdd` intentionally exposes the uncoalesced DRAM sector and atomic-contention bottlenecks analyzed in `[krulis2020detailed]`.
3. **V2 (`src/v2_memory/`) — Solving Coalescing & Read Broadcasts with Capacity Guards:**
   Transposes points to SoA ($D \times N$) and stages centroids in `__constant__` memory `[krulis2020detailed]`, with an explicit $K \cdot D \le 16{,}384$ floats ($64\text{ KB}$) guard that falls back to read-only global memory (`__ldg` / `__restrict__`) when exceeded.
4. **V3 (`src/v3_privatized/`) — Solving Atomic Contention with Shared-Memory Guards:**
   Uses per-block shared-memory accumulators and warp shuffles `[krulis2020detailed]` to collapse global `atomicAdd` contention to one block-level flush ($D+1$ scalar `atomicAdd`s) per active cluster per block, checking `K * (D + 1) * sizeof(float)` against `cudaDevAttrMaxSharedMemoryPerBlock` to guard or tile large $(K, D)$ regimes.
5. **V4 (`src/v4_matmul/`) — Solving High-$(D, K)$ Arithmetic Intensity & Checking Cancellation:**
   Applies the GEMM distance expansion $\|\mathbf{x}\|_2^2 - 2\mathbf{x}\cdot\mathbf{c} + \|\mathbf{c}\|_2^2$ `[johnson2019billion]` to cross over V3 in large-$(D, K)$ regimes while empirically measuring small-$(K, D)$ overhead and checking FP32 cancellation against V0 (`max_centroid_abs_diff < 1e-3`, `sse_rel_diff < 1e-3`).
6. **V5 (`src/v5_streaming/`) — Solving PCIe Transfer Bottlenecks:**
   Addresses the full-batch H2D/D2H PCIe bottleneck using pinned host memory (`cudaMallocHost`) and chunked multi-stream execution (cf. multi-stream tiling in `[johnson2019billion]`) to overlap transfers with kernel compute.

---

## 5. Verified Bibliography Key Index

All citation keys used in this document map verbatim to [`docs/references.bib`](./references.bib):

- `[lloyd1982least]` — S. P. Lloyd, *"Least Squares Quantization in PCM,"* IEEE Trans. Information Theory, Vol. 28, No. 2, pp. 129–137, Mar. 1982.
- `[macqueen1967some]` — J. B. MacQueen, *"Some Methods for Classification and Analysis of Multivariate Observations,"* Proc. 5th Berkeley Symp. Math. Stat. Probab., Vol. 1, pp. 281–297, 1967.
- `[arthur2007kmeanspp]` — D. Arthur and S. Vassilvitskii, *"k-means++: The Advantages of Careful Seeding,"* Proc. 18th ACM-SIAM SODA, pp. 1027–1035, 2007.
- `[sculley2010web]` — D. Sculley, *"Web-Scale K-Means Clustering,"* Proc. 19th ACM WWW, pp. 1177–1178, 2010.
- `[johnson2019billion]` — J. Johnson, M. Douze, and H. Jégou, *"Billion-Scale Similarity Search with GPUs,"* IEEE Trans. Big Data, Vol. 7, No. 3, pp. 535–547, 2021 (`arXiv:1702.08734`).
- `[krulis2020detailed]` — M. Kruliš and M. Kratochvíl, *"Detailed Analysis and Optimization of CUDA K-means Algorithm,"* Proc. 49th ICPP, pp. 69:1–69:11, 2020.
- `[bessa2026emeans]` — S. Bessa et al., *"e-means: Outperforming k-means (in One-Tenth the Epochs),"* `arXiv:2611.09501`, Nov. 2026.
