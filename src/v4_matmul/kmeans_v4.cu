// V4 — Distance as MatMul: tiled shared-memory GEMM with a fused argmin.
//
// V3 profiling showed that, once global atomics are privatized, K-Means is compute-bound in the
// O(N*K*D) distance loop. V4 rewrites that loop with the expansion
//     ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2          (Johnson et al., 2019)
// so the dominant work becomes the dense product X * C^T, computed with a classic register-tiled
// shared-memory GEMM:
//   * a 16x16 thread block owns 64 points and sweeps all centroids in tiles of 64,
//   * 16-dimension slices of X (SoA, D x N) and C (K x D) are staged in shared memory,
//   * each thread keeps a 4x4 micro-tile of dot products in registers (float4 shared reads),
//   * the argmin is fused into the epilogue, so the N x K distance matrix is never stored.
// ||x||^2 is constant per point and does not change the argmin, so by default centroids are ranked
// by the shifted distance ||c||^2 - 2 x.c. Setting KUDA_V4_FULL_EXPANSION=1 ranks by the full
// float expansion instead, to measure floating-point cancellation against V0.
// Either form loses precision when ||x||, ||c|| >> ||x - c|| (cancellation), which flipped a few
// near-tie assignments vs. V0. The GEMM keeps each point's top-2 candidates and, only when their
// gap is within a float error bound (a close call), re-scores both with the exact (x - c)^2 form.
// KUDA_V4_EXACT_RERANK=0 disables this (raw error); =all re-scores every point (cost reference).
// Accumulation reuses V3's privatized shared-memory reduction (global-atomic fallback
// when the accumulators do not fit). Centroids are read with __ldg (see docs/v2_analysis.md).

#include "cuda_utils.cuh"
#include "kmeans_common.h"
#include "kmeans_gpu.cuh"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error "V4 requires sm_70+ (__match_any_sync in the accumulation kernel)."
#endif

namespace kuda {

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr size_t kDefaultSharedLimit = 48 * 1024;

// GEMM tiling: block tile = kTileM points x kTileN centroids, staged kTileK dimensions at a time.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kMicro = 4;                           // per-thread micro-tile: 4 points x 4 centroids
constexpr int kThreadsX = kTileN / kMicro;          // 16 threads across centroids
constexpr int kThreadsY = kTileM / kMicro;          // 16 threads across points
constexpr int kGemmThreads = kThreadsX * kThreadsY; // 256
constexpr int kCentroidPad = 4;                     // keeps float4 alignment, staggers banks
constexpr int kNoCentroid = INT_MAX;                // empty top-2 slot

// Exact re-rank modes (KUDA_V4_EXACT_RERANK): off, close calls only (default), every point.
constexpr int kRerankOff = 0;
constexpr int kRerankCloseCalls = 1;
constexpr int kRerankAll = 2;
// A point is a close call when its two best GEMM distances differ by less than
// kRerankSafety * (D + 2) * FLT_EPSILON * (||x||^2 + max ||c||^2). (D + 2) * eps * scale bounds the
// float error of one expansion term sum; the safety factor covers both candidates' GEMM error and
// the rounding of the exact form, so any pair whose order could differ from V0 is re-scored.
constexpr float kRerankSafety = 8.0f;

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a whole number of warps");
static_assert(kThreadsX == 16, "Argmin reduction assumes 16 lanes share one point row");
static_assert(kGemmThreads % kWarpSize == 0, "GEMM block must be a whole number of warps");

__global__ void point_norms_v4_kernel(const float *__restrict__ points, float *xnorm, int n,
                                      int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float s = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float v = points[static_cast<size_t>(j) * static_cast<size_t>(n) + idx];
        s = fmaf(v, v, s);
    }
    xnorm[idx] = s;
}

__global__ void centroid_norms_v4_kernel(const float *__restrict__ centroids, float *cnorm, int d,
                                         int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) {
        return;
    }
    const float *row = centroids + static_cast<size_t>(c) * static_cast<size_t>(d);
    float s = 0.0f;
    for (int j = 0; j < d; ++j) {
        s = fmaf(row[j], row[j], s);
    }
    cnorm[c] = s;
}

// Strict ranking on (distance, centroid index): lower distance first, ties to the lower index
// (the same first-minimum rule as V0's ascending scan).
__device__ __forceinline__ bool ranks_before(float a, int ka, float b, int kb) {
    return a < b || (a == b && ka < kb);
}

// Exact squared distance in the same form as V0-V3 (accumulate diff * diff), used to re-rank.
__device__ __forceinline__ float exact_dist_sq(const float *__restrict__ points,
                                               const float *__restrict__ centroids, int p, int c,
                                               int n, int d) {
    const float *ctr = centroids + static_cast<size_t>(c) * static_cast<size_t>(d);
    float dist_sq = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float diff =
            points[static_cast<size_t>(j) * static_cast<size_t>(n) + p] - __ldg(&ctr[j]);
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// Tiled GEMM (X * C^T) fused with a top-2 argmin. The expansion loses precision when ||x||, ||c||
// are large relative to ||x - c|| (cancellation), so the two best GEMM candidates of a close call
// are re-scored with the exact (x - c)^2 form before the final choice.
template <bool FullExpansion, int RerankMode>
__global__ void __launch_bounds__(kGemmThreads)
    gemm_argmin_v4_kernel(const float *__restrict__ points, const float *__restrict__ centroids,
                          const float *__restrict__ cnorm, const float *__restrict__ xnorm,
                          int *assignments, unsigned int *rerank_count, int n, int d, int k) {
    __shared__ __align__(16) float s_x[kTileK][kTileM];
    __shared__ __align__(16) float s_c[kTileK][kTileN + kCentroidPad];

    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid % kThreadsX; // centroid micro-tile column
    const int ty = tid / kThreadsX; // point micro-tile row
    const int p0 = blockIdx.x * kTileM;
    const auto n_sz = static_cast<size_t>(n);
    const auto d_sz = static_cast<size_t>(d);

    // Per point row: best and second-best (distance, centroid) seen by this thread.
    float best1[kMicro];
    float best2[kMicro];
    int best1_k[kMicro];
    int best2_k[kMicro];
    float x_norm[kMicro];
#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
        best1[i] = FLT_MAX;
        best2[i] = FLT_MAX;
        best1_k[i] = kNoCentroid;
        best2_k[i] = kNoCentroid;
        x_norm[i] = 0.0f;
        if constexpr (FullExpansion) {
            const int p = p0 + ty * kMicro + i;
            x_norm[i] = p < n ? xnorm[p] : 0.0f;
        }
    }

    for (int c0 = 0; c0 < k; c0 += kTileN) {
        float acc[kMicro][kMicro];
#pragma unroll
        for (int i = 0; i < kMicro; ++i) {
#pragma unroll
            for (int j = 0; j < kMicro; ++j) {
                acc[i][j] = 0.0f;
            }
        }

        for (int d0 = 0; d0 < d; d0 += kTileK) {
            // Stage X tile: consecutive threads read consecutive points (coalesced SoA rows).
            for (int e = tid; e < kTileK * kTileM; e += kGemmThreads) {
                const int kk = e / kTileM;
                const int m = e % kTileM;
                const int p = p0 + m;
                const int dd = d0 + kk;
                s_x[kk][m] = (p < n && dd < d) ? points[static_cast<size_t>(dd) * n_sz + p] : 0.0f;
            }
            // Stage C tile: consecutive threads read consecutive dimensions of one centroid.
            for (int e = tid; e < kTileK * kTileN; e += kGemmThreads) {
                const int cc = e / kTileK;
                const int kk = e % kTileK;
                const int c = c0 + cc;
                const int dd = d0 + kk;
                s_c[kk][cc] = (c < k && dd < d)
                                  ? __ldg(&centroids[static_cast<size_t>(c) * d_sz + dd])
                                  : 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                const float4 a = *reinterpret_cast<const float4 *>(&s_x[kk][ty * kMicro]);
                const float4 b = *reinterpret_cast<const float4 *>(&s_c[kk][tx * kMicro]);
                const float av[kMicro] = {a.x, a.y, a.z, a.w};
                const float bv[kMicro] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < kMicro; ++i) {
#pragma unroll
                    for (int j = 0; j < kMicro; ++j) {
                        acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // Epilogue: distances for this centroid tile, scanned in ascending centroid order.
#pragma unroll
        for (int j = 0; j < kMicro; ++j) {
            const int c = c0 + tx * kMicro + j;
            if (c < k) {
                const float cn = cnorm[c];
#pragma unroll
                for (int i = 0; i < kMicro; ++i) {
                    float dist;
                    if constexpr (FullExpansion) {
                        dist = (x_norm[i] + cn) - 2.0f * acc[i][j];
                    } else {
                        dist = cn - 2.0f * acc[i][j];
                    }
                    if (ranks_before(dist, c, best1[i], best1_k[i])) {
                        best2[i] = best1[i];
                        best2_k[i] = best1_k[i];
                        best1[i] = dist;
                        best1_k[i] = c;
                    } else if (ranks_before(dist, c, best2[i], best2_k[i])) {
                        best2[i] = dist;
                        best2_k[i] = c;
                    }
                }
            }
        }
    }

    // Merge each point row's top-2 across the 16 lanes (tx) with an xor butterfly; lanes hold
    // disjoint centroid sets, so every lane ends with the same global top-2.
    float top_v[kMicro];
    float second_v[kMicro];
    int top_k[kMicro];
    int second_k[kMicro];
#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
        float v1 = best1[i];
        float v2 = best2[i];
        int k1 = best1_k[i];
        int k2 = best2_k[i];
#pragma unroll
        for (int offset = kThreadsX / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(kFullMask, v1, offset);
            const int ok1 = __shfl_xor_sync(kFullMask, k1, offset);
            const float o2 = __shfl_xor_sync(kFullMask, v2, offset);
            const int ok2 = __shfl_xor_sync(kFullMask, k2, offset);
            if (ranks_before(v1, k1, o1, ok1)) {
                // Mine stays first; second is the better of my second and their first.
                if (ranks_before(o1, ok1, v2, k2)) {
                    v2 = o1;
                    k2 = ok1;
                }
            } else {
                // Theirs becomes first; second is the better of my first and their second.
                if (ranks_before(v1, k1, o2, ok2)) {
                    v2 = v1;
                    k2 = k1;
                } else {
                    v2 = o2;
                    k2 = ok2;
                }
                v1 = o1;
                k1 = ok1;
            }
        }
        top_v[i] = v1;
        second_v[i] = v2;
        top_k[i] = k1;
        second_k[i] = k2;
    }

    // Lanes tx = 0..3 each finalize one of the row's 4 points (spreads the exact re-rank work).
#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
        const int p = p0 + ty * kMicro + i;
        if (tx == i && p < n) {
            int choice = top_k[i];
            if constexpr (RerankMode != kRerankOff) {
                const int alt = second_k[i];
                if (alt != kNoCentroid) {
                    bool close_call = true;
                    if constexpr (RerankMode == kRerankCloseCalls) {
                        const float scale = xnorm[p] + fmaxf(cnorm[choice], cnorm[alt]);
                        const float tol =
                            kRerankSafety * static_cast<float>(d + 2) * FLT_EPSILON * scale;
                        close_call = (second_v[i] - top_v[i]) <= tol;
                    }
                    if (close_call) {
                        atomicAdd(rerank_count, 1u);
                        const float e1 = exact_dist_sq(points, centroids, p, choice, n, d);
                        const float e2 = exact_dist_sq(points, centroids, p, alt, n, d);
                        if (ranks_before(e2, alt, e1, choice)) {
                            choice = alt;
                        }
                    }
                }
            }
            assignments[p] = choice;
        }
    }
}

// Segmented warp reduction over lanes that share a cluster (same as V3).
__device__ __forceinline__ double reduce_peers(unsigned peers, double x) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    int rank = __popc(peers & ((1u << lane) - 1u));
    unsigned higher = peers & (0xfffffffeu << lane);

    while (__any_sync(kFullMask, higher != 0u)) {
        const int next = __ffs(static_cast<int>(higher)); // 1-based lane, 0 if none
        const double other = __shfl_sync(kFullMask, x, next > 0 ? next - 1 : lane);
        if (next > 0) {
            x += other;
        }
        higher &= ~__ballot_sync(kFullMask, rank & 1);
        rank >>= 1;
    }
    return x;
}

// Privatized accumulation (V3 scheme) driven by the assignments written by the GEMM kernel.
__global__ void accumulate_privatized_v4_kernel(const float *__restrict__ points,
                                                const int *__restrict__ assignments, double *sums,
                                                int *counts, int n, int d, int k) {
    extern __shared__ double s_mem[];
    const int kd = k * d;
    double *s_sums = s_mem;
    int *s_counts = reinterpret_cast<int *>(s_mem + kd);

    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        s_sums[i] = 0.0;
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const auto n_sz = static_cast<size_t>(n);
    const int stride = gridDim.x * blockDim.x;

    // `base` is block-uniform so all 32 lanes reach the *_sync intrinsics together.
    for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
        const int idx = base + static_cast<int>(threadIdx.x);
        const bool valid = idx < n;
        const int best_k = valid ? assignments[idx] : -1;

        const unsigned peers = __match_any_sync(kFullMask, best_k);
        const bool is_leader = lane == (__ffs(static_cast<int>(peers)) - 1);
        const bool writes = valid && is_leader;

        if (writes) {
            atomicAdd(&s_counts[best_k], __popc(peers));
        }
        for (int j = 0; j < d; ++j) {
            const double x =
                valid ? static_cast<double>(points[static_cast<size_t>(j) * n_sz + idx]) : 0.0;
            const double group_sum = reduce_peers(peers, x);
            if (writes) {
                atomicAdd(&s_sums[best_k * d + j], group_sum);
            }
        }
    }
    __syncthreads();

    // Flush: at most one global atomic per centroid coordinate (and per count) per block.
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        const double v = s_sums[i];
        if (v != 0.0) {
            atomicAdd(&sums[i], v);
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int c = s_counts[i];
        if (c != 0) {
            atomicAdd(&counts[i], c);
        }
    }
}

// Capacity-wall fallback: one global atomic per point per coordinate.
__global__ void accumulate_global_v4_kernel(const float *__restrict__ points,
                                            const int *__restrict__ assignments, double *sums,
                                            int *counts, int n, int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int best_k = assignments[idx];
    atomicAdd(&counts[best_k], 1);
    double *sum_row = sums + static_cast<size_t>(best_k) * static_cast<size_t>(d);
    for (int j = 0; j < d; ++j) {
        const float pt_val = points[static_cast<size_t>(j) * static_cast<size_t>(n) + idx];
        atomicAdd(&sum_row[j], static_cast<double>(pt_val));
    }
}

__global__ void update_centroids_v4_kernel(float *centroids, const double *sums, const int *counts,
                                           double *shift_sq_out, int d, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) {
        return;
    }

    const int count = counts[c];
    if (count == 0) {
        // Retain previous centroid to avoid NaN on empty clusters.
        shift_sq_out[c] = 0.0;
        return;
    }

    const double inv_count = 1.0 / static_cast<double>(count);
    const auto row_offset = static_cast<size_t>(c) * static_cast<size_t>(d);
    float *ctr = centroids + row_offset;
    const double *sum_row = sums + row_offset;

    double shift_sq = 0.0;
    for (int j = 0; j < d; ++j) {
        const float new_val = static_cast<float>(sum_row[j] * inv_count);
        const double diff = static_cast<double>(new_val) - static_cast<double>(ctr[j]);
        shift_sq += diff * diff;
        ctr[j] = new_val;
    }
    shift_sq_out[c] = shift_sq;
}

struct V4DeviceBuffers {
    float *points = nullptr;
    float *centroids = nullptr;
    float *cnorm = nullptr;
    float *xnorm = nullptr; // ||x||^2: full expansion and the close-call tolerance
    double *sums = nullptr;
    int *counts = nullptr;
    int *assignments = nullptr;
    double *shift_sq = nullptr;
    unsigned int *rerank_count = nullptr; // points re-scored exactly (diagnostic)
};

struct V4LaunchPlan {
    bool full_expansion = false;
    int rerank_mode = kRerankCloseCalls;
    bool use_shared = false;
    size_t smem_bytes = 0;
    int grid_n = 0;    // one thread per point
    int grid_k = 0;    // one thread per centroid
    int grid_gemm = 0; // one block per kTileM points
    int grid_acc = 0;  // persistent grid for the privatized accumulation kernel
};

// Reads a 0/1 switch from the environment; unset or empty keeps the default.
bool env_flag(const char *name, bool default_value) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0;
}

// KUDA_V4_EXACT_RERANK: "0" = off, "all" = every point, anything else / unset = close calls only.
int rerank_mode_from_env() {
    const char *env = std::getenv("KUDA_V4_EXACT_RERANK");
    if (env != nullptr && std::strcmp(env, "0") == 0) {
        return kRerankOff;
    }
    if (env != nullptr && std::strcmp(env, "all") == 0) {
        return kRerankAll;
    }
    return kRerankCloseCalls;
}

const char *rerank_mode_name(int mode) {
    if (mode == kRerankOff) {
        return "off";
    }
    return mode == kRerankAll ? "all" : "close_calls";
}

V4LaunchPlan make_launch_plan(const KMeansConfig &cfg) {
    const auto k = static_cast<size_t>(cfg.k_clusters);
    const auto d = static_cast<size_t>(cfg.dim);

    int device = 0;
    int smem_optin = 0;
    int sm_count = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    V4LaunchPlan plan;
    plan.full_expansion = env_flag("KUDA_V4_FULL_EXPANSION", false);
    plan.rerank_mode = rerank_mode_from_env();
    plan.smem_bytes = k * d * sizeof(double) + k * sizeof(int);
    plan.use_shared = plan.smem_bytes <= static_cast<size_t>(smem_optin);
    plan.grid_n = (cfg.n_points + kBlockSize - 1) / kBlockSize;
    plan.grid_k = (cfg.k_clusters + kBlockSize - 1) / kBlockSize;
    plan.grid_gemm = (cfg.n_points + kTileM - 1) / kTileM;
    plan.grid_acc = plan.grid_n;

    std::fprintf(stderr,
                 "[v4] distance: gemm (tile %dx%dx%d, %d threads), expansion=%s, exact_rerank=%s\n",
                 kTileM, kTileN, kTileK, kGemmThreads, plan.full_expansion ? "full" : "shifted",
                 rerank_mode_name(plan.rerank_mode));

    if (plan.use_shared) {
        if (plan.smem_bytes > kDefaultSharedLimit) {
            CHECK_CUDA(cudaFuncSetAttribute(accumulate_privatized_v4_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(plan.smem_bytes)));
        }
        int blocks_per_sm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, accumulate_privatized_v4_kernel, kBlockSize, plan.smem_bytes));
        plan.grid_acc = std::min(plan.grid_n, std::max(1, blocks_per_sm) * sm_count);
        std::fprintf(stderr,
                     "[v4] reduction path: shared (smem=%zu B/block, limit=%d B, grid=%d blocks)\n",
                     plan.smem_bytes, smem_optin, plan.grid_acc);
    } else {
        std::fprintf(
            stderr, "[v4] reduction path: global atomic fallback (need %zu B/block > limit %d B)\n",
            plan.smem_bytes, smem_optin);
    }
    return plan;
}

template <bool FullExpansion, int RerankMode>
void launch_gemm_mode(const V4LaunchPlan &plan, const V4DeviceBuffers &buf,
                      const KMeansConfig &cfg) {
    gemm_argmin_v4_kernel<FullExpansion, RerankMode><<<plan.grid_gemm, kGemmThreads>>>(
        buf.points, buf.centroids, buf.cnorm, buf.xnorm, buf.assignments, buf.rerank_count,
        cfg.n_points, cfg.dim, cfg.k_clusters);
}

template <bool FullExpansion>
void launch_gemm(const V4LaunchPlan &plan, const V4DeviceBuffers &buf, const KMeansConfig &cfg) {
    if (plan.rerank_mode == kRerankOff) {
        launch_gemm_mode<FullExpansion, kRerankOff>(plan, buf, cfg);
    } else if (plan.rerank_mode == kRerankAll) {
        launch_gemm_mode<FullExpansion, kRerankAll>(plan, buf, cfg);
    } else {
        launch_gemm_mode<FullExpansion, kRerankCloseCalls>(plan, buf, cfg);
    }
}

void launch_assign(const V4LaunchPlan &plan, const V4DeviceBuffers &buf, const KMeansConfig &cfg) {
    centroid_norms_v4_kernel<<<plan.grid_k, kBlockSize>>>(buf.centroids, buf.cnorm, cfg.dim,
                                                          cfg.k_clusters);
    CHECK_LAST_CUDA_ERROR();
    if (plan.full_expansion) {
        launch_gemm<true>(plan, buf, cfg);
    } else {
        launch_gemm<false>(plan, buf, cfg);
    }
    CHECK_LAST_CUDA_ERROR();
}

void launch_accumulate(const V4LaunchPlan &plan, const V4DeviceBuffers &buf,
                       const KMeansConfig &cfg) {
    if (plan.use_shared) {
        accumulate_privatized_v4_kernel<<<plan.grid_acc, kBlockSize, plan.smem_bytes>>>(
            buf.points, buf.assignments, buf.sums, buf.counts, cfg.n_points, cfg.dim,
            cfg.k_clusters);
    } else {
        accumulate_global_v4_kernel<<<plan.grid_n, kBlockSize>>>(
            buf.points, buf.assignments, buf.sums, buf.counts, cfg.n_points, cfg.dim);
    }
    CHECK_LAST_CUDA_ERROR();
}

void run_single_lloyd_v4(const std::vector<float> &points_soa,
                         const std::vector<float> &init_centroids, const KMeansConfig &cfg,
                         const V4LaunchPlan &plan, const V4DeviceBuffers &buf,
                         std::vector<double> &h_shift_sq, std::vector<int> &assignments,
                         std::vector<float> &centroids, int &iterations_out, bool record_timers,
                         double &kernel_ms_out, double &e2e_ms_out, unsigned int &reranked_out,
                         CudaTimer &kernel_timer, CudaTimer &e2e_timer) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);

    const size_t points_bytes = n * d * sizeof(float);
    const size_t centroids_bytes = k * d * sizeof(float);
    const size_t sums_bytes = k * d * sizeof(double);
    const size_t counts_bytes = k * sizeof(int);
    const size_t assignments_bytes = n * sizeof(int);
    const size_t shift_bytes = k * sizeof(double);

    const double eps_sq = static_cast<double>(cfg.epsilon) * static_cast<double>(cfg.epsilon);

    double run_kernel_ms = 0.0;
    double run_e2e_ms = 0.0;
    int iter_done = 0;

    if (record_timers) {
        e2e_timer.start();
    }

    CHECK_CUDA(cudaMemcpy(buf.points, points_soa.data(), points_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(buf.centroids, init_centroids.data(), centroids_bytes, cudaMemcpyHostToDevice));

    // ||x||^2 (full expansion + close-call tolerance): computed once per run, counted as kernel
    // time.
    if (record_timers) {
        kernel_timer.start();
    }
    point_norms_v4_kernel<<<plan.grid_n, kBlockSize>>>(buf.points, buf.xnorm, cfg.n_points,
                                                       cfg.dim);
    CHECK_LAST_CUDA_ERROR();
    if (record_timers) {
        run_kernel_ms += static_cast<double>(kernel_timer.stop_ms());
    }

    for (int iter = 0; iter < cfg.max_iters; ++iter) {
        if (record_timers) {
            kernel_timer.start();
        }

        CHECK_CUDA(cudaMemsetAsync(buf.sums, 0, sums_bytes, 0));
        CHECK_CUDA(cudaMemsetAsync(buf.counts, 0, counts_bytes, 0));

        launch_assign(plan, buf, cfg);
        launch_accumulate(plan, buf, cfg);

        update_centroids_v4_kernel<<<plan.grid_k, kBlockSize>>>(
            buf.centroids, buf.sums, buf.counts, buf.shift_sq, cfg.dim, cfg.k_clusters);
        CHECK_LAST_CUDA_ERROR();

        if (record_timers) {
            run_kernel_ms += static_cast<double>(kernel_timer.stop_ms());
        }

        CHECK_CUDA(
            cudaMemcpy(h_shift_sq.data(), buf.shift_sq, shift_bytes, cudaMemcpyDeviceToHost));

        double max_shift_sq = 0.0;
        for (size_t c = 0; c < k; ++c) {
            if (h_shift_sq[c] > max_shift_sq) {
                max_shift_sq = h_shift_sq[c];
            }
        }

        ++iter_done;
        if (max_shift_sq < eps_sq) {
            break;
        }
    }

    // Final assignment pass so assignments & SSE match the returned centroids exactly. The
    // re-rank counter is reset here so it reports how many points this pass re-scored.
    CHECK_CUDA(cudaMemsetAsync(buf.rerank_count, 0, sizeof(unsigned int), 0));
    if (record_timers) {
        kernel_timer.start();
    }
    launch_assign(plan, buf, cfg);
    if (record_timers) {
        run_kernel_ms += static_cast<double>(kernel_timer.stop_ms());
    }

    CHECK_CUDA(
        cudaMemcpy(centroids.data(), buf.centroids, centroids_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(assignments.data(), buf.assignments, assignments_bytes, cudaMemcpyDeviceToHost));

    if (record_timers) {
        run_e2e_ms = static_cast<double>(e2e_timer.stop_ms());
    }

    CHECK_CUDA(
        cudaMemcpy(&reranked_out, buf.rerank_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    iterations_out = iter_done;
    kernel_ms_out = run_kernel_ms;
    e2e_ms_out = run_e2e_ms;
}

} // namespace

KMeansResult run_kmeans_v4(const std::vector<float> &points,
                           const std::vector<float> &init_centroids, const KMeansConfig &cfg) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);

    // Transpose row-major AoS points into Structure-of-Arrays (SoA) layout (D x N), as in V2/V3.
    std::vector<float> points_soa(n * d);
    for (size_t idx = 0; idx < n; ++idx) {
        for (size_t j = 0; j < d; ++j) {
            points_soa[j * n + idx] = points[idx * d + j];
        }
    }

    const V4LaunchPlan plan = make_launch_plan(cfg);

    V4DeviceBuffers buf;
    CHECK_CUDA(cudaMalloc(&buf.points, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.centroids, k * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.cnorm, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.xnorm, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.sums, k * d * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&buf.counts, k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buf.assignments, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buf.shift_sq, k * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&buf.rerank_count, sizeof(unsigned int)));
    unsigned int reranked = 0;

    KMeansResult res;
    res.assignments.resize(n);
    res.centroids.resize(k * d);
    std::vector<double> h_shift_sq(k, 0.0);

    CudaTimer kernel_timer;
    CudaTimer e2e_timer;

    // Warm-up runs (untimed).
    for (int w = 0; w < cfg.warmup_runs; ++w) {
        double dummy_kernel_ms = 0.0;
        double dummy_e2e_ms = 0.0;
        run_single_lloyd_v4(points_soa, init_centroids, cfg, plan, buf, h_shift_sq, res.assignments,
                            res.centroids, res.iterations, false, dummy_kernel_ms, dummy_e2e_ms,
                            reranked, kernel_timer, e2e_timer);
    }

    // Timed benchmark runs.
    double total_kernel_ms = 0.0;
    double total_e2e_ms = 0.0;
    for (int r = 0; r < cfg.bench_runs; ++r) {
        double run_kernel_ms = 0.0;
        double run_e2e_ms = 0.0;
        run_single_lloyd_v4(points_soa, init_centroids, cfg, plan, buf, h_shift_sq, res.assignments,
                            res.centroids, res.iterations, true, run_kernel_ms, run_e2e_ms,
                            reranked, kernel_timer, e2e_timer);
        total_kernel_ms += run_kernel_ms;
        total_e2e_ms += run_e2e_ms;
    }

    std::fprintf(stderr, "[v4] exact re-rank (final assignment pass): %u of %d points (%.4f%%)\n",
                 reranked, cfg.n_points,
                 100.0 * static_cast<double>(reranked) / static_cast<double>(cfg.n_points));

    CHECK_CUDA(cudaFree(buf.rerank_count));
    CHECK_CUDA(cudaFree(buf.shift_sq));
    CHECK_CUDA(cudaFree(buf.assignments));
    CHECK_CUDA(cudaFree(buf.counts));
    CHECK_CUDA(cudaFree(buf.sums));
    CHECK_CUDA(cudaFree(buf.xnorm));
    CHECK_CUDA(cudaFree(buf.cnorm));
    CHECK_CUDA(cudaFree(buf.centroids));
    CHECK_CUDA(cudaFree(buf.points));

    res.kernel_ms = total_kernel_ms / static_cast<double>(cfg.bench_runs);
    res.e2e_ms = total_e2e_ms / static_cast<double>(cfg.bench_runs);
    res.sse = compute_sse(points, res.centroids, res.assignments, cfg.n_points, cfg.dim);
    return res;
}

KUDA_REGISTER_GPU_RUNNER("v4", kuda::run_kmeans_v4);

} // namespace kuda
