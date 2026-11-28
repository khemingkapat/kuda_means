// V3 — Privatized Reduction: per-block shared-memory accumulators + warp shuffles.
//
// Keeps everything from V2 (coalesced SoA D x N points, centroids in __constant__ memory with an
// __ldg fallback) and replaces only the accumulation step, which V1/V2 do with one global
// atomicAdd per point per coordinate:
//   1. Lanes of a warp that chose the same cluster are grouped with __match_any_sync and their
//      coordinates are summed with a segmented __shfl_sync tree; only the group leader issues a
//      shared-memory atomic.
//   2. Each block accumulates into private double s_sums[K * D] / int s_counts[K] in shared memory.
//   3. After __syncthreads(), each block flushes with at most one global atomicAdd per coordinate.
//   4. The accumulation grid is persistent (occupancy x SM count, grid-stride), so the flush is
//      amortized over many points per block.
// When K * D * 8 + K * 4 bytes exceed the opt-in shared-memory limit per block, V3 falls back to
// V2-style global atomics (logged on stderr). Requires sm_70+ for __match_any_sync.

#include "cuda_utils.cuh"
#include "kmeans_common.h"
#include "kmeans_gpu.cuh"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error "V3 requires sm_70+ (__match_any_sync)."
#endif

namespace kuda {

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kMaxConstantCentroids = 16384; // 64 KB limit: 16384 floats
constexpr size_t kDefaultSharedLimit = 48 * 1024;

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a whole number of warps");

__constant__ float c_centroids_v3[kMaxConstantCentroids];

template <bool UseConstant>
__device__ __forceinline__ int nearest_centroid(const float *__restrict__ points,
                                                const float *__restrict__ d_centroids, int idx,
                                                int n, int d, int k) {
    float best_dist = FLT_MAX;
    int best_k = 0;

    for (int c = 0; c < k; ++c) {
        const auto c_offset = static_cast<size_t>(c) * static_cast<size_t>(d);
        float dist_sq = 0.0f;

        for (int j = 0; j < d; ++j) {
            // Structure-of-Arrays (SoA) layout: D x N, indexed by j * n + idx
            const float pt_val = points[static_cast<size_t>(j) * static_cast<size_t>(n) + idx];
            float ctr_val;
            if constexpr (UseConstant) {
                ctr_val = c_centroids_v3[c_offset + j];
            } else {
                ctr_val = __ldg(&d_centroids[c_offset + j]);
            }
            const float diff = pt_val - ctr_val;
            dist_sq += diff * diff;
        }

        if (dist_sq < best_dist) {
            best_dist = dist_sq;
            best_k = c;
        }
    }
    return best_k;
}

// Segmented warp reduction: sums x over the lanes in `peers` (lanes that chose the same cluster).
// The full sum ends up in the lowest lane of `peers`. All 32 lanes must call this together.
// Each round, every lane adds the value of its next-higher remaining peer, then the peers whose
// rank is odd drop out (tree reduction over ranks, at most 5 rounds).
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

template <bool UseConstant>
__global__ void accumulate_privatized_v3_kernel(const float *__restrict__ points,
                                                const float *__restrict__ d_centroids, double *sums,
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

    // `base` is block-uniform, so every warp runs the same number of iterations and all 32 lanes
    // reach the *_sync intrinsics together. Out-of-range lanes stay active with best_k = -1.
    for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
        const int idx = base + static_cast<int>(threadIdx.x);
        const bool valid = idx < n;
        const int best_k =
            valid ? nearest_centroid<UseConstant>(points, d_centroids, idx, n, d, k) : -1;

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

// Capacity-wall fallback: V2-style accumulation with one global atomic per point per coordinate.
template <bool UseConstant>
__global__ void accumulate_global_v3_kernel(const float *__restrict__ points,
                                            const float *__restrict__ d_centroids, double *sums,
                                            int *counts, int n, int d, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const int best_k = nearest_centroid<UseConstant>(points, d_centroids, idx, n, d, k);
    atomicAdd(&counts[best_k], 1);
    double *sum_row = sums + static_cast<size_t>(best_k) * static_cast<size_t>(d);
    for (int j = 0; j < d; ++j) {
        const float pt_val = points[static_cast<size_t>(j) * static_cast<size_t>(n) + idx];
        atomicAdd(&sum_row[j], static_cast<double>(pt_val));
    }
}

template <bool UseConstant>
__global__ void assign_v3_kernel(const float *__restrict__ points,
                                 const float *__restrict__ d_centroids, int *assignments, int n,
                                 int d, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    assignments[idx] = nearest_centroid<UseConstant>(points, d_centroids, idx, n, d, k);
}

__global__ void update_centroids_v3_kernel(float *centroids, const double *sums, const int *counts,
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

struct V3DeviceBuffers {
    float *points = nullptr;
    float *centroids = nullptr;
    double *sums = nullptr;
    int *counts = nullptr;
    int *assignments = nullptr;
    double *shift_sq = nullptr;
};

struct V3LaunchPlan {
    bool use_constant = false;
    bool use_shared = false;
    size_t smem_bytes = 0;
    int grid_n = 0;   // one thread per point (fallback + final assignment)
    int grid_acc = 0; // persistent grid for the privatized accumulation kernel
};

template <bool UseConstant> int persistent_grid(size_t smem_bytes, int grid_n, int sm_count) {
    if (smem_bytes > kDefaultSharedLimit) {
        CHECK_CUDA(cudaFuncSetAttribute(accumulate_privatized_v3_kernel<UseConstant>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes)));
    }
    int blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, accumulate_privatized_v3_kernel<UseConstant>, kBlockSize, smem_bytes));
    return std::min(grid_n, std::max(1, blocks_per_sm) * sm_count);
}

V3LaunchPlan make_launch_plan(const KMeansConfig &cfg) {
    const auto k = static_cast<size_t>(cfg.k_clusters);
    const auto d = static_cast<size_t>(cfg.dim);

    int device = 0;
    int smem_optin = 0;
    int sm_count = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    V3LaunchPlan plan;
    plan.use_constant = k * d <= static_cast<size_t>(kMaxConstantCentroids);
    plan.smem_bytes = k * d * sizeof(double) + k * sizeof(int);
    plan.use_shared = plan.smem_bytes <= static_cast<size_t>(smem_optin);
    plan.grid_n = (cfg.n_points + kBlockSize - 1) / kBlockSize;
    plan.grid_acc = plan.grid_n;

    if (plan.use_shared) {
        plan.grid_acc = plan.use_constant
                            ? persistent_grid<true>(plan.smem_bytes, plan.grid_n, sm_count)
                            : persistent_grid<false>(plan.smem_bytes, plan.grid_n, sm_count);
        std::fprintf(stderr,
                     "[v3] reduction path: shared (smem=%zu B/block, limit=%d B, grid=%d blocks, "
                     "centroids=%s)\n",
                     plan.smem_bytes, smem_optin, plan.grid_acc,
                     plan.use_constant ? "__constant__" : "__ldg");
    } else {
        std::fprintf(stderr,
                     "[v3] reduction path: global atomic fallback (need %zu B/block > limit %d B, "
                     "centroids=%s)\n",
                     plan.smem_bytes, smem_optin, plan.use_constant ? "__constant__" : "__ldg");
    }
    return plan;
}

void launch_accumulate(const V3LaunchPlan &plan, const V3DeviceBuffers &buf,
                       const KMeansConfig &cfg) {
    if (plan.use_shared) {
        if (plan.use_constant) {
            accumulate_privatized_v3_kernel<true><<<plan.grid_acc, kBlockSize, plan.smem_bytes>>>(
                buf.points, buf.centroids, buf.sums, buf.counts, cfg.n_points, cfg.dim,
                cfg.k_clusters);
        } else {
            accumulate_privatized_v3_kernel<false><<<plan.grid_acc, kBlockSize, plan.smem_bytes>>>(
                buf.points, buf.centroids, buf.sums, buf.counts, cfg.n_points, cfg.dim,
                cfg.k_clusters);
        }
    } else {
        if (plan.use_constant) {
            accumulate_global_v3_kernel<true>
                <<<plan.grid_n, kBlockSize>>>(buf.points, buf.centroids, buf.sums, buf.counts,
                                              cfg.n_points, cfg.dim, cfg.k_clusters);
        } else {
            accumulate_global_v3_kernel<false>
                <<<plan.grid_n, kBlockSize>>>(buf.points, buf.centroids, buf.sums, buf.counts,
                                              cfg.n_points, cfg.dim, cfg.k_clusters);
        }
    }
    CHECK_LAST_CUDA_ERROR();
}

void launch_assign(const V3LaunchPlan &plan, const V3DeviceBuffers &buf, const KMeansConfig &cfg) {
    if (plan.use_constant) {
        assign_v3_kernel<true><<<plan.grid_n, kBlockSize>>>(
            buf.points, buf.centroids, buf.assignments, cfg.n_points, cfg.dim, cfg.k_clusters);
    } else {
        assign_v3_kernel<false><<<plan.grid_n, kBlockSize>>>(
            buf.points, buf.centroids, buf.assignments, cfg.n_points, cfg.dim, cfg.k_clusters);
    }
    CHECK_LAST_CUDA_ERROR();
}

void run_single_lloyd_v3(const std::vector<float> &points_soa,
                         const std::vector<float> &init_centroids, const KMeansConfig &cfg,
                         const V3LaunchPlan &plan, const V3DeviceBuffers &buf,
                         std::vector<double> &h_shift_sq, std::vector<int> &assignments,
                         std::vector<float> &centroids, int &iterations_out, bool record_timers,
                         double &kernel_ms_out, double &e2e_ms_out, CudaTimer &kernel_timer,
                         CudaTimer &e2e_timer) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);

    const size_t points_bytes = n * d * sizeof(float);
    const size_t centroids_bytes = k * d * sizeof(float);
    const size_t sums_bytes = k * d * sizeof(double);
    const size_t counts_bytes = k * sizeof(int);
    const size_t assignments_bytes = n * sizeof(int);
    const size_t shift_bytes = k * sizeof(double);

    const int grid_k = (cfg.k_clusters + kBlockSize - 1) / kBlockSize;
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

    if (plan.use_constant) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_centroids_v3, init_centroids.data(), centroids_bytes, 0,
                                      cudaMemcpyHostToDevice));
    }

    for (int iter = 0; iter < cfg.max_iters; ++iter) {
        if (record_timers) {
            kernel_timer.start();
        }

        CHECK_CUDA(cudaMemsetAsync(buf.sums, 0, sums_bytes, 0));
        CHECK_CUDA(cudaMemsetAsync(buf.counts, 0, counts_bytes, 0));

        launch_accumulate(plan, buf, cfg);

        update_centroids_v3_kernel<<<grid_k, kBlockSize>>>(buf.centroids, buf.sums, buf.counts,
                                                           buf.shift_sq, cfg.dim, cfg.k_clusters);
        CHECK_LAST_CUDA_ERROR();

        if (plan.use_constant) {
            CHECK_CUDA(cudaMemcpyToSymbol(c_centroids_v3, buf.centroids, centroids_bytes, 0,
                                          cudaMemcpyDeviceToDevice));
        }

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

    // Final assignment pass so assignments & SSE match the returned centroids exactly.
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

    iterations_out = iter_done;
    kernel_ms_out = run_kernel_ms;
    e2e_ms_out = run_e2e_ms;
}

} // namespace

KMeansResult run_kmeans_v3(const std::vector<float> &points,
                           const std::vector<float> &init_centroids, const KMeansConfig &cfg) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);

    // Transpose row-major AoS points into Structure-of-Arrays (SoA) layout (D x N), as in V2.
    std::vector<float> points_soa(n * d);
    for (size_t idx = 0; idx < n; ++idx) {
        for (size_t j = 0; j < d; ++j) {
            points_soa[j * n + idx] = points[idx * d + j];
        }
    }

    const V3LaunchPlan plan = make_launch_plan(cfg);

    V3DeviceBuffers buf;
    CHECK_CUDA(cudaMalloc(&buf.points, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.centroids, k * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf.sums, k * d * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&buf.counts, k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buf.assignments, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buf.shift_sq, k * sizeof(double)));

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
        run_single_lloyd_v3(points_soa, init_centroids, cfg, plan, buf, h_shift_sq, res.assignments,
                            res.centroids, res.iterations, false, dummy_kernel_ms, dummy_e2e_ms,
                            kernel_timer, e2e_timer);
    }

    // Timed benchmark runs.
    double total_kernel_ms = 0.0;
    double total_e2e_ms = 0.0;
    for (int r = 0; r < cfg.bench_runs; ++r) {
        double run_kernel_ms = 0.0;
        double run_e2e_ms = 0.0;
        run_single_lloyd_v3(points_soa, init_centroids, cfg, plan, buf, h_shift_sq, res.assignments,
                            res.centroids, res.iterations, true, run_kernel_ms, run_e2e_ms,
                            kernel_timer, e2e_timer);
        total_kernel_ms += run_kernel_ms;
        total_e2e_ms += run_e2e_ms;
    }

    CHECK_CUDA(cudaFree(buf.shift_sq));
    CHECK_CUDA(cudaFree(buf.assignments));
    CHECK_CUDA(cudaFree(buf.counts));
    CHECK_CUDA(cudaFree(buf.sums));
    CHECK_CUDA(cudaFree(buf.centroids));
    CHECK_CUDA(cudaFree(buf.points));

    res.kernel_ms = total_kernel_ms / static_cast<double>(cfg.bench_runs);
    res.e2e_ms = total_e2e_ms / static_cast<double>(cfg.bench_runs);
    res.sse = compute_sse(points, res.centroids, res.assignments, cfg.n_points, cfg.dim);
    return res;
}

KUDA_REGISTER_GPU_RUNNER("v3", kuda::run_kmeans_v3);

} // namespace kuda
