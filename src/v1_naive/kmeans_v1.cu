#include "cuda_utils.cuh"
#include "kmeans_common.h"
#include "kmeans_gpu.cuh"

#include <cfloat>
#include <cstddef>
#include <vector>

namespace kuda {

namespace {

constexpr int kBlockSize = 256;

template <bool Accumulate>
__global__ void assign_v1_kernel(const float *points, const float *centroids, double *sums,
                                 int *counts, int *assignments, int n, int d, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const auto d_sz = static_cast<size_t>(d);
    const float *pt = points + static_cast<size_t>(idx) * d_sz;

    float best_dist = FLT_MAX;
    int best_k = 0;

    for (int c = 0; c < k; ++c) {
        const float *ctr = centroids + static_cast<size_t>(c) * d_sz;
        float dist_sq = 0.0f;
        for (int j = 0; j < d; ++j) {
            const float diff = pt[j] - ctr[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < best_dist) {
            best_dist = dist_sq;
            best_k = c;
        }
    }

    if constexpr (Accumulate) {
        atomicAdd(&counts[best_k], 1);
        double *sum_row = sums + static_cast<size_t>(best_k) * d_sz;
        for (int j = 0; j < d; ++j) {
            atomicAdd(&sum_row[j], static_cast<double>(pt[j]));
        }
    } else {
        assignments[idx] = best_k;
    }
}

__global__ void update_centroids_v1_kernel(float *centroids, const double *sums, const int *counts,
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

void run_single_lloyd_v1(const std::vector<float> &points, const std::vector<float> &init_centroids,
                         const KMeansConfig &cfg, float *d_points, float *d_centroids,
                         double *d_sums, int *d_counts, int *d_assignments, double *d_shift_sq,
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

    const int grid_n = (cfg.n_points + kBlockSize - 1) / kBlockSize;
    const int grid_k = (cfg.k_clusters + kBlockSize - 1) / kBlockSize;
    const double eps_sq = static_cast<double>(cfg.epsilon) * static_cast<double>(cfg.epsilon);

    double run_kernel_ms = 0.0;
    double run_e2e_ms = 0.0;
    int iter_done = 0;

    if (record_timers) {
        e2e_timer.start();
    }

    CHECK_CUDA(cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_centroids, init_centroids.data(), centroids_bytes, cudaMemcpyHostToDevice));

    for (int iter = 0; iter < cfg.max_iters; ++iter) {
        if (record_timers) {
            kernel_timer.start();
        }

        CHECK_CUDA(cudaMemsetAsync(d_sums, 0, sums_bytes, 0));
        CHECK_CUDA(cudaMemsetAsync(d_counts, 0, counts_bytes, 0));

        assign_v1_kernel<true><<<grid_n, kBlockSize>>>(d_points, d_centroids, d_sums, d_counts,
                                                       d_assignments, cfg.n_points, cfg.dim,
                                                       cfg.k_clusters);
        CHECK_LAST_CUDA_ERROR();

        update_centroids_v1_kernel<<<grid_k, kBlockSize>>>(d_centroids, d_sums, d_counts,
                                                           d_shift_sq, cfg.dim, cfg.k_clusters);
        CHECK_LAST_CUDA_ERROR();

        if (record_timers) {
            run_kernel_ms += static_cast<double>(kernel_timer.stop_ms());
        }

        CHECK_CUDA(cudaMemcpy(h_shift_sq.data(), d_shift_sq, shift_bytes, cudaMemcpyDeviceToHost));

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
    assign_v1_kernel<false><<<grid_n, kBlockSize>>>(d_points, d_centroids, d_sums, d_counts,
                                                    d_assignments, cfg.n_points, cfg.dim,
                                                    cfg.k_clusters);
    CHECK_LAST_CUDA_ERROR();
    if (record_timers) {
        run_kernel_ms += static_cast<double>(kernel_timer.stop_ms());
    }

    CHECK_CUDA(cudaMemcpy(centroids.data(), d_centroids, centroids_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(assignments.data(), d_assignments, assignments_bytes, cudaMemcpyDeviceToHost));

    if (record_timers) {
        run_e2e_ms = static_cast<double>(e2e_timer.stop_ms());
    }

    iterations_out = iter_done;
    kernel_ms_out = run_kernel_ms;
    e2e_ms_out = run_e2e_ms;
}

} // namespace

KMeansResult run_kmeans_v1(const std::vector<float> &points,
                           const std::vector<float> &init_centroids, const KMeansConfig &cfg) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);

    float *d_points = nullptr;
    float *d_centroids = nullptr;
    double *d_sums = nullptr;
    int *d_counts = nullptr;
    int *d_assignments = nullptr;
    double *d_shift_sq = nullptr;

    CHECK_CUDA(cudaMalloc(&d_points, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_centroids, k * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sums, k * d * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_counts, k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_shift_sq, k * sizeof(double)));

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
        run_single_lloyd_v1(points, init_centroids, cfg, d_points, d_centroids, d_sums, d_counts,
                            d_assignments, d_shift_sq, h_shift_sq, res.assignments, res.centroids,
                            res.iterations, false, dummy_kernel_ms, dummy_e2e_ms, kernel_timer,
                            e2e_timer);
    }

    // Timed benchmark runs.
    double total_kernel_ms = 0.0;
    double total_e2e_ms = 0.0;
    for (int r = 0; r < cfg.bench_runs; ++r) {
        double run_kernel_ms = 0.0;
        double run_e2e_ms = 0.0;
        run_single_lloyd_v1(points, init_centroids, cfg, d_points, d_centroids, d_sums, d_counts,
                            d_assignments, d_shift_sq, h_shift_sq, res.assignments, res.centroids,
                            res.iterations, true, run_kernel_ms, run_e2e_ms, kernel_timer,
                            e2e_timer);
        total_kernel_ms += run_kernel_ms;
        total_e2e_ms += run_e2e_ms;
    }

    CHECK_CUDA(cudaFree(d_shift_sq));
    CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_sums));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_points));

    res.kernel_ms = total_kernel_ms / static_cast<double>(cfg.bench_runs);
    res.e2e_ms = total_e2e_ms / static_cast<double>(cfg.bench_runs);
    res.sse = compute_sse(points, res.centroids, res.assignments, cfg.n_points, cfg.dim);
    return res;
}

KUDA_REGISTER_GPU_RUNNER("v1", kuda::run_kmeans_v1);

} // namespace kuda
