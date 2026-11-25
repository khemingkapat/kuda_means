#include "kmeans_common.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <vector>

namespace kuda {

namespace {

void run_single_lloyd_cpu(const std::vector<float> &points,
                          const std::vector<float> &init_centroids, const KMeansConfig &cfg,
                          std::vector<int> &assignments, std::vector<float> &centroids,
                          int &iterations_out) {
    const auto n = static_cast<size_t>(cfg.n_points);
    const auto d = static_cast<size_t>(cfg.dim);
    const auto k = static_cast<size_t>(cfg.k_clusters);
    const double eps_sq = static_cast<double>(cfg.epsilon) * static_cast<double>(cfg.epsilon);

    centroids = init_centroids;
    assignments.assign(n, 0);

    std::vector<double> sums(k * d, 0.0);
    std::vector<int> counts(k, 0);

    int iter_done = 0;
    for (int iter = 0; iter < cfg.max_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        // Step 1: Assignment (Voronoi partition)
        for (size_t i = 0; i < n; ++i) {
            const float *pt = &points[i * d];
            float best_dist = std::numeric_limits<float>::max();
            int best_k = 0;

            for (size_t c = 0; c < k; ++c) {
                const float *ctr = &centroids[c * d];
                float dist_sq = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    const float diff = pt[j] - ctr[j];
                    dist_sq += diff * diff;
                }
                if (dist_sq < best_dist) {
                    best_dist = dist_sq;
                    best_k = static_cast<int>(c);
                }
            }

            assignments[i] = best_k;
            const auto bk = static_cast<size_t>(best_k);
            ++counts[bk];
            double *sum_row = &sums[bk * d];
            for (size_t j = 0; j < d; ++j) {
                sum_row[j] += static_cast<double>(pt[j]);
            }
        }

        // Step 2: Update centroids & compute max squared shift
        double max_shift_sq = 0.0;
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                // Retain previous centroid to avoid NaN on empty clusters
                continue;
            }
            const double inv_count = 1.0 / static_cast<double>(counts[c]);
            float *ctr = &centroids[c * d];
            const double *sum_row = &sums[c * d];

            double shift_sq = 0.0;
            for (size_t j = 0; j < d; ++j) {
                const float new_val = static_cast<float>(sum_row[j] * inv_count);
                const double diff = static_cast<double>(new_val) - static_cast<double>(ctr[j]);
                shift_sq += diff * diff;
                ctr[j] = new_val;
            }
            if (shift_sq > max_shift_sq) {
                max_shift_sq = shift_sq;
            }
        }

        ++iter_done;
        if (max_shift_sq < eps_sq) {
            break;
        }
    }

    // Final assignment pass so assignments & SSE match the returned centroids exactly
    for (size_t i = 0; i < n; ++i) {
        const float *pt = &points[i * d];
        float best_dist = std::numeric_limits<float>::max();
        int best_k = 0;
        for (size_t c = 0; c < k; ++c) {
            const float *ctr = &centroids[c * d];
            float dist_sq = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                const float diff = pt[j] - ctr[j];
                dist_sq += diff * diff;
            }
            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_k = static_cast<int>(c);
            }
        }
        assignments[i] = best_k;
    }

    iterations_out = iter_done;
}

} // namespace

KMeansResult run_kmeans_cpu(const std::vector<float> &points,
                            const std::vector<float> &init_centroids, const KMeansConfig &cfg) {
    KMeansResult res;
    res.hw.gpu_name = "CPU (" + detect_cpu_name() + ")";
    res.hw.compute_cap = "none";
    res.hw.sm_count = 0;
    res.hw.global_mem_mb = 0.0;
    res.hw.cuda_version = "none";
    res.hw.git_commit = cfg.git_commit.empty() ? std::string(KUDA_GIT_COMMIT) : cfg.git_commit;

    // Warm-up runs
    for (int w = 0; w < cfg.warmup_runs; ++w) {
        run_single_lloyd_cpu(points, init_centroids, cfg, res.assignments, res.centroids,
                             res.iterations);
    }

    // Timed benchmark runs
    double total_ms = 0.0;
    for (int r = 0; r < cfg.bench_runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run_single_lloyd_cpu(points, init_centroids, cfg, res.assignments, res.centroids,
                             res.iterations);
        const auto t1 = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        total_ms += elapsed.count();
    }

    const double avg_ms = total_ms / static_cast<double>(cfg.bench_runs);
    res.kernel_ms = avg_ms;
    res.e2e_ms = avg_ms;
    res.sse = compute_sse(points, res.centroids, res.assignments, cfg.n_points, cfg.dim);
    return res;
}

} // namespace kuda
