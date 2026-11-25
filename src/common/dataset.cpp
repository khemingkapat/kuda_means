#include "kmeans_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <unordered_set>

namespace kuda {

void generate_gaussian_blobs(int n_points, int dim, int k_clusters, uint32_t seed,
                             std::vector<float> &points_out,
                             std::vector<float> &init_centroids_out) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_dist(-10.0f, 10.0f);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> cluster_dist(0, k_clusters - 1);

    const auto n = static_cast<size_t>(n_points);
    const auto d = static_cast<size_t>(dim);
    const auto k = static_cast<size_t>(k_clusters);

    std::vector<float> true_centers(k * d);
    for (size_t i = 0; i < k * d; ++i) {
        true_centers[i] = center_dist(rng);
    }

    points_out.resize(n * d);
    for (size_t i = 0; i < n; ++i) {
        const auto blob = static_cast<size_t>(cluster_dist(rng));
        const float *center = &true_centers[blob * d];
        float *pt = &points_out[i * d];
        for (size_t j = 0; j < d; ++j) {
            pt[j] = center[j] + noise_dist(rng);
        }
    }

    // O(K) deterministic Forgy initialization (Floyd's sampling without replacement)
    init_centroids_out.resize(k * d);
    const size_t sample_count = std::min(k, n);
    std::unordered_set<size_t> chosen_set;
    chosen_set.reserve(sample_count * 2);
    std::vector<size_t> chosen;
    chosen.reserve(sample_count);

    for (size_t i = n - sample_count; i < n; ++i) {
        std::uniform_int_distribution<size_t> pick(0, i);
        const size_t candidate = pick(rng);
        const size_t idx = chosen_set.insert(candidate).second ? candidate : i;
        if (idx == i) {
            chosen_set.insert(i);
        }
        chosen.push_back(idx);
    }

    for (size_t c = 0; c < k; ++c) {
        const size_t src_idx = chosen[c % sample_count];
        const float *src = &points_out[src_idx * d];
        float *dst = &init_centroids_out[c * d];
        std::copy(src, src + d, dst);
    }
}

double compute_sse(const std::vector<float> &points, const std::vector<float> &centroids,
                   const std::vector<int> &assignments, int n_points, int dim) {
    if (n_points < 0 || dim <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto n = static_cast<size_t>(n_points);
    const auto d = static_cast<size_t>(dim);
    const size_t k = centroids.size() / d;

    if (assignments.size() < n || points.size() < n * d || k == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const int a = assignments[i];
        if (a < 0 || static_cast<size_t>(a) >= k) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto c = static_cast<size_t>(a);
        const float *pt = &points[i * d];
        const float *ctr = &centroids[c * d];
        double dist_sq = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double diff = static_cast<double>(pt[j]) - static_cast<double>(ctr[j]);
            dist_sq += diff * diff;
        }
        sse += dist_sq;
    }
    return sse;
}

VerificationReport verify_against_baseline(const KMeansResult &ref, const KMeansResult &cand,
                                           int n_points, int dim, int k_clusters,
                                           double centroid_tol, double sse_rel_tol) {
    VerificationReport report;

    const auto n = static_cast<size_t>(n_points);
    const auto kd = static_cast<size_t>(k_clusters) * static_cast<size_t>(dim);

    if (ref.assignments.size() != n || cand.assignments.size() != n || ref.centroids.size() != kd ||
        cand.centroids.size() != kd) {
        report.passed = false;
        return report;
    }

    int mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (ref.assignments[i] != cand.assignments[i]) {
            ++mismatches;
        }
    }
    report.assignment_mismatches = mismatches;

    double max_diff = 0.0;
    for (size_t i = 0; i < kd; ++i) {
        const double diff = std::fabs(static_cast<double>(ref.centroids[i]) -
                                      static_cast<double>(cand.centroids[i]));
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    report.max_centroid_abs_diff = max_diff;

    const double denom = std::max(std::fabs(ref.sse), 1e-12);
    report.sse_rel_diff = std::fabs(ref.sse - cand.sse) / denom;

    report.passed = std::isfinite(ref.sse) && std::isfinite(cand.sse) &&
                    (report.max_centroid_abs_diff <= centroid_tol) &&
                    (report.sse_rel_diff <= sse_rel_tol);
    return report;
}

} // namespace kuda
