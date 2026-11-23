#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kuda {

struct KMeansConfig {
    int n_points = 100000;
    int dim = 16;
    int k_clusters = 8;
    int max_iters = 50;
    float epsilon = 1e-4f;
    uint32_t seed = 42;
    int warmup_runs = 1;
    int bench_runs = 5;
    std::string version = "cpu";
    std::string csv_out = "";
};

struct HardwareProvenance {
    std::string gpu_name = "CPU";
    std::string compute_cap = "none";
    int sm_count = 0;
    double global_mem_mb = 0.0;
    std::string cuda_version = "none";
};

struct KMeansResult {
    std::vector<int> assignments; // Length N
    std::vector<float> centroids; // Length K * D (row-major)
    int iterations = 0;
    double sse = 0.0;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
    HardwareProvenance hw;
};

struct VerificationReport {
    bool passed = false;
    int assignment_mismatches = 0;
    double max_centroid_abs_diff = 0.0;
    double sse_rel_diff = 0.0;
};

// Deterministic Gaussian-blob dataset generator (flat row-major N x D)
void generate_gaussian_blobs(int n_points, int dim, int k_clusters, uint32_t seed,
                             std::vector<float> &points_out,
                             std::vector<float> &init_centroids_out);

// Compute within-cluster Sum of Squared Errors (SSE)
double compute_sse(const std::vector<float> &points, const std::vector<float> &centroids,
                   const std::vector<int> &assignments, int n_points, int dim);

// Compare a candidate GPU result against the CPU golden reference
VerificationReport verify_against_baseline(const KMeansResult &ref, const KMeansResult &cand,
                                           int n_points, int dim, int k_clusters,
                                           double centroid_tol = 1e-3, double sse_rel_tol = 1e-3);

} // namespace kuda
