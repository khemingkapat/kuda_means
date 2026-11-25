#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifndef KUDA_GIT_COMMIT
#define KUDA_GIT_COMMIT "unknown"
#endif

namespace kuda {

enum class CliParseStatus {
    Ok,
    Help,
    Error,
};

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
    std::string json_out = "";
    std::string git_commit = "";
};

struct HardwareProvenance {
    std::string gpu_name = "CPU";
    std::string compute_cap = "none";
    int sm_count = 0;
    double global_mem_mb = 0.0;
    std::string cuda_version = "none";
    std::string git_commit = KUDA_GIT_COMMIT;
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

// Deterministic Gaussian-blob dataset generator (flat row-major N x D).
// Uses O(K) sparse Forgy sampling for initial centroids.
void generate_gaussian_blobs(int n_points, int dim, int k_clusters, uint32_t seed,
                             std::vector<float> &points_out,
                             std::vector<float> &init_centroids_out);

// Compute within-cluster Sum of Squared Errors (SSE).
// Returns NaN if any assignment index is out of range [0, K).
double compute_sse(const std::vector<float> &points, const std::vector<float> &centroids,
                   const std::vector<int> &assignments, int n_points, int dim);

// Compare a candidate result against the CPU golden reference.
VerificationReport verify_against_baseline(const KMeansResult &ref, const KMeansResult &cand,
                                           int n_points, int dim, int k_clusters,
                                           double centroid_tol = 1e-3, double sse_rel_tol = 1e-3);

// V0: Single-threaded CPU Lloyd's K-Means baseline (golden reference).
// Contract replicated by V1-V5:
//   - Stopping rule: max_k ||c_k^(t) - c_k^(t-1)||^2 < epsilon^2 or max_iters reached.
//   - Empty clusters: retain previous centroid (no NaNs).
//   - Final assignment pass: runs once after the loop (inside timing scope) so returned
//     assignments and SSE correspond exactly to the returned centroids.
KMeansResult run_kmeans_cpu(const std::vector<float> &points,
                            const std::vector<float> &init_centroids, const KMeansConfig &cfg);

// CLI parsing and result I/O helpers
CliParseStatus parse_cli_args(int argc, char **argv, KMeansConfig &cfg);
void print_usage(const char *prog_name);
std::string detect_cpu_name();
bool write_result_json(const std::string &path, const KMeansConfig &cfg, const KMeansResult &res,
                       const VerificationReport *verify = nullptr, bool default_verified = false);
bool append_result_csv(const std::string &path, const KMeansConfig &cfg, const KMeansResult &res,
                       const VerificationReport *verify = nullptr, bool default_verified = false);

} // namespace kuda
