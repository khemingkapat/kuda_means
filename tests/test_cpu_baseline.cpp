#include "kmeans_common.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check(bool cond, const char *msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        return false;
    }
    std::cout << "[PASS] " << msg << "\n";
    return true;
}

} // namespace

int main() {
    bool all_ok = true;

    // Test 1: Deterministic dataset & identical repeated runs
    kuda::KMeansConfig cfg;
    cfg.n_points = 4000;
    cfg.dim = 8;
    cfg.k_clusters = 6;
    cfg.max_iters = 50;
    cfg.epsilon = 1e-4f;
    cfg.seed = 42;
    cfg.warmup_runs = 0;
    cfg.bench_runs = 1;

    std::vector<float> points;
    std::vector<float> init_centroids;
    kuda::generate_gaussian_blobs(cfg.n_points, cfg.dim, cfg.k_clusters, cfg.seed, points,
                                  init_centroids);

    const kuda::KMeansResult r1 = kuda::run_kmeans_cpu(points, init_centroids, cfg);
    const kuda::KMeansResult r2 = kuda::run_kmeans_cpu(points, init_centroids, cfg);

    const kuda::VerificationReport rep =
        kuda::verify_against_baseline(r1, r2, cfg.n_points, cfg.dim, cfg.k_clusters, 0.0, 0.0);
    all_ok &= check(rep.passed && rep.assignment_mismatches == 0,
                    "Determinism: repeated CPU runs match bit-for-bit");
    all_ok &= check(r1.iterations > 0 && r1.iterations <= cfg.max_iters,
                    "Convergence: iterations within [1, max_iters]");

    // Test 2: Monotonic SSE decrease across iterations (Lloyd's guarantee)
    double prev_sse = 1e300;
    bool monotonic = true;
    for (int it = 1; it <= 5; ++it) {
        kuda::KMeansConfig step_cfg = cfg;
        step_cfg.max_iters = it;
        step_cfg.epsilon = 0.0f;
        const kuda::KMeansResult step_res = kuda::run_kmeans_cpu(points, init_centroids, step_cfg);
        if (step_res.sse > prev_sse + 1e-6) {
            monotonic = false;
            std::cerr << "  SSE increased at iter " << it << ": " << prev_sse << " -> "
                      << step_res.sse << "\n";
        }
        prev_sse = step_res.sse;
    }
    all_ok &= check(monotonic, "Lloyd's monotonicity: SSE is non-increasing over iterations 1..5");

    // Test 3: Exact recovery on trivially separated 2D clusters
    const std::vector<float> simple_pts = {
        0.0f,  0.0f,  0.2f,  0.0f,  0.0f,  0.2f,  0.2f,  0.2f,  // Cluster around (0.1, 0.1)
        10.0f, 10.0f, 10.2f, 10.0f, 10.0f, 10.2f, 10.2f, 10.2f, // Cluster around (10.1, 10.1)
    };
    const std::vector<float> simple_init = {0.0f, 0.0f, 10.0f, 10.0f};
    kuda::KMeansConfig simple_cfg;
    simple_cfg.n_points = 8;
    simple_cfg.dim = 2;
    simple_cfg.k_clusters = 2;
    simple_cfg.max_iters = 20;
    simple_cfg.epsilon = 1e-6f;
    simple_cfg.warmup_runs = 0;
    simple_cfg.bench_runs = 1;

    const kuda::KMeansResult simple_res = kuda::run_kmeans_cpu(simple_pts, simple_init, simple_cfg);
    const bool exact_centers = std::fabs(simple_res.centroids[0] - 0.1f) < 1e-5f &&
                               std::fabs(simple_res.centroids[1] - 0.1f) < 1e-5f &&
                               std::fabs(simple_res.centroids[2] - 10.1f) < 1e-5f &&
                               std::fabs(simple_res.centroids[3] - 10.1f) < 1e-5f;
    all_ok &= check(exact_centers, "Exact recovery: centroids match (0.1, 0.1) and (10.1, 10.1)");

    // Test 4: Empty cluster safety (no NaN when a centroid starts far away)
    const std::vector<float> far_init = {0.0f, 0.0f, 1000.0f, 1000.0f};
    const kuda::KMeansResult empty_res = kuda::run_kmeans_cpu(simple_pts, far_init, simple_cfg);
    bool no_nan = true;
    for (float v : empty_res.centroids) {
        if (std::isnan(v) || std::isinf(v)) {
            no_nan = false;
        }
    }
    all_ok &= check(no_nan, "Empty cluster safety: no NaN/Inf in centroids");

    // Test 5: Out-of-range assignment bounds safety in compute_sse
    const std::vector<int> bad_assign = {0, 99, 0, 0, 1, 1, 1, 1};
    const double bad_sse = kuda::compute_sse(simple_pts, simple_res.centroids, bad_assign, 8, 2);
    all_ok &= check(std::isnan(bad_sse), "Bounds safety: compute_sse returns NaN on bad index");

    // Test 6: CLI strict validation & --help handling
    {
        char p[] = "kuda_means";
        char h[] = "--help";
        char *argv_help[] = {p, h};
        kuda::KMeansConfig c;
        all_ok &= check(kuda::parse_cli_args(2, argv_help, c) == kuda::CliParseStatus::Help,
                        "CLI: --help returns Help status");

        char eps_flag[] = "--eps";
        char neg[] = "-0.5";
        char *argv_neg[] = {p, eps_flag, neg};
        all_ok &= check(kuda::parse_cli_args(3, argv_neg, c) == kuda::CliParseStatus::Error,
                        "CLI: negative --eps rejected");

        char n_flag[] = "-n";
        char junk[] = "100abc";
        char *argv_junk[] = {p, n_flag, junk};
        all_ok &= check(kuda::parse_cli_args(3, argv_junk, c) == kuda::CliParseStatus::Error,
                        "CLI: trailing garbage in -n rejected");

        kuda::KMeansConfig c_verify;
        all_ok &= check(c_verify.verify, "CLI: verify defaults to true");

        char no_verify[] = "--no-verify";
        char *argv_nv[] = {p, no_verify};
        all_ok &= check(kuda::parse_cli_args(2, argv_nv, c_verify) == kuda::CliParseStatus::Ok &&
                            !c_verify.verify,
                        "CLI: --no-verify disables verification");
    }

    // Test 7: JSON & CSV provenance output (git_commit, verified default, N column)
    {
        const std::string json_path = "/tmp/kuda_test_v0.json";
        const std::string csv_path = "/tmp/kuda_test_v0.csv";
        std::remove(json_path.c_str());
        std::remove(csv_path.c_str());

        const bool j_ok = kuda::write_result_json(json_path, simple_cfg, simple_res);
        const bool c_ok = kuda::append_result_csv(csv_path, simple_cfg, simple_res);

        std::ifstream jf(json_path);
        const std::string jcontent((std::istreambuf_iterator<char>(jf)),
                                   std::istreambuf_iterator<char>());
        std::ifstream cf(csv_path);
        std::string csv_header;
        std::getline(cf, csv_header);

        all_ok &= check(j_ok && jcontent.find("\"git_commit\"") != std::string::npos &&
                            jcontent.find("\"verified\": false") != std::string::npos,
                        "JSON I/O: includes git_commit and safe verified=false default");
        all_ok &= check(c_ok && csv_header.find(",git_commit,N,D,K,") != std::string::npos,
                        "CSV I/O: header includes git_commit and N column for plot_results.py");

        std::remove(json_path.c_str());
        std::remove(csv_path.c_str());
    }

    return all_ok ? 0 : 1;
}
