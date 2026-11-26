#include "kmeans_common.h"
#ifdef KUDA_GPU_BUILD
#include "kmeans_gpu.cuh"
#endif

#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    kuda::KMeansConfig cfg;
    const kuda::CliParseStatus status = kuda::parse_cli_args(argc, argv, cfg);
    if (status == kuda::CliParseStatus::Help) {
        return 0;
    }
    if (status == kuda::CliParseStatus::Error) {
        return 1;
    }

    const bool is_reference = (cfg.version == "cpu" || cfg.version == "v0");
    if (!is_reference) {
#ifdef KUDA_GPU_BUILD
        if (!kuda::has_gpu_runner(cfg.version)) {
            const auto available = kuda::list_gpu_runners();
            std::cerr << "Error: GPU version '" << cfg.version
                      << "' is not linked in this build. Available GPU versions: ";
            if (available.empty()) {
                std::cerr << "(none linked yet)";
            } else {
                for (std::size_t i = 0; i < available.size(); ++i) {
                    if (i > 0) {
                        std::cerr << ", ";
                    }
                    std::cerr << available[i];
                }
            }
            std::cerr << "\n";
            return 1;
        }
#else
        std::cerr << "Error: version '" << cfg.version
                  << "' is not available in this build (CPU-only binary supports 'cpu'/'v0').\n";
        return 1;
#endif
    }

    std::vector<float> points;
    std::vector<float> init_centroids;
    kuda::generate_gaussian_blobs(cfg.n_points, cfg.dim, cfg.k_clusters, cfg.seed, points,
                                  init_centroids);

    kuda::KMeansResult res;
#ifdef KUDA_GPU_BUILD
    kuda::VerificationReport verify_report;
    bool has_verify = false;
    if (is_reference) {
        res = kuda::run_kmeans_cpu(points, init_centroids, cfg);
    } else {
        res = kuda::run_kmeans_gpu(cfg.version, points, init_centroids, cfg);
        if (cfg.verify) {
            kuda::KMeansConfig ref_cfg = cfg;
            ref_cfg.warmup_runs = 0;
            ref_cfg.bench_runs = 1;
            const kuda::KMeansResult ref = kuda::run_kmeans_cpu(points, init_centroids, ref_cfg);
            verify_report =
                kuda::verify_against_baseline(ref, res, cfg.n_points, cfg.dim, cfg.k_clusters);
            has_verify = true;
        }
    }
#else
    res = kuda::run_kmeans_cpu(points, init_centroids, cfg);
#endif

    const double ms_per_iter =
        (res.iterations > 0) ? (res.kernel_ms / static_cast<double>(res.iterations)) : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[kuda_means] version=" << cfg.version << " hw=\"" << res.hw.gpu_name << "\""
              << " commit=" << res.hw.git_commit << " N=" << cfg.n_points << " D=" << cfg.dim
              << " K=" << cfg.k_clusters << " iters=" << res.iterations << " sse=" << res.sse
              << " kernel_ms=" << res.kernel_ms << " e2e_ms=" << res.e2e_ms
              << " ms_per_iter=" << ms_per_iter << "\n";

#ifdef KUDA_GPU_BUILD
    const kuda::VerificationReport *verify_ptr = has_verify ? &verify_report : nullptr;
    if (has_verify) {
        std::cout << "[verify] status=" << (verify_report.passed ? "PASS" : "FAIL")
                  << " assignment_mismatches=" << verify_report.assignment_mismatches
                  << std::scientific << std::setprecision(6)
                  << " max_centroid_abs_diff=" << verify_report.max_centroid_abs_diff
                  << " sse_rel_diff=" << verify_report.sse_rel_diff << "\n";
    }
    if (!cfg.json_out.empty()) {
        if (!kuda::write_result_json(cfg.json_out, cfg, res, verify_ptr, is_reference)) {
            return 1;
        }
    }
    if (!cfg.csv_out.empty()) {
        if (!kuda::append_result_csv(cfg.csv_out, cfg, res, verify_ptr, is_reference)) {
            return 1;
        }
    }
    if (has_verify && !verify_report.passed) {
        return 1;
    }
#else
    if (!cfg.json_out.empty()) {
        if (!kuda::write_result_json(cfg.json_out, cfg, res, nullptr, is_reference)) {
            return 1;
        }
    }
    if (!cfg.csv_out.empty()) {
        if (!kuda::append_result_csv(cfg.csv_out, cfg, res, nullptr, is_reference)) {
            return 1;
        }
    }
#endif

    return 0;
}
