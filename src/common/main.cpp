#include "kmeans_common.h"

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

    std::vector<float> points;
    std::vector<float> init_centroids;
    kuda::generate_gaussian_blobs(cfg.n_points, cfg.dim, cfg.k_clusters, cfg.seed, points,
                                  init_centroids);

    kuda::KMeansResult res;
    bool is_reference = false;
    if (cfg.version == "cpu" || cfg.version == "v0") {
        res = kuda::run_kmeans_cpu(points, init_centroids, cfg);
        is_reference = true;
    } else {
        std::cerr << "Error: version '" << cfg.version
                  << "' is not available in this build (CPU-only binary supports 'cpu'/'v0').\n";
        return 1;
    }

    const double ms_per_iter =
        (res.iterations > 0) ? (res.kernel_ms / static_cast<double>(res.iterations)) : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[kuda_means] version=" << cfg.version << " hw=\"" << res.hw.gpu_name << "\""
              << " commit=" << res.hw.git_commit << " N=" << cfg.n_points << " D=" << cfg.dim
              << " K=" << cfg.k_clusters << " iters=" << res.iterations << " sse=" << res.sse
              << " kernel_ms=" << res.kernel_ms << " e2e_ms=" << res.e2e_ms
              << " ms_per_iter=" << ms_per_iter << "\n";

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

    return 0;
}
