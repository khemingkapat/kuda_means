#include "cuda_utils.cuh"
#include "kmeans_gpu.cuh"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace kuda {

namespace {

std::map<std::string, GpuKMeansFn> &runner_registry() {
    static std::map<std::string, GpuKMeansFn> registry;
    return registry;
}

} // namespace

bool register_gpu_runner(const std::string &version, GpuKMeansFn fn) {
    if (fn == nullptr) {
        std::fprintf(stderr, "Fatal: attempted to register null GPU runner for version '%s'.\n",
                     version.c_str());
        std::abort();
    }
    if (version.empty() || version == "cpu" || version == "v0") {
        std::fprintf(stderr, "Fatal: invalid or reserved GPU runner version name '%s'.\n",
                     version.c_str());
        std::abort();
    }
    auto &registry = runner_registry();
    if (!registry.emplace(version, fn).second) {
        std::fprintf(stderr, "Fatal: duplicate GPU runner registration for version '%s'.\n",
                     version.c_str());
        std::abort();
    }
    return true;
}

bool has_gpu_runner(const std::string &version) {
    const auto &registry = runner_registry();
    return registry.find(version) != registry.end();
}

std::vector<std::string> list_gpu_runners() {
    std::vector<std::string> names;
    const auto &registry = runner_registry();
    names.reserve(registry.size());
    for (const auto &entry : registry) {
        names.push_back(entry.first);
    }
    return names;
}

HardwareProvenance detect_gpu_provenance(const KMeansConfig &cfg, int device_id) {
    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0 || device_id < 0 || device_id >= dev_count) {
        std::fprintf(stderr, "Error: invalid CUDA device_id=%d (available devices: %d).\n",
                     device_id, dev_count);
        std::exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    int runtime_ver = 0;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_ver));

    HardwareProvenance hw;
    hw.gpu_name = prop.name;
    hw.compute_cap = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
    hw.sm_count = prop.multiProcessorCount;
    hw.global_mem_mb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);
    hw.cuda_version =
        std::to_string(runtime_ver / 1000) + "." + std::to_string((runtime_ver % 1000) / 10);
    hw.git_commit = cfg.git_commit.empty() ? std::string(KUDA_GIT_COMMIT) : cfg.git_commit;
    return hw;
}

KMeansResult run_kmeans_gpu(const std::string &version, const std::vector<float> &points,
                            const std::vector<float> &init_centroids, const KMeansConfig &cfg) {
    const auto &registry = runner_registry();
    const auto it = registry.find(version);
    if (it == registry.end()) {
        std::fprintf(stderr, "Error: GPU runner '%s' is not registered in this build.\n",
                     version.c_str());
        std::exit(EXIT_FAILURE);
    }

    KMeansResult res = it->second(points, init_centroids, cfg);
    if (res.hw.compute_cap == "none") {
        res.hw = detect_gpu_provenance(cfg);
    }
    if (!cfg.git_commit.empty()) {
        res.hw.git_commit = cfg.git_commit;
    }
    return res;
}

} // namespace kuda
