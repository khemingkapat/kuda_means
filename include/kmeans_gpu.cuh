#pragma once

#include "kmeans_common.h"

#include <string>
#include <vector>

namespace kuda {

// Uniform GPU K-Means stage signature matching run_kmeans_cpu().
using GpuKMeansFn = KMeansResult (*)(const std::vector<float> &points,
                                     const std::vector<float> &init_centroids,
                                     const KMeansConfig &cfg);

// Register a GPU stage implementation (e.g. "v1".."v5") at static init time.
// Aborts loudly on nullptr, empty/reserved ("cpu"/"v0"), or duplicate keys.
bool register_gpu_runner(const std::string &version, GpuKMeansFn fn);

// Check whether a GPU stage version is linked and registered in this binary.
bool has_gpu_runner(const std::string &version);

// Return sorted names of all registered GPU stages.
std::vector<std::string> list_gpu_runners();

// Invoke a registered GPU stage, ensuring HardwareProvenance and git_commit are populated.
KMeansResult run_kmeans_gpu(const std::string &version, const std::vector<float> &points,
                            const std::vector<float> &init_centroids, const KMeansConfig &cfg);

// Populate HardwareProvenance for the selected CUDA device (compute_cap formatted as "sm_XX").
HardwareProvenance detect_gpu_provenance(const KMeansConfig &cfg, int device_id = 0);

} // namespace kuda

#define KUDA_CONCAT_IMPL(a, b) a##b
#define KUDA_CONCAT(a, b) KUDA_CONCAT_IMPL(a, b)

// Place at file scope (inside or outside namespace kuda) in each src/v*/*.cu translation unit:
//   KUDA_REGISTER_GPU_RUNNER("v1", kuda::run_kmeans_v1);
#define KUDA_REGISTER_GPU_RUNNER(version_str, fn)                                                  \
    [[maybe_unused]] static const bool KUDA_CONCAT(kuda_gpu_runner_reg_, __LINE__) =               \
        ::kuda::register_gpu_runner((version_str), (fn))
