#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", __FILE__, __LINE__, #call,           \
                    cudaGetErrorString(err__));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                                    \
    do {                                                                                           \
        cudaError_t err__ = cudaGetLastError();                                                    \
        if (err__ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

struct CudaTimer {
    cudaEvent_t start_evt{};
    cudaEvent_t stop_evt{};

    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start_evt));
        CHECK_CUDA(cudaEventCreate(&stop_evt));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }

    CudaTimer(const CudaTimer &) = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

    void start(cudaStream_t stream = 0) { CHECK_CUDA(cudaEventRecord(start_evt, stream)); }

    float stop_ms(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_evt, stream));
        CHECK_CUDA(cudaEventSynchronize(stop_evt));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
        return elapsed_ms;
    }
};
