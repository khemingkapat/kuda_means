# CUDA Coding & Benchmarking Standards (`kuda_means`)

Follow these rules for all C++ and CUDA code in this repository:

## 1. Error Checking (Mandatory)
- Wrap **every** CUDA runtime API call with `CHECK_CUDA(call)` from `include/cuda_utils.cuh`.
- After every kernel launch, check launch errors with `CHECK_LAST_CUDA_ERROR()`.
- Never ignore return codes or silent out-of-bounds memory accesses.

## 2. Deterministic Reproducibility
- All versions (V0–V5) must accept the same CLI flags: `-n <N> -d <D> -k <K> --iters <max_iters> --eps <epsilon> --seed <seed> --runs <repeats>`.
- Use a deterministic PRNG (`std::mt19937` with fixed `--seed`) to generate Gaussian-blob datasets and pick initial centroids identically across CPU and GPU runs.

## 3. Timing Methodology
- Use `cudaEvent_t` (`CudaTimer` in `include/cuda_utils.cuh`) for GPU timings.
- Always perform at least **1 warm-up run** before recording benchmark timings.
- Report **two distinct timings** in milliseconds (`ms`), averaged over `--runs` (default 5):
  1. `kernel_ms`: Pure GPU compute time per run (and average `ms/iter`).
  2. `e2e_ms`: End-to-end time including H2D and D2H PCIe transfers.
- Compute speedup against the CPU baseline (`cpu_ms / gpu_kernel_ms` and `cpu_ms / gpu_e2e_ms`).

## 4. Memory & Architecture Rules
- **V2 (`__constant__` memory):** Check size limits (`K * D * sizeof(float) <= 64 KB`, i.e., `<= 16384` floats). Fall back cleanly to read-only global memory (`__ldg` / `const float* __restrict__`) when $K \cdot D$ exceeds constant memory capacity.
- **V3 (Shared Memory Privatization):** Check dynamic shared memory requirements (`K * (D + 1) * sizeof(float)`) against `cudaDevAttrMaxSharedMemoryPerBlock`; guard or tile when $K \cdot D$ is large.
- **V5 (Streaming):** Always use pinned host memory (`cudaMallocHost` / `cudaFreeHost`) for `cudaMemcpyAsync`. Never pass pageable `malloc`/`std::vector` pointers to async transfers.

## 5. Formatting
- Format C++/CUDA files according to `.clang-format` (LLVM style, 4-space indent, 100-column limit).
