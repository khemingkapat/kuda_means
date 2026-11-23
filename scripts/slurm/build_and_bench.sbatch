#!/bin/bash
#SBATCH --job-name=kuda_bench
#SBATCH --output=logs/slurm-%j.log
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

source /etc/profile
set -euo pipefail

mkdir -p logs results

echo "=== Host, Partition & Timestamp ==="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} PARTITION=${SLURM_JOB_PARTITION:-unknown}"
date

echo "=== Load CUDA Module ==="
module purge
module load cuda/12.2
module list

echo "=== Hardware Provenance ==="
lscpu | grep -E "Model name|CPU\(s\):" | head -n 2 || true
g++-12 --version | head -n 1
nvcc --version | tail -n 2
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

# Auto-detect SM architecture from the allocated GPU node (e.g. 8.9 -> sm_89)
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' .')"
GPU_ARCH="sm_${COMPUTE_CAP}"
echo "Detected GPU Architecture: ${GPU_ARCH}"

echo "=== Build CPU & GPU Targets (${GPU_ARCH}) ==="
make clean
make cpu CXX=g++-12
make gpu ARCH="${GPU_ARCH}" NVCC_CCBIN=/usr/bin/g++-12

echo "=== Run Benchmark ==="
if [[ -x bin/kuda_means_gpu ]]; then
    ./bin/kuda_means_gpu "$@"
elif [[ -x bin/kuda_means_cpu ]]; then
    echo "GPU binary not yet built; running CPU baseline..."
    ./bin/kuda_means_cpu "$@"
else
    echo "No executable produced yet. Implement src/cpu/ or src/v1_naive/ first."
fi
