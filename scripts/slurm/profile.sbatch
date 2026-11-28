#!/bin/bash
#SBATCH --job-name=kuda_prof
#SBATCH --output=logs/slurm-%j.log
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

# shellcheck source=/dev/null
source /etc/profile
set -euo pipefail

VERSION="${1:-v1}"
N_PTS="${2:-1000000}"
DIM="${3:-32}"
K_CLUSTERS="${4:-16}"

mkdir -p logs profiles

echo "=== Host, Partition & CUDA Module ==="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} PARTITION=${SLURM_JOB_PARTITION:-unknown}"
date
module purge
module load cuda/12.2
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' .')"
GPU_ARCH="sm_${COMPUTE_CAP}"
echo "Detected GPU Architecture: ${GPU_ARCH}"

echo "=== Build (${GPU_ARCH}) ==="
make clean
if [[ -x /usr/bin/g++-12 ]]; then
    make gpu ARCH="${GPU_ARCH}" NVCC_CCBIN=/usr/bin/g++-12
elif [[ -x /usr/bin/g++-11 ]]; then
    make gpu ARCH="${GPU_ARCH}" NVCC_CCBIN=/usr/bin/g++-11
else
    make gpu ARCH="${GPU_ARCH}" NVCC="nvcc -allow-unsupported-compiler"
fi

if [[ ! -x bin/kuda_means_gpu ]]; then
    echo "ERROR: bin/kuda_means_gpu not found." >&2
    exit 1
fi

TAG="${VERSION}_${GPU_ARCH}_N${N_PTS}_D${DIM}_K${K_CLUSTERS}_job${SLURM_JOB_ID:-local}"

echo "=== [1/4] compute-sanitizer (memcheck + racecheck + synccheck) ==="
compute-sanitizer --tool memcheck ./bin/kuda_means_gpu \
    --version "$VERSION" -n 10000 -d "$DIM" -k "$K_CLUSTERS" --iters 5 --runs 1 \
    > "profiles/memcheck_${TAG}.txt" 2>&1 || true

compute-sanitizer --tool racecheck ./bin/kuda_means_gpu \
    --version "$VERSION" -n 10000 -d "$DIM" -k "$K_CLUSTERS" --iters 5 --runs 1 \
    > "profiles/racecheck_${TAG}.txt" 2>&1 || true

compute-sanitizer --tool synccheck ./bin/kuda_means_gpu \
    --version "$VERSION" -n 10000 -d "$DIM" -k "$K_CLUSTERS" --iters 5 --runs 1 \
    > "profiles/synccheck_${TAG}.txt" 2>&1 || true

echo "=== [2/4] Nsight Systems (nsys) timeline & stats ==="
nsys profile --stats=true -o "profiles/nsys_${TAG}" --force-overwrite=true \
    ./bin/kuda_means_gpu --version "$VERSION" -n "$N_PTS" -d "$DIM" -k "$K_CLUSTERS" --iters 10 --runs 1 --no-verify \
    > "profiles/nsys_${TAG}.txt" 2>&1 || true

echo "=== [3/4] Nsight Compute (ncu) kernel metrics ==="
set +e
ncu --set full --launch-skip 0 --launch-count 2 \
    -o "profiles/ncu_${TAG}" --force-overwrite \
    ./bin/kuda_means_gpu --version "$VERSION" -n "$N_PTS" -d "$DIM" -k "$K_CLUSTERS" \
    --warmup 0 --iters 1 --runs 1 --no-verify \
    > "profiles/ncu_${TAG}.txt" 2>&1
NCU_RC=$?
set -e

if [[ "$NCU_RC" -ne 0 ]] || grep -q "ERR_NVGPUCTRPERM" "profiles/ncu_${TAG}.txt"; then
    echo "ERROR: ncu profiling failed (exit=${NCU_RC})." >&2
    cat "profiles/ncu_${TAG}.txt" >&2
    exit 1
fi

if [[ ! -s "profiles/ncu_${TAG}.ncu-rep" ]]; then
    echo "ERROR: profiles/ncu_${TAG}.ncu-rep is missing or empty." >&2
    exit 1
fi

ncu --import "profiles/ncu_${TAG}.ncu-rep" --page details >> "profiles/ncu_${TAG}.txt"

echo "=== [4/4] Profiling Complete ==="
ls -lh profiles/*"${TAG}"* || true
