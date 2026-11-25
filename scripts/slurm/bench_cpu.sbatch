#!/bin/bash
#SBATCH --job-name=kuda_cpu
#SBATCH --output=logs/slurm-%j.log
#SBATCH --partition=cpu
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

echo "=== Hardware Provenance ==="
lscpu | grep -E "Model name|CPU\(s\):" | head -n 2 || true
g++ --version | head -n 1

echo "=== Build CPU Baseline ==="
make clean
make cpu

echo "=== Run CPU Baseline ==="
./bin/kuda_means_cpu "$@"
