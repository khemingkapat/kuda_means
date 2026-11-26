#!/bin/bash
#SBATCH --job-name=kuda_sweep
#SBATCH --output=logs/slurm-%j.log
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ==============================================================================
# sweep_regimes.sbatch — Reproducible (N, D, K) Regime Sweep for V0–V4
# ==============================================================================
# Usage (via slurm_ctl.sh):
#   ./.agents/scripts/slurm_ctl.sh sync
#   ./.agents/scripts/slurm_ctl.sh submit --script sweep_regimes.sbatch \
#       [--partition <p>] [--time <HH:MM:SS>] \
#       [-- --versions v0,v1,v2,v3,v4 --git-commit "$(git rev-parse --short HEAD)"]
#
# Notes:
#   - Locks canonical benchmark constants (SEED=42, ITERS=50, EPS=1e-4) so
#     results/*.json files remain directly comparable across runs.
#   - Uses a 13-point representative (N, D, K) matrix:
#       * N-scaling at (D=16, K=8): N in {10^4, 10^5, 10^6, 10^7}
#       * Crossover grid at N=10^5 across D in {16, 128}, K in {8, 64, 256}
#       * Crossover grid at N=10^6 across D in {16, 128}, K in {8, 64, 256}
#         excluding (10^6, 128, 256), which alone requires ~17 min per CPU
#         verification pass (already covered by (10^5, 128, 256) for V2/V3/V4).
#   - Probes linked GPU versions once via `--version __probe__` and skips
#     unimplemented versions cleanly; any real runtime or verification failure
#     deletes the target JSON artifact and aborts non-zero.
# ==============================================================================

set -euo pipefail

VERSIONS_CSV="${VERSIONS:-v0,v1,v2,v3,v4}"
GIT_COMMIT_ARG="${GIT_COMMIT:-}"

usage() {
    cat <<'EOF'
Usage: sweep_regimes.sbatch [options]

Options:
  --versions <csv>     Comma-separated versions to sweep (default: v0,v1,v2,v3,v4)
  --git-commit <sha>   Git commit SHA to record in HardwareProvenance
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --versions)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --versions requires a non-empty comma-separated value." >&2
                exit 1
            fi
            VERSIONS_CSV="$2"
            shift 2
            ;;
        --git-commit)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --git-commit requires a non-empty value." >&2
                exit 1
            fi
            GIT_COMMIT_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

IFS=',' read -r -a REQUESTED_VERSIONS <<< "${VERSIONS_CSV}"
VALIDATED_VERSIONS=()
for raw_version in "${REQUESTED_VERSIONS[@]}"; do
    version="${raw_version#"${raw_version%%[![:space:]]*}"}"
    version="${version%"${version##*[![:space:]]}"}"
    if [[ -z "${version}" ]]; then
        continue
    fi
    case "${version}" in
        v0|cpu|v1|v2|v3|v4|v5)
            VALIDATED_VERSIONS+=("${version}")
            ;;
        *)
            echo "ERROR: Unknown version '${version}' in --versions (expected one of: v0, cpu, v1, v2, v3, v4, v5)." >&2
            exit 1
            ;;
    esac
done
if [[ "${#VALIDATED_VERSIONS[@]}" -eq 0 ]]; then
    echo "ERROR: No valid versions specified in --versions." >&2
    exit 1
fi

set +euo pipefail
# shellcheck source=/dev/null
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
if [[ -z "${COMPUTE_CAP}" ]]; then
    echo "ERROR: Failed to detect GPU compute capability via nvidia-smi." >&2
    exit 1
fi
GPU_ARCH="sm_${COMPUTE_CAP}"
echo "Detected GPU Architecture: ${GPU_ARCH}"

echo "=== Build Unified GPU/CPU Binary (${GPU_ARCH}) ==="
make clean
make gpu ARCH="${GPU_ARCH}" NVCC_CCBIN=/usr/bin/g++-12

if [[ ! -x bin/kuda_means_gpu ]]; then
    echo "ERROR: bin/kuda_means_gpu was not produced by make gpu." >&2
    exit 1
fi

# Zero-cost probe of linked GPU versions (exits before data allocation)
PROBE_OUT="$(./bin/kuda_means_gpu --version __probe__ 2>&1 || true)"
if [[ "${PROBE_OUT}" != *"Available GPU versions:"* ]]; then
    echo "ERROR: Failed to probe bin/kuda_means_gpu (unexpected startup failure):" >&2
    echo "${PROBE_OUT}" >&2
    exit 1
fi
AVAILABLE_GPU_VERSIONS=()
avail_raw="${PROBE_OUT#*Available GPU versions: }"
avail_raw="${avail_raw%%$'\n'*}"
if [[ "${avail_raw}" != "(none linked yet)" && -n "${avail_raw}" ]]; then
    IFS=',' read -r -a raw_arr <<< "${avail_raw}"
    for item in "${raw_arr[@]}"; do
        trimmed="${item#"${item%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        if [[ -n "${trimmed}" ]]; then
            AVAILABLE_GPU_VERSIONS+=("${trimmed}")
        fi
    done
fi

is_gpu_version_linked() {
    local target="$1"
    if [[ "${#AVAILABLE_GPU_VERSIONS[@]}" -eq 0 ]]; then
        return 1
    fi
    local linked
    for linked in "${AVAILABLE_GPU_VERSIONS[@]}"; do
        if [[ "${linked}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

# Canonical locked benchmark constants
readonly SEED=42
readonly ITERS=50
readonly EPS="1e-4"
readonly GPU_WARMUP=1
readonly GPU_RUNS=5
readonly CPU_LIGHT_WARMUP=1
readonly CPU_LIGHT_RUNS=5
readonly CPU_HEAVY_WARMUP=0
readonly CPU_HEAVY_RUNS=1
readonly CPU_HEAVY_THRESHOLD=100000000

# Representative 13-point (N, D, K) regime matrix
readonly REGIMES=(
    # N-scaling at baseline (D=16, K=8)
    "10000 16 8"
    "100000 16 8"
    "1000000 16 8"
    "10000000 16 8"
    # Low vs. high (D, K) crossover at N=100,000
    "100000 16 64"
    "100000 16 256"
    "100000 128 8"
    "100000 128 64"
    "100000 128 256"
    # Low vs. high (D, K) crossover at N=1,000,000
    "1000000 16 64"
    "1000000 16 256"
    "1000000 128 8"
    "1000000 128 64"
)

if [[ "${#AVAILABLE_GPU_VERSIONS[@]}" -gt 0 ]]; then
    linked_summary="${AVAILABLE_GPU_VERSIONS[*]}"
else
    linked_summary="(none)"
fi

echo "=== Sweep Configuration ==="
echo "Requested versions : ${VALIDATED_VERSIONS[*]}"
echo "Linked GPU versions: ${linked_summary}"
echo "Regime count       : ${#REGIMES[@]}"
echo "Constants          : seed=${SEED} iters=${ITERS} eps=${EPS}"
if [[ -n "${GIT_COMMIT_ARG}" ]]; then
    echo "Git commit override: ${GIT_COMMIT_ARG}"
fi

completed_runs=0
skipped_versions=0

for version in "${VALIDATED_VERSIONS[@]}"; do

    is_cpu=0
    if [[ "${version}" == "v0" || "${version}" == "cpu" ]]; then
        is_cpu=1
    elif ! is_gpu_version_linked "${version}"; then
        echo "[SKIP] version='${version}' is not linked in this build."
        skipped_versions=$((skipped_versions + 1))
        continue
    fi

    echo "=== Sweeping version=${version} ==="
    for regime in "${REGIMES[@]}"; do
        read -r n d k <<< "${regime}"

        if [[ "${is_cpu}" -eq 1 ]]; then
            json_out="results/v0_cpu_n${n}_d${d}_k${k}.json"
            if (( n * d * k >= CPU_HEAVY_THRESHOLD )); then
                warmup="${CPU_HEAVY_WARMUP}"
                runs="${CPU_HEAVY_RUNS}"
            else
                warmup="${CPU_LIGHT_WARMUP}"
                runs="${CPU_LIGHT_RUNS}"
            fi
        else
            json_out="results/${version}_${GPU_ARCH}_n${n}_d${d}_k${k}.json"
            warmup="${GPU_WARMUP}"
            runs="${GPU_RUNS}"
        fi

        cmd=(
            ./bin/kuda_means_gpu
            --version "${version}"
            -n "${n}"
            -d "${d}"
            -k "${k}"
            --iters "${ITERS}"
            --eps "${EPS}"
            --seed "${SEED}"
            --warmup "${warmup}"
            --runs "${runs}"
            --json "${json_out}"
        )
        if [[ -n "${GIT_COMMIT_ARG}" ]]; then
            cmd+=(--git-commit "${GIT_COMMIT_ARG}")
        fi

        echo ">> Running version=${version} N=${n} D=${d} K=${k} (warmup=${warmup}, runs=${runs}) -> ${json_out}"
        if ! "${cmd[@]}"; then
            echo "ERROR: Benchmark/verification failed for version=${version} N=${n} D=${d} K=${k}; removing ${json_out}" >&2
            rm -f "${json_out}"
            exit 1
        fi
        completed_runs=$((completed_runs + 1))
    done
done

echo "=== Sweep Complete ==="
echo "Completed runs  : ${completed_runs}"
echo "Skipped versions: ${skipped_versions}"
date
