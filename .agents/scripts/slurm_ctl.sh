#!/usr/bin/env bash
# ==============================================================================
# slurm_ctl.sh — Local-to-HPC Slurm Control CLI for kuda_means
# ==============================================================================
# Usage:
#   ./.agents/scripts/slurm_ctl.sh nodes
#   ./.agents/scripts/slurm_ctl.sh sync [--dry-run]
#   ./.agents/scripts/slurm_ctl.sh submit [--script <file.sbatch>] [--partition <p>] [--gres <g>] [--time <t>] [--force] [-- <args...>]
#   ./.agents/scripts/slurm_ctl.sh profile [--partition <p>] [--gres <g>] [version] [N] [D] [K]
#   ./.agents/scripts/slurm_ctl.sh status [job_id | --all | -p <partition>]
#   ./.agents/scripts/slurm_ctl.sh logs <job_id> [lines]
#   ./.agents/scripts/slurm_ctl.sh fetch [job_id]
#   ./.agents/scripts/slurm_ctl.sh cancel <job_id>
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SSH_HOST="${KUDA_MEANS_SSH_HOST:-kmutt-slurm-kuda}"
REMOTE_DIR="~/kuda_means"

usage() {
    cat <<EOF
Usage: ./.agents/scripts/slurm_ctl.sh <command> [args]

SSH Target: ${SSH_HOST}:${REMOTE_DIR}

Commands:
  nodes                               Show cluster partitions, GPU nodes, and availability (sinfo)
  sync [--dry-run]                    Stage Makefile, include/, src/, tests/, scripts/slurm/ to cluster
  submit [options] [-- <args...>]     Submit scripts/slurm/<script>.sbatch (default: build_and_bench.sbatch)
      --script <name.sbatch>          Choose any sbatch file in scripts/slurm/
      --partition <name>              Override #SBATCH --partition (e.g. gpu4090)
      --gres <spec>                   Override #SBATCH --gres (e.g. gpu:1)
      --time <HH:MM:SS>               Override #SBATCH --time
      --force                         Skip active job ceiling check
  profile [opts] [ver] [N] [D] [K]    Submit scripts/slurm/profile.sbatch (ncu + nsys + sanitizer)
  status [job_id | --all | -p <part>] Check your queue, a job's sacct record, or partition queue
  logs <job_id> [lines]               View last N lines of slurm-<job_id>.log (default: 80)
  fetch [job_id]                      Pull results/, profiles/*.txt, and logs/slurm-*.log from cluster
  cancel <job_id>                     Cancel a specific numeric Slurm job ID
EOF
}

validate_job_id() {
    local job_id="${1:-}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid or missing numeric <job_id>: '${job_id}'" >&2
        exit 1
    fi
}

cmd_nodes() {
    echo ">> Cluster GPU Nodes & Partitions (sinfo):"
    ssh "$SSH_HOST" "sinfo -N -o %N,%P,%t,%G,%c,%m" 2>/dev/null | column -t -s ',' || ssh "$SSH_HOST" "sinfo"
    echo ""
    echo ">> Active Queue Summary (squeue):"
    ssh "$SSH_HOST" "squeue"
}

cmd_sync() {
    local dry_run_flag=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                dry_run_flag=("--dry-run")
                shift
                ;;
            *)
                echo "ERROR: Unknown sync argument '$1'" >&2
                exit 1
                ;;
        esac
    done

    if [[ "${#dry_run_flag[@]}" -gt 0 ]]; then
        echo ">> [DRY-RUN] Previewing outbound rsync payload to ${SSH_HOST}:${REMOTE_DIR}/ ..."
    else
        echo ">> Staging source & build files to ${SSH_HOST}:${REMOTE_DIR}/ ..."
    fi

    rsync -avz "${dry_run_flag[@]}" \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.direnv/' \
        --exclude='.agents/' \
        --exclude='.env*' \
        --exclude='build/' \
        --exclude='bin/' \
        --exclude='logs/' \
        --exclude='docs/' \
        --exclude='__pycache__/' \
        --include='Makefile' \
        --include='include/' \
        --include='include/**' \
        --include='src/' \
        --include='src/**' \
        --include='tests/' \
        --include='tests/**' \
        --include='scripts/' \
        --include='scripts/slurm/' \
        --include='scripts/slurm/*.sbatch' \
        --exclude='*' \
        "$REPO_ROOT/" "${SSH_HOST}:${REMOTE_DIR}/"
}

check_queue_ceiling() {
    echo ">> Checking active Slurm queue..."
    local active_queue
    active_queue="$(ssh "$SSH_HOST" "squeue --me -o %i,%j,%T,%M,%R" 2>/dev/null || ssh "$SSH_HOST" "squeue --me" || true)"
    local job_count
    job_count="$(echo "$active_queue" | grep -v "^JOBID" | grep -c "kuda_" || true)"
    if [[ "$job_count" -ge 3 ]]; then
        echo "ERROR: Safety ceiling reached ($job_count active kuda_* jobs in queue)." >&2
        echo "$active_queue" >&2
        echo "Wait for jobs to finish, or pass --force if intentional." >&2
        exit 1
    fi
}

cmd_submit() {
    local script_name="build_and_bench.sbatch"
    local force=0
    local sbatch_opts=()
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --script)
                script_name="$(basename "$2")"
                shift 2
                ;;
            --partition|-p)
                sbatch_opts+=("--partition=$2")
                shift 2
                ;;
            --gres)
                sbatch_opts+=("--gres=$2")
                shift 2
                ;;
            --time|-t)
                sbatch_opts+=("--time=$2")
                shift 2
                ;;
            --force)
                force=1
                shift
                ;;
            --)
                shift
                extra_args+=("$@")
                break
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done

    if [[ ! "$script_name" =~ ^[a-zA-Z0-9_-]+\.sbatch$ ]]; then
        echo "ERROR: Invalid sbatch script name '${script_name}'" >&2
        exit 1
    fi

    if [[ "$force" -eq 0 ]]; then
        check_queue_ceiling
    fi

    local cmd="sbatch"
    if [[ "${#sbatch_opts[@]}" -gt 0 ]]; then
        cmd+=" ${sbatch_opts[*]}"
    fi
    cmd+=" scripts/slurm/${script_name}"
    if [[ "${#extra_args[@]}" -gt 0 ]]; then
        cmd+=" ${extra_args[*]}"
    fi

    echo ">> Submitting: ${cmd} (on ${SSH_HOST})"
    local submit_out
    submit_out="$(ssh "$SSH_HOST" "$cmd")"
    echo "$submit_out"

    if [[ "$submit_out" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]+([0-9]+) ]]; then
        echo "JOB_ID=${BASH_REMATCH[1]}"
    fi
}

cmd_profile() {
    local sbatch_opts=()
    local pos_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --partition|-p) sbatch_opts+=("--partition=$2"); shift 2 ;;
            --gres)         sbatch_opts+=("--gres=$2"); shift 2 ;;
            --time|-t)      sbatch_opts+=("--time=$2"); shift 2 ;;
            *)              pos_args+=("$1"); shift ;;
        esac
    done

    local version="${pos_args[0]:-v1}"
    local n_pts="${pos_args[1]:-1000000}"
    local dim="${pos_args[2]:-32}"
    local k_clusters="${pos_args[3]:-16}"

    check_queue_ceiling
    local cmd="sbatch"
    if [[ "${#sbatch_opts[@]}" -gt 0 ]]; then
        cmd+=" ${sbatch_opts[*]}"
    fi
    cmd+=" scripts/slurm/profile.sbatch ${version} ${n_pts} ${dim} ${k_clusters}"

    echo ">> Submitting: ${cmd} (on ${SSH_HOST})"
    local submit_out
    submit_out="$(ssh "$SSH_HOST" "$cmd")"
    echo "$submit_out"

    if [[ "$submit_out" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]+([0-9]+) ]]; then
        echo "JOB_ID=${BASH_REMATCH[1]}"
    fi
}

cmd_status() {
    local arg="${1:-}"
    if [[ -z "$arg" ]]; then
        echo ">> Active Queue (squeue --me):"
        ssh "$SSH_HOST" "squeue --me -o %i,%j,%T,%M,%R" 2>/dev/null || ssh "$SSH_HOST" "squeue --me"
    elif [[ "$arg" == "--all" ]]; then
        echo ">> Full Cluster Queue (squeue):"
        ssh "$SSH_HOST" "squeue"
    elif [[ "$arg" == "-p" || "$arg" == "--partition" ]]; then
        local part="${2:-gpu4090}"
        echo ">> Partition '${part}' Queue:"
        ssh "$SSH_HOST" "squeue -p ${part}"
    else
        validate_job_id "$arg"
        echo ">> Queue Status (job ${arg}):"
        ssh "$SSH_HOST" "squeue --me -j ${arg}" || true
        echo ">> Accounting Record (sacct job ${arg}):"
        ssh "$SSH_HOST" "sacct -j ${arg} --format=JobID,JobName,Partition,State,ExitCode,Elapsed -P -n"
    fi
}

cmd_logs() {
    local job_id="${1:-}"
    local lines="${2:-80}"
    validate_job_id "$job_id"
    if [[ ! "$lines" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid line count '${lines}'" >&2
        exit 1
    fi
    ssh "$SSH_HOST" "tail -n ${lines} slurm-${job_id}.log"
}

cmd_fetch() {
    local job_id="${1:-}"
    if [[ -n "$job_id" ]]; then
        validate_job_id "$job_id"
    fi

    mkdir -p "$REPO_ROOT/results" "$REPO_ROOT/profiles" "$REPO_ROOT/logs"

    echo ">> Fetching benchmark results from ${SSH_HOST}:${REMOTE_DIR}/results/ ..."
    rsync -avz --max-size=50m \
        --include='*/' \
        --include='*.csv' \
        --include='*.json' \
        --include='*.txt' \
        --exclude='*' \
        "${SSH_HOST}:${REMOTE_DIR}/results/" "$REPO_ROOT/results/" || true

    echo ">> Fetching profiling text summaries from ${SSH_HOST}:${REMOTE_DIR}/profiles/ ..."
    rsync -avz --max-size=50m \
        --include='*/' \
        --include='*.txt' \
        --include='*.csv' \
        --exclude='*' \
        "${SSH_HOST}:${REMOTE_DIR}/profiles/" "$REPO_ROOT/profiles/" || true

    if [[ -n "$job_id" ]]; then
        echo ">> Fetching slurm-${job_id}.log to logs/ ..."
        rsync -avz --max-size=50m \
            "${SSH_HOST}:${REMOTE_DIR}/logs/slurm-${job_id}.log" "$REPO_ROOT/logs/" || true
    else
        echo ">> Fetching slurm-*.log files to logs/ ..."
        rsync -avz --max-size=50m \
            --include='slurm-*.log' \
            --exclude='*' \
            "${SSH_HOST}:${REMOTE_DIR}/logs/" "$REPO_ROOT/logs/" || true
    fi
}

cmd_cancel() {
    local job_id="${1:-}"
    validate_job_id "$job_id"
    echo ">> Cancelling Slurm job ${job_id} on ${SSH_HOST}..."
    ssh "$SSH_HOST" "scancel ${job_id}"
    echo ">> Cancelled job ${job_id}."
}

SUBCOMMAND="${1:-}"
if [[ -z "$SUBCOMMAND" ]]; then
    usage
    exit 1
fi
shift

case "$SUBCOMMAND" in
    nodes)      cmd_nodes "$@" ;;
    sync)       cmd_sync "$@" ;;
    submit)     cmd_submit "$@" ;;
    profile)    cmd_profile "$@" ;;
    status)     cmd_status "$@" ;;
    logs)       cmd_logs "$@" ;;
    fetch)      cmd_fetch "$@" ;;
    cancel)     cmd_cancel "$@" ;;
    -h|--help|help) usage ;;
    *)
        echo "ERROR: Unknown command '${SUBCOMMAND}'" >&2
        usage >&2
        exit 1
        ;;
esac
