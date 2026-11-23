#!/usr/bin/env bash
# ==============================================================================
# kuda_means_gatekeeper.sh — Server-Side SSH ForceCommand Gatekeeper (ctl01)
# ==============================================================================
# Installed on the Slurm login node (portal.slurm.cpe.kmutt.ac.th) at:
#   ~/bin/kuda_means_gatekeeper.sh (chmod 700)
#
# Bound in ~/.ssh/authorized_keys on the cluster (coexists with kv_bench):
#   command="$HOME/bin/kuda_means_gatekeeper.sh",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty <KUDA_PUBKEY>
#
# Enforces:
#   1. No interactive shell access.
#   2. No shell metacharacters (; & | ` $ > < ( ) { } newlines) or path traversal (..).
#   3. Outbound/inbound rsync strictly restricted to ~/kuda_means/ (no --delete).
#   4. Slurm submission restricted to scripts/slurm/<name>.sbatch (with optional safe
#      sbatch flags like --partition, --gres, --time and safe script args).
#   5. Read-only cluster inspection: sinfo, squeue, sacct, tail/cat slurm-<job_id>.log.
#   6. Job cancellation restricted to a single numeric job ID (scancel <job_id>).
#   7. Full audit logging to ~/.ssh/kuda_means_audit.log.
# ==============================================================================

set -euo pipefail

REMOTE_DIR="${KUDA_MEANS_REMOTE_DIR:-$HOME/kuda_means}"
AUDIT_LOG="${KUDA_MEANS_AUDIT_LOG:-$HOME/.ssh/kuda_means_audit.log}"
DRY_RUN="${KUDA_MEANS_GATEKEEPER_DRY_RUN:-0}"

mkdir -p "$(dirname "$AUDIT_LOG")"

log_audit() {
    local status="$1"
    local reason="$2"
    local ts
    ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local client="${SSH_CLIENT:-local}"
    printf "%s | %s | client=%s | reason=%s | cmd=%s\n" \
        "$ts" "$status" "$client" "$reason" "${SSH_ORIGINAL_COMMAND:-<empty>}" >> "$AUDIT_LOG" 2>/dev/null || true
}

deny() {
    local reason="$1"
    log_audit "DENIED" "$reason"
    echo "ERROR [kuda_means_gatekeeper]: Access denied ($reason)." >&2
    exit 1
}

run_or_dry() {
    log_audit "ALLOWED" "$1"
    shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "ALLOWED: $*"
        exit 0
    fi
    exec "$@"
}

# 1. Reject empty command (interactive shell attempt)
if [[ -z "${SSH_ORIGINAL_COMMAND:-}" ]]; then
    deny "interactive_shell_prohibited"
fi

CMD="$SSH_ORIGINAL_COMMAND"

# 2. Reject multiline commands
if [[ "$CMD" == *$'\n'* ]] || [[ "$CMD" == *$'\r'* ]]; then
    deny "multiline_command_prohibited"
fi

# 3. Reject dangerous shell metacharacters and path traversal
if [[ "$CMD" =~ [\;\&\|\`\$\>\<\(\)\{\}] ]]; then
    deny "shell_metacharacters_prohibited"
fi

if [[ "$CMD" == *".."* ]]; then
    deny "path_traversal_prohibited"
fi

# 4. Match allowed command patterns
# ------------------------------------------------------------------------------
# Pattern A: Rsync Transfer (Local <-> Cluster ~/kuda_means/)
# ------------------------------------------------------------------------------
if [[ "$CMD" =~ ^rsync[[:space:]]+--server[[:space:]]+ ]]; then
    if [[ "$CMD" == *"--delete"* ]]; then
        deny "rsync_delete_prohibited"
    fi

    # Inbound rsync (Cluster -> Local) has --sender
    if [[ "$CMD" =~ ^rsync[[:space:]]+--server[[:space:]]+--sender[[:space:]]+-[a-zA-Z0-9.]+([[:space:]]+--max-size=[0-9]+[kmgKMG]?)?[[:space:]]+(\.[[:space:]]+)?(~/kuda_means|kuda_means)(/[a-zA-Z0-9._/*-]*)?$ ]]; then
        mkdir -p "$REMOTE_DIR/results" "$REMOTE_DIR/profiles" "$REMOTE_DIR/logs"
        SAFE_CMD="${CMD//\~\/kuda_means/$REMOTE_DIR}"
        read -r -a ARGS <<< "$SAFE_CMD"
        run_or_dry "rsync_inbound" "${ARGS[@]}"
    fi

    # Outbound rsync (Local -> Cluster)
    if [[ "$CMD" =~ ^rsync[[:space:]]+--server[[:space:]]+-[a-zA-Z0-9.]+[[:space:]]+\.[[:space:]]+(~/kuda_means|kuda_means)/?$ ]]; then
        mkdir -p "$REMOTE_DIR/results" "$REMOTE_DIR/profiles" "$REMOTE_DIR/logs"
        SAFE_CMD="${CMD//\~\/kuda_means/$REMOTE_DIR}"
        read -r -a ARGS <<< "$SAFE_CMD"
        run_or_dry "rsync_outbound" "${ARGS[@]}"
    fi

    deny "rsync_target_out_of_bounds"
fi

# ------------------------------------------------------------------------------
# Pattern B: Job Submission
# Allows any scripts/slurm/<name>.sbatch plus optional safe sbatch flags
# (--partition=..., --gres=..., --time=..., --job-name=...) and safe script args.
# ------------------------------------------------------------------------------
if [[ "$CMD" =~ ^sbatch[[:space:]]+((--(partition|gres|time|job-name)=[a-zA-Z0-9._:/-]+)[[:space:]]+)*scripts/slurm/[a-zA-Z0-9_-]+\.sbatch([[:space:]]+[a-zA-Z0-9._,=:/-]+)*$ ]]; then
    mkdir -p "$REMOTE_DIR/results" "$REMOTE_DIR/profiles" "$REMOTE_DIR/logs"
    cd "$REMOTE_DIR" || deny "missing_remote_dir"
    read -r -a ARGS <<< "$CMD"
    run_or_dry "sbatch_submit" "${ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# Pattern C: Cluster Availability, Queue & Accounting Status (sinfo / squeue / sacct)
# ------------------------------------------------------------------------------
if [[ "$CMD" == "sinfo" ]]; then
    run_or_dry "sinfo_default" sinfo
fi

if [[ "$CMD" == "sinfo -o %P,%a,%l,%D,%t,%G,%N" ]]; then
    run_or_dry "sinfo_formatted" sinfo -o "%P,%a,%l,%D,%t,%G,%N"
fi

if [[ "$CMD" == "sinfo -N -o %N,%P,%t,%G,%c,%m" ]]; then
    run_or_dry "sinfo_nodes" sinfo -N -o "%N,%P,%t,%G,%c,%m"
fi

if [[ "$CMD" == "squeue" ]]; then
    run_or_dry "squeue_all" squeue
fi

if [[ "$CMD" == "squeue --me" ]]; then
    run_or_dry "squeue_me" squeue --me
fi

if [[ "$CMD" == "squeue --me -o %i,%j,%T,%M,%R" ]]; then
    run_or_dry "squeue_me_formatted" squeue --me -o "%i,%j,%T,%M,%R"
fi

if [[ "$CMD" =~ ^squeue[[:space:]]+-p[[:space:]]+([a-zA-Z0-9_-]+)$ ]]; then
    PART="${BASH_REMATCH[1]}"
    run_or_dry "squeue_partition" squeue -p "$PART"
fi

if [[ "$CMD" =~ ^squeue[[:space:]]+--me[[:space:]]+-j[[:space:]]+([0-9]+)$ ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    run_or_dry "squeue_job" squeue --me -j "$JOB_ID"
fi

if [[ "$CMD" =~ ^sacct[[:space:]]+-j[[:space:]]+([0-9]+)[[:space:]]+--format=JobID,JobName,Partition,State,ExitCode,Elapsed[[:space:]]+-P[[:space:]]+-n$ ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    run_or_dry "sacct_job" sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed -P -n
fi

# ------------------------------------------------------------------------------
# Pattern D: Job Cancellation (scancel <numeric_job_id>)
# ------------------------------------------------------------------------------
if [[ "$CMD" =~ ^scancel[[:space:]]+([0-9]+)$ ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    run_or_dry "scancel_job" scancel "$JOB_ID"
fi

# ------------------------------------------------------------------------------
# Pattern E: Log Inspection (tail / cat logs/slurm-<job_id>.log or slurm-<job_id>.log)
# ------------------------------------------------------------------------------
resolve_log_file() {
    local job_id="$1"
    if [[ -f "$REMOTE_DIR/logs/slurm-${job_id}.log" ]]; then
        echo "$REMOTE_DIR/logs/slurm-${job_id}.log"
        return 0
    fi
    if [[ -f "$REMOTE_DIR/slurm-${job_id}.log" ]]; then
        echo "$REMOTE_DIR/slurm-${job_id}.log"
        return 0
    fi
    return 1
}

if [[ "$CMD" =~ ^tail[[:space:]]+-n[[:space:]]+([0-9]+)[[:space:]]+slurm-([0-9]+)\.log$ ]]; then
    LINES="${BASH_REMATCH[1]}"
    JOB_ID="${BASH_REMATCH[2]}"
    LOG_PATH="$(resolve_log_file "$JOB_ID" || echo "$REMOTE_DIR/logs/slurm-${JOB_ID}.log")"
    run_or_dry "tail_slurm_log" tail -n "$LINES" "$LOG_PATH"
fi

if [[ "$CMD" =~ ^cat[[:space:]]+slurm-([0-9]+)\.log$ ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    LOG_PATH="$(resolve_log_file "$JOB_ID" || echo "$REMOTE_DIR/logs/slurm-${JOB_ID}.log")"
    run_or_dry "cat_slurm_log" cat "$LOG_PATH"
fi

# Anything else is strictly rejected
deny "unrecognized_command"
