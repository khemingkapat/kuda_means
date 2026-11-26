#!/usr/bin/env bash
# ==============================================================================
# gpu_lock.sh — flock + Persistent Lease Lock for Slurm & Colab GPU Resources
# ==============================================================================
# Prevents parallel git worktrees / agents on the same machine from:
#   1. Simultaneously syncing to ~/kuda_means or submitting Slurm jobs with the
#      same SSH key / cluster account.
#   2. Spawning two Google Colab GPU sessions simultaneously.
#
# Usage:
#   ./.agents/scripts/gpu_lock.sh acquire <slurm|colab> [--wait <sec>] [--note <text>] [--force]
#   ./.agents/scripts/gpu_lock.sh release <slurm|colab> [--force]
#   ./.agents/scripts/gpu_lock.sh status [slurm|colab]
#   ./.agents/scripts/gpu_lock.sh set-job <slurm|colab> <job_id>
#   ./.agents/scripts/gpu_lock.sh run <slurm|colab> [--wait <sec>] [--note <text>] [--ephemeral] -- <cmd...>
# ==============================================================================

set -euo pipefail

# Consistent UID-scoped lock dir across login/non-login shells (no XDG split)
LOCK_DIR="${KUDA_MEANS_LOCK_DIR:-/tmp/kuda_means_locks_$(id -u)}"
mkdir -p "$LOCK_DIR"
chmod 700 "$LOCK_DIR" 2>/dev/null || true

usage() {
    cat <<EOF
Usage: ./.agents/scripts/gpu_lock.sh <command> [args]

Resources: slurm | colab
Lock dir:  ${LOCK_DIR}

Commands:
  acquire <resource> [--wait <sec>] [--note <text>] [--force]
      Acquire (or refresh for the same worktree) the exclusive lease.
  release <resource> [--force]
      Release the lease (must be owned by this worktree unless --force).
  status [resource]
      Show current lease status for slurm and/or colab.
  set-job <resource> <job_id>
      Record an active job/session ID on the current worktree's lease.
  run <resource> [--wait <sec>] [--note <text>] [--ephemeral] -- <cmd...>
      Serialize execution with flock, verify/acquire the lease, and run <cmd...>.
      With --ephemeral, releases the lease when <cmd...> finishes if newly acquired.
      If <cmd...> fails and the lease was newly acquired, the lease is rolled back.
EOF
}

validate_resource() {
    local res="${1:-}"
    case "$res" in
        slurm|colab) ;;
        *)
            echo "ERROR: Invalid resource '${res}'. Expected 'slurm' or 'colab'." >&2
            exit 1
            ;;
    esac
}

validate_uint() {
    local val="${1:-}"
    local flag="${2:---wait}"
    if [[ ! "$val" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${flag} requires a non-negative integer, got '${val}'." >&2
        exit 1
    fi
}

current_worktree_path() {
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
        (cd "$(git rev-parse --show-toplevel)" && pwd -P)
    else
        pwd -P
    fi
}

current_branch_name() {
    git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
}

lease_file_for() {
    echo "${LOCK_DIR}/$1.lease"
}

flock_file_for() {
    echo "${LOCK_DIR}/$1.flock"
}

read_lease_field() {
    local file="$1"
    local key="$2"
    if [[ ! -f "$file" ]]; then
        return 0
    fi
    awk -F'=' -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

write_lease_file() {
    local file="$1"
    local owner_path="$2"
    local owner_branch="$3"
    local job_id="$4"
    local note="$5"
    local acquired_at
    acquired_at="$(LC_ALL=C date -u '+%Y-%m-%dT%H:%M:%SZ')"

    local tmp="${file}.tmp.$$"
    {
        printf 'OWNER_PATH=%s\n' "$owner_path"
        printf 'OWNER_BRANCH=%s\n' "$owner_branch"
        printf 'JOB_ID=%s\n' "$job_id"
        printf 'NOTE=%s\n' "$note"
        printf 'ACQUIRED_AT=%s\n' "$acquired_at"
    } > "$tmp"
    mv -f "$tmp" "$file"
}

acquire_flock_fd9() {
    local res="$1"
    local wait_sec="$2"
    local ffile
    ffile="$(flock_file_for "$res")"
    exec 9>"$ffile"
    if [[ "$wait_sec" -gt 0 ]]; then
        if ! flock -w "$wait_sec" 9; then
            echo "ERROR: Timed out after ${wait_sec}s waiting for '${res}' flock (${ffile})." >&2
            echo "Another process is actively running a '${res}' operation." >&2
            exit 2
        fi
    else
        if ! flock -n 9; then
            echo "ERROR: Another process is actively running a '${res}' operation (flock busy: ${ffile})." >&2
            echo "Retry shortly or pass --wait <sec>." >&2
            exit 2
        fi
    fi
}

check_and_claim_lease() {
    local res="$1"
    local note="$2"
    local force="$3"
    local lfile
    lfile="$(lease_file_for "$res")"
    local my_path my_branch
    my_path="$(current_worktree_path)"
    my_branch="$(current_branch_name)"

    local existing_owner="" existing_branch="" existing_job="" existing_note="" existing_at=""
    if [[ -f "$lfile" ]]; then
        existing_owner="$(read_lease_field "$lfile" "OWNER_PATH")"
        existing_branch="$(read_lease_field "$lfile" "OWNER_BRANCH")"
        existing_job="$(read_lease_field "$lfile" "JOB_ID")"
        existing_note="$(read_lease_field "$lfile" "NOTE")"
        existing_at="$(read_lease_field "$lfile" "ACQUIRED_AT")"

        if [[ "$existing_owner" != "$my_path" && "$force" -ne 1 ]]; then
            echo "ERROR: '${res}' GPU resource is locked by another worktree!" >&2
            echo "  Owner worktree : ${existing_owner}" >&2
            echo "  Owner branch   : ${existing_branch}" >&2
            if [[ -n "$existing_job" ]]; then
                echo "  Active job/ID  : ${existing_job}" >&2
            fi
            if [[ -n "$existing_note" ]]; then
                echo "  Note           : ${existing_note}" >&2
            fi
            echo "  Acquired at    : ${existing_at}" >&2
            echo "Wait for it to finish and release, or run:" >&2
            echo "  ./.agents/scripts/gpu_lock.sh release ${res} --force" >&2
            exit 2
        fi
    fi

    local keep_job="$existing_job"
    local keep_note="$existing_note"
    if [[ "$existing_owner" != "$my_path" ]]; then
        keep_job=""
        keep_note=""
    fi
    local final_note="${note:-$keep_note}"
    write_lease_file "$lfile" "$my_path" "$my_branch" "$keep_job" "$final_note"
}

cmd_acquire() {
    local res="${1:-}"
    validate_resource "$res"
    shift

    local wait_sec=0
    local note=""
    local force=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wait)
                [[ $# -ge 2 ]] || { echo "ERROR: --wait requires a value." >&2; exit 1; }
                validate_uint "$2" "--wait"
                wait_sec="$2"
                shift 2
                ;;
            --note)
                [[ $# -ge 2 ]] || { echo "ERROR: --note requires a value." >&2; exit 1; }
                note="$2"
                shift 2
                ;;
            --force) force=1; shift ;;
            *)
                echo "ERROR: Unknown acquire option '$1'" >&2
                exit 1
                ;;
        esac
    done

    local held_var="KUDA_MEANS_LOCK_HELD_${res^^}"
    if [[ "${!held_var:-0}" != "1" ]]; then
        acquire_flock_fd9 "$res" "$wait_sec"
    fi
    check_and_claim_lease "$res" "$note" "$force"
    echo "Acquired '${res}' lock for $(current_worktree_path) (branch: $(current_branch_name))."
}

cmd_release() {
    local res="${1:-}"
    validate_resource "$res"
    shift

    local force=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force) force=1; shift ;;
            *)
                echo "ERROR: Unknown release option '$1'" >&2
                exit 1
                ;;
        esac
    done

    local held_var="KUDA_MEANS_LOCK_HELD_${res^^}"
    if [[ "${!held_var:-0}" != "1" ]]; then
        acquire_flock_fd9 "$res" 5
    fi

    local lfile
    lfile="$(lease_file_for "$res")"
    if [[ ! -f "$lfile" ]]; then
        echo "No active '${res}' lock to release."
        return 0
    fi

    local existing_owner my_path
    existing_owner="$(read_lease_field "$lfile" "OWNER_PATH")"
    my_path="$(current_worktree_path)"

    if [[ "$existing_owner" != "$my_path" && "$force" -ne 1 ]]; then
        echo "ERROR: Cannot release '${res}' lock owned by another worktree (${existing_owner})." >&2
        echo "Pass --force to override." >&2
        exit 2
    fi

    rm -f "$lfile"
    echo "Released '${res}' lock."
}

cmd_set_job() {
    local res="${1:-}"
    local job_id="${2:-}"
    validate_resource "$res"
    if [[ -z "$job_id" ]]; then
        echo "ERROR: Missing <job_id>." >&2
        exit 1
    fi

    local held_var="KUDA_MEANS_LOCK_HELD_${res^^}"
    if [[ "${!held_var:-0}" != "1" ]]; then
        acquire_flock_fd9 "$res" 5
    fi

    local lfile
    lfile="$(lease_file_for "$res")"
    if [[ ! -f "$lfile" ]]; then
        echo "ERROR: Cannot set job ID — no active '${res}' lock held." >&2
        exit 2
    fi

    local existing_owner my_path my_branch existing_note
    existing_owner="$(read_lease_field "$lfile" "OWNER_PATH")"
    my_path="$(current_worktree_path)"
    if [[ "$existing_owner" != "$my_path" ]]; then
        echo "ERROR: Cannot set job ID — '${res}' lock is owned by '${existing_owner}'." >&2
        exit 2
    fi

    my_branch="$(current_branch_name)"
    existing_note="$(read_lease_field "$lfile" "NOTE")"
    write_lease_file "$lfile" "$my_path" "$my_branch" "$job_id" "$existing_note"
}

print_one_status() {
    local res="$1"
    local lfile
    lfile="$(lease_file_for "$res")"
    if [[ ! -f "$lfile" ]]; then
        echo "${res}: UNLOCKED"
        return 0
    fi
    echo "${res}: LOCKED"
    echo "  worktree : $(read_lease_field "$lfile" "OWNER_PATH")"
    echo "  branch   : $(read_lease_field "$lfile" "OWNER_BRANCH")"
    local job_id note
    job_id="$(read_lease_field "$lfile" "JOB_ID")"
    note="$(read_lease_field "$lfile" "NOTE")"
    [[ -n "$job_id" ]] && echo "  job_id   : ${job_id}"
    [[ -n "$note" ]] && echo "  note     : ${note}"
    echo "  since    : $(read_lease_field "$lfile" "ACQUIRED_AT")"
}

cmd_status() {
    local res="${1:-}"
    if [[ -n "$res" ]]; then
        validate_resource "$res"
        print_one_status "$res"
    else
        print_one_status "slurm"
        print_one_status "colab"
    fi
}

cmd_run() {
    local res="${1:-}"
    validate_resource "$res"
    shift

    local wait_sec=0
    local note=""
    local ephemeral=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wait)
                [[ $# -ge 2 ]] || { echo "ERROR: --wait requires a value." >&2; exit 1; }
                validate_uint "$2" "--wait"
                wait_sec="$2"
                shift 2
                ;;
            --note)
                [[ $# -ge 2 ]] || { echo "ERROR: --note requires a value." >&2; exit 1; }
                note="$2"
                shift 2
                ;;
            --ephemeral) ephemeral=1; shift ;;
            --)          shift; break ;;
            *)
                echo "ERROR: Unknown run option '$1' (separate command with --)" >&2
                exit 1
                ;;
        esac
    done

    if [[ $# -eq 0 ]]; then
        echo "ERROR: Missing command after --" >&2
        exit 1
    fi

    local held_var="KUDA_MEANS_LOCK_HELD_${res^^}"
    local already_held=0
    if [[ "${!held_var:-0}" == "1" ]]; then
        already_held=1
    else
        acquire_flock_fd9 "$res" "$wait_sec"
    fi

    local lfile had_lease=0
    lfile="$(lease_file_for "$res")"
    [[ -f "$lfile" ]] && had_lease=1

    check_and_claim_lease "$res" "$note" 0

    # Roll back newly acquired lease if interrupted by SIGINT/SIGTERM
    if [[ "$had_lease" -eq 0 ]]; then
        # shellcheck disable=SC2064
        trap "rm -f '$lfile'; exit 130" INT TERM
    fi

    local rc=0
    # Close FD 9 inside child subshell so child daemons (e.g. SSH ControlMaster)
    # do not inherit the flock descriptor while parent still holds it.
    if [[ "$already_held" -eq 1 ]]; then
        "$@" || rc=$?
    else
        (exec 9>&- && exec env "${held_var}=1" "$@") || rc=$?
    fi

    trap - INT TERM

    # Release newly acquired lease if ephemeral OR if the command failed
    if [[ "$had_lease" -eq 0 && ( "$ephemeral" -eq 1 || "$rc" -ne 0 ) ]]; then
        rm -f "$lfile"
    fi
    return "$rc"
}

SUBCOMMAND="${1:-}"
if [[ -z "$SUBCOMMAND" ]]; then
    usage
    exit 1
fi
shift

case "$SUBCOMMAND" in
    acquire) cmd_acquire "$@" ;;
    release) cmd_release "$@" ;;
    status)  cmd_status "$@" ;;
    set-job) cmd_set_job "$@" ;;
    run)     cmd_run "$@" ;;
    -h|--help|help) usage ;;
    *)
        echo "ERROR: Unknown command '${SUBCOMMAND}'" >&2
        usage >&2
        exit 1
        ;;
esac
