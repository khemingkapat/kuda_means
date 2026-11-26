#!/usr/bin/env bash
set -euo pipefail

FORCE=false
TARGET_BRANCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: post_merge_cleanup.sh [branch] [--force]"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option '$1'" >&2
            exit 1
            ;;
        *)
            if [[ -z "${TARGET_BRANCH}" ]]; then
                TARGET_BRANCH="$1"
                shift
            else
                echo "ERROR: Unexpected extra argument '$1'" >&2
                exit 1
            fi
            ;;
    esac
done

# Helper: print worktree path if a branch is currently checked out in any worktree
branch_worktree_path() {
    local target_ref="refs/heads/$1"
    git worktree list --porcelain | awk -v ref="${target_ref}" '
        /^worktree / { wt = substr($0, 10) }
        /^branch / { if (substr($0, 8) == ref) { print wt; exit } }
    '
}

# Helper: verify PR merge state on GitHub via gh CLI (handles squash merges)
verify_branch_merged() {
    local br="$1"
    local merged_pr
    merged_pr="$(gh pr list --head "${br}" --state merged --json number,title --jq 'if length > 0 then "#\(.[0].number) \(.[0].title)" else empty end' 2>/dev/null || true)"
    if [[ -n "${merged_pr}" ]]; then
        echo "Verified merged PR for '${br}': ${merged_pr}"
        return 0
    elif [[ "${FORCE}" == "true" ]]; then
        echo "WARNING: --force set; proceeding for '${br}' without merged PR confirmation."
        return 0
    else
        echo "ERROR: No merged PR found on GitHub for branch '${br}'." >&2
        echo "Verify the PR is merged, or re-run with --force to delete anyway." >&2
        exit 2
    fi
}

# 1. Refuse to run with uncommitted changes in current working tree
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Working tree has uncommitted changes. Commit or stash them first." >&2
    git status -sb >&2
    exit 1
fi

# 2. Detect default branch (fallback to main) and linked worktree status
DEFAULT_BRANCH="$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || true)"
if [[ -z "${DEFAULT_BRANCH}" ]]; then
    DEFAULT_BRANCH="main"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_DIR_ABS="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
GIT_COMMON_ABS="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
IS_LINKED_WORKTREE=false
if [[ "${GIT_DIR_ABS}" != "${GIT_COMMON_ABS}" ]]; then
    IS_LINKED_WORKTREE=true
fi

# 3A. Linked worktree mode: verify PR, remove this worktree, sync primary, delete branch
if [[ "${IS_LINKED_WORKTREE}" == "true" ]]; then
    LINKED_WT="$(cd "$(git rev-parse --show-toplevel)" && pwd -P)"
    PRIMARY_WT="$(git worktree list --porcelain | awk '/^worktree / {print substr($0, 10); exit}')"

    DELETE_BRANCH=""
    if [[ -n "${TARGET_BRANCH}" && "${TARGET_BRANCH}" != "${CURRENT_BRANCH}" ]]; then
        echo "ERROR: Inside linked worktree on '${CURRENT_BRANCH}'; omit [branch] or run from primary worktree to clean '${TARGET_BRANCH}'." >&2
        exit 3
    fi
    if [[ "${CURRENT_BRANCH}" != "${DEFAULT_BRANCH}" && "${CURRENT_BRANCH}" != "HEAD" ]]; then
        verify_branch_merged "${CURRENT_BRANCH}"
        DELETE_BRANCH="${CURRENT_BRANCH}"
    fi

    cd "${PRIMARY_WT}"
    git worktree remove "${LINKED_WT}"
    echo "Removed linked worktree: ${LINKED_WT}"

    git fetch --prune origin
    PRIMARY_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "${PRIMARY_BRANCH}" == "${DEFAULT_BRANCH}" && -z "$(git status --porcelain)" ]]; then
        git pull --ff-only origin "${DEFAULT_BRANCH}"
    else
        echo "NOTE: Primary worktree (${PRIMARY_WT}) is on '${PRIMARY_BRANCH}' or dirty; skipped fast-forward pull of '${DEFAULT_BRANCH}'."
    fi

    if [[ -n "${DELETE_BRANCH}" ]]; then
        git branch -D "${DELETE_BRANCH}"
        echo "Deleted merged local branch '${DELETE_BRANCH}'."
    fi

    echo ""
    echo "Primary worktree (${PRIMARY_WT}) at: $(git log -1 --oneline)"
    echo "NOTE: Your previous shell directory (${LINKED_WT}) was removed; close this pane or cd to ${PRIMARY_WT}."
    exit 0
fi

# 3B. Primary worktree mode
DELETE_BRANCH=""
if [[ -n "${TARGET_BRANCH}" ]]; then
    if [[ "${TARGET_BRANCH}" == "${DEFAULT_BRANCH}" ]]; then
        echo "ERROR: Refusing to delete default branch '${DEFAULT_BRANCH}'." >&2
        exit 1
    fi
    if ! git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
        echo "ERROR: Local branch '${TARGET_BRANCH}' does not exist." >&2
        exit 1
    fi
    verify_branch_merged "${TARGET_BRANCH}"
    DELETE_BRANCH="${TARGET_BRANCH}"
elif [[ "${CURRENT_BRANCH}" != "${DEFAULT_BRANCH}" && "${CURRENT_BRANCH}" != "HEAD" ]]; then
    verify_branch_merged "${CURRENT_BRANCH}"
    DELETE_BRANCH="${CURRENT_BRANCH}"
fi

# 4. Sync default branch from origin
if [[ "${CURRENT_BRANCH}" != "${DEFAULT_BRANCH}" ]]; then
    git checkout "${DEFAULT_BRANCH}"
fi
git fetch --prune origin
git pull --ff-only origin "${DEFAULT_BRANCH}"

# 5. Delete the merged feature branch
if [[ -n "${DELETE_BRANCH}" ]]; then
    ACTIVE_WT="$(branch_worktree_path "${DELETE_BRANCH}")"
    if [[ -n "${ACTIVE_WT}" ]]; then
        echo "ERROR: Cannot delete '${DELETE_BRANCH}' — it is checked out in worktree: ${ACTIVE_WT}" >&2
        echo "Remove that worktree first (git worktree remove '${ACTIVE_WT}'), then delete the branch." >&2
        exit 4
    fi
    git branch -D "${DELETE_BRANCH}"
    echo "Deleted merged local branch '${DELETE_BRANCH}'."
fi

# 6. Report synced HEAD and any other local branches whose PRs are already merged
echo ""
echo "Synced '${DEFAULT_BRANCH}' at: $(git log -1 --oneline)"
echo "Remaining local branches:"
git branch --format="  %(refname:short)"

while IFS= read -r br; do
    [[ -z "${br}" || "${br}" == "${DEFAULT_BRANCH}" ]] && continue
    OTHER_MERGED="$(gh pr list --head "${br}" --state merged --json number --jq '.[0].number // empty' 2>/dev/null || true)"
    if [[ -n "${OTHER_MERGED}" ]]; then
        WT_PATH="$(branch_worktree_path "${br}")"
        if [[ -n "${WT_PATH}" ]]; then
            echo "NOTE: Local branch '${br}' already has merged PR #${OTHER_MERGED} (checked out in worktree: ${WT_PATH})."
        else
            echo "NOTE: Local branch '${br}' already has merged PR #${OTHER_MERGED} (not auto-deleted)."
        fi
    fi
done < <(git for-each-ref --format='%(refname:short)' refs/heads/)
