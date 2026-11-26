#!/usr/bin/env bash
set -euo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# 1. Refuse to run with uncommitted changes
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Working tree has uncommitted changes. Commit or stash them first." >&2
    git status -sb >&2
    exit 1
fi

# 2. Detect default branch (fallback to main)
DEFAULT_BRANCH="$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || true)"
if [[ -z "${DEFAULT_BRANCH}" ]]; then
    DEFAULT_BRANCH="main"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
DELETE_BRANCH=""

# 3. If on a feature branch, verify it was merged via gh CLI (handles squash merges)
if [[ "${CURRENT_BRANCH}" != "${DEFAULT_BRANCH}" && "${CURRENT_BRANCH}" != "HEAD" ]]; then
    MERGED_PR="$(gh pr list --head "${CURRENT_BRANCH}" --state merged --json number,title --jq '.[0] | "#\(.number) \(.title)"' 2>/dev/null || true)"
    if [[ -n "${MERGED_PR}" ]]; then
        echo "Verified merged PR for '${CURRENT_BRANCH}': ${MERGED_PR}"
        DELETE_BRANCH="${CURRENT_BRANCH}"
    elif [[ "${FORCE}" == "true" ]]; then
        echo "WARNING: --force set; will delete '${CURRENT_BRANCH}' without merged PR confirmation."
        DELETE_BRANCH="${CURRENT_BRANCH}"
    else
        echo "ERROR: No merged PR found on GitHub for branch '${CURRENT_BRANCH}'." >&2
        echo "Verify the PR is merged, or re-run with --force to delete anyway." >&2
        exit 2
    fi
fi

# 4. Sync default branch from origin
git checkout "${DEFAULT_BRANCH}"
git fetch --prune origin
git pull --ff-only origin "${DEFAULT_BRANCH}"

# 5. Delete the merged feature branch
if [[ -n "${DELETE_BRANCH}" ]]; then
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
        echo "NOTE: Local branch '${br}' already has merged PR #${OTHER_MERGED} (not auto-deleted)."
    fi
done < <(git for-each-ref --format='%(refname:short)' refs/heads/)
