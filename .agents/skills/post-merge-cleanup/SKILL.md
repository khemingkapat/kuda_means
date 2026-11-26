---
name: post-merge-cleanup
description: >-
  Sync local default branch (main/master) with origin after a PR is merged on
  GitHub and safely delete the merged local feature branch (including squash-merged
  PRs verified via gh CLI).
---

# Post-Merge Cleanup Workflow

Use this skill after a Pull Request has been merged on GitHub to sync local `main` (or the repo's default branch) with `origin`, prune stale remote refs, and safely delete the merged local feature branch.

## Helper Script

Run the cleanup script located in this skill's `scripts/` directory:
```bash
bash .agents/skills/post-merge-cleanup/scripts/post_merge_cleanup.sh
```
*(If invoked from another repo via the global skill, run `bash ~/.gemini/config/skills/post-merge-cleanup/scripts/post_merge_cleanup.sh`.)*

## What the Script Does Deterministically
1. **Working tree safety check:** Aborts immediately if there are uncommitted changes.
2. **PR merge verification (`gh`):** If currently on a feature branch, checks `gh pr list --head <branch> --state merged` so squash-merged PRs are verified safely before deletion.
3. **Sync default branch:** Checks out `main` (auto-detected from `origin/HEAD`), runs `git fetch --prune origin`, and runs `git pull --ff-only origin <default-branch>`.
4. **Delete merged branch:** Removes the merged feature branch locally.
5. **Report:** Prints the synced `HEAD` commit, remaining local branches, and flags any other local branches whose PRs are already `MERGED` on GitHub.

## Handling Edge Cases (Agent Judgment)
If the script exits non-zero, **do not force-delete or discard work automatically**:
- **Dirty working tree:** Show the uncommitted files and ask whether to commit, stash, or keep them.
- **Branch PR not merged yet:** Tell the user the PR is still open/unmerged and ask if they really want `--force`.
- **Fast-forward pull failed (`main` diverged):** Show `git status` / `git log` and ask how to reconcile.
- **Other stale merged branches listed:** Report them and ask before deleting any extra branches.
