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
bash .agents/skills/post-merge-cleanup/scripts/post_merge_cleanup.sh [branch] [--force]
```
*(If invoked from another repo via the global skill, run `bash ~/.gemini/config/skills/post-merge-cleanup/scripts/post_merge_cleanup.sh`.)*

## What the Script Does Deterministically
1. **Working tree safety check:** Aborts immediately if there are uncommitted changes.
2. **PR merge verification (`gh`):** Verifies the current feature branch (or explicit `[branch]` argument when run from `main`) via `gh pr list --head <branch> --state merged` so squash-merged PRs are verified safely before deletion.
3. **Linked worktree mode:** If run from inside a linked git worktree, automatically removes that worktree after PR verification, syncs `main` in the primary checkout (if clean and on `main`), and deletes the merged branch.
4. **Primary checkout mode:** Checks out `main` (auto-detected from `origin/HEAD`), runs `git fetch --prune origin` + `git pull --ff-only origin <default-branch>`, and deletes the merged feature branch (refusing if still checked out in another worktree).
5. **Report:** Prints the synced `HEAD` commit, remaining local branches, and flags any other local branches whose PRs are already `MERGED` on GitHub.

## Handling Edge Cases (Agent Judgment)
If the script exits non-zero, **do not force-delete or discard work automatically**:
- **Dirty working tree:** Show the uncommitted files and ask whether to commit, stash, or keep them.
- **Branch PR not merged yet:** Tell the user the PR is still open/unmerged and ask if they really want `--force`.
- **Fast-forward pull failed (`main` diverged):** Show `git status` / `git log` and ask how to reconcile.
- **Branch still checked out in another worktree:** Remove that secondary worktree first (`git worktree remove <path>`), then re-run cleanup from the primary checkout.
- **Other stale merged branches listed:** Report them and ask before deleting any extra branches.
