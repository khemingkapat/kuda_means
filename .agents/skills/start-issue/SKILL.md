---
name: start-issue
description: >-
  Pick, claim (assign @me), and create a feature branch for the next GitHub issue
  to work on. Uses gh CLI and git via a deterministic queue script, with agent
  intelligence for unblocked issue ranking, branch naming, and kickoff briefing.
---

# Start Issue Workflow

Use this skill when the user wants to start work on the next issue (or a specific issue number) in this repository.

## Helper Script

All deterministic `gh` and `git` operations live in:
- `.agents/skills/start-issue/scripts/issue_queue.py`

## Procedure

### Case A: User did not specify an issue number
1. Run the queue script to fetch and deterministically sort open issues (use `--json` so issue bodies are available in one call):
   ```bash
   python3 .agents/skills/start-issue/scripts/issue_queue.py list --json
   ```
   The script automatically:
   - Checks working tree cleanliness and remote sync status.
   - Filters out parent `epic` issues.
   - Lists issues already assigned to `@me` first, followed by unassigned issues sorted by earliest milestone due date and lowest issue number.
2. **Use judgment to rank, filter, and tag for parallel work:**
   - Inspect issue descriptions for blockers (e.g., "Depends on #X" where `#X` is still open).
   - Inspect each unblocked issue's **Files to Touch** section and note which candidates are **parallel-safe** (zero file overlap) vs. overlapping.
   - Tag each candidate's workload profile (`GPU / long-running` vs. `local / fast`) so long Slurm/CUDA jobs can be paired with fast local tasks in separate worktrees.
   - If the user already has an open assigned issue, highlight it first.
   - Present the top 2–4 unblocked candidates concisely (number, title, milestone, workload tag, parallel-safe pairings) and ask the user which issue (or disjoint pair of issues) to pick (use `ask_question` when there are multiple good choices).

### Case B: User selected or specified one or more issues `#N`
1. Determine the branch name for each selected issue:
   - Check the issue body first — if it explicitly names a branch (e.g., `docs/references-bib`), use that exact name.
   - Otherwise, infer a concise conventional branch name matching `.agents/rules/team-workflow.md` (`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `exp/<topic>`).
2. Run the claim-and-branch script:
   - **Default (single issue, switch current checkout):**
     ```bash
     python3 .agents/skills/start-issue/scripts/issue_queue.py claim <number> <branch-name>
     ```
   - **Branch ref only (leave current working tree untouched, e.g., before opening a separate worktree):**
     ```bash
     python3 .agents/skills/start-issue/scripts/issue_queue.py claim <number> <branch-name> --no-checkout
     ```
   - **Isolated git worktree:**
     ```bash
     python3 .agents/skills/start-issue/scripts/issue_queue.py claim <number> <branch-name> --worktree <path>
     ```
   Default mode verifies a clean working tree, fetches `origin`, assigns the issue to `@me` via `gh issue edit`, and creates/switches to `<branch-name>` from `origin/main`. `--no-checkout` and `--worktree` leave the current working tree untouched.
3. Give a brief kickoff summary for each claimed issue:
   - Branch created (and worktree path, if used) and issue assigned
   - Files to touch
   - Acceptance criteria to satisfy
   - Note if GPU execution is required (remind to acquire `.agents/scripts/gpu_lock.sh` / use `slurm_ctl.sh` so parallel worktrees do not collide on Slurm or Colab).
