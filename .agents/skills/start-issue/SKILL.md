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
1. Run the queue script to fetch and deterministically sort open issues:
   ```bash
   python3 .agents/skills/start-issue/scripts/issue_queue.py list
   ```
   The script automatically:
   - Checks working tree cleanliness and remote sync status.
   - Filters out parent `epic` issues.
   - Lists issues already assigned to `@me` first, followed by unassigned issues sorted by earliest milestone due date and lowest issue number.
2. **Use judgment to rank & filter:**
   - Inspect issue descriptions for blockers (e.g., "Depends on #X" where `#X` is still open).
   - If the user already has an open assigned issue, highlight it first.
   - Present the top 2–3 unblocked candidates concisely (number, title, milestone, why it's ready) and ask the user which one to pick (use `ask_question` when there are multiple good choices).

### Case B: User selected or specified an issue `#N`
1. Determine the branch name:
   - Check the issue body first — if it explicitly names a branch (e.g., `docs/references-bib`), use that exact name.
   - Otherwise, infer a concise conventional branch name matching `.agents/rules/team-workflow.md` (`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `exp/<topic>`).
2. Run the claim-and-branch script:
   ```bash
   python3 .agents/skills/start-issue/scripts/issue_queue.py claim <number> <branch-name>
   ```
   This verifies a clean working tree, fetches `origin`, assigns the issue to `@me` via `gh issue edit`, and creates/switches to `<branch-name>` from `origin/main`.
3. Give a brief kickoff summary:
   - Branch created and issue assigned
   - Files to touch
   - Acceptance criteria to satisfy
