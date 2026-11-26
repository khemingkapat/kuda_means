#!/usr/bin/env python3
"""Deterministic GitHub issue queue and claim-and-branch helper for start-issue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def get_current_user() -> str:
    res = run_cmd(["gh", "api", "user", "--jq", ".login"])
    return res.stdout.strip()


def get_default_branch() -> str:
    res = run_cmd(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], check=False)
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip().rsplit("/", 1)[-1]
    return "main"


def check_working_tree() -> list[str]:
    res = run_cmd(["git", "status", "--porcelain"])
    return [line for line in res.stdout.splitlines() if line.strip()]


def fetch_open_issues() -> list[dict[str, Any]]:
    fields = "number,title,assignees,labels,milestone,body,url"
    res = run_cmd(["gh", "issue", "list", "--state", "open", "--limit", "100", "--json", fields])
    return json.loads(res.stdout or "[]")


def sort_key(issue: dict[str, Any], me: str) -> tuple[int, str, int]:
    assignees = [a.get("login", "") for a in issue.get("assignees", [])]
    if me in assignees:
        bucket = 0
    elif not assignees:
        bucket = 1
    else:
        bucket = 2
    milestone = issue.get("milestone") or {}
    due_on = milestone.get("dueOn") or "9999-12-31T23:59:59Z"
    return (bucket, due_on, int(issue["number"]))


def cmd_list(args: argparse.Namespace) -> int:
    me = get_current_user()
    dirty = check_working_tree()
    issues = fetch_open_issues()

    filtered: list[dict[str, Any]] = []
    for issue in issues:
        labels = {lbl.get("name", "").lower() for lbl in issue.get("labels", [])}
        if not args.include_epics and "epic" in labels:
            continue
        assignees = [a.get("login", "") for a in issue.get("assignees", [])]
        if not args.include_others and assignees and me not in assignees:
            continue
        filtered.append(issue)

    filtered.sort(key=lambda item: sort_key(item, me))

    if args.json:
        payload = {
            "user": me,
            "dirty_files": dirty,
            "issues": filtered,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"User: {me}")
    if dirty:
        print(f"WARNING: {len(dirty)} uncommitted path(s) in working tree.")
    if not filtered:
        print("No matching open issues found.")
        return 0

    for issue in filtered:
        assignees = [a.get("login", "") for a in issue.get("assignees", [])]
        status = "ASSIGNED(me)" if me in assignees else ("UNASSIGNED" if not assignees else "OTHER")
        milestone = (issue.get("milestone") or {}).get("title") or "No milestone"
        due = ((issue.get("milestone") or {}).get("dueOn") or "")[:10]
        due_str = f" (due {due})" if due else ""
        labels = ", ".join(lbl.get("name", "") for lbl in issue.get("labels", []))
        print(f"#{issue['number']:<4} [{status}] {issue['title']}")
        print(f"      Milestone: {milestone}{due_str} | Labels: {labels or 'none'}")
        print(f"      URL: {issue['url']}")
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    dirty = check_working_tree()
    if dirty and not args.allow_dirty:
        print("ERROR: Working tree has uncommitted changes:", file=sys.stderr)
        for line in dirty:
            print(f"  {line}", file=sys.stderr)
        print("Commit/stash changes first, or pass --allow-dirty.", file=sys.stderr)
        return 1

    default_branch = get_default_branch()
    run_cmd(["git", "fetch", "origin"])
    run_cmd(["gh", "issue", "edit", str(args.number), "--add-assignee", "@me"])

    branch_exists = (
        run_cmd(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{args.branch}"],
            check=False,
        ).returncode
        == 0
    )
    if branch_exists:
        run_cmd(["git", "checkout", args.branch])
        action = f"Switched to existing branch '{args.branch}'"
    else:
        run_cmd(["git", "checkout", "-b", args.branch, f"origin/{default_branch}"])
        action = f"Created and switched to '{args.branch}' from origin/{default_branch}"

    print(f"Claimed issue #{args.number} (@me). {action}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List, sort, claim, and branch GitHub issues.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List and sort open issues.")
    p_list.add_argument("--json", action="store_true", help="Output structured JSON.")
    p_list.add_argument("--include-epics", action="store_true", help="Include epic-labeled issues.")
    p_list.add_argument(
        "--include-others", action="store_true", help="Include issues assigned to teammates."
    )
    p_list.set_defaults(func=cmd_list)

    p_claim = sub.add_parser("claim", help="Assign issue to @me and create/switch feature branch.")
    p_claim.add_argument("number", type=int, help="GitHub issue number.")
    p_claim.add_argument("branch", help="Feature branch name to create or switch to.")
    p_claim.add_argument(
        "--allow-dirty", action="store_true", help="Allow running with uncommitted changes."
    )
    p_claim.set_defaults(func=cmd_claim)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
