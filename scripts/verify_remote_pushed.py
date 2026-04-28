#!/usr/bin/env python3
"""Provenance-honesty gate.

Verifies that the local working tree's commit has been pushed to
`SharathSPhD/pratyabhijna` on the active branch (or a configured branch).

Checks:

1. The working tree is clean (no unstaged or untracked-but-tracked-elsewhere
   files); a phase that declared `done` cannot have dirty state.
2. The local HEAD commit SHA matches the remote branch head SHA reported
   by `gh api repos/SharathSPhD/pratyabhijna/branches/<branch>`.
3. The branch exists remotely.

Exit code:

* 0 - all green;
* 1 - violation;
* 2 - infra failure.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE = "SharathSPhD/pratyabhijna"


def _run(cmd: list[str], cwd: Path, strip: bool = True) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), check=False, capture_output=True, text=True, timeout=30,
    )
    if strip:
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    return p.returncode, p.stdout, p.stderr.strip()


_PORCELAIN_LINE = re.compile(r"^(?P<status>..)\s(?P<path>.+)$")


def _parse_porcelain(output: str) -> list[tuple[str, str]]:
    """Parse `git status --porcelain=v1` output into (status_code, path) pairs.

    The two-character status field at the start can include leading spaces
    (e.g. ' M file' for a tracked-but-modified-in-worktree file). Naive
    string splitting that calls `.strip()` on the whole output eats the
    leading space of the first line, so we always parse via a regex over the
    raw output.
    """
    pairs: list[tuple[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        m = _PORCELAIN_LINE.match(line)
        if m:
            pairs.append((m.group("status"), m.group("path")))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PCE remote-push honesty gate.")
    parser.add_argument("--repo", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="GitHub owner/repo slug.")
    parser.add_argument(
        "--branch", default=None,
        help="Branch to verify; defaults to current local branch.",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Skip the clean-tree check (use only for in-flight diagnostic runs).",
    )
    parser.add_argument(
        "--strict-dirty", action="store_true",
        help="Include audit/ and .ralph-loop entries in the dirty-tree check.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(json.dumps({"ok": False, "error": f"not a git repo: {repo}"}))
        return 2

    if shutil.which("gh") is None:
        print(json.dumps({"ok": False, "error": "gh CLI not on PATH"}))
        return 2

    rc, branch_out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    if rc != 0:
        print(json.dumps({"ok": False, "error": "git rev-parse failed"}))
        return 2
    branch = args.branch or branch_out

    rc, head_sha, _ = _run(["git", "rev-parse", "HEAD"], repo)
    if rc != 0:
        print(json.dumps({"ok": False, "error": "could not resolve HEAD"}))
        return 2

    rc, status_out, _ = _run(["git", "status", "--porcelain=v1"], repo, strip=False)
    if rc != 0:
        print(json.dumps({"ok": False, "error": "git status failed"}))
        return 2
    # The gate itself writes to audit/<phase>/, so audit/ entries are
    # excluded from the dirty-tree check by default. Use --strict-dirty to
    # include them.
    EXCLUDED_PREFIXES = ("audit/", ".ralph-loop")
    pairs = _parse_porcelain(status_out)
    if args.strict_dirty:
        dirty_lines = [f"{st} {pth}" for st, pth in pairs]
    else:
        dirty_lines = [
            f"{st} {pth}" for st, pth in pairs
            if not any(pth.startswith(pref) for pref in EXCLUDED_PREFIXES)
        ]

    rc, gh_out, gh_err = _run(
        ["gh", "api", f"repos/{args.remote}/branches/{branch}"], repo,
    )
    remote_sha = ""
    branch_exists = False
    if rc == 0:
        try:
            data = json.loads(gh_out)
            remote_sha = (data.get("commit") or {}).get("sha") or ""
            branch_exists = bool(remote_sha)
        except json.JSONDecodeError:
            pass
    elif "Branch not found" not in gh_err and "404" not in gh_err:
        print(json.dumps({"ok": False, "error": f"gh api failed: {gh_err[:200]}"}))
        return 2

    payload = {
        "branch": branch,
        "local_head": head_sha,
        "remote_head": remote_sha,
        "branch_exists_remote": branch_exists,
        "dirty_files": dirty_lines,
    }
    failures: list[str] = []
    if not branch_exists:
        failures.append(f"branch {branch} does not exist on {args.remote}")
    if remote_sha and remote_sha != head_sha:
        failures.append(f"local HEAD {head_sha[:8]} != remote {remote_sha[:8]} on {branch}")
    if dirty_lines and not args.allow_dirty:
        failures.append(f"working tree has {len(dirty_lines)} dirty entries")

    payload["ok"] = not failures
    payload["failures"] = failures
    print(json.dumps(payload, indent=2))
    if failures:
        for f in failures:
            print(f"[verify_remote_pushed] FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
