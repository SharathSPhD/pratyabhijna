#!/usr/bin/env python3
"""Phase 7+8 plugin verification.

Loads the plugin manifest, the .mcp.json, the MCP server module, and walks the
plugin tree to confirm:

* `plugin/.claude-plugin/plugin.json` exists and parses;
* `.claude-plugin/marketplace.json` (at the **repo root**, where Claude
  Code's marketplace loader looks) exists and parses;
* `.mcp.json` has the expected server stanza;
* MCP server registers exactly 15 tools;
* `skills/`, `agents/`, `commands/`, `hooks/` each have the right count;
* Each markdown component carries a frontmatter with `name` and `description`.

Writes `audit/phase8/smoke.json` and exits 0 only if every count matches.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "plugin"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED = {
    # v0.2 adds two arm-aware tools: `pce_cascade` (arm-switchable cascade)
    # and `haiku_bare` (single Haiku call). The original 15 v0.1 tools are
    # all preserved for backward compatibility.
    "tools": 17,
    "skills": 5,
    "agents": 5,
    "commands": 5,
    "hooks": 3,
}

FRONTMATTER_RE = re.compile(r"^---\n(.+?)\n---", re.DOTALL)


def _has_frontmatter(p: Path) -> tuple[bool, list[str]]:
    text = p.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return False, ["missing frontmatter"]
    body = m.group(1)
    issues = []
    for key in ("name", "description"):
        if not re.search(rf"^{key}\s*:", body, re.MULTILINE):
            issues.append(f"missing key '{key}'")
    return len(issues) == 0, issues


def _count_tools_in_server() -> int:
    server_dir = str(PLUGIN_ROOT / "mcp")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    import server as plugin_server
    tools = plugin_server.mcp._tool_manager.list_tools() if hasattr(plugin_server.mcp, "_tool_manager") else []
    return len(tools)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "audit" / "phase8" / "smoke.json")
    args = parser.parse_args()

    checks: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {
        "plugin_root": str(PLUGIN_ROOT),
        "checks": checks,
        "issues": [],
    }
    issues: list[str] = []

    plugin_json = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
    # Per Claude Code docs, the marketplace manifest must live at the
    # repository root (`.claude-plugin/marketplace.json`); the plugin
    # entry's `source: "./plugin"` points back at PLUGIN_ROOT.
    marketplace_json = REPO_ROOT / ".claude-plugin" / "marketplace.json"
    mcp_json = PLUGIN_ROOT / ".mcp.json"

    for label, p in (
        ("plugin.json", plugin_json),
        ("marketplace.json", marketplace_json),
        (".mcp.json", mcp_json),
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            checks[label] = {"ok": True, "size": p.stat().st_size}
            if label == "plugin.json" and data.get("name") != "pratyabhijna-creative-engine":
                issues.append(f"{label}: name != pratyabhijna-creative-engine")
        except FileNotFoundError:
            checks[label] = {"ok": False, "error": "missing"}
            issues.append(f"{label}: missing")
        except json.JSONDecodeError as e:
            checks[label] = {"ok": False, "error": f"json: {e}"}
            issues.append(f"{label}: json parse error")

    skill_dirs = sorted([d for d in (PLUGIN_ROOT / "skills").iterdir() if d.is_dir()]) if (PLUGIN_ROOT / "skills").exists() else []
    skill_files = [d / "SKILL.md" for d in skill_dirs]
    agent_files = sorted((PLUGIN_ROOT / "agents").glob("*.md")) if (PLUGIN_ROOT / "agents").exists() else []
    command_files = sorted((PLUGIN_ROOT / "commands").glob("*.md")) if (PLUGIN_ROOT / "commands").exists() else []
    hook_files = sorted((PLUGIN_ROOT / "hooks").glob("*.sh")) if (PLUGIN_ROOT / "hooks").exists() else []

    counts = {
        "skills": len(skill_files),
        "agents": len(agent_files),
        "commands": len(command_files),
        "hooks": len(hook_files),
    }
    for kind, expected in EXPECTED.items():
        if kind == "tools":
            continue
        actual = counts.get(kind, -1)
        checks[kind] = {"ok": actual == expected, "expected": expected, "actual": actual}
        if actual != expected:
            issues.append(f"{kind}: expected {expected}, got {actual}")

    for kind, files in (("skills", skill_files), ("agents", agent_files), ("commands", command_files)):
        for f in files:
            ok, problems = _has_frontmatter(f)
            if not ok:
                issues.append(f"{kind}/{f.name}: {','.join(problems)}")

    try:
        n_tools = _count_tools_in_server()
        checks["tools"] = {"ok": n_tools == EXPECTED["tools"], "expected": EXPECTED["tools"], "actual": n_tools}
        if n_tools != EXPECTED["tools"]:
            issues.append(f"tools: expected {EXPECTED['tools']}, got {n_tools}")
    except Exception as e:
        checks["tools"] = {"ok": False, "error": str(e)}
        issues.append(f"tools: import failed: {e}")

    report["issues"] = issues
    report["ok"] = len(issues) == 0
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
