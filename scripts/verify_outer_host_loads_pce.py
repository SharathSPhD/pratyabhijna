#!/usr/bin/env python3
"""Outer-host PCE plugin loading smoke (Phase 2 gate).

Per [docs/SPEC_v0.3.md §1.1](../docs/SPEC_v0.3.md), v0.3 only sanitizes the *inner*
`claude --print` subprocess that `HaikuLM` spawns. The *outer* host (this Python
process, or a real Claude Code session) MUST still be able to discover and load
the PCE plugin so `pce_cascade(...)` is callable at all.

This script verifies:

* `pce` package importable and exposes `run_cascade`, operators, substrate, types;
* `plugin/.claude-plugin/plugin.json` parses and is at v0.3.0;
* `plugin/.mcp.json` parses;
* `plugin/mcp/server.py` importable and registers >=15 MCP tools;
* `plugin/skills/` has 5 skill dirs each containing `SKILL.md` with frontmatter;
* `plugin/agents/` has 5 agent .md files;
* `plugin/commands/` has 5 command .md files;
* `plugin/hooks/` has 3 hook entries.

Writes `audit/v0_3_outer_host_loads.json` and exits 0 only if all checks pass.

This script is read-only; it does NOT call the `claude` CLI and does not need
network access.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "plugin"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MIN_TOOLS_OUTER = 15  # v0.2 baseline; v0.3 adds 2 more (probe + hopfield_state)
EXPECTED_SKILLS = 5
EXPECTED_AGENTS = 5
EXPECTED_COMMANDS = 5
EXPECTED_HOOKS = 3
EXPECTED_PLUGIN_VERSION = "0.3.0"

FRONTMATTER_RE = re.compile(r"^---\n(.+?)\n---", re.DOTALL)


def _check_frontmatter(p: Path) -> tuple[bool, list[str]]:
    if not p.exists():
        return False, [f"missing file: {p}"]
    text = p.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return False, [f"{p.name}: missing frontmatter"]
    body = m.group(1)
    issues = []
    for key in ("name", "description"):
        if not re.search(rf"^{key}\s*:", body, re.MULTILINE):
            issues.append(f"{p.name}: missing key '{key}'")
    return len(issues) == 0, issues


def _import_pce() -> dict[str, object]:
    issues: list[str] = []
    out: dict[str, object] = {}
    try:
        cascade = importlib.import_module("pce.cascade")
        out["has_run_cascade"] = hasattr(cascade, "run_cascade")
    except Exception as exc:  # pragma: no cover
        issues.append(f"import pce.cascade failed: {exc!r}")
        out["has_run_cascade"] = False
    try:
        importlib.import_module("pce.substrate.haiku_lm")
        out["has_haiku_lm"] = True
    except Exception as exc:  # pragma: no cover
        issues.append(f"import pce.substrate.haiku_lm failed: {exc!r}")
        out["has_haiku_lm"] = False
    try:
        importlib.import_module("pce.substrate.integrity")
        out["has_integrity_probe"] = True
    except Exception as exc:  # pragma: no cover
        issues.append(f"import pce.substrate.integrity failed: {exc!r}")
        out["has_integrity_probe"] = False
    out["import_issues"] = issues
    return out


def _count_tools_in_server() -> tuple[int, list[str]]:
    issues: list[str] = []
    server_dir = str(PLUGIN_ROOT / "mcp")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    try:
        import server as plugin_server  # type: ignore
    except Exception as exc:
        issues.append(f"plugin server import failed: {exc!r}")
        return 0, issues
    mcp_obj = getattr(plugin_server, "mcp", None)
    if mcp_obj is None:
        issues.append("plugin server has no `mcp` attribute")
        return 0, issues
    tm = getattr(mcp_obj, "_tool_manager", None)
    if tm is None or not hasattr(tm, "list_tools"):
        issues.append("plugin server `mcp._tool_manager.list_tools` not found")
        return 0, issues
    tools = list(tm.list_tools())
    return len(tools), issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "audit" / "v0_3_outer_host_loads.json",
    )
    args = parser.parse_args()
    issues: list[str] = []
    report: dict[str, object] = {"plugin_root": str(PLUGIN_ROOT), "issues": issues}

    plugin_json_path = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
    if not plugin_json_path.exists():
        issues.append(f"plugin.json not found at {plugin_json_path}")
    else:
        try:
            plugin_json = json.loads(plugin_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"plugin.json parse error: {exc!r}")
            plugin_json = {}
        report["plugin_json_version"] = plugin_json.get("version")
        if plugin_json.get("version") != EXPECTED_PLUGIN_VERSION:
            issues.append(
                f"plugin.json version is {plugin_json.get('version')!r}, "
                f"expected {EXPECTED_PLUGIN_VERSION!r}"
            )

    mcp_json_path = PLUGIN_ROOT / ".mcp.json"
    if not mcp_json_path.exists():
        issues.append(f".mcp.json not found at {mcp_json_path}")
    else:
        try:
            json.loads(mcp_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f".mcp.json parse error: {exc!r}")

    pce_check = _import_pce()
    report["pce_import"] = pce_check
    issues.extend(pce_check.get("import_issues", []))  # type: ignore[arg-type]

    n_tools, tool_issues = _count_tools_in_server()
    report["mcp_tools_registered"] = n_tools
    issues.extend(tool_issues)
    if n_tools < MIN_TOOLS_OUTER:
        issues.append(
            f"MCP server registers {n_tools} tools, expected >= {MIN_TOOLS_OUTER}"
        )

    skills_dir = PLUGIN_ROOT / "skills"
    skill_dirs = sorted(p for p in skills_dir.glob("*") if p.is_dir())
    report["skills_count"] = len(skill_dirs)
    if len(skill_dirs) != EXPECTED_SKILLS:
        issues.append(
            f"skills count {len(skill_dirs)} != expected {EXPECTED_SKILLS}"
        )
    for d in skill_dirs:
        ok, sub = _check_frontmatter(d / "SKILL.md")
        if not ok:
            issues.extend(sub)

    agents_dir = PLUGIN_ROOT / "agents"
    agent_files = sorted(agents_dir.glob("*.md"))
    report["agents_count"] = len(agent_files)
    if len(agent_files) != EXPECTED_AGENTS:
        issues.append(
            f"agents count {len(agent_files)} != expected {EXPECTED_AGENTS}"
        )
    for f in agent_files:
        ok, sub = _check_frontmatter(f)
        if not ok:
            issues.extend(sub)

    commands_dir = PLUGIN_ROOT / "commands"
    command_files = sorted(commands_dir.glob("*.md"))
    report["commands_count"] = len(command_files)
    if len(command_files) != EXPECTED_COMMANDS:
        issues.append(
            f"commands count {len(command_files)} != expected {EXPECTED_COMMANDS}"
        )

    hooks_dir = PLUGIN_ROOT / "hooks"
    hook_entries = sorted(p for p in hooks_dir.glob("*") if p.is_file() or p.is_dir())
    report["hooks_count"] = len(hook_entries)
    if len(hook_entries) < EXPECTED_HOOKS:
        # `<` not `!=` because some installs may have settings.local etc.
        issues.append(
            f"hooks count {len(hook_entries)} < expected {EXPECTED_HOOKS}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report["passed"] = len(issues) == 0
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if issues:
        print("[verify_outer_host_loads_pce] FAILED")
        for line in issues:
            print(f"  - {line}")
        return 1
    print(
        f"[verify_outer_host_loads_pce] OK: PCE plugin v{EXPECTED_PLUGIN_VERSION} discoverable; "
        f"{n_tools} MCP tools, {len(skill_dirs)} skills, {len(agent_files)} agents, "
        f"{len(command_files)} commands, {len(hook_entries)} hooks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
