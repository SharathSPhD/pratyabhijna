"""v0.4.1 review fix #11: assert plugin manifests are well-formed.

Validates:
  - plugin/.cursor-plugin/plugin.json: required fields, version
  - plugin/.claude-plugin/plugin.json: required fields, version
  - plugin/.mcp.json: server entry shape; every command/arg path
    referenced under ${CLAUDE_PLUGIN_ROOT} exists on disk
  - plugin/README.md: declared MCP-tool count matches the server's
    runtime tool registration count

This is a manifest-level check, not a runtime check; it deliberately
does not import the FastMCP server (which loads heavy substrate
dependencies). The runtime tool count is approximated by counting
``@mcp.tool`` decorators in plugin/mcp/server.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugin"

CURSOR_MANIFEST = PLUGIN / ".cursor-plugin" / "plugin.json"
CLAUDE_MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
MCP_MANIFEST = PLUGIN / ".mcp.json"
PLUGIN_README = PLUGIN / "README.md"
SERVER_PY = PLUGIN / "mcp" / "server.py"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_cursor_manifest_well_formed() -> None:
    m = _load(CURSOR_MANIFEST)
    assert m["name"] == "pratyabhijna-creative-engine"
    assert m["version"].startswith("0.4")
    assert "displayName" in m
    assert m["license"] == "MIT"


def test_claude_code_manifest_well_formed() -> None:
    m = _load(CLAUDE_MANIFEST)
    assert m["name"] == "pratyabhijna-creative-engine"
    assert m["version"].startswith("0.4")
    assert m["license"] == "MIT"


def test_manifest_versions_agree() -> None:
    cur = _load(CURSOR_MANIFEST)
    cc = _load(CLAUDE_MANIFEST)
    assert cur["version"] == cc["version"], (
        f"manifest versions diverge: cursor={cur['version']!r} "
        f"claude={cc['version']!r}"
    )


def test_mcp_manifest_paths_exist_under_plugin_root() -> None:
    m = _load(MCP_MANIFEST)
    servers = m["mcpServers"]
    assert "pratyabhijna" in servers
    args = servers["pratyabhijna"]["args"]
    # Substitute CLAUDE_PLUGIN_ROOT -> plugin/ so referenced paths can be
    # resolved against the working tree. Skip pure CLI flags.
    placeholder = "${CLAUDE_PLUGIN_ROOT}"
    for arg in args:
        if placeholder not in arg:
            continue
        on_disk = arg.replace(placeholder, str(PLUGIN))
        # arg may contain `..` segments — resolve before existence check.
        path = Path(on_disk).resolve()
        assert path.exists(), f"manifest references missing path: {arg!r} -> {path}"


def test_plugin_readme_tool_count_matches_decorators() -> None:
    src = SERVER_PY.read_text(encoding="utf-8")
    # FastMCP is registered as ``mcp = FastMCP(...)``; tools are decorated
    # with ``@mcp.tool``.
    decorated = len(re.findall(r"^\s*@mcp\.tool\b", src, flags=re.MULTILINE))
    readme = PLUGIN_README.read_text(encoding="utf-8")
    # Pull the "MCP tools | <N>" cell out of the README's component table.
    match = re.search(r"\|\s*MCP tools\s*\|\s*(\d+)\s*\|", readme)
    assert match, "plugin/README.md must declare an MCP-tool count in its component table"
    declared = int(match.group(1))
    assert declared == decorated, (
        f"plugin/README.md declares {declared} MCP tools but plugin/mcp/server.py "
        f"actually registers {decorated} via @mcp.tool decorators"
    )


@pytest.mark.parametrize("manifest", [CURSOR_MANIFEST, CLAUDE_MANIFEST, MCP_MANIFEST])
def test_manifest_is_valid_json(manifest: Path) -> None:
    json.loads(manifest.read_text(encoding="utf-8"))
