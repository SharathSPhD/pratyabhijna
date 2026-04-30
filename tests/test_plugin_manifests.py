"""Validate the dual Claude Code + Cursor plugin manifests.

Both manifests must:
  * be valid JSON
  * declare the same ``name``, ``version``, ``repository``, ``license``
  * point at component subdirectories that actually exist under ``plugin/``
  * have ``version`` aligned with ``pyproject.toml`` (single source of truth)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "plugin"
CLAUDE_MANIFEST = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
CURSOR_MANIFEST = PLUGIN_ROOT / ".cursor-plugin" / "plugin.json"
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def claude_manifest() -> dict:
    return json.loads(CLAUDE_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cursor_manifest() -> dict:
    return json.loads(CURSOR_MANIFEST.read_text(encoding="utf-8"))


def test_both_manifests_exist() -> None:
    assert CLAUDE_MANIFEST.exists(), CLAUDE_MANIFEST
    assert CURSOR_MANIFEST.exists(), CURSOR_MANIFEST


def test_required_keys_present(claude_manifest: dict, cursor_manifest: dict) -> None:
    for m, name in ((claude_manifest, "claude"), (cursor_manifest, "cursor")):
        for key in ("name", "version", "description", "author", "license"):
            assert key in m, f"{name} manifest missing key {key!r}"
        assert isinstance(m["author"], dict), f"{name} author must be object"


def test_name_version_repo_match(claude_manifest: dict, cursor_manifest: dict) -> None:
    assert claude_manifest["name"] == cursor_manifest["name"]
    assert claude_manifest["version"] == cursor_manifest["version"]
    assert claude_manifest["repository"] == cursor_manifest["repository"]
    assert claude_manifest["license"] == cursor_manifest["license"]


def test_version_matches_pyproject(claude_manifest: dict) -> None:
    txt = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', txt, re.M)
    assert m, "could not find version in pyproject.toml"
    assert claude_manifest["version"] == m.group(1), (
        f"plugin manifest version {claude_manifest['version']!r} != "
        f"pyproject {m.group(1)!r}"
    )


def test_component_dirs_exist() -> None:
    """Cursor and Claude Code both auto-discover from these subdirs."""
    for sub in ("agents", "commands", "hooks", "mcp", "skills"):
        d = PLUGIN_ROOT / sub
        assert d.is_dir(), f"plugin component dir missing: {d}"
        # at least one entry per subdir (otherwise the manifest's claim of
        # "ships 19 MCP tools, 5 skills, 5 agents, …" is a lie)
        children = [c for c in d.iterdir() if not c.name.startswith(".")
                    and not c.name.startswith("__")]
        assert children, f"plugin component dir is empty: {d}"


def test_cursor_manifest_has_displayname(cursor_manifest: dict) -> None:
    """Cursor manifests conventionally carry a displayName for marketplace UI."""
    assert "displayName" in cursor_manifest
    assert cursor_manifest["displayName"]
