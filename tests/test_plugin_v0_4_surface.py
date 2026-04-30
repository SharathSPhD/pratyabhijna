"""Phase 6 prove-gate: v0.4 plugin/MCP surface contract.

Asserts the v0.4 plugin manifest version, the new ``judge_pair`` MCP
tool round-trips a pair under the deterministic dry-run responder, and
``_v3_arm_overrides`` accepts the new ``commit_policy="learned_gate"``
option per ADR-002. No Haiku / Sonnet calls — those are exercised by
``smoke_plugin.py --with-haiku`` and the Phase 7 powered pilot.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_MCP = REPO_ROOT / "plugin" / "mcp"
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
for p in (str(SRC), str(PLUGIN_MCP), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import server  # noqa: E402


def test_plugin_manifest_version_is_v0_4_0() -> None:
    """ADR / Phase 6 acceptance: plugin.json + pyproject.toml at 0.4.0."""
    manifest = json.loads(
        (REPO_ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["version"] == "0.4.0"

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.4.0"' in pyproject


def test_v3_arm_overrides_accepts_learned_gate() -> None:
    """Phase 6: cascade MCP tool now exposes ``commit_policy="learned_gate"``."""
    K, cp, brief = server._v3_arm_overrides(
        "haiku_cascade", K=4, commit_policy="learned_gate"
    )
    assert K == 4
    assert cp == "learned_gate"
    assert brief is None


def test_v3_arm_overrides_rejects_unknown_commit_policy() -> None:
    with pytest.raises(ValueError, match="commit_policy"):
        server._v3_arm_overrides(
            "haiku_cascade", K=4, commit_policy="cleverly_made_up"
        )


def test_judge_pair_tool_dry_run_round_trips() -> None:
    """Phase 6 prove-gate: ``judge_pair`` returns a frozen-prompt verdict.

    Uses ``dry_run=True`` so no Sonnet quota is consumed; this is the
    same code path that downstream MCP callers will use to validate
    their wiring before calling the real Sonnet bridge in Phase 7.
    """

    async def _call() -> dict[str, object]:
        result = await server.mcp._tool_manager.call_tool(
            "judge_pair",
            {
                "prompt": "Write one striking line of imagery about autumn.",
                "text_a": (
                    "the maples burn slow as bronze coins falling through dusk"
                ),
                "text_b": "leaves are red.",
                "dry_run": True,
            },
            convert_result=False,
        )
        return result if isinstance(result, dict) else result.model_dump()

    out = asyncio.run(_call())
    assert out["winner"] == "A", "dry-run picks longer block; treatment is longer"
    assert out["model"] == "fake-responder"
    assert isinstance(out["prompt_sha256"], str)
    assert len(out["prompt_sha256"]) == 64
    assert out["error"] is False
    assert isinstance(out["cost_usd"], float)
    assert out["cost_usd"] >= 0.0
    assert isinstance(out["rationale"], str)
    assert "dry-run" in out["rationale"]


def test_judge_pair_tool_dry_run_swap_invariant() -> None:
    """Swapping A and B should flip the verdict under the deterministic responder."""

    async def _call(text_a: str, text_b: str) -> dict[str, object]:
        result = await server.mcp._tool_manager.call_tool(
            "judge_pair",
            {
                "prompt": "Write one striking line.",
                "text_a": text_a,
                "text_b": text_b,
                "dry_run": True,
            },
            convert_result=False,
        )
        return result if isinstance(result, dict) else result.model_dump()

    long_first = asyncio.run(_call("long " * 30, "short"))
    short_first = asyncio.run(_call("short", "long " * 30))
    assert long_first["winner"] == "A"
    assert short_first["winner"] == "B"


def test_pce_cascade_tool_accepts_learned_gate_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP ``pce_cascade`` tool's signature includes ``commit_policy``
    and accepts ``"learned_gate"`` without raising at the validation layer.

    We don't run a full cascade here (that would require Haiku); we only
    assert that the override path resolves correctly so the live
    ``--with-haiku`` smoke probe is the only place real LLM calls happen.
    """
    K, cp, brief = server._v3_arm_overrides(
        "haiku_cascade", K=2, commit_policy="learned_gate"
    )
    assert (K, cp, brief) == (2, "learned_gate", None)
    K2, cp2, brief2 = server._v3_arm_overrides(
        "haiku_cascade", K=2, commit_policy=None
    )
    assert (K2, cp2, brief2) == (2, "event_gated", None)
