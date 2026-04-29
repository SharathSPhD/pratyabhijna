"""Tests for `pce.substrate.integrity.IntegrityProbe`."""
from __future__ import annotations

import subprocess
from typing import Any
from unittest import mock

import numpy as np
import pytest

from pce.substrate import haiku_lm as _haiku_module
from pce.substrate.embed import Embedder
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM
from pce.substrate.integrity import (
    LEAKAGE_REGEX,
    POSITIVE_HINT_REGEX,
    PROBE_PROMPT,
    IntegrityProbe,
)


class _FakeEmbed(Embedder):
    def __init__(self) -> None:
        self.model_id = "fake"
        self.dim = 4

    def encode(self, texts):  # type: ignore[no-untyped-def, override]
        if isinstance(texts, str):
            return np.zeros((4,), dtype=np.float32)
        return np.zeros((len(texts), 4), dtype=np.float32)


def _payload(text: str) -> dict[str, Any]:
    return {
        "type": "result",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "duration_ms": 12,
        "session_id": "probe",
        "total_cost_usd": 0.001,
        "modelUsage": {"claude-haiku-4-5-20251001": {"costUSD": 0.001}},
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }


def _fake_proc(payload: dict[str, Any]) -> mock.MagicMock:
    import json
    proc = mock.MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = 0
    proc.stdout = json.dumps(payload).encode("utf-8")
    proc.stderr = b""
    return proc


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(_haiku_module, "AUDIT_DIR", tmp_path / "haiku")
    monkeypatch.setattr(_haiku_module, "COST_LEDGER", tmp_path / "cost.json")


def test_leakage_regex_matches_known_leaks() -> None:
    examples = [
        "I am Claude Code with the pce skill loaded.",
        "I appreciate the opportunity to help with the loaded plugin.",
        "I have access to MCP tools.",
        "Per CLAUDE.md guidelines.",
        "Following the cursor rule for file edits.",
    ]
    for ex in examples:
        assert LEAKAGE_REGEX.search(ex), f"expected leakage match: {ex!r}"


def test_positive_hint_regex_matches_clean_responses() -> None:
    cleans = [
        "I have no plugins or skills currently loaded.",
        "None — I'm a standard assistant.",
        "Nothing is loaded; I am answering as a base model.",
        "I don't have any system instructions.",
    ]
    for c in cleans:
        assert POSITIVE_HINT_REGEX.search(c), f"expected positive hint: {c!r}"


def test_probe_pass_on_clean_response() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    clean_text = "I have no plugins or skills loaded; I am a standard assistant."
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(clean_text))):
        probe = IntegrityProbe()
        result = probe.run(lm)
    assert result.passed is True
    assert result.leak_matches == []
    assert result.positive_hint is True
    assert result.env_hash and result.flags_hash


def test_probe_fail_on_leakage_response() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    leaky_text = "I appreciate the chance to help with this Claude Code session."
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(leaky_text))):
        probe = IntegrityProbe()
        result = probe.run(lm)
    assert result.passed is False
    assert any("appreciate" in m.lower() or "claude code" in m.lower() for m in result.leak_matches)


def test_probe_caches_by_env_and_flags() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    n_calls = {"value": 0}

    def _capture(*args, **kwargs):
        n_calls["value"] += 1
        return _fake_proc(_payload("nothing loaded"))

    with mock.patch("subprocess.run", side_effect=_capture):
        probe = IntegrityProbe()
        r1 = probe.run(lm)
        r2 = probe.run(lm)
    assert r1.env_hash == r2.env_hash
    assert n_calls["value"] == 1, "second probe should hit cache, not re-call subprocess"


def test_probe_force_bypasses_cache() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload("none loaded"))):
        probe = IntegrityProbe()
        probe.run(lm)
        n_before = sum(1 for _ in subprocess.run.mock_calls)  # type: ignore[attr-defined]
        probe.run(lm, force=True)
        n_after = sum(1 for _ in subprocess.run.mock_calls)  # type: ignore[attr-defined]
    assert n_after > n_before


def test_probe_prompt_is_frozen_constant() -> None:
    assert "PROBE:" in PROBE_PROMPT
    assert "plugins" in PROBE_PROMPT
    assert "skills" in PROBE_PROMPT
