"""HaikuLM unit tests: cost ledger, audit logs, budget enforcement.

These tests stub out the subprocess call so they run without invoking the
real `claude` CLI. A separate manual probe in Phase 2 (`scripts/haiku_one_shot.py`)
calls the real CLI once.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from pce.substrate import haiku_lm as _haiku_module
from pce.substrate.embed import Embedder
from pce.substrate.haiku_lm import (
    HaikuBudgetExceededError,
    HaikuConfig,
    HaikuLM,
)


class _FakeEmbed(Embedder):
    """Stand-in embedder so we don't load sentence-transformers in fast tests."""

    def __init__(self) -> None:
        self.model_id = "fake-embedder"
        self.dim = 4

    def encode(self, texts):  # type: ignore[no-untyped-def, override]
        if isinstance(texts, str):
            return np.zeros((4,), dtype=np.float32)
        return np.zeros((len(texts), 4), dtype=np.float32)


def _fake_proc(payload: dict[str, Any], rc: int = 0) -> mock.MagicMock:
    proc = mock.MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = rc
    proc.stdout = json.dumps(payload).encode("utf-8")
    proc.stderr = b""
    return proc


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect audit + ledger to a tmp dir so tests don't pollute the repo."""
    monkeypatch.setattr(_haiku_module, "AUDIT_DIR", tmp_path / "haiku")
    monkeypatch.setattr(_haiku_module, "COST_LEDGER", tmp_path / "cost_ledger.json")


def _payload(text: str = "hello world", cost: float = 0.001, in_tok: int = 5, out_tok: int = 2) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "duration_ms": 1234,
        "session_id": "test-session",
        "total_cost_usd": cost,
        "modelUsage": {"claude-haiku-4-5-20251001": {"costUSD": cost}},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def test_generate_returns_candidate_with_text() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload("the duck-rabbit shifts"))):
        cand = lm.generate("Interpret the duck-rabbit", max_tokens=64, sampler={"tau": 0.9}, seed=0)
    assert cand.text == "the duck-rabbit shifts"
    assert cand.embedding.shape == (4,)
    assert cand.sampler["tau"] == 0.9
    assert cand.tokens == ()


def test_generate_records_cost_in_ledger() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(cost=0.05))):
        lm.generate("ping", max_tokens=8, sampler={"tau": 0.9}, seed=1)
    raw = json.loads(_haiku_module.COST_LEDGER.read_text(encoding="utf-8"))
    assert abs(float(raw["total_usd"]) - 0.05) < 1e-9
    assert int(raw["n_calls"]) == 1


def test_two_calls_accumulate_cost() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(cost=0.02))):
        lm.generate("a", max_tokens=8, sampler={"tau": 0.9}, seed=0)
        lm.generate("b", max_tokens=8, sampler={"tau": 0.9}, seed=1)
    raw = json.loads(_haiku_module.COST_LEDGER.read_text(encoding="utf-8"))
    assert abs(float(raw["total_usd"]) - 0.04) < 1e-9
    assert int(raw["n_calls"]) == 2


def test_budget_cap_raises_before_call() -> None:
    lm = HaikuLM(
        config=HaikuConfig(cli_bin="claude", cost_cap_usd=0.01), embedder=_FakeEmbed()
    )
    _haiku_module.COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    _haiku_module.COST_LEDGER.write_text(
        json.dumps({"total_usd": 0.5, "n_calls": 1, "by_model": {}})
    )
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload())), \
         pytest.raises(HaikuBudgetExceededError):
        lm.generate("anything", max_tokens=8, sampler={"tau": 0.9}, seed=0)


def test_cli_nonzero_rc_raises() -> None:
    """v0.4 (ADR-006): rc != 0 with no JSON body raises HaikuCLIError."""
    from pce.substrate.errors import HaikuCLIError

    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    bad = mock.MagicMock(spec=subprocess.CompletedProcess)
    bad.returncode = 2
    bad.stdout = b""
    bad.stderr = b"unexpected cli failure"
    with mock.patch("subprocess.run", return_value=bad), \
         pytest.raises(HaikuCLIError, match="rc=2"):
        lm.generate("x", max_tokens=4, sampler={"tau": 0.9}, seed=0)


def test_cli_is_error_payload_raises() -> None:
    """v0.4 (ADR-006): is_error=True with non-429 status raises HaikuApiError."""
    from pce.substrate.errors import HaikuApiError

    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    bad_payload = _payload()
    bad_payload["is_error"] = True
    bad_payload["api_error_status"] = 500
    bad_payload["result"] = "internal server error"
    with mock.patch("subprocess.run", return_value=_fake_proc(bad_payload, rc=1)), \
         pytest.raises(HaikuApiError):
        lm.generate("x", max_tokens=4, sampler={"tau": 0.9}, seed=0)


def test_seed_changes_prompt_prefix() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    seen: list[list[str]] = []

    def _capture(*args: Any, **kwargs: Any) -> Any:
        seen.append(list(args[0]) if args else [])
        return _fake_proc(_payload())

    with mock.patch("subprocess.run", side_effect=_capture):
        lm.generate("same prompt", max_tokens=4, sampler={"tau": 0.9}, seed=11)
        lm.generate("same prompt", max_tokens=4, sampler={"tau": 0.9}, seed=42)
    p11 = seen[0][-1]
    p42 = seen[1][-1]
    assert p11 != p42


def test_audit_call_writes_a_file() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(text="hi"))):
        lm.generate("audit me", max_tokens=4, sampler={"tau": 0.9}, seed=0)
    files = list(_haiku_module.AUDIT_DIR.glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    assert rec["model"]
    assert rec["response"] == "hi"
    assert "ts" in rec


def test_report_includes_ledger_total() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    with mock.patch("subprocess.run", return_value=_fake_proc(_payload(cost=0.03))):
        lm.generate("x", max_tokens=4, sampler={"tau": 0.9}, seed=0)
    rep = lm.report()
    assert rep["name"] == "claude-haiku"
    assert abs(float(rep["ledger_total_usd"]) - 0.03) < 1e-9
    assert int(rep["ledger_n_calls"]) == 1


# --- v0.3 clean substrate tests ---------------------------------------------


def test_clean_substrate_uses_isolation_flags_in_cmd() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    captured: list[list[str]] = []

    def _capture(*args, **kwargs):
        captured.append(list(args[0]) if args else [])
        return _fake_proc(_payload())

    with mock.patch("subprocess.run", side_effect=_capture):
        lm.generate("hi", max_tokens=8, sampler={"tau": 0.9}, seed=0)
    cmd = captured[0]
    for flag in (
        "--print",
        "--system-prompt",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--setting-sources",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
    ):
        assert flag in cmd, f"clean substrate cmd missing required flag: {flag}"


def test_clean_substrate_uses_explicit_env_and_cwd() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    captured: list[dict[str, Any]] = []

    def _capture(*args, **kwargs):
        captured.append(dict(kwargs))
        return _fake_proc(_payload())

    with mock.patch("subprocess.run", side_effect=_capture):
        lm.generate("hi", max_tokens=8, sampler={"tau": 0.9}, seed=0)
    kw = captured[0]
    assert "env" in kw, "clean substrate must pass an explicit env to subprocess.run"
    env = kw["env"]
    assert "PATH" in env
    assert env.get("HOME", "").startswith("/"), f"HOME must be set, got {env.get('HOME')!r}"
    assert env["HOME"] != os.environ.get("HOME", ""), (
        "clean HOME must differ from parent HOME"
    )
    # No Claude Code env vars should leak in.
    for k in env:
        assert not k.startswith("CLAUDE_CODE_"), f"clean env leaked Claude Code var: {k}"
    assert "cwd" in kw and kw["cwd"]
    assert kw["cwd"] != os.getcwd(), "clean substrate must use a temp cwd, not the repo"


def test_clean_substrate_disabled_falls_back_to_v02_path() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=False)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    captured_kwargs: list[dict[str, Any]] = []

    def _capture(*args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return _fake_proc(_payload())

    with mock.patch("subprocess.run", side_effect=_capture):
        lm.generate("hi", max_tokens=8, sampler={"tau": 0.9}, seed=0)
    kw = captured_kwargs[0]
    # When clean_substrate is off we don't pass env/cwd — subprocess inherits parent env.
    assert "env" not in kw
    assert "cwd" not in kw


def test_clean_substrate_env_never_inherits_blindly() -> None:
    """Regression guard: we must NEVER pass env=os.environ.copy() to subprocess.run."""
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    captured: list[dict[str, Any]] = []

    def _capture(*args, **kwargs):
        captured.append(dict(kwargs))
        return _fake_proc(_payload())

    with mock.patch("subprocess.run", side_effect=_capture), \
         mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "test", "MY_SECRET": "leak-me"}):
        lm.generate("hi", max_tokens=8, sampler={"tau": 0.9}, seed=0)
    env = captured[0]["env"]
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert "MY_SECRET" not in env


def test_capability_flags_present() -> None:
    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    assert lm.supports_logprobs is False
    assert lm.supports_score is False
    assert lm.supports_entropy is False


def test_length_proxy_logp_is_monotone_in_length() -> None:
    from pce.types import Candidate

    lm = HaikuLM(config=HaikuConfig(cli_bin="claude"), embedder=_FakeEmbed())
    short = Candidate(seed=0, sampler={"tau": 0.9}, tokens=(), text="hi", logp=0.0, embedding=np.zeros(4, dtype=np.float32))
    long = Candidate(seed=0, sampler={"tau": 0.9}, tokens=(), text="hi" * 200, logp=0.0, embedding=np.zeros(4, dtype=np.float32))
    assert lm.length_proxy_logp(long) < lm.length_proxy_logp(short)


def test_report_exposes_clean_substrate_state() -> None:
    cfg = HaikuConfig(cli_bin="claude", clean_substrate=True)
    lm = HaikuLM(config=cfg, embedder=_FakeEmbed())
    rep = lm.report()
    assert rep["clean_substrate"] is True
    assert rep["clean_home"] is not None
    assert rep["clean_cwd"] is not None
    assert rep["supports_logprobs"] is False
    assert "isolation_flags" in rep
    assert "--disable-slash-commands" in rep["isolation_flags"]
