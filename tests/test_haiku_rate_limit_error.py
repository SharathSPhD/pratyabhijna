"""v0.4 Phase 2 (ADR-006) gate: typed Haiku error hierarchy.

Verifies that ``HaikuLM._call_cli_once`` parses ``claude --print`` stdout JSON
even when ``rc != 0`` and raises typed exceptions:

* ``api_error_status == 429`` -> ``HaikuRateLimitError``
* ``is_error == True`` with non-429 status -> ``HaikuApiError``
* ``rc != 0`` with no parseable JSON body -> ``HaikuCLIError``

The v0.3 path collapsed all three into a generic ``RuntimeError`` and threw
away the JSON body on stdout; smoke summaries could not distinguish quota
exhaustion from real bugs. v0.4 makes the distinction observable.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from pce.substrate.errors import (
    HaikuApiError,
    HaikuCLIError,
    HaikuRateLimitError,
)
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM


def _patched_subprocess_run(stdout: bytes, stderr: bytes, returncode: int) -> Any:
    """Build a fake subprocess.run replacement that returns canned bytes."""

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def _fake_run(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return _FakeProc()

    return _fake_run


@pytest.fixture()
def haiku() -> HaikuLM:
    cfg = HaikuConfig(
        clean_substrate=False,  # avoid scrubbed-HOME setup in unit tests
        cli_retry=0,
    )
    return HaikuLM(config=cfg)


def test_429_stdout_raises_rate_limit_error(haiku: HaikuLM) -> None:
    """Synthetic 429 stdout body raises HaikuRateLimitError with the parsed body."""
    body = {
        "is_error": True,
        "api_error_status": 429,
        "result": "You're out of extra usage. Resets at 3pm.",
        "total_cost_usd": 0,
    }
    with patch(
        "pce.substrate.haiku_lm.subprocess.run",
        new=_patched_subprocess_run(
            stdout=json.dumps(body).encode("utf-8"),
            stderr=b"",
            returncode=1,
        ),
    ), pytest.raises(HaikuRateLimitError) as exc_info:
        haiku._call_cli_once("hello")
    err = exc_info.value
    assert err.api_error_status() == 429
    assert "extra usage" in err.result()
    assert err.parsed["is_error"] is True


def test_non_429_api_error_raises_api_error(haiku: HaikuLM) -> None:
    body = {
        "is_error": True,
        "api_error_status": 500,
        "result": "internal server error",
    }
    with patch(
        "pce.substrate.haiku_lm.subprocess.run",
        new=_patched_subprocess_run(
            stdout=json.dumps(body).encode("utf-8"),
            stderr=b"",
            returncode=1,
        ),
    ), pytest.raises(HaikuApiError) as exc_info:
        haiku._call_cli_once("hello")
    err = exc_info.value
    assert not isinstance(err, HaikuRateLimitError)
    assert err.api_error_status() == 500
    assert "internal" in err.result()


def test_non_zero_rc_with_no_body_raises_cli_error(haiku: HaikuLM) -> None:
    """rc != 0 with empty stdout (no JSON body) raises HaikuCLIError."""
    with patch(
        "pce.substrate.haiku_lm.subprocess.run",
        new=_patched_subprocess_run(
            stdout=b"",
            stderr=b"some unrelated stderr noise",
            returncode=2,
        ),
    ), pytest.raises(HaikuCLIError) as exc_info:
        haiku._call_cli_once("hello")
    err = exc_info.value
    assert "rc=2" in str(err)
    # The HaikuCLIError carries stderr/stdout for diagnostic purposes.
    assert "stderr" in err.parsed


def test_successful_response_returns_text(haiku: HaikuLM) -> None:
    """Sanity: the fast path still works after the v0.4 rewrite."""
    body = {
        "is_error": False,
        "result": "hi there",
        "total_cost_usd": 0.001,
        "duration_ms": 42,
        "stop_reason": "stop",
        "session_id": "x",
        "modelUsage": {"haiku": {}},
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    with patch(
        "pce.substrate.haiku_lm.subprocess.run",
        new=_patched_subprocess_run(
            stdout=json.dumps(body).encode("utf-8"),
            stderr=b"",
            returncode=0,
        ),
    ):
        text, meta = haiku._call_cli_once("hello")
    assert text == "hi there"
    assert meta["cost_usd"] == pytest.approx(0.001)
    assert meta["output_tokens"] == 2


def test_rate_limit_propagates_through_call_cli_no_retry(haiku: HaikuLM) -> None:
    """``_call_cli`` must NOT retry rate-limit errors -- they propagate immediately."""
    body = {
        "is_error": True,
        "api_error_status": 429,
        "result": "rate limited",
    }
    call_count = {"n": 0}

    def _counting_run(*_args: Any, **_kwargs: Any) -> Any:
        call_count["n"] += 1
        return _patched_subprocess_run(
            stdout=json.dumps(body).encode("utf-8"),
            stderr=b"",
            returncode=1,
        )()

    with patch("pce.substrate.haiku_lm.subprocess.run", new=_counting_run), pytest.raises(HaikuRateLimitError):
        haiku._call_cli("hello")
    assert call_count["n"] == 1, "rate-limit must not trigger retry"
