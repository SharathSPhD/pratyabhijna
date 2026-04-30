"""Phase 5 (ADR-004) gate: Sonnet judge bridge dry-run.

Exercises ``scripts/judge_subset.py`` end-to-end with the deterministic
fake responder so the pipeline (stratification, frozen-prompt sha256,
JSONL row schema, H9.v4 aggregate) is verified without spending Sonnet
quota. Real Sonnet integration is gated on a $5 cost cap and a healthy
``--dry-run`` first.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for _p in (str(SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.judge_subset import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    INPUT_TOKEN_ESTIMATE,
    OUTPUT_TOKEN_ESTIMATE,
    SONNET_INPUT_USD_PER_TOK,
    SONNET_OUTPUT_USD_PER_TOK,
    JudgePair,
    _aggregate_h9,
    _fake_responder,
    _format_judge_prompt,
    _load_prompt_template,
    _project_cost,
    _resolve_winner,
    _stratify_pairs,
)


def _make_results_dir(tmp_path: Path, *, n_per_domain: int = 4) -> Path:
    """Build a synthetic ``results_v0.4`` tree with n_per_domain items per domain."""
    domains = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
    out = tmp_path / "results_v0.4"
    out.mkdir(parents=True, exist_ok=True)
    for dom in domains:
        rows = {}
        for i in range(n_per_domain):
            row_id = f"{dom}_{i}"
            # Treatment text grows with i so the deterministic fake responder
            # picks treatment more often as i grows. Composite mirrors length.
            t_text = "treatment" + " . " * (i + 1)
            c_text = "control" + " . " * 1
            rows[row_id] = {
                "item": {
                    "id": row_id,
                    "line": f"a line of poetry {i}",
                    "topic": f"topic {i}",
                    "object": f"object {i}",
                    "form": "haiku",
                    "prompt": f"sci probe {i}",
                },
                "haiku_cascade": {
                    "text": t_text,
                    "composite": 0.5 + 0.05 * i,
                    "n_words": len(t_text.split()),
                    "meta": {"committed": "draft"},
                },
                "haiku_bare": {
                    "text": c_text,
                    "composite": 0.5,
                    "n_words": len(c_text.split()),
                    "meta": {},
                },
            }
        payload = {"domain": dom, "version": "v0.4", "rows": rows}
        (out / f"{dom}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return out


def test_frozen_prompt_exists_and_is_versioned() -> None:
    """ADR-004 acceptance: prompt file exists and has a stable sha256."""
    assert DEFAULT_PROMPT_PATH.exists()
    raw = DEFAULT_PROMPT_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert len(sha) == 64
    assert b"__PROMPT__" in raw
    assert b"__TEXT_A__" in raw
    assert b"__TEXT_B__" in raw
    assert b"STRICT JSON" in raw


def test_load_prompt_template_returns_text_and_hash() -> None:
    text, sha = _load_prompt_template(DEFAULT_PROMPT_PATH)
    assert text
    assert isinstance(sha, str) and len(sha) == 64


def test_format_judge_prompt_swap_orientation() -> None:
    template = "P:__PROMPT__\nA:__TEXT_A__\nB:__TEXT_B__\n"
    pair = JudgePair(
        domain="poetry_gen",
        item_id="x",
        item_prompt="prompt-text",
        treatment_arm="haiku_cascade",
        control_arm="haiku_bare",
        treatment_text="TREAT-TEXT",
        control_text="CONTROL-TEXT",
        treatment_composite=0.7,
        control_composite=0.3,
        proxy_delta=0.4,
        quartile=2,
    )
    no_swap = _format_judge_prompt(template, pair=pair, swap=False)
    swapped = _format_judge_prompt(template, pair=pair, swap=True)
    assert "A:TREAT-TEXT" in no_swap
    assert "B:CONTROL-TEXT" in no_swap
    assert "A:CONTROL-TEXT" in swapped
    assert "B:TREAT-TEXT" in swapped


def test_resolve_winner_swap_inversion() -> None:
    assert _resolve_winner(raw="A", swap=False) == "treatment"
    assert _resolve_winner(raw="B", swap=False) == "control"
    assert _resolve_winner(raw="A", swap=True) == "control"
    assert _resolve_winner(raw="B", swap=True) == "treatment"
    assert _resolve_winner(raw="tie", swap=False) == "tie"
    assert _resolve_winner(raw="tie", swap=True) == "tie"
    assert _resolve_winner(raw="garbage", swap=False) == "tie"


def test_fake_responder_picks_longer_block() -> None:
    """Dry-run responder is deterministic: longer block wins."""
    prompt_long_a = (
        "<prompt>x</prompt>\n<A>" + "a" * 200 + "</A>\n<B>" + "b" * 50 + "</B>"
    )
    out = _fake_responder(prompt_long_a)
    assert out["winner"] == "A"
    prompt_long_b = (
        "<prompt>x</prompt>\n<A>" + "a" * 50 + "</A>\n<B>" + "b" * 200 + "</B>"
    )
    out = _fake_responder(prompt_long_b)
    assert out["winner"] == "B"
    prompt_eq = "<prompt>x</prompt>\n<A>aaa</A>\n<B>aaa</B>"
    out = _fake_responder(prompt_eq)
    assert out["winner"] == "tie"


def test_fake_responder_with_real_frozen_prompt_picks_longer_block() -> None:
    """Regression: the real frozen prompt mentions ``<A>...</A>`` in its
    explanatory prose; the fake responder must still find the actual
    response blocks (last occurrence) and pick the longer one.
    """
    template, _ = _load_prompt_template(DEFAULT_PROMPT_PATH)
    pair = JudgePair(
        domain="poetry_gen",
        item_id="x",
        item_prompt="Write one striking line of imagery about autumn.",
        treatment_arm="A",
        control_arm="B",
        treatment_text="the maples burn slow as bronze coins falling through dusk",
        control_text="leaves are red.",
        treatment_composite=0.0,
        control_composite=0.0,
        proxy_delta=0.0,
        quartile=-1,
    )
    formatted = _format_judge_prompt(template, pair=pair, swap=False)
    out = _fake_responder(formatted)
    assert out["winner"] == "A", "long treatment text should win when not swapped"
    formatted_swapped = _format_judge_prompt(template, pair=pair, swap=True)
    out = _fake_responder(formatted_swapped)
    assert out["winner"] == "B", "swapped: treatment text now occupies B"


def test_stratify_pairs_per_domain_count(tmp_path: Path) -> None:
    """Stratifier picks exactly ``n_per_domain`` items per domain when available."""
    import random

    pairs = []
    for dom in ("poetry_gen", "poetry_interp", "aut", "sci_creativity"):
        for i in range(8):
            pairs.append(
                JudgePair(
                    domain=dom,
                    item_id=f"{dom}_{i}",
                    item_prompt="x",
                    treatment_arm="t",
                    control_arm="c",
                    treatment_text="T",
                    control_text="C",
                    treatment_composite=float(i),
                    control_composite=0.0,
                    proxy_delta=float(i),
                    quartile=-1,
                )
            )
    rng = random.Random(123)
    selected = _stratify_pairs(pairs, n_per_domain=4, rng=rng)
    by_dom: dict[str, int] = {}
    for p in selected:
        by_dom[p.domain] = by_dom.get(p.domain, 0) + 1
    for dom in ("poetry_gen", "poetry_interp", "aut", "sci_creativity"):
        assert by_dom[dom] == 4
    quartiles = {p.quartile for p in selected}
    assert quartiles == {0, 1, 2, 3}, f"missing quartile coverage: {quartiles}"


def test_project_cost_under_five_dollars_for_default_subset() -> None:
    """ADR-004 cost-cap check: 32 items × Sonnet stays well under $5."""
    cost = _project_cost(32)
    assert cost < 5.0, f"projected cost {cost} blows the $5 cap"
    # Sanity: a 1-pair cost matches the per-token math.
    one = (
        INPUT_TOKEN_ESTIMATE * SONNET_INPUT_USD_PER_TOK
        + OUTPUT_TOKEN_ESTIMATE * SONNET_OUTPUT_USD_PER_TOK
    )
    assert _project_cost(1) == pytest.approx(one)


def test_aggregate_h9_handles_empty() -> None:
    out = _aggregate_h9([])
    assert out["status"] == "empty"
    assert out["n"] == 0
    assert out["supported"] is False


def test_dryrun_cli_writes_jsonl_and_agreement(tmp_path: Path) -> None:
    """End-to-end dry-run: CLI invocation produces valid jsonl + agreement.

    This is the ADR-004 Phase 5 gate: 4-item dry-run must produce a
    valid ``judge_agreement.json`` and a JSONL with one row per item.
    """
    results = _make_results_dir(tmp_path, n_per_domain=4)
    out_jsonl = tmp_path / "judge.jsonl"
    out_agreement = tmp_path / "judge_agreement.json"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "judge_subset.py"),
            "--dry-run",
            "--results-dir", str(results),
            "--out-jsonl", str(out_jsonl),
            "--out-agreement", str(out_agreement),
            "--n-per-domain", "1",  # 1 per domain × 4 domains = 4 items
            "--seed", "4242",
            "--cost-cap-usd", "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0, res.stderr
    assert out_jsonl.exists()
    assert out_agreement.exists()

    # JSONL has exactly 4 rows.
    lines = [
        line for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 4
    for line in lines:
        row = json.loads(line)
        for k in (
            "domain", "item_id", "treatment_arm", "control_arm", "position_swap",
            "proxy_delta", "judge_delta", "winner_raw", "winner_resolved",
            "confidence", "rationale", "quartile", "prompt_sha256", "model",
            "input_tokens", "output_tokens", "cost_usd", "elapsed_s",
        ):
            assert k in row, f"missing key {k!r} in {row}"
        assert row["winner_raw"] in ("A", "B", "tie")
        assert row["winner_resolved"] in ("treatment", "control", "tie")
        assert row["model"] == "fake-responder"
        assert len(row["prompt_sha256"]) == 64
        assert row["cost_usd"] >= 0.0

    agreement = json.loads(out_agreement.read_text(encoding="utf-8"))
    assert agreement["status"] == "ok"
    assert agreement["n"] == 4
    assert agreement["model"] == "fake-responder"
    assert len(agreement["prompt_sha256"]) == 64
    assert agreement["total_cost_usd"] >= 0.0
    assert "sign_agreement_rate" in agreement
    assert "by_quartile" in agreement
    # Cost stays trivially under cap (the dry-run fake responder uses
    # the token estimate only).
    assert agreement["total_cost_usd"] < 1.0


def test_dryrun_cli_aborts_when_no_results(tmp_path: Path) -> None:
    """Empty results dir -> CLI exits with non-zero (rc=2)."""
    out_jsonl = tmp_path / "j.jsonl"
    out_agreement = tmp_path / "a.json"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "judge_subset.py"),
            "--dry-run",
            "--results-dir", str(tmp_path / "missing"),
            "--out-jsonl", str(out_jsonl),
            "--out-agreement", str(out_agreement),
            "--n-per-domain", "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 2, res.stderr


def test_dryrun_cli_rejects_oversized_subset(tmp_path: Path) -> None:
    """Cost-cap guard: when projected cost exceeds the cap, CLI returns 3."""
    results = _make_results_dir(tmp_path, n_per_domain=8)
    out_jsonl = tmp_path / "j.jsonl"
    out_agreement = tmp_path / "a.json"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "judge_subset.py"),
            "--dry-run",
            "--results-dir", str(results),
            "--out-jsonl", str(out_jsonl),
            "--out-agreement", str(out_agreement),
            "--n-per-domain", "8",
            "--cost-cap-usd", "0.0001",  # absurdly low to force rejection
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 3, res.stderr
    assert "exceeds cap" in res.stderr or "exceeds cap" in res.stdout
