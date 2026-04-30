"""Smoke tests for the v0.4 autoreport / placeholder pack.

Asserts:
  * ``_build_v04_placeholders`` returns every key the rewritten paper sections
    expect to bind (and never raw ``None``).
  * ``_v04_headline`` produces a single-paragraph string mentioning the
    Phase 7 anchor numbers (pooled g + supported H8a if applicable).
  * ``_enforce_no_unbound_v04`` correctly flags missing tokens.
  * The CLI ``main()`` path with ``--version v0.4`` writes
    ``autoreport_v0.4.tex`` and binds tokens in the in-paper sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import autoreport as A

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results_v0.4"
STATS = RESULTS / "stats.json"
LEDGER = REPO_ROOT / "audit" / "v0.4" / "cost_ledger_merged.json"


def _has_artefacts() -> bool:
    return STATS.exists() and LEDGER.exists()


pytestmark = pytest.mark.skipif(
    not _has_artefacts(),
    reason="v0.4 Phase 7 result tree absent",
)

EXPECTED_KEYS = {
    "V04_HEADLINE",
    "V04_COST_TOTAL_USD",
    "V04_COST_N_CALLS",
    "V04_H5_POOLED_G",
    "V04_H5_CI_LO",
    "V04_H5_CI_HI",
    "V04_H5_METHOD",
    "V04_H8A_G",
    "V04_H8A_P",
    "V04_H8A_N",
    "V04_H8B_LEARNED_F1",
    "V04_H8B_EVENT_F1",
    "V04_H8B_SUPPORTED",
    "V04_H8C_WINNER",
    "V04_H8C_SUPPORTED",
    "V04_H9_RHO",
    "V04_H9_SIGN_AGREEMENT",
    "V04_H9_N",
    "V04_JUDGE_COST",
    "V04_N_PER_DOMAIN",
    "V04_N_PAIRED_TOTAL",
    "V04_SHOWCASE_TOTAL",
    "V04_SHOWCASE_SANSKRIT",
    "V04_SHOWCASE_ENGLISH",
    "V04_SHOWCASE_SCIENCE",
}


def test_build_v04_placeholders_complete() -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ja_path = RESULTS / "judge_agreement.json"
    ja = json.loads(ja_path.read_text(encoding="utf-8")) if ja_path.exists() else None
    sc = A._v04_showcase_count(REPO_ROOT / "benchmarks" / "showcase_v0.4")
    out = A._build_v04_placeholders(
        stats, cost_ledger=ledger, judge_agreement=ja, showcase_counts=sc,
    )
    missing = EXPECTED_KEYS - set(out.keys())
    assert not missing, f"missing v04 placeholder keys: {sorted(missing)}"
    for k, v in out.items():
        assert isinstance(v, str), f"{k} -> {type(v).__name__}"
        assert v != "", f"{k} -> empty string"
        assert v != "None", f"{k} -> 'None' string (likely a None.value bug)"


def test_v04_headline_mentions_pool_and_h8a() -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    out = A._v04_headline(stats)
    assert "pool" in out or "Pool" in out, "headline should mention the FE pool"
    assert "H8a" in out, "headline should mention H8a"
    assert "H8c" in out or "H8c leaderboard" in out
    assert "v0.4" in out or "PCE" in out


def test_unbound_v04_token_detected() -> None:
    bad = "Some prose with {V04_HEADLINE} bound and {V04_NEVER_DEFINED} dangling."
    bound = bad.replace("{V04_HEADLINE}", "BOUND")
    remaining = A._enforce_no_unbound_v04(bound)
    assert remaining == ["{V04_NEVER_DEFINED}"]


def test_v04_main_writes_autoreport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}{V04_HEADLINE} {V04_COST_TOTAL_USD}\\end{document}",
        encoding="utf-8",
    )
    sections = paper_dir / "sections"
    sections.mkdir()
    (sections / "09_results.tex").write_text(
        "% v0.4 results: pooled {V04_H5_POOLED_G} CI [{V04_H5_CI_LO}, {V04_H5_CI_HI}].",
        encoding="utf-8",
    )

    argv = [
        "autoreport",
        "--version", "v0.4",
        "--paper-dir", str(paper_dir),
        "--strict",
    ]
    monkeypatch.setattr("sys.argv", argv)
    rc = A.main()
    assert rc == 0
    assert (paper_dir / "autoreport_v0.4.tex").exists()
    main_text = (paper_dir / "main.tex").read_text(encoding="utf-8")
    sec_text = (sections / "09_results.tex").read_text(encoding="utf-8")
    assert "{V04_HEADLINE}" not in main_text
    assert "{V04_H5_POOLED_G}" not in sec_text
