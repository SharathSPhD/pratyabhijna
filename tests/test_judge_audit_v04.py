"""v0.4.1 review fix #D: assert the recovered judge.jsonl is replay-auditable.

Properties under test:
  - every row carries both prompt_sha256 and formatted_prompt_sha256
  - the formatted hash recomputed against the current cascade JSON
    matches the stored hash, i.e. nothing has drifted since recovery
  - prompt_sha256 is the SAME for all rows (it's the template hash);
    formatted_prompt_sha256 is row-specific (different rows produce
    different formatted prompts)
  - judge_agreement.json carries the recovery audit block
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "benchmarks" / "results_v0.4"
JUDGE_JSONL = RESULTS / "judge.jsonl"
JUDGE_AGREEMENT = RESULTS / "judge_agreement.json"
PROMPT_PATH = SCRIPTS / "judge_prompt_v0_4.txt"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from judge_subset import (  # type: ignore  # noqa: E402
    JudgePair,
    _format_judge_prompt,
    _item_prompt,
)


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    with JUDGE_JSONL.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _rebuild_pair(record: dict) -> JudgePair | None:
    domain = record["domain"]
    item_id = record["item_id"]
    path = RESULTS / f"{domain}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or {}
    row = rows.get(item_id)
    if not row:
        return None
    t = row.get(record["treatment_arm"]) or {}
    c = row.get(record["control_arm"]) or {}
    treatment_text = (t.get("text") or "").strip()
    control_text = (c.get("text") or "").strip()
    if not treatment_text or not control_text:
        return None
    proxy_t = float((t.get("scores") or {}).get("composite") or 0.0)
    proxy_c = float((c.get("scores") or {}).get("composite") or 0.0)
    return JudgePair(
        domain=domain,
        item_id=item_id,
        item_prompt=_item_prompt(row.get("item") or {"id": item_id}, domain),
        treatment_arm=record["treatment_arm"],
        control_arm=record["control_arm"],
        treatment_text=treatment_text,
        control_text=control_text,
        treatment_composite=proxy_t,
        control_composite=proxy_c,
        proxy_delta=proxy_t - proxy_c,
        quartile=0,
    )


def test_judge_jsonl_exists_and_nonempty() -> None:
    assert JUDGE_JSONL.exists(), f"missing {JUDGE_JSONL}"
    rows = _load_rows()
    assert rows, "judge.jsonl is empty"
    assert len(rows) >= 20, f"unexpectedly few rows: {len(rows)}"


def test_every_row_has_both_sha_fields() -> None:
    for r in _load_rows():
        assert r.get("prompt_sha256"), f"row missing prompt_sha256: {r['domain']}/{r['item_id']}"
        assert r.get("formatted_prompt_sha256"), (
            f"row missing formatted_prompt_sha256: {r['domain']}/{r['item_id']}"
        )


def test_prompt_sha_is_constant_across_rows() -> None:
    seen = {r["prompt_sha256"] for r in _load_rows()}
    assert len(seen) == 1, f"prompt_sha256 should be a single template hash, got {seen}"


def test_formatted_sha_is_row_specific() -> None:
    rows = _load_rows()
    seen = {r["formatted_prompt_sha256"] for r in rows}
    # 23 rows over different items + position swaps must produce
    # 23 different formatted hashes if the recovery is faithful.
    assert len(seen) == len(rows), (
        f"expected {len(rows)} unique formatted hashes, got {len(seen)}"
    )


def test_recovered_formatted_sha_matches_replay() -> None:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    rows = _load_rows()
    mismatches: list[str] = []
    for r in rows:
        if r.get("formatted_prompt_sha256") == "unrecoverable_v0_4_legacy_run":
            continue
        pair = _rebuild_pair(r)
        if pair is None:
            mismatches.append(f"{r['domain']}/{r['item_id']}: cannot rebuild pair")
            continue
        prompt = _format_judge_prompt(template, pair=pair, swap=bool(r.get("position_swap")))
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if sha != r["formatted_prompt_sha256"]:
            mismatches.append(
                f"{r['domain']}/{r['item_id']}: stored {r['formatted_prompt_sha256'][:12]}... "
                f"replayed {sha[:12]}..."
            )
    assert not mismatches, "formatted_prompt_sha256 drift:\n  " + "\n  ".join(mismatches)


def test_judge_agreement_carries_recovery_audit_block() -> None:
    agree = json.loads(JUDGE_AGREEMENT.read_text(encoding="utf-8"))
    audit = agree.get("formatted_prompt_recovery") or {}
    assert audit.get("version") == "v0.4.1"
    assert audit.get("recovered", 0) >= 20
    index = agree.get("formatted_prompt_sha256_index") or []
    assert len(index) == len(_load_rows())


def test_input_tokens_is_placeholder_contract() -> None:
    """v0.4.2 hardening: ``input_tokens`` in judge.jsonl is a documented placeholder.

    The OAuth ``claude --print`` substrate did not expose per-call token
    counts in the v0.4 pilot. We can either record the counts truthfully
    or remove the field; we chose to keep the placeholder and document
    it. The contract this test enforces:

    1. Every row has an ``input_tokens`` field (so consumers do not need
       to handle a missing key).
    2. Every row reports the **same** placeholder value (otherwise the
       field looks like a real measurement).
    """
    rows = _load_rows()
    assert all("input_tokens" in r for r in rows), (
        "all judge.jsonl rows must carry input_tokens (even if a placeholder)"
    )
    seen = {r["input_tokens"] for r in rows}
    assert len(seen) == 1, (
        f"input_tokens must be a single placeholder across all rows, "
        f"got {len(seen)} distinct values: {seen}"
    )


def test_judge_provenance_keys_present() -> None:
    """v0.4.2 hardening: judge_agreement.json names the provenance of
    its two audit fields so consumers know what was measured vs.
    recovered post-hoc."""
    agree = json.loads(JUDGE_AGREEMENT.read_text(encoding="utf-8"))
    assert agree.get("input_tokens_provenance") == "placeholder_substrate_did_not_record"
    assert agree.get("formatted_prompt_sha256_provenance") == "post_hoc_v0_4_1_recovery"
    note = agree.get("input_tokens_provenance_note") or ""
    assert "placeholder" in note.lower(), (
        "input_tokens_provenance_note must explain the placeholder"
    )
