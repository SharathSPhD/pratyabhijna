#!/usr/bin/env python3
"""v0.4.1 review fix #D: backfill ``formatted_prompt_sha256`` into the
existing ``benchmarks/results_v0.4/judge.jsonl``.

The original Phase 7 judge run (``scripts/judge_subset.py``) recorded
only ``prompt_sha256``, the hash of the *template*. That field is
identical for every row, so it cannot detect substitution drift in the
two cascade outputs that were fed to the judge for that row. The v0.4.1
review flagged this as a replay-audit gap.

This script reconstructs the *formatted* prompt for every row by
replaying ``_format_judge_prompt(template, pair=pair, swap=row.position_swap)``
against the same per-domain results files the judge read, and writes a
new ``formatted_prompt_sha256`` field next to ``prompt_sha256``. Rows
already carrying a non-empty ``formatted_prompt_sha256`` are kept
untouched.

Behaviour:
  - Reads benchmarks/results_v0.4/judge.jsonl + judge_prompt_v0_4.txt
    + benchmarks/results_v0.4/<domain>.json for the cascade text.
  - Writes a backup at <jsonl>.pre_v0_4_1_backup so the recovery is
    reversible.
  - Writes the recovered file in place.
  - Re-emits the judge_agreement.json with the same metrics plus a
    new ``formatted_prompt_sha256_index`` listing the unique formatted
    prompt hashes (one per row).

Usage:
    python scripts/recover_judge_formatted_sha.py [--dry-run]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "benchmarks" / "results_v0.4"
JUDGE_JSONL = RESULTS / "judge.jsonl"
JUDGE_AGREEMENT = RESULTS / "judge_agreement.json"
PROMPT_PATH = SCRIPTS / "judge_prompt_v0_4.txt"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the canonical formatter so we hash exactly what the judge saw.
from judge_subset import (  # type: ignore  # noqa: E402
    JudgePair,
    _format_judge_prompt,
    _item_prompt,
)


def _load_pair(domain: str, item_id: str, treatment_arm: str, control_arm: str) -> JudgePair | None:
    path = RESULTS / f"{domain}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or {}
    row = rows.get(item_id)
    if not row:
        return None
    t = row.get(treatment_arm) or {}
    c = row.get(control_arm) or {}
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
        treatment_arm=treatment_arm,
        control_arm=control_arm,
        treatment_text=treatment_text,
        control_text=control_text,
        treatment_composite=proxy_t,
        control_composite=proxy_c,
        proxy_delta=proxy_t - proxy_c,
        quartile=0,  # not needed for hash recomputation
    )


def _recover_one(record: dict, template: str) -> tuple[dict, str]:
    """Return (updated_record, formatted_prompt_sha256). Idempotent."""
    if record.get("formatted_prompt_sha256"):
        return record, str(record["formatted_prompt_sha256"])
    pair = _load_pair(
        domain=record["domain"],
        item_id=record["item_id"],
        treatment_arm=record["treatment_arm"],
        control_arm=record["control_arm"],
    )
    if pair is None:
        # We cannot rebuild the surfaces — record an explicit sentinel
        # rather than guessing. This is rare (only if the per-domain
        # JSON has been deleted) but preserves auditability.
        sha = "unrecoverable_v0_4_legacy_run"
        record["formatted_prompt_sha256"] = sha
        record["_recovery_note"] = (
            "v0.4.1 recovery could not reconstruct the formatted prompt; "
            "per-domain results file missing or item id absent."
        )
        return record, sha
    prompt = _format_judge_prompt(template, pair=pair, swap=bool(record.get("position_swap")))
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record["formatted_prompt_sha256"] = sha
    return record, sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts without writing files")
    args = ap.parse_args()

    if not JUDGE_JSONL.exists():
        print(f"FATAL: {JUDGE_JSONL} missing", file=sys.stderr)
        return 2
    template = PROMPT_PATH.read_text(encoding="utf-8")

    rows: list[dict] = []
    formatted_index: list[dict] = []
    recovered = 0
    already = 0
    unrecoverable = 0
    with JUDGE_JSONL.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            had = bool(rec.get("formatted_prompt_sha256"))
            rec, sha = _recover_one(rec, template)
            rows.append(rec)
            formatted_index.append({
                "domain": rec["domain"],
                "item_id": rec["item_id"],
                "treatment_arm": rec["treatment_arm"],
                "control_arm": rec["control_arm"],
                "position_swap": rec.get("position_swap"),
                "formatted_prompt_sha256": sha,
            })
            if had:
                already += 1
            elif sha == "unrecoverable_v0_4_legacy_run":
                unrecoverable += 1
            else:
                recovered += 1

    print(
        f"[recover-judge-sha] rows={len(rows)} recovered={recovered} "
        f"already_present={already} unrecoverable={unrecoverable}",
    )

    if args.dry_run:
        return 0

    # Backup, then rewrite in place.
    backup = JUDGE_JSONL.with_suffix(JUDGE_JSONL.suffix + ".pre_v0_4_1_backup")
    if not backup.exists():
        shutil.copy2(JUDGE_JSONL, backup)
        print(f"[recover-judge-sha] backed up original to {backup.name}")
    with JUDGE_JSONL.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[recover-judge-sha] wrote {JUDGE_JSONL}")

    # Mirror the new field into judge_agreement.json so anyone reading
    # only the aggregate also sees the recovery state.
    if JUDGE_AGREEMENT.exists():
        agree = json.loads(JUDGE_AGREEMENT.read_text(encoding="utf-8"))
        agree["formatted_prompt_sha256_index"] = formatted_index
        agree["formatted_prompt_recovery"] = {
            "recovered": recovered,
            "already_present": already,
            "unrecoverable": unrecoverable,
            "version": "v0.4.1",
        }
        JUDGE_AGREEMENT.write_text(
            json.dumps(agree, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[recover-judge-sha] updated {JUDGE_AGREEMENT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
