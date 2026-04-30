"""v0.4.2 content-expansion: no "Bedrock" in user-facing prose.

Mirrors the ``gate_no_bedrock_in_user_prose`` Phase 8 gate at the unit-test
level for fast pytest feedback. The user-facing prose surfaces (paper, the
Astro site sources, README, RUN_LOCAL, RELEASE_NOTES_v0.4) must not mention
"Bedrock" — the v0.4.2 rewrite normalises every such reference to "API
calls" or "managed Anthropic-API substrate".

Whitelist (intentionally retains the term):

* ``docs/RUN_ON_BEDROCK.md`` — operator runbook for users specifically
  running on AWS Bedrock.
* ``scripts/`` — code identifiers like ``run_v0_4_bedrock.py``.
* ``audit/`` — raw audit JSON with substrate provenance fields.
* ``docs/adr/`` — ADRs are append-only historical records.
* ``docs/reviews/`` — adversarial-review records are immutable.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"[Bb]edrock")


def _user_prose_files() -> list[Path]:
    files: list[Path] = []
    files.append(REPO / "README.md")
    files.append(REPO / "docs" / "RUN_LOCAL.md")
    files.append(REPO / "docs" / "RELEASE_NOTES_v0.4.md")
    paper = REPO / "paper"
    files.append(paper / "main.tex")
    files.extend((paper / "sections").glob("*.tex"))
    appendices = paper / "appendices"
    if appendices.is_dir():
        files.extend(appendices.glob("*.tex"))
    site_src = REPO / "docs" / "site" / "src"
    if site_src.is_dir():
        files.extend(site_src.rglob("*.astro"))
        files.extend(site_src.rglob("*.ts"))
        files.extend(site_src.rglob("*.tsx"))
    return [f for f in files if f.is_file()]


def test_no_bedrock_token_in_user_prose() -> None:
    hits: list[str] = []
    for f in _user_prose_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                hits.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:120]}")
    assert not hits, "Bedrock references found in user-facing prose:\n  " + "\n  ".join(hits)


def test_whitelist_files_intentionally_retain_term() -> None:
    """Smoke-check that the whitelist files DO still contain the term — if they
    don't, either the whitelist is stale or the term has been removed everywhere
    (in which case the gate could be retired)."""
    runbook = REPO / "docs" / "RUN_ON_BEDROCK.md"
    if runbook.exists():
        text = runbook.read_text(encoding="utf-8", errors="ignore")
        assert PATTERN.search(text), (
            "docs/RUN_ON_BEDROCK.md is whitelisted because it operationally targets AWS "
            "Bedrock; if it no longer mentions the term, drop the whitelist or rename the file."
        )
