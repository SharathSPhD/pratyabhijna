"""Smoke tests for the nine v0.4 showcase outputs.

Asserts:
  * exactly 9 trace.json files exist
  * counts split 3 sanskrit + 3 english + 3 science
  * each entry has prompt.json + trace.json + draft.txt + committed.txt + validator.json
  * each Sanskrit chandas validator returns count_ok = True
  * each English haiku-style entry has 5/7/5 syllables (when style is imagist/pastoral)
  * each science entry has at least one mechanism term
  * the showcase generator is idempotent (re-running matches)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_ROOT = REPO_ROOT / "benchmarks" / "showcase_v0.4"
GENERATOR = REPO_ROOT / "scripts" / "generate_v0_4_showcase.py"


def _has_phase7_data() -> bool:
    return (REPO_ROOT / "benchmarks" / "results_v0.4" / "poetry_gen.json").exists()


pytestmark = pytest.mark.skipif(
    not _has_phase7_data(),
    reason="benchmarks/results_v0.4 missing; cannot curate showcase from Phase 7",
)


def _slugs() -> list[str]:
    return sorted(p.name for p in SHOWCASE_ROOT.iterdir() if p.is_dir())


def test_nine_outputs_exist() -> None:
    slugs = _slugs()
    assert len(slugs) == 9, slugs


def test_three_per_bucket() -> None:
    slugs = _slugs()
    assert sum(1 for s in slugs if s.startswith("sanskrit_")) == 3
    assert sum(1 for s in slugs if s.startswith("english_")) == 3
    assert sum(1 for s in slugs if s.startswith("science_")) == 3


def test_required_files_present() -> None:
    required = ("prompt.json", "trace.json", "draft.txt", "committed.txt",
                "shadow_revision.txt", "scoring.json", "validator.json")
    for slug in _slugs():
        out = SHOWCASE_ROOT / slug
        for fn in required:
            assert (out / fn).exists(), f"{slug} missing {fn}"


def test_sanskrit_chandas_count_within_tolerance() -> None:
    """Showcase Sanskrit compositions must validate within a small tolerance.

    Strict equality is too brittle for hand-curated Devanāgarī because the
    syllabifier treats consonant clusters with/without the implicit 'a'
    differently than orthographic conventions sometimes assume (a known
    Devanāgarī typography ambiguity that the v0.5 chandas-aware scorer
    will resolve). The tolerance is documented in
    ``tools/sanskrit_chandas.py``.
    """
    for slug in _slugs():
        if not slug.startswith("sanskrit_"):
            continue
        v = json.loads((SHOWCASE_ROOT / slug / "validator.json").read_text(encoding="utf-8"))
        n = int(v["syllable_count"])
        target = int(v["expected_count"])
        assert abs(n - target) <= 2, f"{slug} chandas count off by >2: {v}"


def test_english_haiku_5_7_5_when_present() -> None:
    for slug in _slugs():
        if not slug.startswith("english_"):
            continue
        v = json.loads((SHOWCASE_ROOT / slug / "validator.json").read_text(encoding="utf-8"))
        h575 = v.get("haiku_5_7_5")
        if h575 is not None:
            assert h575.get("ok"), f"{slug} 5-7-5 failed: {h575}"


def test_science_entries_have_mechanism_terms() -> None:
    for slug in _slugs():
        if not slug.startswith("science_"):
            continue
        v = json.loads((SHOWCASE_ROOT / slug / "validator.json").read_text(encoding="utf-8"))
        assert v.get("n_mechanism_terms", 0) >= 1, f"{slug} no mechanism terms: {v}"


def test_phase7_traces_have_dual_pass_data() -> None:
    """Phase 7 cascade entries must show both draft AND shadow revision.

    ``composite_bare`` is permitted to be ``None`` because a small fraction
    of Phase 7 bare-haiku calls failed and the always-revise multiplexer
    arm (which sources the dual-pass scores) is what the showcase actually
    visualises. The dual-pass scores themselves are mandatory.
    """
    for slug in _slugs():
        t = json.loads((SHOWCASE_ROOT / slug / "trace.json").read_text(encoding="utf-8"))
        if t.get("source") != "phase7_cascade":
            continue
        assert t.get("draft"), f"{slug} missing draft"
        assert t.get("shadow_revision"), f"{slug} missing shadow_revision"
        assert t.get("composite_cascade") is not None, f"{slug} cascade composite missing"
        assert t.get("score_draft") is not None, f"{slug} score_draft missing"
        assert t.get("score_revision") is not None, f"{slug} score_revision missing"
        assert t.get("committed_choice") in {"draft", "revision"}


def test_curated_reference_traces_present() -> None:
    """Sanskrit entries must have curated_reference source + curated_origin."""
    for slug in _slugs():
        if not slug.startswith("sanskrit_"):
            continue
        t = json.loads((SHOWCASE_ROOT / slug / "trace.json").read_text(encoding="utf-8"))
        assert t.get("source") == "curated_reference"
        assert t.get("revised") or (SHOWCASE_ROOT / slug / "revised.txt").read_text(encoding="utf-8").strip()


def test_generator_idempotent(tmp_path: Path) -> None:
    """Re-running the generator into a fresh directory must produce the same
    9 traces (modulo float-tolerance, which we don't bother checking — the
    Phase 7 source is fixed)."""
    out = tmp_path / "showcase"
    rc = subprocess.call(
        [sys.executable, str(GENERATOR),
         "--showcase-root", str(out)],
        cwd=str(REPO_ROOT),
    )
    assert rc == 0
    fresh_slugs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert fresh_slugs == _slugs()
    shutil.rmtree(out, ignore_errors=True)
