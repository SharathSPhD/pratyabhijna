"""Release-notes / paper statistical-label consistency.

v0.4 adversarial review flagged that several headline labels paired
``g`` with a CI in a way that read as "CI for g" when the CI was
actually on the paired mean ``Δ`` (or, for H5, on the pooled ``g`` from
a fixed-effects Wald interval rather than a BCa bootstrap). The v0.4.1
amend relabels every CI inline.

This test asserts the contract: any line in the release notes or the
main paper that contains a ``BCa`` token followed by a bracketed
interval must also name the estimand it belongs to (``Δ``, ``delta``,
or ``paired mean``) on the same line. The H5 fixed-effects pool, which
is **not** a BCa interval, must use ``Wald`` wording when bracketed
intervals appear next to it.

This is a documentation lint, not a numerical check. It runs without
the cascade.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_TARGETS = [
    REPO_ROOT / "docs" / "RELEASE_NOTES_v0.4.md",
    REPO_ROOT / "paper" / "main.tex",
    REPO_ROOT / "paper" / "sections" / "07_methods.tex",
    REPO_ROOT / "paper" / "sections" / "09_results.tex",
    REPO_ROOT / "paper" / "sections" / "10_discussion.tex",
    REPO_ROOT / "paper" / "sections" / "10c_showcase_examples.tex",
]

ESTIMAND_TOKENS_RE = re.compile(
    r"(\\Delta\b|paired mean|delta|\bΔ\b|on pooled\s*\$?g\$?|on the pooled\s*\$?g\$?|pooled\s*\$?g\$?)",
    re.IGNORECASE,
)
BCA_LINE_RE = re.compile(r"BCa.*\[", re.IGNORECASE)
H5_LINE_RE = re.compile(r"\bH5(\.v4)?\b")
WALD_RE = re.compile(r"\bWald\b", re.IGNORECASE)


def _readlines(p: Path) -> list[str]:
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8").splitlines()


def test_every_bca_line_names_its_estimand() -> None:
    """Every release-notes / paper line containing ``BCa.*\\[...]`` must
    name its estimand on the same line."""
    offenders: list[str] = []
    for tgt in DOC_TARGETS:
        for i, raw in enumerate(_readlines(tgt), start=1):
            if not BCA_LINE_RE.search(raw):
                continue
            if ESTIMAND_TOKENS_RE.search(raw):
                continue
            offenders.append(f"{tgt.relative_to(REPO_ROOT)}:{i}: {raw.strip()[:200]}")
    assert not offenders, (
        "Lines with `BCa ... [` must also name the estimand "
        "(Δ / paired mean / delta / pooled g):\n  " + "\n  ".join(offenders)
    )


def test_h5_lines_with_intervals_use_wald_not_bca() -> None:
    """Any line that mentions H5 and contains a bracketed CI must use
    Wald wording (the H5 pool is reported with a fixed-effects Wald CI,
    not a BCa bootstrap)."""
    offenders: list[str] = []
    for tgt in DOC_TARGETS:
        for i, raw in enumerate(_readlines(tgt), start=1):
            if not H5_LINE_RE.search(raw):
                continue
            if "[" not in raw:
                continue
            if "BCa" in raw and not WALD_RE.search(raw):
                offenders.append(f"{tgt.relative_to(REPO_ROOT)}:{i}: {raw.strip()[:200]}")
    assert not offenders, (
        "Lines that mention H5 with a bracketed interval should not say BCa "
        "without also saying Wald:\n  " + "\n  ".join(offenders)
    )


def test_release_notes_v041_amend_section_present() -> None:
    """Release notes must carry the v0.4.1 amend section so the GitHub
    release body reflects the amended state."""
    notes = (REPO_ROOT / "docs" / "RELEASE_NOTES_v0.4.md").read_text(encoding="utf-8")
    assert "v0.4.1 amend" in notes, "missing 'v0.4.1 amend' header in release notes"
    assert "showcase --regenerate" in notes, "missing 'showcase --regenerate' CLI label"
    assert "live_cascade_v0_4_1" in notes, "missing live_cascade_v0_4_1 mention"
    assert "input_tokens" in notes and "placeholder" in notes, (
        "missing input_tokens placeholder disclosure"
    )


def test_release_notes_config_precedence_correct() -> None:
    """Release notes must describe the actual ``PCEConfig.load`` order:
    defaults < repo TOML < user TOML < env < CLI flags."""
    notes = (REPO_ROOT / "docs" / "RELEASE_NOTES_v0.4.md").read_text(encoding="utf-8")
    bad_pattern = re.compile(
        r"~/\.config/pce/config\.toml.{0,40}repo\s+`?pce\.toml`?",
        re.DOTALL | re.IGNORECASE,
    )
    assert not bad_pattern.search(notes), (
        "release notes still describe user-TOML before repo-TOML; "
        "PCEConfig.load layers repo TOML before user TOML"
    )
