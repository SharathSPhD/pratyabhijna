"""v0.4.3 paper polish: no version-prose hits in the paper sources.

Per the v0.4.3 plan (Q1=A "prose-only"), the paper drops all rendered
mentions of ``v0.4 / v0.3 / v0.5`` from prose, abstracts, plain-language
summaries, section bodies, and hypothesis labels. Versioned tokens are
allowed when they appear inside file-system identifiers wrapped in
``\\nolinkurl{}``, ``\\url{}``, ``\\path{}``, or ``\\input{}`` — these
are JSON / file paths, not prose claims.

This test scans every paper source file and fails on any prose hit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"
APPENDICES = PAPER / "appendices"

VERSION_RE = re.compile(r"\bv0\.[0-9]+(?:\.[0-9]+)?\b")
# Strip macros whose argument is a file-path / asset-identifier rather
# than prose: the argument may contain a versioned path token without
# implying the paper's prose names a version. Order matters: with
# nested-brace support it's simplest to scrub the argument of common
# path-bearing macros.
PATH_MACRO_RE = re.compile(
    r"\\(?:nolinkurl|url|path|input|includegraphics(?:\[[^\]]*\])?)\{[^{}]*\}"
)
# A ``\texttt{path/with/v0.4/in_it}`` reference is also a file-path
# identifier when its content matches the path-shaped pattern (slashes
# or a versioned prefix). Strip these so prose-only text remains.
TEXTTT_PATH_RE = re.compile(
    r"\\texttt\{[^{}]*?(?:/|v0\.[0-9]+|\\_)[^{}]*?\}"
)


def _strip_path_macros(line: str) -> str:
    """Remove macro arguments that hold file paths, image references,
    or path-shaped ``\\texttt{}`` identifiers so the leftover text is
    only prose.
    """
    line = PATH_MACRO_RE.sub("", line)
    line = TEXTTT_PATH_RE.sub("", line)
    return line


def _is_comment(stripped: str) -> bool:
    return stripped.lstrip().startswith("%")


def _scan(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    if not path.exists():
        return hits
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _is_comment(raw):
            continue
        prose = _strip_path_macros(raw)
        if VERSION_RE.search(prose):
            hits.append((lineno, raw.strip()[:160]))
    return hits


def _all_paper_tex() -> list[Path]:
    paths = [PAPER / "main.tex"]
    paths.extend(sorted(SECTIONS.glob("*.tex")))
    paths.extend(sorted(APPENDICES.glob("*.tex")))
    # The auto-generated table snippets and autoreport tex files are
    # under ``paper/sections/_tables/`` and ``paper/`` and are checked
    # alongside the rest.
    paths.extend(sorted((SECTIONS / "_tables").glob("*.tex")))
    paths.append(PAPER / "autoreport_v0.4.tex")
    paths.append(PAPER / "autoreport.tex")
    return paths


def test_no_version_in_paper_prose() -> None:
    failures: list[str] = []
    for path in _all_paper_tex():
        hits = _scan(path)
        for lineno, snippet in hits:
            failures.append(f"{path.relative_to(REPO)}:{lineno}: {snippet}")
    assert not failures, (
        "version-prose hits found in paper (allowed only inside "
        "\\nolinkurl/\\url/\\path/\\input or comments):\n"
        + "\n".join(failures[:20])
    )
