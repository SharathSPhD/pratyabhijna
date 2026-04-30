"""v0.4.1 review fix #3: assert documented CLI examples actually parse.

Scans the public docs (README.md, docs/RUN_LOCAL.md, the Astro plugin
and methods pages, and the standalone CLI's own docstring) for
``pce ...`` invocations, splits them safely, and feeds each one to
``pce.cli.build_parser().parse_args()``. A doc example that argparse
rejects is a documentation bug we want to catch in CI.

Multi-line backslash continuations are joined; subshells (``$(...)``)
inside arguments are stubbed out before parsing. ``pce`` examples that
include placeholders the user must substitute (``<slug>``, ``out/...``)
are kept; argparse only validates flag names + types, not the values.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from pce.cli import build_parser

REPO = Path(__file__).resolve().parents[1]

DOC_FILES = [
    REPO / "README.md",
    REPO / "docs" / "RUN_LOCAL.md",
    REPO / "docs" / "site" / "src" / "pages" / "plugin.astro",
    REPO / "docs" / "site" / "src" / "pages" / "methods.astro",
    REPO / "src" / "pce" / "cli.py",
]

# Lines starting with `pce ` (after optional `$ ` prompt or whitespace).
_PCE_LINE_RE = re.compile(r"(?m)^\s*\$?\s*(pce\s+\S.*)$")
# Match argparse's "all subcommands have global flags" surface — every example
# must hit one of these top-level subcommands.
KNOWN_SUBCOMMANDS = {"config", "smoke", "cascade", "judge-pair", "showcase"}


def _join_continuations(text: str) -> str:
    """Collapse line-continuation markers into single logical lines.

    Real shell continuations are ``\\\\\n``. Astro template literals also
    encode an escaped backslash as ``\\\\\\\\\n``, which after JSX runs
    becomes ``\\\\`` followed by a real newline. Both forms collapse here.
    """
    text = re.sub(r"\\\\\s*\n\s*", " ", text)
    text = re.sub(r"\\\s*\n\s*", " ", text)
    return text


def _strip_subshells(line: str) -> str:
    """Replace shell sub-expressions with safe placeholders so shlex can
    split a documented invocation without choking on unmatched parens."""
    line = re.sub(r"\$\([^)]+\)", "PROMPT_FROM_FILE", line)
    line = re.sub(r"`[^`]+`", "INLINE_BACKTICK_PLACEHOLDER", line)
    return line


def _extract_pce_invocations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = _join_continuations(text)
    invocations: list[str] = []
    for match in _PCE_LINE_RE.finditer(text):
        line = match.group(1).strip()
        # Drop trailing markdown / template syntax tokens.
        line = line.rstrip("`").rstrip("$").rstrip()
        line = re.sub(r"\s*#.*$", "", line)
        # Astro example pages wrap CLI snippets in template literals.
        line = line.replace("`}</pre>", "").replace("{`", "").replace("`}", "")
        line = line.strip()
        if not line.startswith("pce "):
            continue
        # Drop placeholder-only forms like `pce showcase --regenerate <slug|all>`
        # by stripping angle-bracketed metavar tokens; argparse accepts the
        # value as a string regardless of content.
        line = re.sub(r"<[^>]+>", "PLACEHOLDER", line)
        invocations.append(_strip_subshells(line))
    return invocations


def _all_invocations() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in DOC_FILES:
        if not path.exists():
            continue
        for line in _extract_pce_invocations(path):
            out.append((str(path.relative_to(REPO)), line))
    return out


INVOCATIONS = _all_invocations()


def test_some_examples_were_collected() -> None:
    """Sanity check — if regex breaks we want to know."""
    assert INVOCATIONS, (
        "no `pce ...` examples found in any documented file; the regex "
        "or the doc files moved"
    )


@pytest.mark.parametrize(
    "where,line",
    INVOCATIONS,
    ids=[f"{w}::{ln[:60]!r}" for w, ln in INVOCATIONS],
)
def test_doc_pce_invocation_parses(where: str, line: str) -> None:
    parser = build_parser()
    tokens = shlex.split(line)
    assert tokens[0] == "pce"
    argv = tokens[1:]
    if not argv:
        pytest.skip(f"{where}: bare `pce` with no subcommand is the help case")
    subcommand = argv[0]
    if subcommand not in KNOWN_SUBCOMMANDS:
        pytest.skip(f"{where}: subcommand {subcommand!r} is not part of the standalone CLI")
    parser.parse_args(argv)
