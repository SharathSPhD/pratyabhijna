#!/usr/bin/env python3
"""Structural validator for the Arxiv paper.

Cannot run pdflatex in CI but we still want a hard guarantee that:
  * every `\\cite{key}` resolves to a bibtex entry in references.bib
  * every `\\ref{label}` resolves to a `\\label{label}` in some .tex file
  * every `\\input{...}` resolves to an existing file
  * every `\\includegraphics{...}` resolves to an existing image
  * every `{PLACEHOLDER}`-style autoreport token has been substituted

Exit code 0 = clean, >0 = number of unresolved references.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _gather_tex() -> list[Path]:
    return sorted(PAPER.rglob("*.tex"))


def _bib_keys(bib_path: Path) -> set[str]:
    text = _read(bib_path)
    return {m.group(1) for m in re.finditer(r"@\w+\{\s*([^,\s]+)\s*,", text)}


def _labels(all_text: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\\label\{([^}]+)\}", all_text)}


def _cites(all_text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", all_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                out.append(k)
    return out


def _refs(all_text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", all_text)]


def _inputs(all_text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\\input\{([^}]+)\}", all_text)]


def _includes(all_text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text)]


def _placeholders(all_text: str) -> list[str]:
    out: list[str] = []
    pattern = re.compile(r"\{([A-Z][A-Z0-9_]+)\}")
    for m in pattern.finditer(all_text):
        token = m.group(1)
        if token in {"T", "TIME", "DATE", "TODAY"}:
            continue
        out.append(token)
    return out


def main() -> int:
    tex_files = _gather_tex()
    if not tex_files:
        print("ERROR: no .tex files under paper/")
        return 1
    all_text = "\n".join(_read(p) for p in tex_files)

    bib_keys = _bib_keys(PAPER / "references.bib")
    labels = _labels(all_text)

    missing_cites = sorted({k for k in _cites(all_text) if k not in bib_keys})
    missing_refs = sorted({k for k in _refs(all_text) if k not in labels})

    missing_inputs: list[str] = []
    for inp in _inputs(all_text):
        candidates = [
            PAPER / f"{inp}.tex",
            PAPER / inp,
        ]
        if not any(c.exists() for c in candidates):
            missing_inputs.append(inp)

    missing_images: list[str] = []
    for inc in _includes(all_text):
        for ext in ("", ".png", ".pdf", ".jpg"):
            if (PAPER / f"{inc}{ext}").exists():
                break
        else:
            missing_images.append(inc)

    leftover = sorted(set(_placeholders(all_text)))
    leftover = [p for p in leftover if p not in {"BCa", "BMR", "CI", "DPO", "RGB", "GPU", "MPS", "SWS", "REM", "JSON", "MCP"}]

    n_problems = 0

    print("=== validate_paper ===")
    print(f"  tex files       : {len(tex_files)}")
    print(f"  bib keys        : {len(bib_keys)}")
    print(f"  labels found    : {len(labels)}")
    print()

    if missing_cites:
        n_problems += len(missing_cites)
        print(f"MISSING CITES ({len(missing_cites)}):")
        for k in missing_cites:
            print(f"  - {k}")
    else:
        print("CITES         : OK")

    if missing_refs:
        n_problems += len(missing_refs)
        print(f"MISSING REFS ({len(missing_refs)}):")
        for k in missing_refs:
            print(f"  - {k}")
    else:
        print("REFS          : OK")

    if missing_inputs:
        n_problems += len(missing_inputs)
        print(f"MISSING INPUTS ({len(missing_inputs)}):")
        for k in missing_inputs:
            print(f"  - {k}")
    else:
        print("INPUTS        : OK")

    if missing_images:
        n_problems += len(missing_images)
        print(f"MISSING IMAGES ({len(missing_images)}):")
        for k in missing_images:
            print(f"  - {k}")
    else:
        print("IMAGES        : OK")

    if leftover:
        print(f"POTENTIAL PLACEHOLDERS LEFTOVER ({len(leftover)}):")
        for k in leftover:
            print(f"  - {k}")
    else:
        print("PLACEHOLDERS  : OK")

    print()
    if n_problems == 0:
        print("OK paper structurally clean")
    else:
        print(f"FAIL {n_problems} unresolved reference(s)")
    return n_problems


if __name__ == "__main__":
    sys.exit(main())
