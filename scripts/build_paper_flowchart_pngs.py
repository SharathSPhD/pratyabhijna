#!/usr/bin/env python3
"""Export paper TikZ flowcharts (F1-F5) as PNGs for the GitHub Pages site.

Pipeline per figure:

1. Locate the canonical ``\\begin{figure*}...\\end{figure*}`` block in the
   right section file by scanning for the ``\\label{fig:...}`` token.
2. Extract the inner ``\\begin{tikzpicture}...\\end{tikzpicture}`` body.
3. Wrap it in ``\\documentclass[border=4pt]{standalone}`` with the same TikZ
   libraries the paper loads.
4. Compile via ``tectonic`` to a per-figure PDF in a temp dir.
5. Convert to PNG via ``pdftocairo -png -singlefile -r 200``.
6. Copy the PNG into both ``paper/figures/v0.4/flowcharts/`` (paper-side
   reference; not embedded in the PDF since the figures are inline TikZ)
   and ``docs/site/public/figures/v0.4/flowcharts/`` (site-side asset
   referenced by the placeholder figures on the architecture, background,
   results, and hypotheses pages).

Slug map (must match the site references):

    F1_panchashakti_cascade.png       -> sections/03_pratyabhijna_background.tex
    F2_active_inference_loop.png      -> sections/04_active_inference_background.tex
    F3_commit_policy_multiplexer.png  -> sections/07_methods.tex
    F4_phase7_pipeline.png            -> sections/07_methods.tex
    F5_hypothesis_tree.png            -> sections/07_methods.tex

Run::

    python3 scripts/build_paper_flowchart_pngs.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "paper" / "sections"
PAPER_OUT = REPO / "paper" / "figures" / "v0.4" / "flowcharts"
SITE_OUT = REPO / "docs" / "site" / "public" / "figures" / "v0.4" / "flowcharts"

# Figures: (output PNG slug, source .tex file, fig: label to anchor on)
FIGURES = [
    (
        "F1_panchashakti_cascade.png",
        SECTIONS / "03_pratyabhijna_background.tex",
        "fig:flow_5shakti",
    ),
    (
        "F2_active_inference_loop.png",
        SECTIONS / "04_active_inference_background.tex",
        "fig:flow_active_inference",
    ),
    (
        "F3_commit_policy_multiplexer.png",
        SECTIONS / "07_methods.tex",
        "fig:flow_commit_policy",
    ),
    (
        "F4_phase7_pipeline.png",
        SECTIONS / "07_methods.tex",
        "fig:flow_phase7_pipeline",
    ),
    (
        "F5_hypothesis_tree.png",
        SECTIONS / "07_methods.tex",
        "fig:flow_hypothesis_tree",
    ),
]


STANDALONE_PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,shapes.geometric,fit}
\newcommand{\iast}[1]{\textit{#1}}
\begin{document}
"""

STANDALONE_POSTAMBLE = r"""
\end{document}
"""


def _extract_tikzpicture(tex_path: Path, label: str) -> str:
    """Find the figure block carrying ``\\label{<label>}`` and return its
    inner ``\\begin{tikzpicture}...\\end{tikzpicture}`` body verbatim.
    """
    src = tex_path.read_text(encoding="utf-8")
    # Find every figure*/figure block and pick the one whose body contains
    # the requested fig label.
    fig_pattern = re.compile(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        re.DOTALL,
    )
    label_token = f"\\label{{{label}}}"
    for m in fig_pattern.finditer(src):
        body = m.group(0)
        if label_token in body:
            tikz = re.search(
                r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                body,
                re.DOTALL,
            )
            if tikz is None:
                raise RuntimeError(
                    f"figure with label {label!r} in {tex_path} has no tikzpicture"
                )
            return tikz.group(0)
    raise RuntimeError(f"no figure with label {label!r} in {tex_path}")


def _compile_one(slug: str, tikz_body: str, tmp: Path) -> Path:
    """Wrap ``tikz_body`` in standalone, compile, return the PDF path."""
    tex_path = tmp / f"{slug}.tex"
    tex_path.write_text(
        STANDALONE_PREAMBLE + tikz_body + STANDALONE_POSTAMBLE,
        encoding="utf-8",
    )
    # tectonic compiles the per-figure standalone document.
    proc = subprocess.run(
        ["tectonic", "-X", "compile", tex_path.name, "--outdir", "."],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"tectonic failed for {slug}")
    pdf = tmp / f"{slug}.pdf"
    if not pdf.exists():
        raise RuntimeError(f"tectonic did not emit {pdf}")
    return pdf


def _pdf_to_png(pdf: Path, out_basename: str) -> Path:
    """Convert ``pdf`` to a single PNG via ``pdftocairo -singlefile``.

    Returns the resulting PNG path.
    """
    out_root = pdf.parent / out_basename  # pdftocairo appends .png
    proc = subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-r",
            "200",
            str(pdf),
            str(out_root),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"pdftocairo failed for {pdf}")
    png = pdf.parent / f"{out_basename}.png"
    if not png.exists():
        raise RuntimeError(f"pdftocairo did not emit {png}")
    return png


def main() -> int:
    if shutil.which("tectonic") is None:
        print("ERROR: tectonic not on PATH", file=sys.stderr)
        return 2
    if shutil.which("pdftocairo") is None:
        print("ERROR: pdftocairo not on PATH", file=sys.stderr)
        return 2

    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    SITE_OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pce_flowcharts_") as tmpdir:
        tmp = Path(tmpdir)
        for png_name, tex_src, label in FIGURES:
            slug = png_name.replace(".png", "")
            print(f"[flowcharts] {png_name} <- {tex_src.name} ({label})")
            tikz = _extract_tikzpicture(tex_src, label)
            pdf = _compile_one(slug, tikz, tmp)
            png = _pdf_to_png(pdf, slug)
            for dst in (PAPER_OUT / png_name, SITE_OUT / png_name):
                shutil.copyfile(png, dst)
                size_kb = dst.stat().st_size // 1024
                print(f"  wrote {dst.relative_to(REPO)} ({size_kb} KB)")
    print("[flowcharts] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
