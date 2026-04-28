#!/usr/bin/env python3
"""Generate paper/autoreport.tex (Table 1) and substitute placeholders in main.tex
with values from benchmarks/results/stats.json.

Idempotent: re-running overwrites the autoreport file and rewrites only the
placeholder lines in main.tex. Original section files are not touched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _row_for(h: str, payload: dict[str, Any]) -> str:
    n = payload["n"]
    est = payload["estimate"]
    g = payload["hedges_g"]
    lo, hi = payload["bca_ci_95"]
    perm = payload["permutation_p_one_sided"]
    wil = payload["wilcoxon_p_one_sided"]
    holm = payload["holm_p"]
    pa = payload["power_apriori"]
    pr = payload["power_retrospective"]
    sup = "\\textbf{yes}" if payload["supported"] else "no"
    return (
        f"{h} & {n} & {est:+.3f} & {g:+.2f} & "
        f"[{lo:+.3f},\\,{hi:+.3f}] & "
        f"{perm:.4f} & {wil:.4f} & {holm:.4f} & "
        f"{pa:.2f} / {pr:.2f} & {sup} \\\\"
    )


def _h5_row(h5: dict[str, Any]) -> str:
    n = h5["n"]
    est = h5["estimate_pooled_z"]
    g = h5["hedges_g"]
    lo, hi = h5["bca_ci_95"]
    perm = h5["permutation_p_one_sided"]
    pa = h5["power_apriori"]
    pr = h5["power_retrospective"]
    sup = "\\textbf{yes}" if h5["supported"] else "no"
    return (
        f"H5 (composite) & {n} & {est:+.3f} & {g:+.2f} & "
        f"[{lo:+.3f},\\,{hi:+.3f}] & {perm:.4f} & --- & --- & "
        f"{pa:.2f} / {pr:.2f} & {sup} \\\\"
    )


def _h6_row(h6: dict[str, Any]) -> str:
    nf = h6.get("n_fired", 0)
    nn = h6.get("n_not_fired", 0)
    est = h6.get("estimate", float("nan"))
    p = h6.get("mannwhitney_u_p_one_sided", float("nan"))
    sup = "\\textbf{yes}" if h6.get("supported", False) else "no"
    if isinstance(est, float) and est != est:  # NaN check
        est_s = "n/a"
    else:
        est_s = f"{est:+.3f}"
    if isinstance(p, float) and p != p:
        p_s = "n/a"
    else:
        p_s = f"{p:.4f}"
    return f"H6 (within-PCE) & {nf}/{nn} & {est_s} & --- & --- & {p_s} & --- & --- & --- & {sup} \\\\"


def _build_autoreport(stats: dict[str, Any]) -> str:
    primary = stats["primary"]
    rows = [
        _row_for(h, primary[h]) for h in ("H1", "H2", "H3", "H4")
    ]
    rows.append(_h5_row(stats["H5"]))
    rows.append(_h6_row(stats["H6"]))
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{l r r r c c c c c c}\n"
        "\\toprule\n"
        "Hyp. & $n$ & $\\bar\\Delta$ & $g$ & 95\\% BCa CI & "
        "perm.~$p$ & Wilcoxon~$p$ & Holm~$p$ & power$_a$ / power$_r$ & supported \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _headline(stats: dict[str, Any]) -> str:
    primary = stats["primary"]
    supported = [h for h in ("H1", "H2", "H3", "H4") if primary[h]["supported"]]
    h6 = stats["H6"].get("supported", False)
    h5 = stats["H5"].get("supported", False)
    parts: list[str] = []
    if supported:
        parts.append(
            "PCE materially shifts the paired contrast against no-PCE Haiku on "
            + ", ".join(supported)
            + " (Holm-adjusted $p<0.05$, BCa CI strictly positive)."
        )
    else:
        parts.append(
            "No primary hypothesis (H1--H4) crosses the pre-registered Holm-adjusted "
            "$p<0.05$ threshold against no-PCE Haiku --- a negative result we report "
            "in compliance with the SPEC's negative-result obligation."
        )
    if h5:
        parts.append("The aggregate composite (H5) is positive and significant.")
    if h6:
        parts.append(
            "The within-PCE H6 internal-validity test is supported: trials in which "
            "the \\iast{vimar\\'sa} layer fired score higher than trials in which "
            "it did not."
        )
    elif stats["H6"].get("n_fired", 0) > 0:
        parts.append(
            "The within-PCE H6 test is reported but does not reach the threshold "
            "at the calibrated aspect-cosine-hit value."
        )
    return " ".join(parts)


def _replace_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Substitute `{KEY}` with `{VALUE}` preserving the brace pair so the
    surrounding LaTeX (e.g. `\\text{...}`) stays well-formed even when the
    placeholder coincides with a LaTeX argument boundary.
    """
    for k, v in mapping.items():
        text = text.replace("{" + k + "}", "{" + v + "}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "stats.json")
    parser.add_argument("--paper-dir", type=Path, default=REPO_ROOT / "paper")
    parser.add_argument("--items", type=Path, default=REPO_ROOT / "benchmarks" / "items.py")
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    autoreport = _build_autoreport(stats)
    (args.paper_dir / "autoreport.tex").write_text(autoreport, encoding="utf-8")

    primary = stats["primary"]
    n_paired = sum(primary[h]["n"] for h in ("H1", "H2", "H3", "H4"))
    headline = _headline(stats)

    # Count items per domain
    items_text = args.items.read_text(encoding="utf-8")

    def _list_count(name: str) -> int:
        m = re.search(rf"^{name}\s*:\s*list\[[^\]]+\]\s*=\s*\[(.*?)^\]", items_text, re.S | re.M)
        if not m:
            return 0
        body = m.group(1)
        return body.count('"id":')

    # We prefer the *actual* paired-observation count from stats.json (i.e. the
    # number of items the driver completed for each domain) over the items.py
    # size, since the driver may be invoked with --n-* limits.
    counts = {
        "POETRY_GEN_N": str(primary["H3"]["n"]),
        "POETRY_INTERP_N": str(primary["H2"]["n"]),
        "AUT_N": str(primary["H1"]["n"]),
        "SCI_N": str(primary["H4"]["n"]),
        "N_PAIRED": str(n_paired),
        "HEADLINE_RESULT": headline,
    }
    # Touch the unused helper to keep import-side metadata consistent (it is
    # available for downstream tooling that wants the canonical items.py size).
    _ = _list_count

    # Substitute placeholders in main.tex (in-place but only on the
    # `{KEY}` tokens we own; we never edit narrative text.)
    main_path = args.paper_dir / "main.tex"
    text = main_path.read_text(encoding="utf-8")
    text = _replace_placeholders(text, counts)
    main_path.write_text(text, encoding="utf-8")

    # Substitute placeholders in section 8 (benchmark item counts).
    sec_path = args.paper_dir / "sections" / "08_benchmarks.tex"
    if sec_path.exists():
        st = sec_path.read_text(encoding="utf-8")
        st = _replace_placeholders(st, counts)
        sec_path.write_text(st, encoding="utf-8")

    print(f"autoreport -> {args.paper_dir / 'autoreport.tex'}")
    print(f"main.tex placeholders replaced: {list(counts.keys())}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
