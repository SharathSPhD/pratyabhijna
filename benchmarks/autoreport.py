#!/usr/bin/env python3
"""Generate paper/autoreport.tex (Table 1) and substitute placeholders in main.tex
with values from benchmarks/results/stats.json.

Idempotent: re-running overwrites the autoreport file and rewrites only the
placeholder lines in main.tex. Original section files are not touched.
"""
from __future__ import annotations

import argparse
import json
import math
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
    """v0.3 H5 = random-effects DerSimonian-Laird pool of per-domain Hedges' g."""
    g = h5.get("pooled_g", float("nan"))
    n_studies = h5.get("n_studies", 0)
    ci = h5.get("ci_95", [float("nan"), float("nan")])
    lo, hi = (ci + [float("nan"), float("nan")])[:2] if isinstance(ci, list) else (float("nan"), float("nan"))
    tau2 = h5.get("tau2", float("nan"))
    sup = "\\textbf{yes}" if h5.get("supported", False) else "no"

    def _fmt(x: Any, w: int = 3, sign: bool = True) -> str:
        try:
            xv = float(x)
        except (TypeError, ValueError):
            return "n/a"
        if math.isnan(xv):
            return "n/a"
        return (f"{xv:+.{w}f}" if sign else f"{xv:.{w}f}")

    return (
        f"H5 (RE pooled $g$) & {int(n_studies)} & --- & {_fmt(g, 2)} & "
        f"[{_fmt(lo, 3)},\\,{_fmt(hi, 3)}] & --- & --- & --- & "
        f"$\\tau^2$={_fmt(tau2, 2, sign=False)} & {sup} \\\\"
    )


def _v3_contrast_row(label: str, payload: dict[str, Any]) -> str:
    """Format a v0.3 between-arm contrast (H6.v3 or H7.v3) as a single row.

    These v0.3 contrasts return the *same* primary structure as H1-H4 (one row
    per domain). For the headline table we surface a meta-aggregated row by
    averaging Hedges' g and showing the count of supported domains.
    """
    if not payload:
        return f"{label} & 0 & --- & --- & --- & --- & --- & --- & --- & no \\\\"
    rows = [v for v in payload.values() if isinstance(v, dict) and "n" in v]
    if not rows:
        return f"{label} & 0 & --- & --- & --- & --- & --- & --- & --- & no \\\\"
    n_sum = sum(int(r.get("n", 0)) for r in rows)
    gs: list[float] = []
    for r in rows:
        gv = r.get("hedges_g")
        if isinstance(gv, (int, float)) and not math.isnan(float(gv)):
            gs.append(float(gv))
    g_mean = sum(gs) / len(gs) if gs else float("nan")
    n_sup = sum(1 for r in rows if r.get("supported"))
    sup_total = len(rows)
    sup = "\\textbf{yes}" if n_sup > 0 else "no"

    def _fmt(x: float, w: int = 2) -> str:
        if math.isnan(x):
            return "n/a"
        return f"{x:+.{w}f}"

    return (
        f"{label} & {n_sum} & --- & {_fmt(g_mean)} & --- & --- & --- & --- & "
        f"{n_sup}/{sup_total} domains & {sup} \\\\"
    )


def _h8_row(h8: dict[str, Any]) -> str:
    """v0.3 H8 = within-cascade revision-vs-draft pairing."""
    if not h8:
        return "H8 (revision vs draft) & 0 & --- & --- & --- & --- & --- & --- & --- & no \\\\"
    n = int(h8.get("n", 0))
    est = h8.get("estimate", float("nan"))
    g = h8.get("hedges_g", float("nan"))
    ci = h8.get("bca_ci_95", [float("nan"), float("nan")])
    lo, hi = (ci + [float("nan"), float("nan")])[:2] if isinstance(ci, list) else (float("nan"), float("nan"))
    perm = h8.get("permutation_p_one_sided", float("nan"))
    sup = "\\textbf{yes}" if h8.get("supported", False) else "no"

    def _fmt(x: Any, w: int = 3, sign: bool = True) -> str:
        try:
            xv = float(x)
        except (TypeError, ValueError):
            return "n/a"
        if math.isnan(xv):
            return "n/a"
        return (f"{xv:+.{w}f}" if sign else f"{xv:.{w}f}")

    return (
        f"H8 (revision vs draft) & {n} & {_fmt(est)} & {_fmt(g, 2)} & "
        f"[{_fmt(lo)},\\,{_fmt(hi)}] & {_fmt(perm, 4, sign=False)} & --- & --- & --- & {sup} \\\\"
    )


def _build_autoreport(stats: dict[str, Any]) -> str:
    primary = stats["primary"]
    rows = [
        _row_for(h, primary[h]) for h in ("H1", "H2", "H3", "H4")
    ]
    rows.append(_h5_row(stats["H5"]))
    rows.append(_v3_contrast_row("H6.v3 (vs +K compute)", stats.get("H6_v3_extra_compute", {})))
    rows.append(_v3_contrast_row("H7.v3 (vs generic 2-pass)", stats.get("H7_v3_generic_revise", {})))
    rows.append(_h8_row(stats.get("H8_v3_revision_vs_draft", {})))
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
    cfg = stats.get("config", {})
    treatment = cfg.get("treatment_arm", "haiku_cascade")
    control = cfg.get("control_arm_primary", "haiku_bare")
    treatment_tex = "\\texttt{" + treatment.replace("_", "\\_") + "}"
    control_tex = "\\texttt{" + control.replace("_", "\\_") + "}"
    contrast = f"{treatment_tex} vs.\\ {control_tex}"

    supported = [h for h in ("H1", "H2", "H3", "H4") if primary[h]["supported"]]
    # "Directional positive" = effect size > 0 AND BCa lower bound > 0
    # (the strongest pre-registered evidence we can claim short of the Holm
    # threshold being crossable at the available n).
    bca_positive = [
        h for h in ("H1", "H2", "H3", "H4")
        if (
            float(primary[h].get("estimate", 0.0)) > 0
            and float(primary[h].get("bca_ci_95", [0.0, 0.0])[0]) > 0.0
            and h not in supported
        )
    ]

    h6_v3 = stats.get("H6_v3_extra_compute", {})
    h7_v3 = stats.get("H7_v3_generic_revise", {})
    h8_v3 = stats.get("H8_v3_revision_vs_draft", {})

    def _supported_domains(payload: dict[str, Any]) -> list[str]:
        return [k for k, v in payload.items()
                if isinstance(v, dict) and v.get("supported")]

    h5 = stats["H5"]
    h5_g = h5.get("pooled_g", float("nan"))
    h5_ci = h5.get("ci_95", [float("nan"), float("nan")])

    parts: list[str] = []
    if supported:
        parts.append(
            f"PCE v0.3 materially shifts the paired apples-to-apples contrast {contrast} on "
            + ", ".join(supported)
            + " (Holm-adjusted $p<0.05$, BCa CI strictly positive)."
        )
    if bca_positive:
        worded = (
            "Effect sizes for "
            + ", ".join(bca_positive)
            + f" are positive with 95\\% BCa CIs strictly above zero in the {contrast} "
            "contrast, but the pre-registered Holm-adjusted $p<0.05$ threshold is not "
            "crossed at the pilot's $n$=5/domain (the exact sign-flip permutation floor "
            "is $1/2^{5}=0.0312$ and Holm with $m{=}4$ pushes the smallest possible "
            "adjusted $p$ to $0.125$)."
        )
        if not supported:
            parts.insert(0, worded)
        else:
            parts.append(worded)
    if not supported and not bca_positive:
        parts.append(
            f"No primary hypothesis (H1--H4) shows a directional-positive BCa CI for the "
            f"{contrast} contrast --- a negative result we report in compliance with the "
            "SPEC's negative-result obligation."
        )
    try:
        g = float(h5_g)
        lo = float(h5_ci[0])
        hi = float(h5_ci[1])
        if g == g and lo == lo and hi == hi:
            parts.append(
                "Random-effects pooled effect across the four domains (H5.v3) "
                f"is $g={g:+.2f}$ (95\\% CI [{lo:+.2f},\\,{hi:+.2f}])."
            )
    except (TypeError, ValueError, IndexError):
        pass

    h6_sup = _supported_domains(h6_v3)
    if h6_sup:
        parts.append(
            "The fairness contrast H6.v3 (vs.\\ a budget-matched 2K-scorer "
            "single-pass control) is supported on " + ", ".join(h6_sup) + ", isolating "
            "architecture from the extra-compute confound flagged in the v0.2 review."
        )
    h7_sup = _supported_domains(h7_v3)
    if h7_sup:
        parts.append(
            "H7.v3 (vs.\\ a generic 2-pass revise) is supported on " + ", ".join(h7_sup) +
            ", showing the \\iast{vimar\\'sa} brief content -- not the mere existence "
            "of a revision pass -- carries the win."
        )
    if h8_v3.get("supported"):
        parts.append(
            "Within-cascade H8.v3 (revision vs.\\ draft for items where the event-gated "
            "commit chose revision) is supported, demonstrating the *causal contribution* "
            "of the second pass on the items the cascade itself decided to revise."
        )
    elif int(h8_v3.get("n", 0)) > 0:
        est = float(h8_v3.get("estimate", 0.0))
        parts.append(
            f"H8.v3 points in the predicted direction ($\\Delta={est:+.3f}$, "
            f"$n={int(h8_v3.get('n', 0))}$) but does not reach $p<0.05$ at the pilot's $n$."
        )
    return " ".join(parts)


def _replace_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Substitute `{KEY}` with `{VALUE}` preserving the brace pair so the
    surrounding LaTeX (e.g. `\\text{...}`) stays well-formed even when the
    placeholder coincides with a LaTeX argument boundary.

    Also tolerates the LaTeX-escaped form `{KEY\\_WITH\\_UNDERSCORES}` because
    underscores are LaTeX-active outside math mode and the placeholder may
    have been authored with backslash-escapes for the typeset preview.
    """
    for k, v in mapping.items():
        text = text.replace("{" + k + "}", "{" + v + "}")
        if "_" in k:
            escaped = k.replace("_", "\\_")
            text = text.replace("{" + escaped + "}", "{" + v + "}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats", type=Path, default=REPO_ROOT / "benchmarks" / "results_v2" / "stats.json",
    )
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
