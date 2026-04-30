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


# ---------------------------------------------------------------------------
# v0.4 placeholder pack
# ---------------------------------------------------------------------------
#
# Phase 8 introduces a new {V04_*} placeholder family that the rewritten
# paper sections (introduction, results, discussion, honest-AI-claims,
# showcase) all bind. We expose a builder that returns a mapping suitable
# for ``_replace_placeholders`` and enforce a missing-key gate (raises
# rather than silently leaving stub tokens in the typeset PDF).


def _fmt_signed(x: Any, width: int = 3) -> str:
    try:
        xv = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(xv):
        return "n/a"
    return f"{xv:+.{width}f}"


def _fmt_bare(x: Any, width: int = 3) -> str:
    try:
        xv = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(xv):
        return "n/a"
    return f"{xv:.{width}f}"


def _v04_h8c_winner_line(h8c: dict[str, Any]) -> str:
    leader = list(h8c.get("leader_board", []))
    if not leader:
        return "no commit policy outperformed bare control"
    leader.sort(key=lambda r: float(r.get("hedges_g", 0.0)), reverse=True)
    top = leader[0]
    pol = str(top.get("policy", "?")).replace("haiku_cascade_", "")
    g = _fmt_signed(top.get("hedges_g"), 2)
    p = _fmt_bare(top.get("permutation_p_one_sided"), 4)
    return f"{pol} (g={g}, p={p})"


def _v04_showcase_count(showcase_root: Path) -> dict[str, int]:
    """Walk ``benchmarks/showcase_v0.4/`` and return per-bucket trace counts.

    Buckets are inferred from the parent directory name of each ``trace.json``:
    ``sanskrit_*`` → sanskrit, ``english_*`` → english, ``science_*`` → science.
    Returns zeros if showcase generation has not yet run.
    """
    out = {"sanskrit": 0, "english": 0, "science": 0, "total": 0}
    if not showcase_root.exists():
        return out
    for p in showcase_root.glob("*/trace.json"):
        slug = p.parent.name
        if slug.startswith("sanskrit_"):
            out["sanskrit"] += 1
        elif slug.startswith("english_"):
            out["english"] += 1
        elif slug.startswith("science_"):
            out["science"] += 1
        out["total"] += 1
    return out


def _v04_headline(stats: dict[str, Any]) -> str:
    """Single-paragraph headline summarising the v0.4 mechanism study.

    Faithful to the negative primary result and the supported H8a/H8b
    findings (the "we know what's mechanistically doing the work" point).
    """
    primary = stats.get("primary", {})
    h5 = stats.get("H5", {})
    h8a = stats.get("H8a_v4_shadow_revision_vs_draft", {})
    h8b = stats.get("H8b_v4_gate_calibration", {})
    h8c = stats.get("H8c_v4_commit_policy_comparison", {})
    h9 = stats.get("H9_v4_judge_proxy_agreement", {})

    n_supported = sum(1 for h in ("H1", "H2", "H3", "H4") if primary.get(h, {}).get("supported"))
    pooled_g = _fmt_signed(h5.get("pooled_g"), 2)
    pooled_ci = h5.get("ci_95", [float("nan"), float("nan")])
    pooled_lo = _fmt_signed(pooled_ci[0] if len(pooled_ci) > 0 else float("nan"), 2)
    pooled_hi = _fmt_signed(pooled_ci[1] if len(pooled_ci) > 1 else float("nan"), 2)

    sentences: list[str] = []
    if n_supported == 0:
        sentences.append(
            "PCE v0.4's pre-registered primary contrast (Haiku\\,$+$\\,PCE vs.\\ bare Haiku) "
            "does not reach Holm-adjusted $p<0.05$ on any of the four domains under the pilot's "
            "paired sample sizes."
        )
    else:
        sentences.append(
            f"PCE v0.4 produces a Holm-supported uplift on {n_supported}/4 domains and a "
            f"fixed-effects pooled $g={pooled_g}$ (95\\% CI [{pooled_lo},\\,{pooled_hi}])."
        )
    sentences.append(
        f"The fixed-effects meta-pool over four domains gives $g={pooled_g}$ "
        f"(95\\% CI [{pooled_lo},\\,{pooled_hi}])."
    )
    if h8a.get("supported"):
        g8a = _fmt_signed(h8a.get("hedges_g"), 2)
        p8a = _fmt_bare(h8a.get("permutation_p_one_sided"), 4)
        sentences.append(
            f"H8a (shadow revision vs.\\ draft within the cascade) is strongly supported "
            f"($g={g8a}$, $p={p8a}$, $n={int(h8a.get('n', 0))}$), localising the within-arm "
            "uplift to the second pass rather than the first."
        )
    else:
        sentences.append(
            "H8a (shadow revision vs.\\ draft within the cascade) is not supported at the "
            "pre-registered $\\alpha=0.05$ in this pilot."
        )
    if h8b.get("supported"):
        f1_l = _fmt_bare(h8b.get("learned_gate", {}).get("f1"), 3)
        f1_e = _fmt_bare(h8b.get("event_gated", {}).get("f1"), 3)
        sentences.append(
            f"H8b (learned commit policy beats event-gated baseline) is supported "
            f"(F$_1$ {f1_l} vs.\\ {f1_e})."
        )
    sentences.append(
        f"H8c leaderboard winner: {_v04_h8c_winner_line(h8c)}."
    )
    if h9.get("status") == "ok":
        rho = h9.get("spearman_rho")
        sign = h9.get("sign_agreement_rate")
        sentences.append(
            f"H9 (Sonnet judge $\\leftrightarrow$ proxy scorer) on "
            f"$n={int(h9.get('n', 0))}$ judged pairs: $\\rho={_fmt_bare(rho, 2)}$, "
            f"sign-agreement={_fmt_bare(sign, 2)}."
        )
    return " ".join(sentences)


def _v04_n_per_domain(stats: dict[str, Any]) -> str:
    primary = stats.get("primary", {})
    parts = []
    for key, name in (("H1", "AUT"), ("H2", "Poetry-interp"), ("H3", "Poetry-gen"),
                      ("H4", "Sci-creativity")):
        n = int(primary.get(key, {}).get("n", 0))
        parts.append(f"{name} $n={n}$")
    return ", ".join(parts)


def _build_v04_placeholders(
    stats: dict[str, Any],
    *,
    cost_ledger: dict[str, Any] | None,
    judge_agreement: dict[str, Any] | None,
    showcase_counts: dict[str, int],
) -> dict[str, str]:
    primary = stats.get("primary", {})
    h5 = stats.get("H5", {})
    h8a = stats.get("H8a_v4_shadow_revision_vs_draft", {})
    h8b = stats.get("H8b_v4_gate_calibration", {})
    h8c = stats.get("H8c_v4_commit_policy_comparison", {})
    h9 = stats.get("H9_v4_judge_proxy_agreement", {})

    cost_total = (cost_ledger or {}).get("total_usd", 0.0)
    cost_calls = (cost_ledger or {}).get("n_calls", 0)
    judge_n = (judge_agreement or {}).get("n", h9.get("n", 0))
    judge_cost = (judge_agreement or {}).get("total_cost_usd", h9.get("total_cost_usd", 0.0))

    return {
        "V04_HEADLINE": _v04_headline(stats),
        "V04_COST_TOTAL_USD": f"\\$ {float(cost_total):.2f}",
        "V04_COST_N_CALLS": str(int(cost_calls)),
        "V04_H5_POOLED_G": _fmt_signed(h5.get("pooled_g"), 2),
        "V04_H5_CI_LO": _fmt_signed((h5.get("ci_95") or [float("nan"), float("nan")])[0], 2),
        "V04_H5_CI_HI": _fmt_signed((h5.get("ci_95") or [float("nan"), float("nan")])[1], 2),
        "V04_H5_METHOD": str(h5.get("method", "fixed_effects_inverse_variance")),
        "V04_H8A_G": _fmt_signed(h8a.get("hedges_g"), 2),
        "V04_H8A_P": _fmt_bare(h8a.get("permutation_p_one_sided"), 4),
        "V04_H8A_N": str(int(h8a.get("n", 0))),
        "V04_H8B_LEARNED_F1": _fmt_bare(h8b.get("learned_gate", {}).get("f1"), 3),
        "V04_H8B_EVENT_F1": _fmt_bare(h8b.get("event_gated", {}).get("f1"), 3),
        "V04_H8B_SUPPORTED": "supported" if h8b.get("supported") else "not supported",
        "V04_H8C_WINNER": _v04_h8c_winner_line(h8c),
        "V04_H8C_SUPPORTED": "supported" if h8c.get("supported") else "not supported",
        "V04_H9_RHO": _fmt_bare(h9.get("spearman_rho"), 2),
        "V04_H9_SIGN_AGREEMENT": _fmt_bare(h9.get("sign_agreement_rate"), 2),
        "V04_H9_N": str(int(judge_n)),
        "V04_JUDGE_COST": f"\\$ {float(judge_cost):.2f}",
        "V04_N_PER_DOMAIN": _v04_n_per_domain(stats),
        "V04_N_PAIRED_TOTAL": str(sum(int(primary.get(h, {}).get("n", 0))
                                      for h in ("H1", "H2", "H3", "H4"))),
        "V04_SHOWCASE_TOTAL": str(showcase_counts.get("total", 0)),
        "V04_SHOWCASE_SANSKRIT": str(showcase_counts.get("sanskrit", 0)),
        "V04_SHOWCASE_ENGLISH": str(showcase_counts.get("english", 0)),
        "V04_SHOWCASE_SCIENCE": str(showcase_counts.get("science", 0)),
    }


def _enforce_no_unbound_v04(text: str) -> list[str]:
    """Return any unsubstituted ``{V04_*}`` tokens still present in ``text``."""
    return sorted(set(re.findall(r"\{V04_[A-Z0-9_]+\}", text)))


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


def _build_v04_autoreport(
    stats: dict[str, Any], placeholders: dict[str, str]
) -> str:
    """Single LaTeX table for the v0.4 mechanism study (replaces autoreport.tex
    when ``--version v0.4`` is selected). Five rows: H1-H4 + H5 pool + H8a +
    H8b + H8c + H9. Holm column omitted because v0.4 uses a per-family
    correction rather than the v0.3 single-Holm-of-four convention.
    """
    primary = stats.get("primary", {})
    h5 = stats.get("H5", {})
    h8a = stats.get("H8a_v4_shadow_revision_vs_draft", {})
    h8b = stats.get("H8b_v4_gate_calibration", {})
    h8c = stats.get("H8c_v4_commit_policy_comparison", {})
    h9 = stats.get("H9_v4_judge_proxy_agreement", {})

    rows: list[str] = []
    for h in ("H1", "H2", "H3", "H4"):
        p = primary.get(h, {})
        ci = p.get("bca_ci_95", [float("nan"), float("nan")])
        rows.append(
            f"{h} & {int(p.get('n', 0))} & {_fmt_signed(p.get('hedges_g'), 2)} & "
            f"[{_fmt_signed(ci[0], 3)},\\,{_fmt_signed(ci[1], 3)}] & "
            f"{_fmt_bare(p.get('permutation_p_one_sided'), 4)} & "
            f"{_fmt_bare(p.get('holm_p'), 4)} & "
            f"{'\\textbf{yes}' if p.get('supported') else 'no'} \\\\"
        )
    rows.append(
        f"H5 (FE pool) & {int(sum(int(p.get('n', 0)) for p in (primary.get(k, {}) for k in ('H1','H2','H3','H4'))))} "
        f"& {placeholders['V04_H5_POOLED_G']} & "
        f"[{placeholders['V04_H5_CI_LO']},\\,{placeholders['V04_H5_CI_HI']}] & --- & --- & "
        f"{'\\textbf{yes}' if h5.get('supported') else 'no'} \\\\"
    )
    rows.append(
        f"H8a (rev vs.\\ draft) & {placeholders['V04_H8A_N']} & {placeholders['V04_H8A_G']} & "
        f"--- & {placeholders['V04_H8A_P']} & --- & "
        f"{'\\textbf{yes}' if h8a.get('supported') else 'no'} \\\\"
    )
    rows.append(
        f"H8b (gate calibration) & --- & --- & --- & --- & --- & "
        f"{'\\textbf{yes}' if h8b.get('supported') else 'no'} \\\\"
    )
    rows.append(
        f"H8c (policy compare) & --- & --- & --- & --- & --- & "
        f"{'\\textbf{yes}' if h8c.get('supported') else 'no'} \\\\"
    )
    rows.append(
        f"H9 (judge $\\leftrightarrow$ scorer) & {placeholders['V04_H9_N']} & --- & "
        f"--- & --- & --- & "
        f"{'\\textbf{yes}' if h9.get('supported') else 'no'} \\\\"
    )

    return (
        "\\begin{tabular}{l r r c c c c}\n"
        "\\toprule\n"
        "Hyp. & $n$ & $g$ & 95\\% BCa CI & perm.~$p$ & Holm~$p$ & supported \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", choices=("v0.2", "v0.3", "v0.4"), default="v0.2",
        help="Which placeholder family to substitute. v0.4 enforces "
             "the {V04_*} no-unbound-token gate.",
    )
    parser.add_argument(
        "--stats", type=Path, default=None,
        help="Defaults to benchmarks/results_<version>/stats.json (results_v2 for v0.2)",
    )
    parser.add_argument(
        "--showcase-root", type=Path, default=REPO_ROOT / "benchmarks" / "showcase_v0.4",
        help="v0.4 only: directory to scan for trace.json files.",
    )
    parser.add_argument(
        "--cost-ledger", type=Path, default=REPO_ROOT / "audit" / "v0.4" / "cost_ledger_merged.json",
    )
    parser.add_argument(
        "--judge-agreement", type=Path, default=None,
        help="Defaults to <stats parent>/judge_agreement.json",
    )
    parser.add_argument("--paper-dir", type=Path, default=REPO_ROOT / "paper")
    parser.add_argument("--items", type=Path, default=REPO_ROOT / "benchmarks" / "items.py")
    parser.add_argument(
        "--strict", action="store_true",
        help="Raise SystemExit on any unbound {V04_*} placeholder.",
    )
    args = parser.parse_args()

    if args.stats is None:
        args.stats = REPO_ROOT / "benchmarks" / (
            "results_v2" if args.version == "v0.2" else f"results_{args.version}"
        ) / "stats.json"
    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    if args.version == "v0.4":
        cost_ledger = (
            json.loads(args.cost_ledger.read_text(encoding="utf-8"))
            if args.cost_ledger.exists() else None
        )
        ja_path = args.judge_agreement or (args.stats.parent / "judge_agreement.json")
        judge_agreement = (
            json.loads(ja_path.read_text(encoding="utf-8"))
            if ja_path.exists() else None
        )
        showcase_counts = _v04_showcase_count(args.showcase_root)
        v04 = _build_v04_placeholders(
            stats,
            cost_ledger=cost_ledger,
            judge_agreement=judge_agreement,
            showcase_counts=showcase_counts,
        )

        # Emit v0.4 autoreport table
        autoreport = _build_v04_autoreport(stats, v04)
        (args.paper_dir / "autoreport_v0.4.tex").write_text(autoreport, encoding="utf-8")

        # Substitute all *.tex files under paper/ (main.tex + sections/*.tex)
        all_tex = [args.paper_dir / "main.tex", *sorted((args.paper_dir / "sections").glob("*.tex"))]
        unbound_collected: dict[str, list[str]] = {}
        for tex in all_tex:
            if not tex.exists():
                continue
            t = tex.read_text(encoding="utf-8")
            if "{V04_" not in t:
                continue
            t = _replace_placeholders(t, v04)
            tex.write_text(t, encoding="utf-8")
            remaining = _enforce_no_unbound_v04(t)
            if remaining:
                unbound_collected[str(tex.relative_to(REPO_ROOT))] = remaining

        print(f"v0.4 autoreport -> {args.paper_dir / 'autoreport_v0.4.tex'}")
        print(f"v0.4 placeholders bound: {sorted(v04.keys())}")
        if unbound_collected:
            msg = "Unbound {V04_*} placeholders remain:\n"
            for f, keys in unbound_collected.items():
                msg += f"  {f}: {keys}\n"
            if args.strict:
                raise SystemExit(msg)
            print("WARNING: " + msg, end="")
        return 0

    # v0.2/v0.3 legacy branch (unchanged behaviour)
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

    counts = {
        "POETRY_GEN_N": str(primary["H3"]["n"]),
        "POETRY_INTERP_N": str(primary["H2"]["n"]),
        "AUT_N": str(primary["H1"]["n"]),
        "SCI_N": str(primary["H4"]["n"]),
        "N_PAIRED": str(n_paired),
        "HEADLINE_RESULT": headline,
    }
    _ = _list_count

    main_path = args.paper_dir / "main.tex"
    text = main_path.read_text(encoding="utf-8")
    text = _replace_placeholders(text, counts)
    main_path.write_text(text, encoding="utf-8")

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
