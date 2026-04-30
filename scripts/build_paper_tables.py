#!/usr/bin/env python3
r"""Generate LaTeX table snippets for the v0.4 paper from JSON artefacts.

Emits four ``.tex`` files under ``paper/sections/_tables/`` that the section
files ``\input{}`` directly. The tables regenerate from the same JSON
artefacts that feed the site CostPanel, the figure pack, and the autoreport,
so the v0.4.2 ralph-loop keeps every surface consistent.

Tables produced:
    * ``tab_per_axis_effects.tex`` (T1) — paired Δ per axis × hypothesis.
    * ``tab_per_domain_raw.tex`` (T2) — n / mean / median / sd / min / max
      for treatment vs. control on every primary domain.
    * ``tab_cost_split.tex`` (T3) — Haiku cascade vs. Sonnet judge USD +
      n_calls per domain.
    * ``tab_showcase_registry.tex`` (T4) — slug / domain / source / validator
      / commit policy for every showcase entry.

Run with::

    python scripts/build_paper_tables.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results_v0.4"
SHOWCASE = REPO / "benchmarks" / "showcase_v0.4"
AUDIT = REPO / "audit" / "v0.4"
OUT_DIR = REPO / "paper" / "sections" / "_tables"

DOMAINS = ("aut", "poetry_interp", "poetry_gen", "sci_creativity")
DOMAIN_DISPLAY = {
    "aut": "AUT",
    "poetry_interp": "Poetry-interp",
    "poetry_gen": "Poetry-gen",
    "sci_creativity": "Sci-creativity",
}
HYPOTHESIS_FOR_DOMAIN = {
    "aut": "H1",
    "poetry_interp": "H2",
    "poetry_gen": "H3",
    "sci_creativity": "H4",
}


def _load_results(domain: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{domain}.json").read_text(encoding="utf-8"))


def _load_stats() -> dict[str, Any]:
    return json.loads((RESULTS / "stats.json").read_text(encoding="utf-8"))


def _esc(text: str) -> str:
    """Minimal LaTeX escaping for free-form strings used in table cells."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("$", r"\$")
        .replace("^", r"\textasciicircum{}")
        .replace("\u00d7", r"$\times$")
    )


def _paired_axis_deltas(
    rows: dict[str, Any], axis: str, treatment: str = "haiku_cascade", control: str = "haiku_bare"
) -> list[float]:
    deltas: list[float] = []
    for _, payload in sorted(rows.items()):
        t_row = payload.get(treatment) or {}
        c_row = payload.get(control) or {}
        if not isinstance(t_row, dict) or not isinstance(c_row, dict):
            continue
        t_axes = t_row.get("axes") or {}
        c_axes = c_row.get("axes") or {}
        if axis not in t_axes or axis not in c_axes:
            continue
        try:
            delta = float(t_axes[axis]) - float(c_axes[axis])
        except (TypeError, ValueError):
            continue
        if math.isfinite(delta):
            deltas.append(delta)
    return deltas


def _axis_keys(rows: dict[str, Any], arm: str = "haiku_bare") -> list[str]:
    for _, payload in sorted(rows.items()):
        row = payload.get(arm) or {}
        axes = row.get("axes") if isinstance(row, dict) else None
        if isinstance(axes, dict) and axes:
            return list(axes.keys())
    return []


def _cohen_d_paired(deltas: list[float]) -> float:
    """Standardised mean of paired differences (~Hedges' g for moderate n)."""
    if len(deltas) < 2:
        return 0.0
    mean = statistics.fmean(deltas)
    sd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return mean / sd


# ---------------------------------------------------------------------------
# T1: per-axis effects (paired mean Δ + Cohen's d_z per (domain, axis))
# ---------------------------------------------------------------------------


def build_per_axis_effects() -> str:
    rows_lines: list[str] = []
    for domain in DOMAINS:
        data = _load_results(domain)
        rows = data.get("rows", {})
        axes = _axis_keys(rows)
        if not axes:
            continue
        for ax in axes:
            deltas = _paired_axis_deltas(rows, ax)
            if not deltas:
                continue
            mean = statistics.fmean(deltas)
            d = _cohen_d_paired(deltas)
            n = len(deltas)
            ax_label = ax.replace("_", " ")
            rows_lines.append(
                f"  {DOMAIN_DISPLAY[domain]} & "
                f"{HYPOTHESIS_FOR_DOMAIN[domain]} & "
                f"\\texttt{{{_esc(ax_label)}}} & "
                f"{n} & "
                f"{mean:+.3f} & "
                f"{d:+.2f} \\\\"
            )
    body = "\n".join(rows_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{T1 --- Per-axis paired effects on the four primary "
        "(haiku\\_cascade $-$ haiku\\_bare) contrasts. ``Paired mean $\\Delta$'' is the "
        "raw paired axis difference; ``$d_z$'' is the standardised mean of paired differences "
        "(approx.\\ Hedges' $g$ for moderate $n$). Domain axis vocabularies differ; the "
        "rows are not intended to be cross-domain comparable.}\n"
        "\\label{tab:per_axis_effects}\n"
        "\\begin{tabular}{lllrrr}\n"
        "\\toprule\n"
        "Domain & Hypothesis & Axis & $n$ & Paired mean $\\Delta$ & $d_z$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# T2: per-domain raw composite summary (n / mean / median / sd / min / max)
# ---------------------------------------------------------------------------


def _composites(rows: dict[str, Any], arm: str) -> list[float]:
    out: list[float] = []
    for _, payload in sorted(rows.items()):
        r = payload.get(arm) or {}
        comp = r.get("composite") if isinstance(r, dict) else None
        if comp is None:
            continue
        try:
            v = float(comp)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def build_per_domain_raw() -> str:
    rows_lines: list[str] = []
    for domain in DOMAINS:
        data = _load_results(domain)
        rows = data.get("rows", {})
        for arm_label, arm in (("bare", "haiku_bare"), ("cascade", "haiku_cascade")):
            vals = _composites(rows, arm)
            if not vals:
                continue
            rows_lines.append(
                f"  {DOMAIN_DISPLAY[domain]} & "
                f"{arm_label} & "
                f"{len(vals)} & "
                f"{statistics.fmean(vals):.3f} & "
                f"{statistics.median(vals):.3f} & "
                f"{(statistics.pstdev(vals) if len(vals) > 1 else 0.0):.3f} & "
                f"{min(vals):.3f} & "
                f"{max(vals):.3f} \\\\"
            )
    body = "\n".join(rows_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{T2 --- Per-domain composite-score summary statistics for the "
        "primary cascade-vs-bare contrast. Both arms are paired item-by-item; row "
        "counts may differ from the per-domain $n$ in Table~\\ref{tab:v04headline} "
        "when an arm-specific composite is missing.}\n"
        "\\label{tab:per_domain_raw}\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\toprule\n"
        "Domain & Arm & $n$ & Mean & Median & SD & Min & Max \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# T3: cost ledger split per domain
# ---------------------------------------------------------------------------


def build_cost_split() -> str:
    rows_lines: list[str] = []
    total_cascade_usd = 0.0
    total_cascade_calls = 0
    for domain in DOMAINS:
        ledger_path = AUDIT / f"cost_ledger_{domain}.json"
        if not ledger_path.exists():
            continue
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        usd = float(ledger.get("total_usd", 0.0))
        calls = int(ledger.get("n_calls", 0))
        total_cascade_usd += usd
        total_cascade_calls += calls
        rows_lines.append(
            f"  {DOMAIN_DISPLAY[domain]} & "
            f"Haiku cascade & "
            f"{calls} & "
            f"\\${usd:.2f} \\\\"
        )
    judge_path = RESULTS / "judge_agreement.json"
    if judge_path.exists():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        judge_usd = float(judge.get("total_cost_usd", 0.0))
        judge_calls = int(judge.get("n", 0))
        rows_lines.append(
            "  \\midrule"
        )
        rows_lines.append(
            f"  Cross-domain & Sonnet judge (H9) & {judge_calls} & \\${judge_usd:.2f} \\\\"
        )
        total_usd = total_cascade_usd + judge_usd
        total_calls = total_cascade_calls + judge_calls
    else:
        total_usd = total_cascade_usd
        total_calls = total_cascade_calls
    rows_lines.append("  \\midrule")
    rows_lines.append(
        f"  Total & --- & {total_calls} & \\${total_usd:.2f} \\\\"
    )
    body = "\n".join(rows_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{T3 --- v0.4 Phase 7 pilot cost ledger, split per model line. The Haiku "
        "rows aggregate the four cascade arms running through the managed-API substrate; "
        "the Sonnet judge row covers the 23-pair stratified bridge (H9 agreement). Numbers "
        "match \\nolinkurl{audit/v0.4/cost_ledger_merged.json} and \\nolinkurl{benchmarks/results_v0.4/judge_agreement.json}.}\n"
        "\\label{tab:cost_split}\n"
        "\\begin{tabular}{llrr}\n"
        "\\toprule\n"
        "Domain & Model line & $n$ calls & Spend (USD) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# T4: showcase registry
# ---------------------------------------------------------------------------


def _read_validator(showcase_dir: Path) -> str:
    v = showcase_dir / "validator.json"
    if not v.exists():
        return "n/a"
    try:
        row = json.loads(v.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "bad-json"
    if "ok" in row:
        ok = bool(row["ok"])
        notes = row.get("notes")
        if isinstance(notes, list) and notes:
            return f"{'pass' if ok else 'review'} ({_esc(str(notes[0])[:48])})"
        return "pass" if ok else "review"
    if "haiku_5_7_5" in row:
        ok = bool((row.get("haiku_5_7_5") or {}).get("ok"))
        return "5-7-5 pass" if ok else "5-7-5 review"
    if "imagism" in row:
        return "imagism noted"
    return "n/a"


def _read_trace(showcase_dir: Path) -> dict[str, Any]:
    t = showcase_dir / "trace.json"
    if not t.exists():
        return {}
    try:
        return json.loads(t.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_showcase_registry() -> str:
    rows_lines: list[str] = []
    for slug_dir in sorted(SHOWCASE.iterdir()):
        if not slug_dir.is_dir():
            continue
        trace = _read_trace(slug_dir)
        domain = trace.get("domain") or "?"
        source = trace.get("source") or trace.get("mode") or "?"
        committed = trace.get("committed_choice") or trace.get("committed") or "?"
        validator = _read_validator(slug_dir)
        rows_lines.append(
            f"  \\texttt{{{_esc(slug_dir.name)}}} & "
            f"\\texttt{{{_esc(str(domain))}}} & "
            f"\\texttt{{{_esc(str(source))}}} & "
            f"\\texttt{{{_esc(str(committed))}}} & "
            f"{_esc(validator)} \\\\"
        )
    body = "\n".join(rows_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{T4 --- Showcase items registry. ``Source'' is the v0.4.1 trace "
        "provenance: \\texttt{live\\_cascade\\_v0\\_4\\_1} for the three Sanskrit "
        "items (regenerated under the live-cascade mode), \\texttt{phase7\\_cascade} "
        "for the curated English-poetry and scientific-creativity items. ``Commit'' "
        "names the surface that the cascade actually committed (draft / shadow "
        "revision / committed\\_choice). ``Validator'' is reported informationally; "
        "for Sanskrit items the v0.4 cascade scorer is not chandas-aware "
        "(v0.5 ladder).}\n"
        "\\label{tab:showcase_registry}\n"
        "\\begin{tabular}{lllll}\n"
        "\\toprule\n"
        "Slug & Domain & Source & Commit & Validator \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "tab_per_axis_effects.tex": build_per_axis_effects,
        "tab_per_domain_raw.tex": build_per_domain_raw,
        "tab_cost_split.tex": build_cost_split,
        "tab_showcase_registry.tex": build_showcase_registry,
    }
    for name, fn in builders.items():
        text = fn()
        out_path = OUT_DIR / name
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path.relative_to(REPO)} ({len(text)}b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
