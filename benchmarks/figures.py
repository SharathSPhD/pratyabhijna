#!/usr/bin/env python3
"""Generate matplotlib figures from benchmarks/results/*.json + stats.json.

Outputs to `presentation/figures/` and `paper/figures/` (the same files; copied
to keep both surfaces in sync). All figures are deterministic (no styling beyond
matplotlib defaults; we set seed-free RNG only for jitter in raincloud plots).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
DOMAIN_LABEL = {
    "poetry_gen": "Poetry Generation\n(POEMetric)",
    "poetry_interp": "Poetry Interpretation\n(Wittgenstein aspect-shift)",
    "aut": "Alternative Uses Task\n(CreativityPrism)",
    "sci_creativity": "Scientific Creativity\n(BBH)",
}
ARM_COLOR = {
    "claude_haiku": "#7c8aa8",
    "local_bare": "#bbbbbb",
    "local_cascade": "#3b6ea8",
}


def _load(results_dir: Path, domain: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((results_dir / f"{domain}.json").read_text(encoding="utf-8"))
    return loaded


def _composites_per_arm(rows: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"claude_haiku": [], "local_bare": [], "local_cascade": []}
    for _, payload in sorted(rows.items()):
        for arm in out:
            row = payload.get(arm, {})
            comp = row.get("composite") if isinstance(row, dict) else None
            if comp is not None and math.isfinite(float(comp)):
                out[arm].append(float(comp))
    return out


def _figure_per_domain_box(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=False)
    for ax, dom in zip(axes, DOMAINS, strict=True):
        data = _load(results_dir, dom)
        comps = _composites_per_arm(data["rows"])
        labels = ["Haiku\n(no PCE)", "Local-Qwen\n(no PCE)", "Local-Qwen\n+PCE"]
        vals = [comps["claude_haiku"], comps["local_bare"], comps["local_cascade"]]
        ax.boxplot(vals, tick_labels=labels, patch_artist=True,
                    boxprops={"facecolor": "#eef1f7", "edgecolor": "#345"},
                    medianprops={"color": "#3b6ea8", "linewidth": 2})
        rng = np.random.default_rng(0)
        for i, v in enumerate(vals):
            jitter = rng.uniform(-0.12, 0.12, size=len(v))
            ax.scatter(np.full(len(v), i + 1) + jitter, v,
                       alpha=0.55, s=18, color=ARM_COLOR[list(comps.keys())[i]], edgecolors="white")
        ax.set_title(DOMAIN_LABEL[dom], fontsize=10)
        ax.set_ylabel("composite score")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Per-domain composite scores: Haiku vs Local-Qwen vs Local-Qwen+PCE", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "fig_per_domain_box.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_paired_deltas(results_dir: Path, stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats["primary"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=False)
    for ax, (h, dom) in zip(axes, [("H1", "aut"), ("H2", "poetry_interp"), ("H3", "poetry_gen"), ("H4", "sci_creativity")], strict=True):
        data = _load(results_dir, dom)
        deltas: list[float] = []
        for _, payload in sorted(data["rows"].items()):
            t = payload.get("local_cascade", {})
            c = payload.get("claude_haiku", {})
            if not isinstance(t, dict) or not isinstance(c, dict):
                continue
            ts, cs = t.get("composite"), c.get("composite")
            if ts is None or cs is None:
                continue
            deltas.append(float(ts) - float(cs))
        if not deltas:
            ax.set_visible(False)
            continue
        ax.hist(deltas, bins=12, color="#bccdf0", edgecolor="#345")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        est = primary[h]["estimate"]
        ax.axvline(est, color="#3b6ea8", linewidth=2, label=f"mean Δ={est:+.2f}")
        ci = primary[h]["bca_ci_95"]
        ax.axvspan(ci[0], ci[1], color="#3b6ea8", alpha=0.18, label="95% BCa CI")
        ax.set_title(f"{h}: {dom}\nn={primary[h]['n']}, Holm p={primary[h]['holm_p']:.3f}", fontsize=10)
        ax.set_xlabel("Δ composite (PCE − Haiku)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Paired score deltas with 95% BCa CI (one-sided H₁: Δ>0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paired_deltas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_effects_forest(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats["primary"]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    rows = ["H1 (AUT)", "H2 (Interp)", "H3 (Poetry-gen)", "H4 (Sci-creat)"]
    keys = ["H1", "H2", "H3", "H4"]
    ests = [primary[k]["estimate"] for k in keys]
    ci_lo = [primary[k]["bca_ci_95"][0] for k in keys]
    ci_hi = [primary[k]["bca_ci_95"][1] for k in keys]
    y = np.arange(len(rows))[::-1]
    ax.errorbar(ests, y, xerr=[np.array(ests) - np.array(ci_lo), np.array(ci_hi) - np.array(ests)],
                fmt="o", color="#3b6ea8", ecolor="#3b6ea8", capsize=4, markersize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    for i, k in enumerate(keys):
        sup = primary[k]["supported"]
        ax.text(max(ci_hi) + 0.02, y[i], f"g={primary[k]['hedges_g']:+.2f}, p={primary[k]['holm_p']:.3f} {'★' if sup else ''}",
                va="center", fontsize=9)
    ax.set_yticks(y, rows)
    ax.set_xlabel("paired mean Δ composite (PCE − Haiku)")
    ax.set_title("Pre-registered hypothesis effects (95% BCa CI, Holm-adjusted)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_effects_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_h6_event_vs_no_event(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    h6 = stats["H6"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    n_fired = int(h6.get("n_fired", 0))
    n_not = int(h6.get("n_not_fired", 0))
    if n_fired == 0 and n_not == 0:
        ax.text(0.5, 0.5, "H6: insufficient vimarśa-event data", transform=ax.transAxes, ha="center")
    else:
        labels = [f"vimarśa-event\nfired (n={n_fired})", f"no event\n(n={n_not})"]
        ax.bar(labels, [h6["estimate"] + 0, 0], color=["#3b6ea8", "#bbbbbb"])
        ax.set_ylabel("mean composite (fired) − mean composite (no event)")
        ax.set_title(f"H6: within-PCE event-vs-no-event\np={h6.get('mannwhitney_u_p_one_sided', float('nan')):.3f}")
        ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_h6_event_vs_no_event.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_power(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats["primary"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    keys = ["H1", "H2", "H3", "H4"]
    pa = [primary[k]["power_apriori"] for k in keys]
    pr = [primary[k]["power_retrospective"] for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, pa, width=w, label="a-priori (g=0.5)", color="#bccdf0", edgecolor="#345")
    ax.bar(x + w / 2, pr, width=w, label="retrospective (observed g)", color="#3b6ea8", edgecolor="#345")
    ax.set_xticks(x, keys)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.80, color="red", linestyle=":", linewidth=1, label="0.80 threshold")
    ax.set_ylabel("statistical power")
    ax.set_title("A-priori vs retrospective power per hypothesis")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_power.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_axes_breakdown(results_dir: Path, out_dir: Path) -> None:
    """Spider/bar chart of per-axis means per arm, for poetry_gen (POEMetric) and aut."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, dom, title in zip(
        axes,
        ("poetry_gen", "aut"),
        ("POEMetric axes (poetry_gen)", "CreativityPrism axes (AUT)"),
        strict=True,
    ):
        data = _load(results_dir, dom)
        rows = data["rows"]
        # Collect per-axis means per arm
        sums: dict[str, dict[str, list[float]]] = {a: {} for a in ARM_COLOR}
        for _, payload in rows.items():
            for arm in ARM_COLOR:
                r = payload.get(arm, {})
                if not isinstance(r, dict):
                    continue
                axes_dict = r.get("axes", {})
                if not isinstance(axes_dict, dict):
                    continue
                for ax_name, val in axes_dict.items():
                    sums[arm].setdefault(ax_name, []).append(float(val))
        if not sums.get("local_cascade"):
            ax.set_visible(False)
            continue
        axis_names = list(sums["local_cascade"].keys())
        x = np.arange(len(axis_names))
        w = 0.27
        for i, arm in enumerate(("claude_haiku", "local_bare", "local_cascade")):
            vals = [float(np.mean(sums[arm].get(a, [0.0]) or [0.0])) for a in axis_names]
            label = {"claude_haiku": "Haiku (no PCE)", "local_bare": "Local-Qwen (no PCE)", "local_cascade": "Local-Qwen + PCE"}[arm]
            ax.bar(x + (i - 1) * w, vals, width=w, color=ARM_COLOR[arm], edgecolor="#234", label=label)
        ax.set_xticks(x, [a.replace("_", "\n") for a in axis_names], fontsize=8)
        ax.set_ylabel("mean axis score")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_axes_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results")
    parser.add_argument("--stats", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "stats.json")
    parser.add_argument("--out-dirs", nargs="+", type=Path, default=[REPO_ROOT / "presentation" / "figures", REPO_ROOT / "paper" / "figures"])
    args = parser.parse_args()
    for out_dir in args.out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        _figure_per_domain_box(args.results_dir, out_dir)
        _figure_paired_deltas(args.results_dir, args.stats, out_dir)
        _figure_effects_forest(args.stats, out_dir)
        _figure_h6_event_vs_no_event(args.stats, out_dir)
        _figure_power(args.stats, out_dir)
        _figure_axes_breakdown(args.results_dir, out_dir)
        print(f"figures -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
