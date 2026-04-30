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
# v0.2 four-arm palette + display labels.
ARM_COLOR = {
    "local_bare": "#bbbbbb",
    "local_cascade": "#5a8d5a",
    "haiku_bare": "#7c8aa8",
    "haiku_cascade": "#3b6ea8",
    # v0.1 alias kept for back-compat.
    "claude_haiku": "#7c8aa8",
}
ARM_LABEL = {
    "local_bare": "Local-Qwen\n(no PCE)",
    "local_cascade": "Local-Qwen\n+ PCE",
    "haiku_bare": "Haiku\n(no PCE)",
    "haiku_cascade": "Haiku\n+ PCE",
    "claude_haiku": "Haiku\n(no PCE)",
}
DEFAULT_ARMS = ("local_bare", "local_cascade", "haiku_bare", "haiku_cascade")


def _load(results_dir: Path, domain: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((results_dir / f"{domain}.json").read_text(encoding="utf-8"))
    return loaded


def _composites_per_arm(
    rows: dict[str, dict[str, Any]],
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {a: [] for a in arms}
    for _, payload in sorted(rows.items()):
        for arm in out:
            row = payload.get(arm, {})
            comp = row.get("composite") if isinstance(row, dict) else None
            if comp is not None and math.isfinite(float(comp)):
                out[arm].append(float(comp))
    return out


def _figure_per_domain_box(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=False)
    arms = DEFAULT_ARMS
    for ax, dom in zip(axes, DOMAINS, strict=True):
        data = _load(results_dir, dom)
        comps = _composites_per_arm(data["rows"], arms=arms)
        labels = [ARM_LABEL[a] for a in arms]
        vals = [comps[a] for a in arms]
        if not any(vals):
            ax.set_visible(False)
            continue
        ax.boxplot(vals, tick_labels=labels, patch_artist=True,
                    boxprops={"facecolor": "#eef1f7", "edgecolor": "#345"},
                    medianprops={"color": "#3b6ea8", "linewidth": 2})
        rng = np.random.default_rng(0)
        for i, v in enumerate(vals):
            if not v:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(v))
            ax.scatter(np.full(len(v), i + 1) + jitter, v,
                       alpha=0.55, s=18, color=ARM_COLOR[arms[i]], edgecolors="white")
        ax.set_title(DOMAIN_LABEL[dom], fontsize=10)
        ax.set_ylabel("composite score")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(
        "PCE v0.2: per-domain composites — local & Haiku, with/without PCE",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "fig_per_domain_box.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_paired_deltas(results_dir: Path, stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats["primary"]
    cfg = stats.get("config", {})
    treatment = cfg.get("treatment_arm", "haiku_cascade")
    control = cfg.get("control_arm_primary", "haiku_bare")
    contrast_label = (
        f"Δ composite ({ARM_LABEL[treatment].replace(chr(10), ' ')}"
        f" − {ARM_LABEL[control].replace(chr(10), ' ')})"
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=False)
    for ax, (h, dom) in zip(
        axes,
        [("H1", "aut"), ("H2", "poetry_interp"), ("H3", "poetry_gen"), ("H4", "sci_creativity")],
        strict=True,
    ):
        data = _load(results_dir, dom)
        deltas: list[float] = []
        for _, payload in sorted(data["rows"].items()):
            t = payload.get(treatment, {})
            c = payload.get(control, {})
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
        ax.set_title(
            f"{h}: {dom}\nn={primary[h]['n']}, Holm p={primary[h]['holm_p']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel(contrast_label)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Paired score deltas with 95% BCa CI (one-sided H₁: Δ>0); contrast = "
        f"{treatment} − {control}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paired_deltas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_effects_forest(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats["primary"]
    cfg = stats.get("config", {})
    treatment = cfg.get("treatment_arm", "haiku_cascade")
    control = cfg.get("control_arm_primary", "haiku_bare")
    contrast = f"{treatment} − {control}"
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    rows = ["H1 (AUT)", "H2 (Interp)", "H3 (Poetry-gen)", "H4 (Sci-creat)"]
    keys = ["H1", "H2", "H3", "H4"]
    ests = [float(primary[k]["estimate"]) for k in keys]
    ci_lo = [float(primary[k]["bca_ci_95"][0]) for k in keys]
    ci_hi = [float(primary[k]["bca_ci_95"][1]) for k in keys]
    y = np.arange(len(rows))[::-1]
    ax.errorbar(
        ests, y,
        xerr=[np.array(ests) - np.array(ci_lo), np.array(ci_hi) - np.array(ests)],
        fmt="o", color="#3b6ea8", ecolor="#3b6ea8", capsize=4, markersize=8,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    if ci_hi:
        text_x = max(ci_hi) + 0.02
        for i, k in enumerate(keys):
            sup = bool(primary[k]["supported"])
            ax.text(
                text_x, y[i],
                f"g={primary[k]['hedges_g']:+.2f}, p={primary[k]['holm_p']:.3f} {'★' if sup else ''}",
                va="center", fontsize=9,
            )
    ax.set_yticks(y, rows)
    ax.set_xlabel(f"paired mean Δ composite ({contrast})")
    ax.set_title(
        f"Pre-registered hypothesis effects (95% BCa CI, Holm-adjusted)\nprimary contrast: {contrast}"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_effects_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_h6_event_vs_no_event(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    # v0.2 emits H6 per cascade arm. Older v0.1 stats had a single "H6" key.
    h6_local = stats.get("H6_local_cascade") or stats.get("H6") or {}
    h6_haiku = stats.get("H6_haiku_cascade") or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, h6, title in zip(
        axes, (h6_local, h6_haiku),
        ("H6 (local_cascade)", "H6 (haiku_cascade)"),
        strict=True,
    ):
        n_fired = int(h6.get("n_fired", 0))
        n_not = int(h6.get("n_not_fired", 0))
        if n_fired == 0 and n_not == 0:
            ax.text(0.5, 0.5, f"{title}: insufficient data", transform=ax.transAxes, ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        labels = [f"event fired\n(n={n_fired})", f"no event\n(n={n_not})"]
        ax.bar(labels, [float(h6.get("estimate", 0.0)), 0.0], color=["#3b6ea8", "#bbbbbb"])
        ax.set_ylabel("mean composite (fired) − mean composite (no event)")
        p = float(h6.get("mannwhitney_u_p_one_sided", float("nan")))
        ax.set_title(f"{title}\np={p:.3f}")
        ax.axhline(0, color="black", linewidth=0.6)
    fig.suptitle("H6: within-cascade vimarśa-event uplift", fontsize=11)
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
    """Bar chart of per-axis means per arm for the four v0.2 arms."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    arms = DEFAULT_ARMS
    for ax, dom, title in zip(
        axes,
        ("poetry_gen", "aut"),
        ("POEMetric axes (poetry_gen)", "CreativityPrism axes (AUT)"),
        strict=True,
    ):
        data = _load(results_dir, dom)
        rows = data["rows"]
        sums: dict[str, dict[str, list[float]]] = {a: {} for a in arms}
        for _, payload in rows.items():
            for arm in arms:
                r = payload.get(arm, {})
                if not isinstance(r, dict):
                    continue
                axes_dict = r.get("axes", {})
                if not isinstance(axes_dict, dict):
                    continue
                for ax_name, val in axes_dict.items():
                    sums[arm].setdefault(ax_name, []).append(float(val))
        # Use the largest arm's axis schema as the canonical axis order.
        axis_source = next((a for a in arms if sums[a]), None)
        if axis_source is None:
            ax.set_visible(False)
            continue
        axis_names = list(sums[axis_source].keys())
        x = np.arange(len(axis_names))
        n_arms = len(arms)
        w = 0.8 / n_arms
        offsets = (np.arange(n_arms) - (n_arms - 1) / 2.0) * w
        for i, arm in enumerate(arms):
            vals = [float(np.mean(sums[arm].get(a, [0.0]) or [0.0])) for a in axis_names]
            ax.bar(
                x + offsets[i], vals, width=w,
                color=ARM_COLOR[arm], edgecolor="#234",
                label=ARM_LABEL[arm].replace("\n", " "),
            )
        ax.set_xticks(x, [a.replace("_", "\n") for a in axis_names], fontsize=8)
        ax.set_ylabel("mean axis score")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_axes_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# v0.4 figure pack
# ---------------------------------------------------------------------------
#
# Six new figures wired to ``benchmarks/results_v0.4/stats.json`` +
# ``judge.jsonl`` + per-domain ledgers. Each writes into the v0.4 subdir
# under each ``--out-dir`` so that ``paper/figures/v0.4/*.png`` and
# ``docs/site/public/figures/v0.4/*.png`` stay in lockstep with the
# rendered LaTeX and the Astro Pages site (Phase 8 deliverables).

V04_PALETTE = {
    "haiku_bare": "#7c8aa8",
    "haiku_cascade": "#3b6ea8",
    "haiku_cascade_event_gated": "#3b6ea8",
    "haiku_cascade_always_draft": "#9bb5d6",
    "haiku_cascade_always_revise": "#1f4d8f",
    "haiku_cascade_learned_gate": "#5a8d5a",
    "haiku_cascade_oracle": "#c8a13b",
    "haiku_bare_2K_scorer": "#a890c8",
    "haiku_generic_revise_2pass": "#c87b7b",
}
V04_POLICY_LABEL = {
    "haiku_cascade_always_draft": "always_draft",
    "haiku_cascade_always_revise": "always_revise",
    "haiku_cascade_event_gated": "event_gated",
    "haiku_cascade_learned_gate": "learned_gate",
    "haiku_cascade_oracle": "oracle (analysis)",
}


def _figure_v04_h5_fixed_forest(stats_path: Path, out_dir: Path) -> None:
    """Fixed-effects forest plot for H5.v4 (per-domain g + pooled diamond)."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    h5 = stats.get("H5", {})
    primary = stats.get("primary", {})
    domain_g = h5.get("per_domain_g", [])
    domain_n = h5.get("per_domain_n", [])
    pooled_g = float(h5.get("pooled_g", 0.0))
    pooled_ci = h5.get("ci_95", [pooled_g, pooled_g])
    domain_keys = ("H1", "H2", "H3", "H4")
    domain_names = {
        "H1": "AUT",
        "H2": "Poetry-interp",
        "H3": "Poetry-gen",
        "H4": "Sci-creativity",
    }
    rows: list[tuple[str, float, tuple[float, float], int]] = []
    for key in domain_keys:
        if key not in domain_g:
            continue
        h = primary.get(key, {})
        ci = h.get("bca_ci_95", [0.0, 0.0])
        rows.append(
            (
                f"{key}: {domain_names[key]}",
                float(domain_g[key]),
                (float(ci[0]), float(ci[1])),
                int(domain_n.get(key, h.get("n", 0)) if isinstance(domain_n, dict) else 0),
            )
        )
    if not rows:
        return
    total_n = (
        sum(int(v) for v in domain_n.values()) if isinstance(domain_n, dict)
        else sum(int(v) for v in domain_n)
    )
    rows.append(
        ("H5: pooled (FE)", pooled_g, (float(pooled_ci[0]), float(pooled_ci[1])), total_n)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(rows))[::-1]
    labels = [r[0] for r in rows]
    ests = [r[1] for r in rows]
    cis_lo = [r[2][0] for r in rows]
    cis_hi = [r[2][1] for r in rows]
    is_pool = [i == len(rows) - 1 for i in range(len(rows))]
    for i, (yi, est, lo, hi, pool) in enumerate(
        zip(y, ests, cis_lo, cis_hi, is_pool, strict=True)
    ):
        color = "#1f4d8f" if pool else "#3b6ea8"
        marker = "D" if pool else "o"
        ax.errorbar(
            est, yi,
            xerr=[[max(0, est - lo)], [max(0, hi - est)]],
            fmt=marker, color=color, ecolor=color, capsize=5,
            markersize=11 if pool else 8,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Hedges' g (haiku_cascade − haiku_bare, paired)")
    ax.set_title(
        f"H5.v4 fixed-effects pool across four domains\n"
        f"pooled g = {pooled_g:+.3f}  95% CI [{pooled_ci[0]:+.3f}, {pooled_ci[1]:+.3f}]",
        fontsize=10,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_h5_fixed_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_h8a_revision_vs_draft(
    results_dir: Path, stats_path: Path, out_dir: Path
) -> None:
    """Paired-delta histogram for H8a (revision composite − draft composite)."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    h8a = stats.get("H8a_v4_shadow_revision_vs_draft", {})
    deltas: list[float] = []
    # Mirror benchmarks/stats.py::_h8a_shadow_revision_vs_draft_all_items
    # which sources scores from haiku_cascade_always_revise.meta.score_{draft,revision}
    # (always-revise multiplexer arm). We deliberately don't re-score from
    # surface_* text — that would require Embedder + scorer machinery and
    # would drift from the stats engine.
    for dom in DOMAINS:
        data = _load(results_dir, dom)
        for _, payload in sorted(data["rows"].items()):
            ar_row = payload.get("haiku_cascade_always_revise", {})
            if not isinstance(ar_row, dict):
                continue
            meta = ar_row.get("meta", {}) if isinstance(ar_row.get("meta"), dict) else {}
            sd = meta.get("score_draft")
            sr = meta.get("score_revision")
            try:
                if sd is None or sr is None:
                    continue
                deltas.append(float(sr) - float(sd))
            except (TypeError, ValueError):
                continue
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if not deltas:
        ax.text(0.5, 0.5, "H8a: no draft/revised pairs in result tree",
                transform=ax.transAxes, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(deltas, bins=20, color="#bccdf0", edgecolor="#345")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        mean_delta = float(np.mean(deltas))
        ax.axvline(mean_delta, color="#1f4d8f", linewidth=2,
                   label=f"mean Δ = {mean_delta:+.3f}")
        ci = h8a.get("bca_ci_95", [0.0, 0.0])
        ax.axvspan(float(ci[0]), float(ci[1]), color="#1f4d8f", alpha=0.18,
                   label=f"95% BCa CI [{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]")
        ax.set_xlabel("Δ composite (revised − draft)")
        ax.set_ylabel("count of items")
        g = float(h8a.get("hedges_g", 0.0))
        p = float(h8a.get("permutation_p_one_sided", 1.0))
        ax.set_title(
            f"H8a.v4 — shadow revision vs draft\n"
            f"n={int(h8a.get('n', len(deltas)))}, g={g:+.3f}, p={p:.4g}",
            fontsize=10,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_h8a_revision_vs_draft.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_h8b_gate_calibration(stats_path: Path, out_dir: Path) -> None:
    """Bar chart: precision/recall/F1/accuracy for event_gated vs learned_gate."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    h8b = stats.get("H8b_v4_gate_calibration", {})
    event = h8b.get("event_gated", {})
    learned = h8b.get("learned_gate", {})
    metrics = ("precision", "recall", "f1", "accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    w = 0.36
    e_vals = [float(event.get(m, 0.0) or 0.0) for m in metrics]
    l_vals = [float(learned.get(m, 0.0) or 0.0) for m in metrics]
    ax.bar(x - w / 2, e_vals, width=w, label="event_gated", color="#7c8aa8", edgecolor="#234")
    ax.bar(x + w / 2, l_vals, width=w, label="learned_gate", color="#5a8d5a", edgecolor="#234")
    for i, v in enumerate(e_vals):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(l_vals):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("classifier metric vs OracleCommit ground truth")
    n_event = int(event.get("n", 0))
    n_learned = int(learned.get("n", 0))
    sup = bool(h8b.get("supported", False))
    ax.set_title(
        f"H8b.v4 — gate calibration (event vs learned, supported={sup})\n"
        f"n_event={n_event}, n_learned={n_learned}",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_h8b_gate_calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_h8c_policy_leaderboard(stats_path: Path, out_dir: Path) -> None:
    """Horizontal bar chart of commit-policy leaderboard vs bare control."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    h8c = stats.get("H8c_v4_commit_policy_comparison", {})
    leader = list(h8c.get("leader_board", []))
    if not leader:
        return
    leader.sort(key=lambda r: float(r.get("hedges_g", 0.0)), reverse=False)
    labels = [V04_POLICY_LABEL.get(r["policy"], r["policy"]) for r in leader]
    g = [float(r.get("hedges_g", 0.0)) for r in leader]
    cis_lo = [float(r.get("bca_ci_95", [0, 0])[0]) for r in leader]
    cis_hi = [float(r.get("bca_ci_95", [0, 0])[1]) for r in leader]
    colors = [V04_PALETTE.get(r["policy"], "#3b6ea8") for r in leader]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    y = np.arange(len(leader))
    for i, (yi, est, lo, hi, c) in enumerate(zip(y, g, cis_lo, cis_hi, colors, strict=True)):
        ax.errorbar(
            est, yi,
            xerr=[[max(0, est - lo)], [max(0, hi - est)]],
            fmt="o", color=c, ecolor=c, capsize=4, markersize=8,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    if cis_hi:
        text_x = max(cis_hi) + 0.02
        for i, r in enumerate(leader):
            p = float(r.get("permutation_p_one_sided", 1.0))
            ax.text(text_x, y[i], f"g={float(r.get('hedges_g', 0.0)):+.2f}, p={p:.3f}",
                    va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Hedges' g (policy_arm − haiku_bare, paired)")
    sup = bool(h8c.get("supported", False))
    ax.set_title(
        f"H8c.v4 — commit-policy leaderboard (supported={sup}, Holm-adjusted)\n"
        f"oracle highlighted as upper-bound reference",
        fontsize=10,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_h8c_policy_leaderboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_h9_judge_scatter(judge_path: Path, agreement_path: Path, out_dir: Path) -> None:
    """Scatter of proxy_delta vs judge_delta with per-domain coloring."""
    rows: list[dict[str, Any]] = []
    if judge_path.exists():
        for line in judge_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if not rows:
        ax.text(0.5, 0.5, "H9: judge.jsonl missing or empty",
                transform=ax.transAxes, ha="center")
    else:
        domain_color = {
            "poetry_gen": "#3b6ea8",
            "poetry_interp": "#5a8d5a",
            "aut": "#c8a13b",
            "sci_creativity": "#c87b7b",
        }
        for dom, color in domain_color.items():
            xs = [float(r.get("proxy_delta") or 0.0) for r in rows if r.get("domain") == dom]
            ys = [float(r.get("judge_delta") or 0.0) for r in rows if r.get("domain") == dom]
            if xs:
                ax.scatter(xs, ys, color=color, alpha=0.7, s=58, edgecolors="white", label=dom)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_xlabel("proxy scorer Δ (treatment − control)")
        ax.set_ylabel("Sonnet judge Δ (+1=treatment / −1=control / 0=tie)")
        agg: dict[str, Any] = {}
        if agreement_path.exists():
            try:
                agg = json.loads(agreement_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                agg = {}
        rho = agg.get("spearman_rho")
        sign = agg.get("sign_agreement_rate")
        n = agg.get("n", len(rows))
        ax.set_title(
            f"H9.v4 — judge–scorer agreement\n"
            f"n={n}, ρ={rho if rho is not None else 'NaN'}, "
            f"sign-agreement={sign if sign is not None else 'NaN'}",
            fontsize=10,
        )
        ax.legend(fontsize=9, loc="lower right", title="domain")
        ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_h9_judge_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_cost_per_domain(audit_dir: Path, out_dir: Path) -> None:
    """Stacked bars: per-domain Bedrock cost from per-domain ledgers + n_calls overlay."""
    cost: dict[str, float] = {}
    calls: dict[str, int] = {}
    for dom in DOMAINS:
        ledger_path = audit_dir / f"cost_ledger_{dom}.json"
        if not ledger_path.exists():
            cost[dom] = 0.0
            calls[dom] = 0
            continue
        try:
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        cost[dom] = float(data.get("total_usd", 0.0))
        calls[dom] = int(data.get("n_calls", 0))
    fig, ax_cost = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(DOMAINS))
    bars = ax_cost.bar(
        x, [cost[d] for d in DOMAINS], color="#3b6ea8", edgecolor="#234",
    )
    ax_cost.set_ylabel("Bedrock cost (USD)", color="#3b6ea8")
    ax_cost.tick_params(axis="y", labelcolor="#3b6ea8")
    ax_cost.set_xticks(x, [d.replace("_", "\n") for d in DOMAINS])
    ax_cost.set_title(
        f"v0.4 Phase 7 cost per domain (Bedrock Haiku)\n"
        f"total = ${sum(cost.values()):.2f} over {sum(calls.values())} calls",
        fontsize=10,
    )
    for i, b in enumerate(bars):
        ax_cost.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                     f"${cost[DOMAINS[i]]:.2f}", ha="center", fontsize=9)
    ax_calls = ax_cost.twinx()
    ax_calls.plot(x, [calls[d] for d in DOMAINS], "D-", color="#c8a13b",
                  linewidth=2, markersize=9, label="n_calls")
    ax_calls.set_ylabel("n_calls", color="#c8a13b")
    ax_calls.tick_params(axis="y", labelcolor="#c8a13b")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_cost_per_domain.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_axes_breakdown(results_dir: Path, out_dir: Path) -> None:
    """C1 — Per-axis paired Δ (cascade − bare) per (domain, axis) cell.

    Domain axis vocabularies differ; we render four sub-axes (one per domain)
    so each domain's axis schema is internally comparable.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=False)
    for ax, dom in zip(axes, DOMAINS, strict=True):
        try:
            data = json.loads((results_dir / f"{dom}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        rows = data.get("rows", {})
        # Discover axis schema from first arm with axes.
        axis_keys: list[str] = []
        for _, payload in sorted(rows.items()):
            t = payload.get("haiku_cascade") or {}
            if isinstance(t, dict) and isinstance(t.get("axes"), dict):
                axis_keys = list(t["axes"].keys())
                break
        if not axis_keys:
            ax.set_visible(False)
            continue
        means: list[float] = []
        for ak in axis_keys:
            deltas: list[float] = []
            for _, payload in sorted(rows.items()):
                t = payload.get("haiku_cascade") or {}
                c = payload.get("haiku_bare") or {}
                if not isinstance(t, dict) or not isinstance(c, dict):
                    continue
                t_axes = t.get("axes") or {}
                c_axes = c.get("axes") or {}
                if ak not in t_axes or ak not in c_axes:
                    continue
                try:
                    deltas.append(float(t_axes[ak]) - float(c_axes[ak]))
                except (TypeError, ValueError):
                    continue
            means.append(float(np.mean(deltas)) if deltas else 0.0)
        x = np.arange(len(axis_keys))
        colors = ["#5a8d5a" if m >= 0 else "#c87b7b" for m in means]
        bars = ax.bar(x, means, color=colors, edgecolor="#234")
        for i, m in enumerate(means):
            ax.text(i, m + (0.005 if m >= 0 else -0.005), f"{m:+.3f}",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [a.replace("_", "\n") for a in axis_keys], fontsize=8)
        ax.set_title(DOMAIN_LABEL.get(dom, dom).replace("\n", " "), fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(
        "C1 — Per-axis paired Δ (haiku_cascade − haiku_bare), v0.4 mechanism pilot",
        fontsize=11,
    )
    fig.supylabel("paired mean Δ axis-score", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_axes_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figure_v04_power_vs_realised(stats_path: Path, out_dir: Path) -> None:
    """C2 — Retrospective power vs realised Hedges' g for H1–H4.

    Renders the "inconclusive, not null" framing visually: at the realised
    effect sizes the per-domain contrasts are below 0.8 power; an a-priori
    g=0.5 target is also below 0.8 at the available n.
    """
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    primary = stats.get("primary", {})
    keys = ["H1", "H2", "H3", "H4"]
    domain_label = {"H1": "AUT", "H2": "Poetry-interp", "H3": "Poetry-gen", "H4": "Sci-creat"}
    g = [float(primary[k].get("hedges_g", 0.0)) for k in keys]
    pr = [float(primary[k].get("power_retrospective", 0.0)) for k in keys]
    pa = [float(primary[k].get("power_apriori", 0.0)) for k in keys]
    n = [int(primary[k].get("n", 0)) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5.0))
    sizes = [80 + 8 * ni for ni in n]
    ax.scatter(g, pr, s=sizes, color="#3b6ea8", alpha=0.75, edgecolors="#234",
               label="realised power (observed g)")
    ax.scatter(g, pa, s=sizes, color="#bccdf0", alpha=0.65, edgecolors="#234",
               label="a-priori power (g=0.5 assumption)")
    for i, k in enumerate(keys):
        ax.annotate(
            f"{k} ({domain_label[k]}, n={n[i]})",
            (g[i], pr[i]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0.8, color="red", linestyle=":", linewidth=1, label="0.8 power threshold")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_xlim(-0.6, 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("realised Hedges' $g$ (haiku_cascade − haiku_bare)")
    ax.set_ylabel("statistical power")
    ax.set_title(
        "C2 — Retrospective power vs realised effect, H1–H4\n"
        "every domain falls below 0.8: the headline contrasts are inconclusive at this $n$",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v04_power_vs_realised.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _emit_v04(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _figure_v04_h5_fixed_forest(args.stats, out_dir)
    _figure_v04_h8a_revision_vs_draft(args.results_dir, args.stats, out_dir)
    _figure_v04_h8b_gate_calibration(args.stats, out_dir)
    _figure_v04_h8c_policy_leaderboard(args.stats, out_dir)
    _figure_v04_h9_judge_scatter(
        args.results_dir / "judge.jsonl",
        args.results_dir / "judge_agreement.json",
        out_dir,
    )
    _figure_v04_cost_per_domain(args.audit_dir, out_dir)
    _figure_v04_axes_breakdown(args.results_dir, out_dir)
    _figure_v04_power_vs_realised(args.stats, out_dir)
    print(f"v0.4 figures -> {out_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", choices=("v0.2", "v0.3", "v0.4"), default="v0.2",
        help="Which figure pack to emit. v0.4 emits the mechanism-study panel.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Defaults to benchmarks/results_<version>/",
    )
    parser.add_argument(
        "--stats", type=Path, default=None,
        help="Defaults to <results-dir>/stats.json",
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=None,
        help="Defaults to audit/<version>/ — used by the v0.4 cost-per-domain figure.",
    )
    parser.add_argument(
        "--out-dirs", nargs="+", type=Path,
        default=None,
        help="Default depends on --version: v0.2 emits to presentation/figures + paper/figures; "
             "v0.3 emits to presentation/figures + paper/figures; v0.4 emits to "
             "paper/figures/v0.4 + docs/site/public/figures/v0.4 + presentation/figures/v0.4.",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = REPO_ROOT / "benchmarks" / (
            "results_v2" if args.version == "v0.2" else f"results_{args.version}"
        )
    if args.stats is None:
        args.stats = args.results_dir / "stats.json"
    if args.audit_dir is None:
        args.audit_dir = REPO_ROOT / "audit" / args.version
    if args.out_dirs is None:
        if args.version == "v0.4":
            args.out_dirs = [
                REPO_ROOT / "paper" / "figures" / "v0.4",
                REPO_ROOT / "docs" / "site" / "public" / "figures" / "v0.4",
                REPO_ROOT / "presentation" / "figures" / "v0.4",
            ]
        else:
            args.out_dirs = [
                REPO_ROOT / "presentation" / "figures",
                REPO_ROOT / "paper" / "figures",
            ]

    if args.version == "v0.4":
        for out_dir in args.out_dirs:
            _emit_v04(args, out_dir)
        return 0

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
