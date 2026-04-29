#!/usr/bin/env python3
"""PCE v0.3 benchmark statistics.

Reads ``--results-dir/{poetry_gen,poetry_interp,aut,sci_creativity}.json``
written by the v0.3 :mod:`benchmarks.driver` and emits ``stats.json`` with the
v0.3 pre-registered hypotheses:

* **H1.v3 - H4.v3**: per-domain ``haiku_cascade`` vs ``haiku_bare``.
  Architecture-vs-nothing, paired by item id.
* **H5.v3** (redesign): effect-size meta-aggregation (random-effects DerSimonian-
  Laird estimate of pooled Hedges' g across the four per-domain primary
  contrasts), replacing the v0.2 z-blend.
* **H6.v3**: ``haiku_cascade`` vs ``haiku_bare_2K_scorer``. Architecture-vs-more-
  compute -- the headline fairness contrast that controls the "extra compute"
  confound the v0.2 review flagged.
* **H7.v3**: ``haiku_cascade`` vs ``haiku_generic_revise_2pass``. Architecture-vs-
  generic-2-pass -- isolates the *content* of the vimarsa brief from the
  *existence* of a revision pass.
* **H8.v3**: paired ``revision`` vs ``draft`` *within* ``haiku_cascade`` for
  items where the event-gated commit committed revision. Hedges' g + paired
  permutation; demonstrates the *causal contribution* of the second pass.

Length-controlled scoring (Phase 7 deliverable from ADR-002): in addition to
the raw composite, we report a per-domain length-residualised composite
where the per-arm mean length effect is subtracted before pairing. This
addresses the v0.2 review's worry that the cascade's longer surfaces alone
might explain the win.

JSON serialisation: every numeric leaf goes through :func:`_clean_json` which
maps non-finite floats (NaN, ±inf) to ``None``; the writer then uses
``allow_nan=False`` so the file is RFC-compliant strict JSON. This was a
v0.2 review finding (the v0.2 stats.json had bare ``NaN`` literals).

Determinism: ``--seed`` controls the bootstrap and permutation RNGs.

CLI::

  uv run python benchmarks/stats.py [--results-dir benchmarks/results_v0.3]
                                    [--seed 4242]
                                    [--out benchmarks/results_v0.3/stats.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for _p in (str(SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
HYPOTHESIS_DOMAIN = {
    "H1": "aut",
    "H2": "poetry_interp",
    "H3": "poetry_gen",
    "H4": "sci_creativity",
}


@dataclass
class HypothesisResult:
    name: str
    domain: str
    n: int
    estimate: float | None
    estimate_length_controlled: float | None
    hedges_g: float | None
    hedges_g_length_controlled: float | None
    bca_ci_95: tuple[float | None, float | None]
    permutation_p_one_sided: float | None
    wilcoxon_p_one_sided: float | None
    holm_p: float | None
    power_apriori: float | None
    power_retrospective: float | None
    supported: bool
    treatment: str
    control: str


def _clean_json(obj: Any) -> Any:
    """Recursive: map NaN/inf to None, tuples to lists, numpy scalars to py."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_clean_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(x) for x in obj]
    return obj


def _hedges_g_paired(d: np.ndarray[Any, Any]) -> float:
    n = len(d)
    if n < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return 0.0
    g = float(np.mean(d) / sd)
    j = 1.0 - 3.0 / (4.0 * n - 5.0) if n > 1 else 1.0
    return float(g * j)


def _paired_permutation_p_one_sided(
    d: np.ndarray[Any, Any],
    *,
    rng: np.random.Generator,
    alternative: str = "greater",
    n_permutations: int = 50_000,
) -> float:
    n = len(d)
    obs = float(np.mean(d))
    if n == 0:
        return 1.0
    if 2**n <= max(n_permutations, 1024):
        signs_all = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        means = (signs_all * d[None, :]).mean(axis=1)
        if alternative == "greater":
            count = int(np.sum(means >= obs - 1e-15))
        else:
            count = int(np.sum(means <= obs + 1e-15))
        return float(count / (2**n))
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    means = (signs * d[None, :]).mean(axis=1)
    if alternative == "greater":
        count = int(np.sum(means >= obs - 1e-15))
    else:
        count = int(np.sum(means <= obs + 1e-15))
    return float((count + 1) / (n_permutations + 1))


def _bca_ci_paired_mean(
    d: np.ndarray[Any, Any],
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    obs = float(np.mean(d))
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    z0 = stats.norm.ppf(
        (np.sum(boot < obs) + 0.5 * np.sum(boot == obs)) / n_boot
    )
    if not np.isfinite(z0):
        z0 = 0.0
    jk = np.array([float(np.mean(np.delete(d, i))) for i in range(n)])
    jk_mean = float(np.mean(jk))
    num = float(np.sum((jk_mean - jk) ** 3))
    den = 6.0 * (float(np.sum((jk_mean - jk) ** 2)) ** 1.5 + 1e-30)
    accel = num / den
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(
        z0 + (z0 + z_alpha_lo) / max(1 - accel * (z0 + z_alpha_lo), 1e-12)
    )
    a2 = stats.norm.cdf(
        z0 + (z0 + z_alpha_hi) / max(1 - accel * (z0 + z_alpha_hi), 1e-12)
    )
    a1 = float(np.clip(a1, 0.0, 1.0))
    a2 = float(np.clip(a2, 0.0, 1.0))
    return (float(np.quantile(boot, a1)), float(np.quantile(boot, a2)))


def _power_paired_t(
    g: float, n: int, alpha: float = 0.05, alternative: str = "greater"
) -> float:
    if n < 2 or not math.isfinite(g):
        return float("nan")
    df = n - 1
    nc = float(g) * math.sqrt(n)
    t_crit = (
        float(stats.t.ppf(1 - alpha, df=df))
        if alternative == "greater"
        else float(stats.t.ppf(alpha, df=df))
    )
    if alternative == "greater":
        return float(1.0 - stats.nct.cdf(t_crit, df=df, nc=nc))
    return float(stats.nct.cdf(t_crit, df=df, nc=nc))


def _holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = (m - i) * p
        adj = min(1.0, max(adj, running_max))
        running_max = adj
        out[k] = adj
    return out


def _load_domain(results_dir: Path, domain: str) -> dict[str, Any]:
    p = results_dir / f"{domain}.json"
    if not p.exists():
        raise SystemExit(f"missing results file: {p}")
    loaded: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return loaded


def _paired_arrays(
    rows: dict[str, dict[str, Any]],
    *,
    treatment: str,
    control: str,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[str],
]:
    """Return (t_composite, c_composite, t_n_words, c_n_words, ids)."""
    ids: list[str] = []
    t_comp: list[float] = []
    c_comp: list[float] = []
    t_words: list[float] = []
    c_words: list[float] = []
    for item_id, payload in sorted(rows.items()):
        t = payload.get(treatment, {})
        c = payload.get(control, {})
        ts = t.get("composite") if isinstance(t, dict) else None
        cs = c.get("composite") if isinstance(c, dict) else None
        if ts is None or cs is None:
            continue
        try:
            tsf = float(ts)
            csf = float(cs)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(tsf) and math.isfinite(csf)):
            continue
        ids.append(item_id)
        t_comp.append(tsf)
        c_comp.append(csf)
        t_words.append(
            float(t.get("n_words") or len(str(t.get("text", "")).split()))
        )
        c_words.append(
            float(c.get("n_words") or len(str(c.get("text", "")).split()))
        )
    return (
        np.asarray(t_comp, dtype=float),
        np.asarray(c_comp, dtype=float),
        np.asarray(t_words, dtype=float),
        np.asarray(c_words, dtype=float),
        ids,
    )


def _length_controlled_delta(
    t_comp: np.ndarray[Any, Any],
    c_comp: np.ndarray[Any, Any],
    t_words: np.ndarray[Any, Any],
    c_words: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Return a per-pair delta with the linear word-count effect regressed out.

    We fit ``composite ~ a + b * n_words`` on the pooled (treatment + control)
    arms and return ``(t_resid - c_resid)``. When the regression cannot be fit
    (degenerate variance), we fall back to the raw delta so callers always
    get an array of the same length.
    """
    if t_comp.size == 0:
        return np.array([], dtype=float)
    pooled_y = np.concatenate([t_comp, c_comp])
    pooled_x = np.concatenate([t_words, c_words])
    if float(np.std(pooled_x, ddof=1)) < 1e-9:
        return np.asarray(t_comp - c_comp, dtype=float)
    slope, intercept = np.polyfit(pooled_x, pooled_y, 1)
    t_resid = t_comp - (intercept + slope * t_words)
    c_resid = c_comp - (intercept + slope * c_words)
    return np.asarray(t_resid - c_resid, dtype=float)


def _compute_hypothesis(
    name: str,
    domain: str,
    rows: dict[str, dict[str, Any]],
    *,
    treatment: str,
    control: str,
    rng: np.random.Generator,
    n_permutations: int,
    n_boot: int,
    holm_p: float | None = None,
) -> tuple[HypothesisResult, np.ndarray[Any, Any]]:
    t_arr, c_arr, t_w, c_w, _ids = _paired_arrays(
        rows, treatment=treatment, control=control
    )
    d = t_arr - c_arr
    if len(d) == 0:
        raise SystemExit(
            f"{name}: no paired observations for {treatment} vs {control} in {domain}"
        )
    d_lc = _length_controlled_delta(t_arr, c_arr, t_w, c_w)
    g = _hedges_g_paired(d)
    g_lc = _hedges_g_paired(d_lc)
    estimate = float(np.mean(d))
    estimate_lc = float(np.mean(d_lc))
    perm_p = _paired_permutation_p_one_sided(
        d, rng=rng, alternative="greater", n_permutations=n_permutations
    )
    if np.all(d == 0):
        wil_p = 1.0
    else:
        try:
            wil = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
            wil_p = float(wil.pvalue)
        except ValueError:
            wil_p = 1.0
    bca = _bca_ci_paired_mean(d, rng=rng, n_boot=n_boot)
    pow_a = _power_paired_t(0.5, n=len(d), alpha=0.05, alternative="greater")
    pow_r = _power_paired_t(g, n=len(d), alpha=0.05, alternative="greater")
    holm = float(holm_p) if holm_p is not None else perm_p
    supported = (
        holm < 0.05
        and math.isfinite(bca[0])
        and bca[0] > 0.0
    )
    return (
        HypothesisResult(
            name=name,
            domain=domain,
            n=len(d),
            estimate=estimate,
            estimate_length_controlled=estimate_lc,
            hedges_g=g,
            hedges_g_length_controlled=g_lc,
            bca_ci_95=bca,
            permutation_p_one_sided=perm_p,
            wilcoxon_p_one_sided=wil_p,
            holm_p=holm,
            power_apriori=pow_a,
            power_retrospective=pow_r,
            supported=bool(supported),
            treatment=treatment,
            control=control,
        ),
        d,
    )


def _hypothesis_to_dict(h: HypothesisResult) -> dict[str, Any]:
    out = asdict(h)
    out["bca_ci_95"] = list(h.bca_ci_95)
    return out


def _meta_aggregate_random_effects(
    per_domain: list[tuple[float, int]]
) -> dict[str, Any]:
    """DerSimonian-Laird random-effects pooling of Hedges' g across domains.

    ``per_domain`` is a list of ``(g, n)`` pairs. Returns the pooled g, the
    DL τ², the 95% CI, and the per-study weights. When a single domain or
    n is too small to compute a variance, that study is dropped.
    """
    studies = [
        (float(g), int(n))
        for g, n in per_domain
        if n >= 2 and math.isfinite(float(g))
    ]
    if not studies:
        return {
            "pooled_g": None,
            "tau2": None,
            "ci_95": [None, None],
            "n_studies": 0,
        }
    gs = np.array([g for g, _ in studies], dtype=float)
    ns = np.array([n for _, n in studies], dtype=float)
    var_fixed = (1.0 / ns) + (gs**2) / (2.0 * ns)
    w_fixed = 1.0 / np.maximum(var_fixed, 1e-12)
    g_fixed = float(np.sum(w_fixed * gs) / np.sum(w_fixed))
    q = float(np.sum(w_fixed * (gs - g_fixed) ** 2))
    df_q = max(len(studies) - 1, 1)
    c = float(
        np.sum(w_fixed)
        - (np.sum(w_fixed**2) / max(np.sum(w_fixed), 1e-12))
    )
    tau2 = max(0.0, (q - df_q) / max(c, 1e-12))
    w_re = 1.0 / np.maximum(var_fixed + tau2, 1e-12)
    g_re = float(np.sum(w_re * gs) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    return {
        "pooled_g": g_re,
        "tau2": tau2,
        "ci_95": [g_re - 1.96 * se_re, g_re + 1.96 * se_re],
        "n_studies": len(studies),
        "weights_re": [float(w) for w in (w_re / np.sum(w_re)).tolist()],
    }


def _within_pce_revision_vs_draft(
    results_dir: Path,
    *,
    cascade_arm: str,
    rng: np.random.Generator,
    n_boot: int,
    n_permutations: int,
) -> dict[str, Any]:
    """H8.v3: paired revision-vs-draft within ``cascade_arm`` for items where
    the event-gated commit committed revision.

    We pair ``meta.surface_revision`` and ``meta.surface_draft`` per item and
    score both using the *same* domain scorer; the paired delta is
    ``score(revision) - score(draft)`` over items with ``committed=='revision'``.
    """
    from benchmarks import scoring as bench_scoring  # local import: scorers

    scorers = {
        "poetry_gen": bench_scoring.score_poetry_gen,
        "poetry_interp": bench_scoring.score_poetry_interp,
        "aut": bench_scoring.score_aut,
        "sci_creativity": bench_scoring.score_sci_creativity,
    }
    from pce.substrate.embed import Embedder  # heavy import: only for H8

    embed = Embedder()

    rev: list[float] = []
    drf: list[float] = []
    for dom in DOMAINS:
        try:
            data = _load_domain(results_dir, dom)
        except SystemExit:
            continue
        scorer = scorers[dom]
        for _id, payload in data["rows"].items():
            row = payload.get(cascade_arm, {})
            if not isinstance(row, dict):
                continue
            meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
            if str(meta.get("committed", "")) != "revision":
                continue
            sr = str(meta.get("surface_revision", "")).strip()
            sd = str(meta.get("surface_draft", "")).strip()
            if not sr or not sd:
                continue
            try:
                sr_score = float(
                    scorer(sr, item=payload.get("item", {}), embed=embed).composite
                )
                sd_score = float(
                    scorer(sd, item=payload.get("item", {}), embed=embed).composite
                )
            except Exception:  # noqa: BLE001 — per-item scoring failure is non-fatal
                continue
            if not (math.isfinite(sr_score) and math.isfinite(sd_score)):
                continue
            rev.append(sr_score)
            drf.append(sd_score)
    if not rev:
        return {
            "name": "H8",
            "n": 0,
            "supported": False,
            "note": "no committed=='revision' items with both shadow surfaces",
        }
    rev_arr = np.asarray(rev, dtype=float)
    drf_arr = np.asarray(drf, dtype=float)
    d = rev_arr - drf_arr
    g = _hedges_g_paired(d)
    perm_p = _paired_permutation_p_one_sided(
        d, rng=rng, alternative="greater", n_permutations=n_permutations
    )
    bca = _bca_ci_paired_mean(d, rng=rng, n_boot=n_boot)
    return {
        "name": "H8",
        "cascade_arm": cascade_arm,
        "n": int(len(d)),
        "estimate": float(np.mean(d)),
        "hedges_g": float(g),
        "bca_ci_95": list(bca),
        "permutation_p_one_sided": float(perm_p),
        "supported": bool(perm_p < 0.05 and math.isfinite(bca[0]) and bca[0] > 0.0),
        "note": (
            "paired score(revision) - score(draft) within haiku_cascade for items "
            "where the event-gated commit committed revision"
        ),
    }


def _run_contrast(
    *,
    contrast_label: str,
    treatment: str,
    control: str,
    results_dir: Path,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    raw_p: dict[str, float] = {}
    primary_results: dict[str, HypothesisResult] = {}
    deltas: dict[str, np.ndarray[Any, Any]] = {}
    for h_name, dom in HYPOTHESIS_DOMAIN.items():
        try:
            data = _load_domain(results_dir, dom)
            res, d = _compute_hypothesis(
                h_name,
                dom,
                data["rows"],
                treatment=treatment,
                control=control,
                rng=rng,
                n_permutations=n_permutations,
                n_boot=n_bootstrap,
            )
            primary_results[h_name] = res
            deltas[h_name] = d
            raw_p[h_name] = res.permutation_p_one_sided or 1.0
        except SystemExit as e:
            print(f"[stats] {contrast_label}/{h_name}: skipped ({e})", file=sys.stderr)
    if raw_p:
        holm_adj = _holm_bonferroni(raw_p)
        for h_name in raw_p:
            primary_results[h_name].holm_p = holm_adj[h_name]
            primary_results[h_name].supported = bool(
                holm_adj[h_name] < 0.05
                and math.isfinite(primary_results[h_name].bca_ci_95[0] or float("nan"))
                and (primary_results[h_name].bca_ci_95[0] or 0.0) > 0.0
            )

    # H5: random-effects meta-aggregation of Hedges' g across H1..H4 domains.
    meta_input = [
        (
            float(primary_results[h].hedges_g or 0.0),
            int(primary_results[h].n),
        )
        for h in primary_results
    ]
    h5 = {
        "name": "H5",
        **_meta_aggregate_random_effects(meta_input),
        "note": (
            "random-effects DerSimonian-Laird meta-analysis of per-domain "
            "Hedges' g (replaces the v0.2 z-blend per phase 7 ADR)"
        ),
        "per_domain_g": {
            h: float(primary_results[h].hedges_g or 0.0) for h in primary_results
        },
        "per_domain_n": {h: int(primary_results[h].n) for h in primary_results},
    }
    if h5.get("pooled_g") is not None:
        h5["supported"] = bool(
            (h5.get("ci_95") or [None, None])[0] is not None
            and float((h5.get("ci_95") or [0.0, 0.0])[0]) > 0.0
        )
    else:
        h5["supported"] = False

    return {
        "treatment": treatment,
        "control": control,
        "primary": {h: _hypothesis_to_dict(r) for h, r in primary_results.items()},
        "H5": h5,
    }


def _arm_means(results_dir: Path) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for dom in DOMAINS:
        try:
            data = _load_domain(results_dir, dom)
        except SystemExit:
            continue
        per_arm: dict[str, list[float]] = {}
        for _id, payload in data["rows"].items():
            for k, v in payload.items():
                if k == "item" or k == "_integrity_probes" or not isinstance(v, dict):
                    continue
                comp = v.get("composite")
                if comp is None:
                    continue
                try:
                    cf = float(comp)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(cf):
                    continue
                per_arm.setdefault(k, []).append(cf)
        out[dom] = {
            arm: {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            }
            for arm, vals in sorted(per_arm.items())
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results_v0.3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results_v0.3" / "stats.json",
    )
    parser.add_argument("--treatment", type=str, default="haiku_cascade")
    parser.add_argument("--control", type=str, default="haiku_bare")
    parser.add_argument(
        "--control-2K", type=str, default="haiku_bare_2K_scorer",
        help="control arm for H6.v3 (architecture vs more compute)",
    )
    parser.add_argument(
        "--control-generic", type=str, default="haiku_generic_revise_2pass",
        help="control arm for H7.v3 (architecture vs generic 2-pass)",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--n-permutations", type=int, default=50_000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    primary = _run_contrast(
        contrast_label="primary_v3",
        treatment=args.treatment,
        control=args.control,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    h6_contrast = _run_contrast(
        contrast_label="H6_v3_extra_compute",
        treatment=args.treatment,
        control=args.control_2K,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    h7_contrast = _run_contrast(
        contrast_label="H7_v3_generic_revise",
        treatment=args.treatment,
        control=args.control_generic,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )

    h8 = _within_pce_revision_vs_draft(
        args.results_dir,
        cascade_arm=args.treatment,
        rng=rng,
        n_boot=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    arm_means = _arm_means(args.results_dir)

    out = {
        "config": {
            "treatment_arm": args.treatment,
            "control_arm_primary": args.control,
            "control_arm_h6": args.control_2K,
            "control_arm_h7": args.control_generic,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "alpha": 0.05,
            "version": "v0.3",
        },
        "primary": primary["primary"],
        "H5": primary["H5"],
        "H6_v3_extra_compute": h6_contrast["primary"],
        "H6_v3_extra_compute_meta": h6_contrast["H5"],
        "H7_v3_generic_revise": h7_contrast["primary"],
        "H7_v3_generic_revise_meta": h7_contrast["H5"],
        "H8_v3_revision_vs_draft": h8,
        "arm_means_per_domain": arm_means,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_clean_json(out), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    headline = {
        h: {
            "n": r.n,
            "estimate": r.estimate,
            "estimate_length_controlled": r.estimate_length_controlled,
            "g": r.hedges_g,
            "g_length_controlled": r.hedges_g_length_controlled,
            "bca_ci_95": list(r.bca_ci_95),
            "perm_p": r.permutation_p_one_sided,
            "holm_p": r.holm_p,
            "supported": r.supported,
        }
        for h, r in {
            k: HypothesisResult(**{**v, "bca_ci_95": tuple(v["bca_ci_95"])})
            for k, v in primary["primary"].items()
        }.items()
    }
    print(
        json.dumps(
            _clean_json(
                {
                    "primary_v3": headline,
                    "H5_v3": primary["H5"],
                    "H8_v3": h8,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
