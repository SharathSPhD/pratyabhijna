#!/usr/bin/env python3
"""PCE benchmark statistics (v0.3 + v0.4).

Reads ``--results-dir/{poetry_gen,poetry_interp,aut,sci_creativity}.json``
written by :mod:`benchmarks.driver` and emits ``stats.json`` with all
pre-registered hypotheses for the requested ``--version``.

v0.3 hypotheses (kept for reproducibility of the v0.3 release):

* **H1.v3 - H4.v3**: per-domain ``haiku_cascade`` vs ``haiku_bare``.
  Architecture-vs-nothing, paired by item id.
* **H5.v3**: random-effects DerSimonian-Laird meta-aggregation of per-domain
  Hedges' g.
* **H6.v3**: ``haiku_cascade`` vs ``haiku_bare_2K_scorer`` (compute fairness).
* **H7.v3**: ``haiku_cascade`` vs ``haiku_generic_revise_2pass`` (protocol
  fairness).
* **H8.v3**: paired ``revision`` vs ``draft`` within ``haiku_cascade`` for
  items where the event-gated commit committed revision.

v0.4 hypotheses (Phase 4, ADR-005 locks H5 to fixed-effects):

* **H1.v4 - H4.v4**: per-domain ``haiku_cascade`` (event_gated) vs
  ``haiku_bare``, paired permutation, Holm-corrected.
* **H5.v4**: fixed-effects meta-pool of per-domain Hedges' g (replaces the
  v0.3 random-effects pool — see ADR-005).
* **H6.v4**: vs ``haiku_bare_2K_scorer`` (compute fairness).
* **H7.v4**: vs ``haiku_generic_revise_2pass`` (protocol fairness).
* **H8a.v4**: shadow-revision-vs-draft within cascade across **all** items
  (not only committed-revision items, unlike H8.v3) — answers the
  "shadow revision value" question from the adversarial review.
* **H8b.v4**: gate calibration — treats ``event_gated`` as a binary
  classifier of "revision is better than draft" (the H8a label) and
  reports precision / recall / F1; ``learned_gate`` is reported as a
  paired baseline classifier on the same items.
* **H8c.v4**: commit-policy comparison — paired across items, comparing
  the four cascade commit policies (event_gated, always_draft,
  always_revise, learned_gate) head-to-head against ``haiku_bare`` and
  emitting a leader-board.
* **H9.v4**: judge-proxy agreement (Spearman + sign-agreement) on the
  Sonnet stratified subset. Computed only when ``judge.jsonl`` exists;
  otherwise the entry records ``"status": "missing"`` so downstream
  consumers can detect that the judge stage has not run yet.

Length-controlled scoring: in addition to the raw composite we report a
per-domain length-residualised composite where the per-arm mean length
effect is subtracted before pairing.

JSON serialisation: every numeric leaf goes through :func:`_clean_json`
which maps non-finite floats (NaN, ±inf) to ``None``; the writer uses
``allow_nan=False`` so the file is RFC-compliant strict JSON.

Synthetic-data emission: when ``--version=v0.4`` and the results
directory is missing or empty, the module emits a v0.4 stats payload
populated with synthetic placeholders so downstream tools (HTML,
plugin smoke tests) can validate schema before the pilot runs. This is
the Phase 4 gate "emit all keys on synthetic data".

Determinism: ``--seed`` controls the bootstrap and permutation RNGs.

CLI::

  uv run python benchmarks/stats.py --version v0.4
  uv run python benchmarks/stats.py --version v0.3 \
      --results-dir benchmarks/results_v0.3
"""
from __future__ import annotations

import argparse
import enum
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, assert_never

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


class Hypothesis(enum.Enum):
    """v0.4 pre-registered hypotheses (ADR-005).

    Exhaustive-switch ready (workspace rule); ``stats_for_v0_4`` uses
    :func:`typing.assert_never` so adding a new variant without handling
    it is a type-check error.
    """

    H1 = "H1.v4"  # per-domain primary, aut
    H2 = "H2.v4"  # per-domain primary, poetry_interp
    H3 = "H3.v4"  # per-domain primary, poetry_gen
    H4 = "H4.v4"  # per-domain primary, sci_creativity
    H5 = "H5.v4"  # meta-pool fixed-effects
    H6 = "H6.v4"  # compute fairness vs haiku_bare_2K_scorer
    H7 = "H7.v4"  # protocol fairness vs haiku_generic_revise_2pass
    H8a = "H8a.v4"  # shadow-revision-vs-draft (all items)
    H8b = "H8b.v4"  # event_gated calibration as classifier
    H8c = "H8c.v4"  # commit-policy comparison
    H9 = "H9.v4"  # judge-proxy agreement


def hypothesis_label(h: Hypothesis) -> str:
    """Human-readable description of a v0.4 hypothesis (exhaustive)."""
    if h is Hypothesis.H1:
        return "haiku_cascade(event_gated) > haiku_bare on aut"
    if h is Hypothesis.H2:
        return "haiku_cascade(event_gated) > haiku_bare on poetry_interp"
    if h is Hypothesis.H3:
        return "haiku_cascade(event_gated) > haiku_bare on poetry_gen"
    if h is Hypothesis.H4:
        return "haiku_cascade(event_gated) > haiku_bare on sci_creativity"
    if h is Hypothesis.H5:
        return "fixed-effects pooled g across H1.v4-H4.v4 > 0"
    if h is Hypothesis.H6:
        return "haiku_cascade > haiku_bare_2K_scorer (compute fairness)"
    if h is Hypothesis.H7:
        return "haiku_cascade > haiku_generic_revise_2pass (protocol fairness)"
    if h is Hypothesis.H8a:
        return "shadow_revision > draft within cascade (all items, paired)"
    if h is Hypothesis.H8b:
        return "event_gated F1 > 0.5 as binary classifier of label_revision_better"
    if h is Hypothesis.H8c:
        return "best commit_policy > event_gated on per-item composite"
    if h is Hypothesis.H9:
        return "Spearman(proxy, judge) > 0 on the stratified judge subset"
    assert_never(h)


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


def _meta_aggregate_fixed_effects(
    per_domain: list[tuple[float, int]]
) -> dict[str, Any]:
    """ADR-005 (v0.4) fixed-effects pooling of paired Hedges' g across domains.

    ``per_domain`` is a list of ``(g, n)`` pairs, one per per-domain
    primary contrast. Returns the inverse-variance-weighted pooled g, its
    95% CI under the fixed-effects assumption, and the per-study weights.

    The fixed-effects estimator assumes a single underlying effect across
    studies and is appropriate when (a) all four domains share the same
    measurement scale and (b) we explicitly *don't* want to absorb
    between-domain heterogeneity into the CI. The v0.3 random-effects
    DerSimonian-Laird estimator inflates the CI when domains diverge,
    making the meta-pool look weaker than a fixed-effects synthesis on
    domain-aligned scoring would. ADR-005 locks the v0.4 paper to this
    fixed-effects estimate to keep SPEC and code consistent.
    """
    studies = [
        (float(g), int(n))
        for g, n in per_domain
        if n >= 2 and math.isfinite(float(g))
    ]
    if not studies:
        return {
            "pooled_g": None,
            "method": "fixed_effects_inverse_variance",
            "ci_95": [None, None],
            "n_studies": 0,
        }
    gs = np.array([g for g, _ in studies], dtype=float)
    ns = np.array([n for _, n in studies], dtype=float)
    var = (1.0 / ns) + (gs**2) / (2.0 * ns)
    w = 1.0 / np.maximum(var, 1e-12)
    g_fixed = float(np.sum(w * gs) / np.sum(w))
    se_fixed = float(math.sqrt(1.0 / np.sum(w)))
    return {
        "pooled_g": g_fixed,
        "method": "fixed_effects_inverse_variance",
        "ci_95": [g_fixed - 1.96 * se_fixed, g_fixed + 1.96 * se_fixed],
        "n_studies": len(studies),
        "weights": [float(wi) for wi in (w / np.sum(w)).tolist()],
    }


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


def _h8a_shadow_revision_vs_draft_all_items(
    results_dir: Path,
    *,
    cascade_arm: str,
    rng: np.random.Generator,
    n_boot: int,
    n_permutations: int,
) -> dict[str, Any]:
    """H8a.v4: paired score(revision) vs score(draft) within ``cascade_arm``,
    over **all** items (not only ``committed=='revision'`` like H8.v3).

    The driver records ``meta.score_draft`` and ``meta.score_revision`` on
    each synthesised commit-policy arm row; we prefer those when present
    (reuses the multiplexer's scoring) and fall back to scoring the
    ``surface_*`` strings ourselves when not.
    """
    from benchmarks import scoring as bench_scoring

    scorers = {
        "poetry_gen": bench_scoring.score_poetry_gen,
        "poetry_interp": bench_scoring.score_poetry_interp,
        "aut": bench_scoring.score_aut,
        "sci_creativity": bench_scoring.score_sci_creativity,
    }
    embed = None  # lazily constructed only if we have to score from text

    rev: list[float] = []
    drf: list[float] = []
    for dom in DOMAINS:
        try:
            data = _load_domain(results_dir, dom)
        except SystemExit:
            continue
        for _id, payload in data["rows"].items():
            row = payload.get(cascade_arm, {})
            if not isinstance(row, dict):
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            sd_score: float | None = None
            sr_score: float | None = None
            # Driver multiplex stores both scores on the *_always_revise arm
            # (always revises) so prefer that when available.
            ar_row = payload.get("haiku_cascade_always_revise")
            if isinstance(ar_row, dict):
                ar_meta = ar_row.get("meta") if isinstance(ar_row.get("meta"), dict) else {}
                if isinstance(ar_meta, dict):
                    raw_d = ar_meta.get("score_draft")
                    raw_r = ar_meta.get("score_revision")
                    if (
                        isinstance(raw_d, (int, float))
                        and math.isfinite(float(raw_d))
                        and isinstance(raw_r, (int, float))
                        and math.isfinite(float(raw_r))
                    ):
                        sd_score = float(raw_d)
                        sr_score = float(raw_r)
            if sd_score is None or sr_score is None:
                sr_text = str(meta.get("surface_revision", "")).strip() if isinstance(meta, dict) else ""
                sd_text = str(meta.get("surface_draft", "")).strip() if isinstance(meta, dict) else ""
                if not sr_text or not sd_text:
                    continue
                if embed is None:
                    from pce.substrate.embed import Embedder
                    embed = Embedder()
                scorer = scorers[dom]
                try:
                    sr_score = float(scorer(sr_text, item=payload.get("item", {}), embed=embed).composite)
                    sd_score = float(scorer(sd_text, item=payload.get("item", {}), embed=embed).composite)
                except Exception:  # noqa: BLE001
                    continue
                if not (math.isfinite(sr_score) and math.isfinite(sd_score)):
                    continue
            rev.append(sr_score)
            drf.append(sd_score)
    if not rev:
        return {
            "name": "H8a.v4",
            "n": 0,
            "supported": False,
            "note": "no cascade items with both shadow surfaces available",
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
        "name": "H8a.v4",
        "cascade_arm": cascade_arm,
        "n": int(len(d)),
        "estimate": float(np.mean(d)),
        "hedges_g": float(g),
        "bca_ci_95": list(bca),
        "permutation_p_one_sided": float(perm_p),
        "supported": bool(perm_p < 0.05 and math.isfinite(bca[0]) and bca[0] > 0.0),
        "note": "paired score(revision) - score(draft) over all cascade items",
    }


def _classifier_metrics(
    *,
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Precision / recall / F1 / accuracy for binary 0/1 arrays."""
    if y_true.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "support_pos": 0,
            "support_neg": 0,
        }
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n = int(y_true.size)
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "n": n,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_pos": int(np.sum(y_true == 1)),
        "support_neg": int(np.sum(y_true == 0)),
    }


def _h8b_gate_calibration(
    results_dir: Path,
) -> dict[str, Any]:
    """H8b.v4: event_gated as binary classifier of "revision is better than draft".

    For each cascade item with both surfaces, we compare:

    * label : 1 if score(revision) > score(draft), else 0.
    * event_gated prediction: 1 if vimarsa_event_draft fired, else 0.
    * learned_gate prediction: 1 if the LearnedGate policy committed
      revision (recovered from the synthesised arm), else 0.

    Reports precision / recall / F1 for each predictor.
    """
    labels: list[int] = []
    eg_preds: list[int] = []
    lg_preds: list[int] = []
    for dom in DOMAINS:
        try:
            data = _load_domain(results_dir, dom)
        except SystemExit:
            continue
        for _id, payload in data["rows"].items():
            ar_row = payload.get("haiku_cascade_always_revise")
            if not isinstance(ar_row, dict):
                continue
            ar_meta = ar_row.get("meta") if isinstance(ar_row.get("meta"), dict) else {}
            if not isinstance(ar_meta, dict):
                continue
            raw_d = ar_meta.get("score_draft")
            raw_r = ar_meta.get("score_revision")
            if not (
                isinstance(raw_d, (int, float))
                and math.isfinite(float(raw_d))
                and isinstance(raw_r, (int, float))
                and math.isfinite(float(raw_r))
            ):
                continue
            label = 1 if float(raw_r) > float(raw_d) else 0
            cascade = payload.get("haiku_cascade", {})
            cascade_meta: dict[str, Any] = {}
            if isinstance(cascade, dict) and isinstance(cascade.get("meta"), dict):
                cascade_meta = cascade["meta"]
            eg_pred = 1 if bool(cascade_meta.get("vimarsa_event_draft", cascade_meta.get("vimarsa_event", False))) else 0
            lg_row = payload.get("haiku_cascade_learned_gate")
            lg_pred = 0
            if isinstance(lg_row, dict) and isinstance(lg_row.get("meta"), dict):
                lg_pred = 1 if bool(lg_row["meta"].get("commit_decision_revision", False)) else 0
            labels.append(label)
            eg_preds.append(eg_pred)
            lg_preds.append(lg_pred)
    y_true = np.asarray(labels, dtype=int)
    return {
        "name": "H8b.v4",
        "event_gated": _classifier_metrics(
            y_true=y_true, y_pred=np.asarray(eg_preds, dtype=int)
        ),
        "learned_gate": _classifier_metrics(
            y_true=y_true, y_pred=np.asarray(lg_preds, dtype=int)
        ),
        "supported": bool(
            y_true.size > 0
            and float(
                _classifier_metrics(
                    y_true=y_true, y_pred=np.asarray(eg_preds, dtype=int)
                )["f1"]
            )
            > 0.5
        ),
        "note": "binary classifier metrics: predict 'revision better than draft'",
    }


def _h8c_commit_policy_comparison(
    results_dir: Path,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """H8c.v4: head-to-head comparison of the four cascade commit policies.

    Each policy is paired against ``haiku_bare`` per item and the per-item
    delta is fed through the same paired-permutation / Hedges' g pipeline
    as H1-H4. Returns a leader-board sorted by mean delta and the pairwise
    paired permutation p-values between every pair of policies.
    """
    policies = (
        "haiku_cascade_event_gated",
        "haiku_cascade_always_draft",
        "haiku_cascade_always_revise",
        "haiku_cascade_learned_gate",
    )
    leader: list[dict[str, Any]] = []
    per_policy_deltas: dict[str, list[float]] = {p: [] for p in policies}
    for dom in DOMAINS:
        try:
            data = _load_domain(results_dir, dom)
        except SystemExit:
            continue
        for policy in policies:
            t_arr, c_arr, _t_w, _c_w, _ids = _paired_arrays(
                data["rows"], treatment=policy, control="haiku_bare"
            )
            d = (t_arr - c_arr).tolist()
            per_policy_deltas[policy].extend(d)
    for policy, ds in per_policy_deltas.items():
        if not ds:
            leader.append(
                {
                    "policy": policy,
                    "n": 0,
                    "estimate": None,
                    "hedges_g": None,
                    "bca_ci_95": [None, None],
                    "permutation_p_one_sided": None,
                }
            )
            continue
        d = np.asarray(ds, dtype=float)
        g = _hedges_g_paired(d)
        perm_p = _paired_permutation_p_one_sided(
            d, rng=rng, alternative="greater", n_permutations=n_permutations
        )
        bca = _bca_ci_paired_mean(d, rng=rng, n_boot=n_bootstrap)
        leader.append(
            {
                "policy": policy,
                "n": int(len(d)),
                "estimate": float(np.mean(d)),
                "hedges_g": float(g),
                "bca_ci_95": list(bca),
                "permutation_p_one_sided": float(perm_p),
            }
        )
    leader.sort(
        key=lambda row: (
            -1.0
            if row.get("estimate") is None
            else float(row["estimate"])
        ),
        reverse=True,
    )

    # Pairwise paired permutation: align by index (per-domain order is stable).
    pair_pvals: dict[str, float] = {}
    n_min = min(
        (len(per_policy_deltas[p]) for p in policies),
        default=0,
    )
    if n_min >= 2:
        for i, a in enumerate(policies):
            for b in policies[i + 1 :]:
                da = np.asarray(per_policy_deltas[a][:n_min], dtype=float)
                db = np.asarray(per_policy_deltas[b][:n_min], dtype=float)
                d = da - db
                p = _paired_permutation_p_one_sided(
                    d, rng=rng, alternative="greater", n_permutations=n_permutations
                )
                pair_pvals[f"{a}__vs__{b}"] = float(p)

    best = leader[0] if leader else {"policy": None}
    supported = bool(
        best.get("policy") == "haiku_cascade_learned_gate"
        and best.get("permutation_p_one_sided") is not None
        and float(best.get("permutation_p_one_sided") or 1.0) < 0.05
    )
    return {
        "name": "H8c.v4",
        "leader_board": leader,
        "pairwise_p": pair_pvals,
        "supported": supported,
        "note": "paired delta vs haiku_bare per policy; pairwise paired permutation across policies",
    }


def _h9_judge_proxy_agreement(
    results_dir: Path,
) -> dict[str, Any]:
    """H9.v4: Spearman + sign-agreement between proxy composite and Sonnet judge.

    Reads ``judge.jsonl`` written by Phase 5. When the file is missing,
    returns ``{"status": "missing"}`` so the schema still emits the H9 key
    even before the judge subset has been run.
    """
    judge_path = results_dir / "judge.jsonl"
    if not judge_path.exists():
        return {
            "name": "H9.v4",
            "status": "missing",
            "supported": False,
            "note": (
                f"{judge_path.name} not found; run scripts/judge_subset.py "
                "(Phase 5) to populate"
            ),
        }
    proxy: list[float] = []
    judge: list[float] = []
    sign_agreements: list[int] = []
    for line in judge_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        proxy_v = row.get("proxy_delta")
        judge_v = row.get("judge_delta")
        if proxy_v is None or judge_v is None:
            continue
        try:
            pf = float(proxy_v)
            jf = float(judge_v)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pf) and math.isfinite(jf)):
            continue
        proxy.append(pf)
        judge.append(jf)
        sign_agreements.append(int((pf >= 0) == (jf >= 0)))
    if len(proxy) < 3:
        return {
            "name": "H9.v4",
            "status": "insufficient",
            "n": len(proxy),
            "supported": False,
            "note": "fewer than 3 paired observations in judge.jsonl",
        }
    proxy_arr = np.asarray(proxy, dtype=float)
    judge_arr = np.asarray(judge, dtype=float)
    spearman = stats.spearmanr(proxy_arr, judge_arr)
    sign_rate = float(np.mean(sign_agreements))
    return {
        "name": "H9.v4",
        "status": "ok",
        "n": int(len(proxy_arr)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "sign_agreement_rate": sign_rate,
        "supported": bool(
            float(spearman.statistic) > 0.0 and float(spearman.pvalue) < 0.05
        ),
        "note": "Spearman rho + sign-agreement between proxy composite delta and Sonnet judge delta",
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


def _synthetic_v0_4_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Phase 4 gate: emit a v0.4 stats payload with all H1.v4-H9.v4 keys
    populated by deterministic placeholders so downstream consumers (HTML,
    plugin smoke tests, paper compile) can validate schema before the
    pilot has produced real results.

    The payload is structurally identical to the real v0.4 output (same
    keys, same nesting, same numeric types) but every effect size is
    ``0.0`` and every supported flag is ``False``. ``"status": "synthetic"``
    is set on the top-level config so consumers can refuse to render it
    as a real result.
    """
    h_zero: dict[str, Any] = {
        "name": "synthetic",
        "n": 0,
        "estimate": 0.0,
        "estimate_length_controlled": 0.0,
        "hedges_g": 0.0,
        "hedges_g_length_controlled": 0.0,
        "bca_ci_95": [0.0, 0.0],
        "permutation_p_one_sided": 1.0,
        "wilcoxon_p_one_sided": 1.0,
        "holm_p": 1.0,
        "power_apriori": 0.0,
        "power_retrospective": 0.0,
        "supported": False,
        "treatment": "haiku_cascade",
        "control": "haiku_bare",
    }
    primary = {
        "H1": dict(h_zero, name="H1.v4", domain="aut"),
        "H2": dict(h_zero, name="H2.v4", domain="poetry_interp"),
        "H3": dict(h_zero, name="H3.v4", domain="poetry_gen"),
        "H4": dict(h_zero, name="H4.v4", domain="sci_creativity"),
    }
    h5 = {
        "name": "H5.v4",
        "method": "fixed_effects_inverse_variance",
        "pooled_g": 0.0,
        "ci_95": [0.0, 0.0],
        "n_studies": 0,
        "weights": [],
        "per_domain_g": dict.fromkeys(primary, 0.0),
        "per_domain_n": dict.fromkeys(primary, 0),
        "supported": False,
        "note": "synthetic v0.4 placeholder; ADR-005 fixed-effects",
    }
    h8a = {
        "name": "H8a.v4",
        "n": 0,
        "estimate": 0.0,
        "hedges_g": 0.0,
        "bca_ci_95": [0.0, 0.0],
        "permutation_p_one_sided": 1.0,
        "supported": False,
        "note": "synthetic placeholder",
    }
    h8b = {
        "name": "H8b.v4",
        "event_gated": _classifier_metrics(
            y_true=np.zeros(0, dtype=int), y_pred=np.zeros(0, dtype=int)
        ),
        "learned_gate": _classifier_metrics(
            y_true=np.zeros(0, dtype=int), y_pred=np.zeros(0, dtype=int)
        ),
        "supported": False,
        "note": "synthetic placeholder",
    }
    h8c = {
        "name": "H8c.v4",
        "leader_board": [],
        "pairwise_p": {},
        "supported": False,
        "note": "synthetic placeholder",
    }
    h9 = {
        "name": "H9.v4",
        "status": "missing",
        "supported": False,
        "note": "synthetic placeholder; judge subset not yet run",
    }
    return {
        "config": {
            "version": "v0.4",
            "status": "synthetic",
            "treatment_arm": args.treatment,
            "control_arm_primary": args.control,
            "control_arm_h6": args.control_2K,
            "control_arm_h7": args.control_generic,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "alpha": 0.05,
            "hypotheses": [h.value for h in Hypothesis],
        },
        "primary": primary,
        "H5": h5,
        "H6_v4_extra_compute": {h: dict(h_zero, name=f"{h}.v4") for h in primary},
        "H6_v4_extra_compute_meta": dict(h5, name="H6.v4"),
        "H7_v4_generic_revise": {h: dict(h_zero, name=f"{h}.v4") for h in primary},
        "H7_v4_generic_revise_meta": dict(h5, name="H7.v4"),
        "H8a_v4_shadow_revision_vs_draft": h8a,
        "H8b_v4_gate_calibration": h8b,
        "H8c_v4_commit_policy_comparison": h8c,
        "H9_v4_judge_proxy_agreement": h9,
        "arm_means_per_domain": {dom: {} for dom in DOMAINS},
    }


def _stats_v0_4(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    """Build the v0.4 stats payload from real results.

    Uses fixed-effects meta-pool for H5 (ADR-005) and emits the H8a/H8b/H8c
    splits plus an H9 stub that picks up ``judge.jsonl`` when present.
    """
    primary = _run_contrast(
        contrast_label="primary_v4",
        treatment=args.treatment,
        control=args.control,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    # ADR-005: lock H5 to fixed-effects.
    primary_meta_input = [
        (
            float(primary["primary"][h].get("hedges_g") or 0.0),
            int(primary["primary"][h].get("n") or 0),
        )
        for h in primary["primary"]
    ]
    h5 = {
        "name": "H5.v4",
        **_meta_aggregate_fixed_effects(primary_meta_input),
        "per_domain_g": {
            h: float(primary["primary"][h].get("hedges_g") or 0.0)
            for h in primary["primary"]
        },
        "per_domain_n": {
            h: int(primary["primary"][h].get("n") or 0)
            for h in primary["primary"]
        },
        "note": "ADR-005 fixed-effects meta-pool of H1.v4-H4.v4 Hedges' g",
    }
    if h5.get("pooled_g") is not None:
        h5["supported"] = bool(
            (h5.get("ci_95") or [None, None])[0] is not None
            and float((h5.get("ci_95") or [0.0, 0.0])[0]) > 0.0
        )
    else:
        h5["supported"] = False

    h6 = _run_contrast(
        contrast_label="H6_v4_extra_compute",
        treatment=args.treatment,
        control=args.control_2K,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    h7 = _run_contrast(
        contrast_label="H7_v4_generic_revise",
        treatment=args.treatment,
        control=args.control_generic,
        results_dir=args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    h8a = _h8a_shadow_revision_vs_draft_all_items(
        args.results_dir,
        cascade_arm=args.treatment,
        rng=rng,
        n_boot=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    h8b = _h8b_gate_calibration(args.results_dir)
    h8c = _h8c_commit_policy_comparison(
        args.results_dir,
        rng=rng,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    h9 = _h9_judge_proxy_agreement(args.results_dir)
    arm_means = _arm_means(args.results_dir)

    return {
        "config": {
            "version": "v0.4",
            "status": "real",
            "treatment_arm": args.treatment,
            "control_arm_primary": args.control,
            "control_arm_h6": args.control_2K,
            "control_arm_h7": args.control_generic,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "alpha": 0.05,
            "hypotheses": [h.value for h in Hypothesis],
        },
        "primary": primary["primary"],
        "H5": h5,
        "H6_v4_extra_compute": h6["primary"],
        "H6_v4_extra_compute_meta": h6["H5"],
        "H7_v4_generic_revise": h7["primary"],
        "H7_v4_generic_revise_meta": h7["H5"],
        "H8a_v4_shadow_revision_vs_draft": h8a,
        "H8b_v4_gate_calibration": h8b,
        "H8c_v4_commit_policy_comparison": h8c,
        "H9_v4_judge_proxy_agreement": h9,
        "arm_means_per_domain": arm_means,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=("v0.3", "v0.4"),
        default="v0.3",
        help="Which pre-registered hypothesis set to emit. v0.4 includes H8a/b/c, H9 and fixed-effects H5.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to benchmarks/results_<version>/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/stats.json",
    )
    parser.add_argument("--treatment", type=str, default="haiku_cascade")
    parser.add_argument("--control", type=str, default="haiku_bare")
    parser.add_argument(
        "--control-2K", type=str, default="haiku_bare_2K_scorer",
        help="control arm for H6 (architecture vs more compute)",
    )
    parser.add_argument(
        "--control-generic", type=str, default="haiku_generic_revise_2pass",
        help="control arm for H7 (architecture vs generic 2-pass)",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--n-permutations", type=int, default=50_000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="(v0.4 only) emit synthetic placeholders for every key without reading results.",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = REPO_ROOT / "benchmarks" / f"results_{args.version}"
    if args.out is None:
        args.out = args.results_dir / "stats.json"

    rng = np.random.default_rng(args.seed)

    if args.version == "v0.4":
        if args.synthetic or not args.results_dir.exists() or not any(
            (args.results_dir / f"{d}.json").exists() for d in DOMAINS
        ):
            payload = _synthetic_v0_4_payload(args)
        else:
            payload = _stats_v0_4(args, rng)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_clean_json(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(_clean_json(payload["config"]), indent=2, allow_nan=False))
        return 0

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
