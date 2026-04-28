#!/usr/bin/env python3
"""Phase 9 statistics — H1..H6.

Reads `benchmarks/results/{poetry_gen,poetry_interp,aut,sci_creativity}.json` and
emits `benchmarks/results/stats.json` containing, per hypothesis:

* `estimate` (paired mean delta, treatment - control)
* `n` (number of paired observations)
* `hedges_g` with small-sample correction
* `bca_ci_95` 95% BCa bootstrap CI of the paired mean delta (10k resamples)
* `permutation_p_one_sided` (sign-flip permutation, exact when feasible)
* `wilcoxon_p_one_sided`
* `holm_p` (after Holm-Bonferroni across {H1, H2, H3, H4})
* `power_apriori` and `power_retrospective`
* `supported` (Holm-corrected p<0.05 AND BCa CI strictly > 0)

Treatment arm = `local_cascade` (PCE).
Control arm   = `claude_haiku` (the headline hypothesis).
We *also* report a sensitivity analysis vs `local_bare` so the cascade-vs-bare
delta is recorded as `H1.sensitivity_local_bare`, etc.

Hypotheses:
  H1 :  AUT composite (creativity·diversity·feasibility-blend)
  H2 :  poetry_interp composite (aspect_count + novelty + coverage)
  H3 :  poetry_gen   composite (creativity + lexdiv + idio + emo + dev + img)
  H4 :  sci_creativity composite
  H5 :  aggregate z-blend across H1..H4 with weights {H1:0.5, H2..H4:1/6 each}
  H6 :  within-PCE: vimarsa_event=True vs False, on per-item composite
        across (poetry_interp + poetry_gen + sci_creativity); independent
        groups (Mann-Whitney U + Hedges' g for independent samples).

Determinism: `--seed` controls the bootstrap and permutation RNGs.

CLI:
  uv run python benchmarks/stats.py [--results-dir benchmarks/results]
                                    [--seed 4242]
                                    [--out benchmarks/results/stats.json]
                                    [--n-permutations 50000]
                                    [--n-bootstrap 10000]
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

DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
HYPOTHESIS_DOMAIN = {"H1": "aut", "H2": "poetry_interp", "H3": "poetry_gen", "H4": "sci_creativity"}


@dataclass
class HypothesisResult:
    name: str
    domain: str
    n: int
    estimate: float
    hedges_g: float
    bca_ci_95: tuple[float, float]
    permutation_p_one_sided: float
    wilcoxon_p_one_sided: float
    holm_p: float
    power_apriori: float
    power_retrospective: float
    supported: bool
    treatment: str
    control: str


def _hedges_g_paired(d: np.ndarray) -> float:
    """Hedges' g for paired differences with small-sample correction."""
    n = len(d)
    if n < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return 0.0
    g = float(np.mean(d) / sd)
    # Hedges' bias correction (Borenstein 2009)
    j = 1.0 - 3.0 / (4.0 * n - 5.0) if n > 1 else 1.0
    return float(g * j)


def _paired_permutation_p_one_sided(d: np.ndarray, *, rng: np.random.Generator,
                                     alternative: str = "greater",
                                     n_permutations: int = 50_000) -> float:
    """Exact sign-flip permutation if 2**n <= n_permutations, else Monte Carlo.

    Tests H0: mean(d)=0 vs H1: mean(d)>0 (or <0).
    """
    n = len(d)
    obs = float(np.mean(d))
    if n == 0:
        return 1.0
    if 2 ** n <= max(n_permutations, 1024):
        # Exact: enumerate sign flips
        signs_all = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        means = (signs_all * d[None, :]).mean(axis=1)
        if alternative == "greater":
            count = int(np.sum(means >= obs - 1e-15))
        else:
            count = int(np.sum(means <= obs + 1e-15))
        return float(count / (2 ** n))
    # Monte Carlo
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    means = (signs * d[None, :]).mean(axis=1)
    if alternative == "greater":
        count = int(np.sum(means >= obs - 1e-15))
    else:
        count = int(np.sum(means <= obs + 1e-15))
    return float((count + 1) / (n_permutations + 1))


def _bca_ci_paired_mean(d: np.ndarray, *, rng: np.random.Generator,
                         n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    obs = float(np.mean(d))
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    # Bias-correction
    z0 = stats.norm.ppf((np.sum(boot < obs) + 0.5 * np.sum(boot == obs)) / n_boot)
    if not np.isfinite(z0):
        z0 = 0.0
    # Acceleration via jackknife
    jk = np.array([float(np.mean(np.delete(d, i))) for i in range(n)])
    jk_mean = float(np.mean(jk))
    num = float(np.sum((jk_mean - jk) ** 3))
    den = 6.0 * (float(np.sum((jk_mean - jk) ** 2)) ** 1.5 + 1e-30)
    accel = num / den
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / max(1 - accel * (z0 + z_alpha_lo), 1e-12))
    a2 = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / max(1 - accel * (z0 + z_alpha_hi), 1e-12))
    a1 = float(np.clip(a1, 0.0, 1.0))
    a2 = float(np.clip(a2, 0.0, 1.0))
    lo = float(np.quantile(boot, a1))
    hi = float(np.quantile(boot, a2))
    return (lo, hi)


def _power_paired_t(g: float, n: int, alpha: float = 0.05, alternative: str = "greater") -> float:
    """Approximate power for a paired one-sample t-test, given the effect size g.

    Uses noncentral t. We treat the alternative as one-sided.
    """
    if n < 2:
        return float("nan")
    df = n - 1
    nc = float(g) * math.sqrt(n)
    t_crit = float(stats.t.ppf(1 - alpha, df=df)) if alternative == "greater" else float(stats.t.ppf(alpha, df=df))
    if alternative == "greater":
        # P(T_nc > t_crit) under H1
        return float(1.0 - stats.nct.cdf(t_crit, df=df, nc=nc))
    return float(stats.nct.cdf(t_crit, df=df, nc=nc))


def _holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Return Holm-Bonferroni adjusted p-values."""
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
    return json.loads(p.read_text(encoding="utf-8"))


def _paired_arrays(rows: dict[str, dict[str, Any]], *, treatment: str, control: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids: list[str] = []
    t_vals: list[float] = []
    c_vals: list[float] = []
    for item_id, payload in sorted(rows.items()):
        t = payload.get(treatment, {})
        c = payload.get(control, {})
        ts = t.get("composite") if isinstance(t, dict) else None
        cs = c.get("composite") if isinstance(c, dict) else None
        if ts is None or cs is None:
            continue
        if not (math.isfinite(float(ts)) and math.isfinite(float(cs))):
            continue
        ids.append(item_id)
        t_vals.append(float(ts))
        c_vals.append(float(cs))
    return np.asarray(t_vals, dtype=float), np.asarray(c_vals, dtype=float), ids


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
) -> tuple[HypothesisResult, np.ndarray]:
    t_arr, c_arr, ids = _paired_arrays(rows, treatment=treatment, control=control)
    d = t_arr - c_arr
    if len(d) == 0:
        raise SystemExit(f"{name}: no paired observations for {treatment} vs {control} in {domain}")
    g = _hedges_g_paired(d)
    estimate = float(np.mean(d))
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
    supported = (holm < 0.05) and (bca[0] > 0.0)
    return (
        HypothesisResult(
            name=name,
            domain=domain,
            n=len(d),
            estimate=estimate,
            hedges_g=g,
            bca_ci_95=bca,
            permutation_p_one_sided=perm_p,
            wilcoxon_p_one_sided=wil_p,
            holm_p=holm,
            power_apriori=pow_a,
            power_retrospective=pow_r,
            supported=supported,
            treatment=treatment,
            control=control,
        ),
        d,
    )


def _hypothesis_to_dict(h: HypothesisResult) -> dict[str, Any]:
    out = asdict(h)
    out["bca_ci_95"] = list(h.bca_ci_95)
    return out


def _within_pce_event_test(
    results_dir: Path, *, rng: np.random.Generator, n_boot: int = 10_000
) -> dict[str, Any]:
    """H6: within-PCE, vimarsa_event=True vs False on per-item composite, pooled
    across poetry_interp + poetry_gen + sci_creativity (the directional domains).
    Independent-samples (Mann-Whitney U) since the firing condition partitions
    items unevenly.
    """
    fired: list[float] = []
    not_fired: list[float] = []
    for dom in ("poetry_interp", "poetry_gen", "sci_creativity"):
        data = _load_domain(results_dir, dom)
        for _, payload in data["rows"].items():
            row = payload.get("local_cascade", {})
            if not isinstance(row, dict):
                continue
            comp = row.get("composite")
            if comp is None or not math.isfinite(float(comp)):
                continue
            ev = bool(row.get("meta", {}).get("vimarsa_event", False))
            (fired if ev else not_fired).append(float(comp))
    n_fired = len(fired)
    n_not = len(not_fired)
    if n_fired == 0 or n_not == 0:
        return {
            "name": "H6",
            "n_fired": n_fired,
            "n_not_fired": n_not,
            "estimate": float("nan"),
            "p_one_sided": float("nan"),
            "supported": False,
            "note": "insufficient data: one of the groups is empty",
        }
    a = np.asarray(fired, dtype=float)
    b = np.asarray(not_fired, dtype=float)
    diff = float(np.mean(a) - np.mean(b))
    sd_pooled = math.sqrt((np.var(a, ddof=1) * (n_fired - 1) + np.var(b, ddof=1) * (n_not - 1))
                          / max(n_fired + n_not - 2, 1)) if n_fired + n_not > 2 else float("nan")
    g = float(diff / sd_pooled) if sd_pooled and math.isfinite(sd_pooled) and sd_pooled > 0 else 0.0
    try:
        u = stats.mannwhitneyu(a, b, alternative="greater")
        u_p = float(u.pvalue)
    except ValueError:
        u_p = float("nan")
    # Bootstrap CI for difference of means
    n_total = max(n_boot, 1)
    boot_diffs = (
        rng.choice(a, size=(n_total, n_fired), replace=True).mean(axis=1)
        - rng.choice(b, size=(n_total, n_not), replace=True).mean(axis=1)
    )
    lo, hi = float(np.quantile(boot_diffs, 0.025)), float(np.quantile(boot_diffs, 0.975))
    return {
        "name": "H6",
        "n_fired": n_fired,
        "n_not_fired": n_not,
        "estimate": diff,
        "hedges_g_independent": g,
        "ci_95": [lo, hi],
        "mannwhitney_u_p_one_sided": u_p,
        "supported": (u_p < 0.05 and lo > 0.0),
        "note": "fired vs not-fired vimarsa_event, pooled poetry_interp+poetry_gen+sci_creativity",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "stats.json")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--n-permutations", type=int, default=50_000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    raw_p_directional: dict[str, float] = {}
    primary_results: dict[str, HypothesisResult] = {}
    deltas: dict[str, np.ndarray] = {}

    for h_name, dom in HYPOTHESIS_DOMAIN.items():
        data = _load_domain(args.results_dir, dom)
        rows = data["rows"]
        res, d = _compute_hypothesis(
            h_name, dom, rows,
            treatment="local_cascade", control="claude_haiku",
            rng=rng, n_permutations=args.n_permutations, n_boot=args.n_bootstrap,
        )
        primary_results[h_name] = res
        deltas[h_name] = d
        raw_p_directional[h_name] = res.permutation_p_one_sided

    holm_adj = _holm_bonferroni(raw_p_directional)
    for h_name, res in primary_results.items():
        res.holm_p = holm_adj[h_name]
        res.supported = (res.holm_p < 0.05) and (res.bca_ci_95[0] > 0.0)

    # H5: composite z across H1..H4 with weights
    weights = {"H1": 0.5, "H2": 1.0 / 6.0, "H3": 1.0 / 6.0, "H4": 1.0 / 6.0}

    def _zscore(arr: np.ndarray) -> np.ndarray:
        sd = float(np.std(arr, ddof=1))
        if sd == 0.0:
            return np.zeros_like(arr)
        return (arr - float(np.mean(arr))) / sd

    z_blend_terms: list[np.ndarray] = []
    for h_name, w in weights.items():
        z_blend_terms.append(w * _zscore(deltas[h_name]))
    # Stack by item-position; we treat domains as independent paired samples
    # contributing to a pooled paired delta vector.
    pooled = np.concatenate(z_blend_terms)
    g5 = _hedges_g_paired(pooled)
    est5 = float(np.mean(pooled))
    perm5 = _paired_permutation_p_one_sided(pooled, rng=rng, alternative="greater",
                                             n_permutations=args.n_permutations)
    bca5 = _bca_ci_paired_mean(pooled, rng=rng, n_boot=args.n_bootstrap)
    pow_a5 = _power_paired_t(0.5, n=len(pooled), alpha=0.05, alternative="greater")
    pow_r5 = _power_paired_t(g5, n=len(pooled), alpha=0.05, alternative="greater")
    h5 = {
        "name": "H5",
        "n": int(len(pooled)),
        "estimate_pooled_z": est5,
        "hedges_g": g5,
        "bca_ci_95": list(bca5),
        "permutation_p_one_sided": perm5,
        "power_apriori": pow_a5,
        "power_retrospective": pow_r5,
        "supported": bool(perm5 < 0.05 and bca5[0] > 0.0),
        "note": "pooled z-blend across H1..H4 with weights H1:0.5, H2/H3/H4:1/6 each",
        "weights": weights,
    }

    # Sensitivity: cascade vs local_bare (isolates PCE-vs-bare local)
    sensitivity: dict[str, dict[str, Any]] = {}
    for h_name, dom in HYPOTHESIS_DOMAIN.items():
        data = _load_domain(args.results_dir, dom)
        try:
            res_s, _ = _compute_hypothesis(
                f"{h_name}_sensitivity_local_bare",
                dom,
                data["rows"],
                treatment="local_cascade",
                control="local_bare",
                rng=rng,
                n_permutations=args.n_permutations,
                n_boot=args.n_bootstrap,
            )
            sensitivity[h_name] = _hypothesis_to_dict(res_s)
        except SystemExit as e:
            sensitivity[h_name] = {"error": str(e)}

    # H6
    h6 = _within_pce_event_test(args.results_dir, rng=rng, n_boot=args.n_bootstrap)

    out = {
        "config": {
            "treatment_arm": "local_cascade",
            "control_arm_primary": "claude_haiku",
            "control_arm_sensitivity": "local_bare",
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "alpha": 0.05,
        },
        "primary": {h: _hypothesis_to_dict(r) for h, r in primary_results.items()},
        "H5": h5,
        "H6": h6,
        "sensitivity_vs_local_bare": sensitivity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # Pretty summary to stdout
    print(json.dumps({h: {
        "n": r.n, "estimate": r.estimate, "g": r.hedges_g,
        "bca_ci_95": list(r.bca_ci_95),
        "perm_p": r.permutation_p_one_sided,
        "wilcoxon_p": r.wilcoxon_p_one_sided,
        "holm_p": r.holm_p,
        "power_a": r.power_apriori, "power_r": r.power_retrospective,
        "supported": r.supported,
    } for h, r in primary_results.items()} | {"H5": {
        "n": h5["n"], "est_z": h5["estimate_pooled_z"], "g": h5["hedges_g"],
        "bca_ci_95": h5["bca_ci_95"], "perm_p": h5["permutation_p_one_sided"],
        "supported": h5["supported"],
    }, "H6": {
        "n_fired": h6["n_fired"], "n_not": h6["n_not_fired"],
        "estimate": h6["estimate"], "p": h6.get("mannwhitney_u_p_one_sided"),
        "supported": h6["supported"],
    }}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
