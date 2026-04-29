#!/usr/bin/env python3
"""Train the v0.4 LearnedGate logistic regression on v0.3 cascade traces.

ADR-002 (v0.4): the model decides commit-policy ("revision vs draft") from
five features extracted from each cascade item's audit:

    [delta_F, novelty, aspect_count, ananda, budget_balance]

Labels are constructed post-hoc by re-scoring both ``surface_draft`` and
``surface_revision`` with the existing benchmark scoring functions and
labelling the row ``1`` (revision better) iff
``composite(revision) > composite(draft)``.

Cross-validation is leave-one-domain-out across the four v0.3 domains
``{aut, poetry_gen, poetry_interp, sci_creativity}``. The reported AUROC
is the mean across the four folds. The deployed model is the final fit on
all four domains.

Inputs:

* ``benchmarks/results_v0.3/<domain>.json`` — per-domain cascade rows
  containing ``haiku_cascade.meta.surface_draft`` and
  ``haiku_cascade.meta.surface_revision``.

Outputs:

* ``artifacts/learned_gate_v0_4.joblib`` — pickled
  ``{"model": LogisticRegression, "scaler": StandardScaler}`` so the
  downstream :class:`pce.policies.LearnedGate` can reconstruct both.
* ``artifacts/learned_gate_v0_4.metadata.json`` — fold AUROC, per-fold
  AUROC, feature coefficients, training data SHA-256, n_train / n_test.

Acceptance: ADR-002 sets the gate at mean leave-one-domain-out AUROC
≥ 0.55. If the AUROC bar is missed, the metadata records the failure
and ``LearnedGate`` falls back to ``EventGated`` at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmarks import scoring as bench_scoring  # noqa: E402
from pce.policies.commit import (  # noqa: E402
    PolicyFeatures,
    extract_features_from_audit,
)
from pce.substrate.embed import Embedder  # noqa: E402

DOMAINS = ("aut", "poetry_gen", "poetry_interp", "sci_creativity")
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results_v0.3"
OUT_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = OUT_DIR / "learned_gate_v0_4.joblib"
METADATA_PATH = OUT_DIR / "learned_gate_v0_4.metadata.json"


@dataclass
class TrainingRow:
    domain: str
    item_id: str
    features: PolicyFeatures
    label: int  # 1 if revision better than draft
    composite_draft: float
    composite_revision: float


_SCORE_FUNCS: dict[str, Any] = {
    "aut": bench_scoring.score_aut,
    "poetry_gen": bench_scoring.score_poetry_gen,
    "poetry_interp": bench_scoring.score_poetry_interp,
    "sci_creativity": bench_scoring.score_sci_creativity,
}


def _score(domain: str, text: str, item: dict[str, Any], embed: Embedder) -> float:
    fn = _SCORE_FUNCS[domain]
    if not text.strip():
        return 0.0
    out = fn(text, item=item, embed=embed)
    return float(out.composite)


def _load_domain_rows(
    domain: str, embed: Embedder, *, verbose: bool = False
) -> list[TrainingRow]:
    path = RESULTS_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"v0.3 results missing for domain={domain!r}: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[TrainingRow] = []
    for item_id, payload in data.get("rows", {}).items():
        cascade = payload.get("haiku_cascade") or {}
        meta = cascade.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        if not meta.get("ok", True):
            continue
        if not meta.get("two_pass", False):
            continue
        surface_draft = str(meta.get("surface_draft", "") or "")
        surface_revision = str(meta.get("surface_revision", "") or "")
        if not surface_draft.strip() or not surface_revision.strip():
            continue
        item = payload.get("item") or {}
        c_draft = _score(domain, surface_draft, item, embed)
        c_rev = _score(domain, surface_revision, item, embed)
        # The audit dict in v0.3 traces is sparse; ``extract_features_from_audit``
        # falls back to defaults for fields v0.3 didn't persist (aspect_count,
        # ananda, budget_balance). delta_F_draft and novelty are always there.
        feats = extract_features_from_audit(meta)
        label = 1 if c_rev > c_draft else 0
        rows.append(
            TrainingRow(
                domain=domain,
                item_id=str(item_id),
                features=feats,
                label=label,
                composite_draft=c_draft,
                composite_revision=c_rev,
            )
        )
        if verbose:
            print(
                f"  [{domain}] {item_id}: draft={c_draft:.4f} rev={c_rev:.4f} "
                f"label={label}",
                flush=True,
            )
    return rows


def _to_xy(rows: list[TrainingRow]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([r.features.as_vector() for r in rows], dtype=np.float64)
    y = np.array([r.label for r in rows], dtype=np.int64)
    return X, y


def _train_one(X_train: np.ndarray, y_train: np.ndarray) -> tuple[Any, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=0,
        max_iter=200,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _file_hashes() -> dict[str, str]:
    h: dict[str, str] = {}
    for d in DOMAINS:
        path = RESULTS_DIR / f"{d}.json"
        if path.exists():
            blob = path.read_bytes()
            h[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(blob).hexdigest()
    return h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--auroc-floor", type=float, default=0.55,
                        help="Minimum mean leave-one-domain-out AUROC to pass.")
    parser.add_argument("--strict", action="store_true",
                        help="Return non-zero if AUROC is below floor.")
    args = parser.parse_args()

    print("[train] loading v0.3 cascade traces ...", flush=True)
    embed = Embedder()
    by_domain: dict[str, list[TrainingRow]] = {}
    for domain in DOMAINS:
        by_domain[domain] = _load_domain_rows(domain, embed, verbose=args.verbose)
        n = len(by_domain[domain])
        n_pos = sum(r.label for r in by_domain[domain])
        print(
            f"[train]   {domain}: n={n} (revision_better={n_pos}, "
            f"draft_better={n - n_pos})",
            flush=True,
        )
    rows_all = [r for v in by_domain.values() for r in v]
    if not rows_all:
        print("[train] no rows loaded; aborting", file=sys.stderr)
        return 2

    print("[train] running leave-one-domain-out CV ...", flush=True)
    fold_aurocs: dict[str, float] = {}
    for held_out in DOMAINS:
        train_rows = [r for d, rs in by_domain.items() for r in rs if d != held_out]
        test_rows = by_domain[held_out]
        if not test_rows:
            fold_aurocs[held_out] = float("nan")
            continue
        X_tr, y_tr = _to_xy(train_rows)
        X_te, y_te = _to_xy(test_rows)
        model, scaler = _train_one(X_tr, y_tr)
        X_te_scaled = scaler.transform(X_te)
        proba = model.predict_proba(X_te_scaled)
        # Pick the column for class 1 (revision_better).
        try:
            classes = list(model.classes_)
            pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
        except (AttributeError, ValueError):
            pos_idx = -1
        auroc = _auroc(y_te, proba[:, pos_idx])
        fold_aurocs[held_out] = auroc
        print(
            f"[train]   held_out={held_out}: AUROC={auroc:.4f} "
            f"(n_test={len(test_rows)}, n_train={len(train_rows)})",
            flush=True,
        )

    finite_aurocs = [v for v in fold_aurocs.values() if not np.isnan(v)]
    mean_auroc = float(np.mean(finite_aurocs)) if finite_aurocs else float("nan")
    print(f"[train] mean leave-one-domain-out AUROC = {mean_auroc:.4f}", flush=True)

    print("[train] training final model on all domains ...", flush=True)
    X_all, y_all = _to_xy(rows_all)
    final_model, final_scaler = _train_one(X_all, y_all)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore[import-untyped]

    joblib.dump(
        {"model": final_model, "scaler": final_scaler},
        MODEL_PATH,
    )
    print(f"[train] wrote {MODEL_PATH}", flush=True)

    coefs = final_model.coef_[0].tolist() if hasattr(final_model, "coef_") else []
    intercept = (
        float(final_model.intercept_[0]) if hasattr(final_model, "intercept_") else 0.0
    )
    metadata: dict[str, Any] = {
        "schema_version": "v0.4-learned-gate-1",
        "feature_order": list(PolicyFeatures.feature_names()),
        "coefficients": dict(zip(PolicyFeatures.feature_names(), coefs, strict=False)),
        "intercept": intercept,
        "scaler_mean": [float(x) for x in getattr(final_scaler, "mean_", [])],
        "scaler_scale": [float(x) for x in getattr(final_scaler, "scale_", [])],
        "fold_aurocs": fold_aurocs,
        "mean_auroc": mean_auroc,
        "n_total": int(len(rows_all)),
        "n_per_domain": {d: int(len(by_domain[d])) for d in DOMAINS},
        "label_balance": {
            d: {
                "revision_better": int(sum(r.label for r in by_domain[d])),
                "draft_better": int(len(by_domain[d]) - sum(r.label for r in by_domain[d])),
            }
            for d in DOMAINS
        },
        "auroc_floor": float(args.auroc_floor),
        "auroc_floor_passed": bool(
            not np.isnan(mean_auroc) and mean_auroc >= float(args.auroc_floor)
        ),
        "training_data_sha256": _file_hashes(),
        "model_path": str(MODEL_PATH.relative_to(REPO_ROOT)),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] wrote {METADATA_PATH}", flush=True)

    if not metadata["auroc_floor_passed"]:
        print(
            f"[train] WARN: mean AUROC {mean_auroc:.4f} below floor "
            f"{args.auroc_floor:.4f}; LearnedGate falls back to EventGated at runtime",
            file=sys.stderr,
        )
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
