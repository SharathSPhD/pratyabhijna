#!/usr/bin/env python3
"""Real-model honesty gate.

For each model id passed in (or read from `audit/hf_downloads.jsonl`), this
script verifies:

1. The HuggingFace cache contains the model's weight files (size > 1 MB).
2. The cached size is within +/-2% of the size reported by the HF API.
3. `transformers.AutoModel.from_pretrained` (or the appropriate class) loads
   the model end-to-end without error, including a forward pass on a fixed
   sanity prompt.
4. The forward pass produces non-degenerate logits (variance > 1e-6, no nan,
   no inf, no all-zero rows).
5. For sentence-transformer encoders, an embedding pair on a known similar
   sentence pair has cosine > 0.85, while an unrelated pair has cosine < 0.6.

Exit code:

* 0 - every model passed every check;
* 1 - at least one model is fake / missing / degenerate;
* 2 - script-level failure.

The output is JSON-on-stdout describing each model's gate outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent
SANITY_PROMPT = "The capital of France is"
EMBED_SIM_PAIR = ("A cat sits on a mat.", "A kitten rests on a rug.")
EMBED_UNREL_PAIR = ("A cat sits on a mat.", "Quantum chromodynamics describes the strong force.")


@dataclass
class ModelReport:
    model_id: str
    kind: str  # 'causal-lm' | 'sentence-transformer' | 'cross-encoder'
    cache_present: bool = False
    cache_bytes: int = 0
    hf_bytes: int = 0
    size_match: bool = False
    load_ok: bool = False
    sanity_logit_variance: float = 0.0
    embed_sim_cos: float | None = None
    embed_unrel_cos: float | None = None
    notes: list[str] = field(default_factory=list)
    ok: bool = False


def _huggingface_size(model_id: str) -> int:
    """Sum of file sizes from the HF API for the main revision."""
    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception as e:  # pragma: no cover - dependency must be present at runtime
        raise RuntimeError(f"huggingface_hub import failed: {e}") from e
    api = HfApi()
    info = api.model_info(model_id, files_metadata=True)
    total = 0
    for f in info.siblings or []:
        size = getattr(f, "size", None) or 0
        total += int(size)
    return total


def _cache_size(model_id: str) -> tuple[int, bool]:
    """Walk the HF hub cache and sum the on-disk size for `model_id`."""
    from huggingface_hub import constants  # type: ignore
    cache = Path(constants.HF_HUB_CACHE)
    safe = "models--" + model_id.replace("/", "--")
    target = cache / safe
    if not target.exists():
        return 0, False
    total = 0
    for p in target.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total, True


def _check_causal_lm(model_id: str, rep: ModelReport) -> None:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    inputs = tok(SANITY_PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.float().detach().cpu().numpy()
    rep.load_ok = True
    var = float(logits.var())
    rep.sanity_logit_variance = var
    if math.isnan(var) or math.isinf(var) or var < 1e-6:
        rep.notes.append(f"degenerate logits: var={var}")
    # No row is all zero / nan
    if (logits == 0).all() or (logits != logits).any() or (~ (logits == logits)).any():
        rep.notes.append("logits contain all-zero or NaN rows")


def _check_sentence_transformer(model_id: str, rep: ModelReport) -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore

    model = SentenceTransformer(model_id)
    rep.load_ok = True
    embs_sim = model.encode(list(EMBED_SIM_PAIR), normalize_embeddings=True)
    embs_unrel = model.encode(list(EMBED_UNREL_PAIR), normalize_embeddings=True)
    sim_cos = float(np.dot(embs_sim[0], embs_sim[1]))
    unrel_cos = float(np.dot(embs_unrel[0], embs_unrel[1]))
    rep.embed_sim_cos = sim_cos
    rep.embed_unrel_cos = unrel_cos
    rep.sanity_logit_variance = float(np.asarray(embs_sim).var())
    if sim_cos < 0.85:
        rep.notes.append(f"similar-pair cosine {sim_cos:.3f} < 0.85 (suspicious)")
    if unrel_cos > 0.6:
        rep.notes.append(f"unrelated-pair cosine {unrel_cos:.3f} > 0.60 (suspicious)")


def _check_cross_encoder(model_id: str, rep: ModelReport) -> None:
    from sentence_transformers import CrossEncoder  # type: ignore
    import numpy as np  # type: ignore

    model = CrossEncoder(model_id)
    rep.load_ok = True
    scores = model.predict([
        ["The cat sits on the mat.", "The cat is on the mat."],
        ["The cat sits on the mat.", "Quantum mechanics is hard."],
    ])
    arr = np.asarray(scores).flatten()
    rep.sanity_logit_variance = float(arr.var())
    if arr[0] <= arr[1]:
        rep.notes.append("cross-encoder did not rank similar > unrelated pair")


def _classify(model_id: str, hint: str | None) -> str:
    if hint:
        return hint
    lower = model_id.lower()
    if "sentence-transformers" in lower or "all-minilm" in lower:
        return "sentence-transformer"
    if "cross-encoder" in lower:
        return "cross-encoder"
    return "causal-lm"


def verify_model(model_id: str, kind_hint: str | None = None, tol: float = 0.02) -> ModelReport:
    rep = ModelReport(model_id=model_id, kind=_classify(model_id, kind_hint))
    cache_bytes, present = _cache_size(model_id)
    rep.cache_bytes = cache_bytes
    rep.cache_present = present
    try:
        hf_bytes = _huggingface_size(model_id)
        rep.hf_bytes = hf_bytes
    except Exception as e:
        rep.notes.append(f"hf metadata fetch failed: {e}")
        hf_bytes = 0
    if hf_bytes > 0 and cache_bytes > 0:
        diff = abs(cache_bytes - hf_bytes) / hf_bytes
        rep.size_match = diff <= tol
        if not rep.size_match:
            rep.notes.append(
                f"cache size {cache_bytes} differs from HF size {hf_bytes} by {diff:.1%} > {tol:.0%}"
            )
    elif cache_bytes < 1_000_000:
        rep.notes.append("cache contents under 1 MB - clearly insufficient")

    try:
        if rep.kind == "causal-lm":
            _check_causal_lm(model_id, rep)
        elif rep.kind == "sentence-transformer":
            _check_sentence_transformer(model_id, rep)
        elif rep.kind == "cross-encoder":
            _check_cross_encoder(model_id, rep)
        else:
            rep.notes.append(f"unknown kind: {rep.kind}")
    except Exception as e:
        rep.notes.append(f"load/forward failed: {type(e).__name__}: {e}")

    rep.ok = (
        rep.cache_present
        and rep.size_match
        and rep.load_ok
        and rep.sanity_logit_variance > 1e-6
        and not rep.notes
    )
    return rep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PCE real-model honesty gate.")
    parser.add_argument(
        "--model", action="append", default=[],
        help="HF model id to verify. Repeatable. If omitted, the audit log is read.",
    )
    parser.add_argument(
        "--kind", action="append", default=[],
        help="Model kind hint per --model: causal-lm | sentence-transformer | cross-encoder.",
    )
    parser.add_argument(
        "--audit",
        default=str(REPO_ROOT_DEFAULT / "audit" / "hf_downloads.jsonl"),
        help="Path to audit log (one JSON object per line with at least model_id and kind).",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Phase number; if < 5 this gate is a no-op and exits 0.",
    )
    parser.add_argument("--tol", type=float, default=0.02, help="Size-match tolerance.")
    args = parser.parse_args(argv)

    if args.phase is not None and args.phase < 5:
        print(json.dumps({"ok": True, "skipped": True, "reason": f"phase {args.phase} < 5"}))
        return 0

    requested: list[tuple[str, str | None]] = []
    if args.model:
        kinds = list(args.kind) + [None] * (len(args.model) - len(args.kind))
        requested.extend(zip(args.model, kinds, strict=False))
    else:
        audit_path = Path(args.audit)
        if not audit_path.exists():
            print(json.dumps({"ok": False, "error": f"no audit log at {audit_path} and no --model"}))
            return 1
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mid = obj.get("model_id")
            if not mid:
                continue
            requested.append((mid, obj.get("kind")))

    if not requested:
        print(json.dumps({"ok": False, "error": "no models to verify"}))
        return 1

    reports: list[dict[str, Any]] = []
    all_ok = True
    for mid, hint in requested:
        rep = verify_model(mid, hint, tol=args.tol)
        reports.append(asdict(rep))
        if not rep.ok:
            all_ok = False

    payload = {"ok": all_ok, "models": reports}
    print(json.dumps(payload, indent=2))
    if not all_ok:
        for r in reports:
            if not r["ok"]:
                print(
                    f"[verify_real_model] FAIL {r['model_id']} ({r['kind']}): {r['notes']}",
                    file=sys.stderr,
                )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
