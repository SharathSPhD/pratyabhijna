# ADR-001 — substrate choice

* Status: accepted
* Date: 2026-04-28
* Supersedes: —
* Superseded by: —

## Context

PCE's `cit`-substrate (the luminous-ground generative prior) needs three things from the local LM:

1. token-level logits accessible (so `cit` can verify temperature monotonicity);
2. fast-enough generation to support K=8 candidate continuations per cascade in ≤ 30 s wall-clock on Apple-Silicon CPU;
3. license that permits research and benchmark publication.

We need an ālayavijñāna store too, but the substrate question here is specifically the LM choice.

## Decision

Primary substrate: **Qwen/Qwen2-1.5B-Instruct** (~1.5 B parameters, ~3.1 GB on disk, Apache-2.0, public on HF). Selected as the operational primary for v0.1.0 because:

* The single-session benchmark must fit in ≤ 16 GB resident memory on the dev machine (CPU-only, no GPU).
* Phi-3-mini-4k at ~7.6 GB exceeds the cascade-runtime budget at K=8 candidates × 64-token generation × wall-clock-SLA 30 s on CPU.
* Qwen2-1.5B passes substrate honesty (`scripts/verify_real_model.py`) with logit variance 7.26 and exact size match.

Secondary fallback (deferred to GPU deployment): **microsoft/Phi-3-mini-4k-instruct** (~3.8 B parameters, MIT). Will be the default if a GPU is added in a future release.

Embedding substrate: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, Apache-2.0).

Cross-encoder (optional, for `ananda` reward axis): **cross-encoder/ms-marco-MiniLM-L-6-v2**.

## Consequences

* Apple-Silicon CPU is sufficient; no GPU dependency.
* Phi-3-mini-4k at fp32 is ~7.6 GB; we need fp16 for an 8 GB-RAM dev machine. The `CitSubstrate` wrapper picks fp16 when `torch.cuda.is_available() is False AND psutil.virtual_memory().total < 16 GB`.
* `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)` introduces small floating-point drift; we tolerate this since BMR is downstream and operates on numpy float64.
* Future ADR can swap in a 4-bit quantized Phi-3 if RAM is insufficient.

## Rejected alternatives

* **Llama-3.2-3B-Instruct**: license restrictions for benchmark publication and weight redistribution; would need user agreement.
* **GPT-OSS or Mistral-7B**: too big for the SLA on the dev machine; can be revisited for a future GPU-backed deployment.
* **API-only Claude Haiku as the cit-substrate**: would couple the engine to the same model used at the chat layer (circular) and leak benchmark control.

## Verification

`scripts/verify_real_model.py` enforces:

* HF cache size matches HF API ±2%;
* `AutoModelForCausalLM.from_pretrained(model_id)` succeeds;
* sanity prompt produces non-degenerate logits (var > 1e-6, no nan/inf);
* for `all-MiniLM-L6-v2` specifically: `cosine(embed("A cat sits on a mat."), embed("A kitten rests on a rug.")) > 0.85` and `cosine(embed("A cat..."), embed("Quantum chromodynamics...")) < 0.6`.

These checks form the substrate-honesty gate that Phase 5+ ralph-loops enforce.
