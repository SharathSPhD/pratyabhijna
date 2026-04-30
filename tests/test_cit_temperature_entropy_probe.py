"""v0.4 Phase 2 (ADR-001) gate: best-of-K entropy probe.

The v0.4 plan's prove-gate v0.4-α requires:

    Best-of-K entropy probe: with the same seed and prompt, n-gram entropy
    at cit_temperature=0.9 is strictly greater than at cit_temperature=0.2
    on a 12-item probe set.

This test validates that property at the candidate level, using a fake
``LMProtocol`` whose output deterministically reflects the prompt prefix.
Because the v0.4 best-of-K mechanism scales ``K_runtime`` with
``cit_temperature`` AND prepends one of 8 frozen perturbation strings to
each candidate's prompt, the joint candidate text distribution must have
strictly higher n-gram entropy at high ``cit_temperature``.

Two kinds of evidence are checked:

1. ``K_runtime`` itself is strictly larger at ``cit_temperature=0.9`` than
   at ``cit_temperature=0.2`` for the K_eff=4 default — this is the
   monotonicity guarantee from ADR-001.
2. Aggregate token-bigram entropy across the 12 probe items is strictly
   greater at ``cit_temperature=0.9`` than at ``cit_temperature=0.2``.

The fake LM appends the (truncated) prompt prefix to its output so that
distinct perturbation prefixes yield distinct candidate texts. Real
HaikuLM behavior under prompt perturbation is empirically validated in
the prove-gate Haiku run, not in this unit test.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pce.operators.iccha import iccha, k_runtime_for
from pce.substrate.embed import Embedder
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, Constraint


class _PromptEchoEmbed(Embedder):
    """Tiny stub embedder so iccha's downstream cosine ops do not blow up."""

    def __init__(self) -> None:
        self.model_id = "fake-embedder"
        self.dim = 16

    def encode(self, texts):  # type: ignore[no-untyped-def, override]
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts], axis=0)

    def _vec(self, t: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class _PromptEchoLM:
    """Fake LM whose output deterministically reflects the prompt prefix.

    The first 40 chars of the prompt are echoed back into the response so
    that distinct perturbation prefixes (the 8-element ``PERTURBATION_TABLE``)
    yield distinct candidate texts. This is the substrate-side requirement
    that lets cit_temperature-driven perturbation actually move the n-gram
    entropy.
    """

    name = "prompt-echo-lm"
    supports_logprobs = True
    supports_score = False
    supports_entropy = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int
    ) -> Candidate:
        self.calls.append({"prompt": prompt[:80], "seed": int(seed)})
        prefix = prompt[:40].replace("\n", " ")
        text = (
            f"Reply seed={seed} prefix={prefix} body terms alpha beta gamma "
            f"delta epsilon zeta eta theta iota kappa."
        )
        embedder = _PromptEchoEmbed()
        emb = embedder.encode(text)
        return Candidate(
            seed=int(seed),
            sampler=dict(sampler),
            tokens=tuple(range(max_tokens)),
            text=text,
            logp=-1.0,
            embedding=emb,
        )

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "n_calls": len(self.calls)}

    def length_proxy_logp(self, candidate: Candidate) -> float:
        return float(candidate.logp)


def _protocol(lm: _PromptEchoLM) -> LMProtocol:
    assert isinstance(lm, LMProtocol)
    return lm


def _bigram_entropy_bits(texts: list[str]) -> float:
    """Aggregate token-bigram Shannon entropy across a list of texts (in bits)."""
    counts: dict[tuple[str, str], int] = {}
    total = 0
    for text in texts:
        toks = text.split()
        for i in range(len(toks) - 1):
            key = (toks[i].lower(), toks[i + 1].lower())
            counts[key] = counts.get(key, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return float(h)


# Twelve diverse probe prompts (the v0.4 plan's "12-item probe set").
PROBE_PROMPTS = (
    "Compose a haiku about autumn leaves.",
    "List unusual uses for a brick.",
    "Describe a duck-rabbit aspect shift.",
    "Solve a logical puzzle about three islanders.",
    "Write a one-line elegy for a forgotten language.",
    "Reframe a daily commute as a pilgrimage.",
    "Explain entropy to a five-year-old.",
    "Imagine the smell of an old library.",
    "Argue for or against minimalism.",
    "Tell a one-sentence ghost story.",
    "Describe the sound of dawn breaking.",
    "Invent a verb for the act of waiting in vain.",
)


def _constraint(embed: Embedder, text: str) -> Constraint:
    return Constraint(text=text, embedding=embed.encode(text), must_avoid=())


@pytest.mark.parametrize("k_eff", [4])
def test_k_runtime_strictly_larger_at_higher_cit_temperature(k_eff: int) -> None:
    """ADR-001: K_runtime is strictly monotonic at the gate's two probe points."""
    k_lo = k_runtime_for(k_eff, 0.2)
    k_hi = k_runtime_for(k_eff, 0.9)
    assert k_hi > k_lo, (
        f"K_runtime should be strictly larger at cit_temp=0.9; got {k_lo=} {k_hi=}"
    )


def test_ngram_entropy_increases_with_cit_temperature_on_probe_set() -> None:
    """v0.4-α gate: n-gram entropy at cit_temp=0.9 > cit_temp=0.2 on 12 prompts."""
    embed = _PromptEchoEmbed()
    lm_lo = _PromptEchoLM()
    lm_hi = _PromptEchoLM()
    base_seed = 7
    K_eff = 4

    texts_lo: list[str] = []
    texts_hi: list[str] = []
    for i, prompt in enumerate(PROBE_PROMPTS):
        c = _constraint(embed, prompt)
        cands_lo = iccha(
            prompt,
            c,
            lm=_protocol(lm_lo),
            K=K_eff,
            base_seed=base_seed + i,
            max_tokens=24,
            cit_temperature=0.2,
            cit_temperature_mechanism="best_of_k",
        )
        cands_hi = iccha(
            prompt,
            c,
            lm=_protocol(lm_hi),
            K=K_eff,
            base_seed=base_seed + i,
            max_tokens=24,
            cit_temperature=0.9,
            cit_temperature_mechanism="best_of_k",
        )
        texts_lo.extend(cand.text for cand in cands_lo)
        texts_hi.extend(cand.text for cand in cands_hi)

    h_lo = _bigram_entropy_bits(texts_lo)
    h_hi = _bigram_entropy_bits(texts_hi)
    # Strictly greater per the gate; record both for diagnostics on failure.
    assert h_hi > h_lo, (
        f"v0.4-α gate failed: n-gram entropy did not increase with "
        f"cit_temperature. h_lo={h_lo:.4f} bits, h_hi={h_hi:.4f} bits, "
        f"len(texts_lo)={len(texts_lo)} len(texts_hi)={len(texts_hi)}"
    )


def test_ngram_entropy_off_mechanism_unchanged_by_cit_temperature() -> None:
    """When the mechanism is ``"off"``, cit_temperature must NOT change candidate width.

    This is the negative control for ADR-001: only the ``"best_of_k"``
    mechanism makes cit_temperature causal. ``"off"`` records it on the
    audit but does not enter generation, so K_runtime stays at K_eff and
    no perturbation is applied.
    """
    embed = _PromptEchoEmbed()
    lm_lo = _PromptEchoLM()
    lm_hi = _PromptEchoLM()
    base_seed = 0
    K_eff = 4
    prompt = PROBE_PROMPTS[0]
    c = _constraint(embed, prompt)
    cands_lo = iccha(
        prompt,
        c,
        lm=_protocol(lm_lo),
        K=K_eff,
        base_seed=base_seed,
        max_tokens=24,
        cit_temperature=0.2,
        cit_temperature_mechanism="off",
    )
    cands_hi = iccha(
        prompt,
        c,
        lm=_protocol(lm_hi),
        K=K_eff,
        base_seed=base_seed,
        max_tokens=24,
        cit_temperature=0.9,
        cit_temperature_mechanism="off",
    )
    assert len(cands_lo) == K_eff
    assert len(cands_hi) == K_eff
    # No perturbation prefix at either temperature.
    for c_lo, c_hi in zip(cands_lo, cands_hi, strict=True):
        assert c_lo.text == c_hi.text
