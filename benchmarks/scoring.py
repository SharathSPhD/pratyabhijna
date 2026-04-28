"""Local creativity-scoring functions for Phase 9.

Each scoring function takes the raw text response (and any required metadata)
and returns a dict of axis->score. Scores are in [0, 1] and aggregated as a
weighted mean per task.

Design notes:
* Scoring is *deterministic* and uses only sentence-transformers embeddings,
  basic NLP, and the ananda/apohana operators. No LLM-as-judge.
* Aggregation weights follow the SPEC.md hypothesis tables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from pce.operators.ananda import ananda
from pce.operators.apohana import apohana
from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


@dataclass
class ItemScore:
    item_id: str
    domain: str
    mode: str  # "no_pce" | "with_pce"
    surface: str
    axes: dict[str, float]
    composite: float


def _make_candidate(text: str, embed: Embedder) -> Candidate:
    return Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=tuple(),
        text=text,
        logp=0.0,
        embedding=embed.encode(text or " "),
    )


def _distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return float(len(set(grams)) / len(grams))


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in re.split(r"[\n\r\.,;]+", text) if ln.strip()]


# ---------- Domain: poetry_gen (H3) ----------

def score_poetry_gen(text: str, *, item: dict[str, Any], embed: Embedder) -> ItemScore:
    """POEMetric-inspired axes: creativity, lexical_diversity, idiosyncrasy,
    emotional_resonance, literary_devices, imagery."""
    constraint_text = f"a {item['form']} about {item['topic']}"
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(item.get("must_avoid", [])),
    )
    cand = _make_candidate(text, embed)
    coherence = float(ananda(cand, constraint=constraint, embed=embed))
    apoha = float(apohana((cand,), constraint, embed=embed)[0])
    apoha_norm = float(max(0.0, min(1.0, (apoha + 1.0) / 2.0)))
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    lexical = _distinct_n(tokens, 1)  # type-token ratio at 1-gram
    bigram = _distinct_n(tokens, 2)
    # Idiosyncrasy: very low n-gram repetition + non-cliché.
    idiosyncrasy = float(0.5 * bigram + 0.5 * (1.0 - apoha_norm * 0.5))
    # Imagery: count of concrete-sensory words.
    sensory_keywords = {
        "light", "dark", "shadow", "sound", "silence", "smell", "scent",
        "taste", "touch", "cold", "warm", "wet", "dry", "rough", "smooth",
        "loud", "quiet", "gold", "silver", "red", "blue", "green", "white",
    }
    imagery = float(min(1.0, sum(1 for t in tokens if t in sensory_keywords) / 5.0))
    # Emotional resonance: negative+positive sentiment words.
    emotional_keywords = {
        "love", "loss", "grief", "fear", "hope", "joy", "longing", "ache",
        "tender", "fierce", "still", "alone", "wait", "remember", "forget",
        "weep", "smile", "lonely", "trembling",
    }
    emotional = float(min(1.0, sum(1 for t in tokens if t in emotional_keywords) / 4.0))
    # Literary devices: alliteration + simile + metaphor markers.
    alliteration = float(min(1.0, sum(
        1 for line in _split_lines(text)
        if any(line[i].lower() == line[j].lower() for i in range(len(line)) for j in range(i + 1, min(i + 30, len(line))))
    ) / max(1, len(_split_lines(text)))))
    similes = float(min(1.0, len(re.findall(r"\b(?:like|as if|as though)\b", text.lower())) / 2.0))
    devices = float(0.5 * alliteration + 0.5 * similes)
    creativity = float(0.5 * coherence + 0.5 * (1.0 - apoha_norm * 0.5))

    axes = {
        "creativity": creativity,
        "lexical_diversity": float(lexical),
        "idiosyncrasy": idiosyncrasy,
        "emotional_resonance": emotional,
        "literary_devices": devices,
        "imagery": imagery,
    }
    composite = float(np.mean(list(axes.values())))
    return ItemScore(
        item_id=item["id"],
        domain="poetry_gen",
        mode="",
        surface=text,
        axes=axes,
        composite=composite,
    )


# ---------- Domain: poetry_interp (H2) ----------

def score_poetry_interp(text: str, *, item: dict[str, Any], embed: Embedder) -> ItemScore:
    """Aspect-multiplicity score: how many of the supplied aspects can be
    detected in the response above a cosine threshold."""
    if not text.strip():
        axes = {"aspect_count": 0.0, "novelty": 0.0, "coverage": 0.0}
        return ItemScore(item["id"], "poetry_interp", "", text, axes, 0.0)
    resp = embed.encode(text)
    aspects = list(item.get("aspects", []))
    asp_emb = embed.encode(aspects) if aspects else np.zeros((0, embed.dim), dtype=np.float32)
    if asp_emb.ndim == 1:
        asp_emb = asp_emb[None, :]
    sims = asp_emb @ resp if asp_emb.size > 0 else np.array([0.0])
    threshold = 0.30
    aspect_count = int(np.sum(sims >= threshold))
    coverage = float(np.mean(np.clip(sims, 0.0, 1.0))) if asp_emb.size > 0 else 0.0
    retr = list(item.get("retrieval_set", []))
    if retr:
        r_emb = embed.encode(retr)
        if r_emb.ndim == 1:
            r_emb = r_emb[None, :]
        max_r = float((r_emb @ resp).max())
    else:
        max_r = 0.0
    novelty = float(max(0.0, 1.0 - max_r))
    axes = {
        "aspect_count": float(aspect_count) / max(1.0, float(len(aspects))),
        "novelty": novelty,
        "coverage": coverage,
    }
    composite = float(np.mean(list(axes.values())))
    return ItemScore(item["id"], "poetry_interp", "", text, axes, composite)


# ---------- Domain: aut (H1) ----------

def _split_uses(text: str) -> list[str]:
    """Split a Claude response into individual 'use' lines. Robust to bullets,
    numbered lists, or run-on prose."""
    lines: list[str] = []
    for ln in re.split(r"[\n\r]+", text):
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^[\-\*\u2022\d\.\)\]]+\s*", "", ln)
        if len(ln) > 4:
            lines.append(ln)
    return lines


def score_aut(text: str, *, item: dict[str, Any], embed: Embedder) -> ItemScore:
    """CreativityPrism slice: fluency, flexibility, originality, elaboration."""
    uses = _split_uses(text)
    fluency = float(min(1.0, len(uses) / 8.0))
    if not uses:
        axes = {"fluency": 0.0, "originality": 0.0, "elaboration": 0.0, "flexibility": 0.0}
        return ItemScore(item["id"], "aut", "", text, axes, 0.0)
    obj = item["object"]
    obvious_text = f"the standard everyday use of a {obj}"
    obvious_emb = embed.encode(obvious_text)
    use_emb = embed.encode(uses)
    if use_emb.ndim == 1:
        use_emb = use_emb[None, :]
    obvious_cos = use_emb @ obvious_emb
    originality = float(np.mean(np.clip(1.0 - obvious_cos, 0.0, 1.0)))
    avg_words = float(np.mean([len(u.split()) for u in uses]))
    elaboration = float(min(1.0, avg_words / 18.0))
    # Flexibility: cluster uses by cosine similarity (>= 0.55 = same cluster) and count clusters.
    n = len(uses)
    sims = use_emb @ use_emb.T
    cluster_id = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= 0.55:
                cluster_id[j] = cluster_id[i]
    flexibility = float(min(1.0, len(set(cluster_id)) / max(1, n)))
    axes = {
        "fluency": fluency,
        "originality": originality,
        "elaboration": elaboration,
        "flexibility": flexibility,
    }
    composite = float(np.mean(list(axes.values())))
    return ItemScore(item["id"], "aut", "", text, axes, composite)


# ---------- Domain: sci_creativity (H4) ----------

def score_sci_creativity(text: str, *, item: dict[str, Any], embed: Embedder) -> ItemScore:
    """Cross-frame analogy detection: how many of the supplied framings appear,
    plus a novelty bump for non-textbook explanations."""
    if not text.strip():
        return ItemScore(item["id"], "sci_creativity", "", text, {"frame_coverage": 0.0, "novelty": 0.0, "specificity": 0.0}, 0.0)
    resp = embed.encode(text)
    framings = list(item.get("framings", []))
    if framings:
        f_emb = embed.encode(framings)
        if f_emb.ndim == 1:
            f_emb = f_emb[None, :]
        sims = f_emb @ resp
        frame_coverage = float(np.mean(np.clip(sims, 0.0, 1.0)))
    else:
        frame_coverage = 0.0
    textbook_text = f"the standard textbook explanation of {item['question']}"
    textbook_emb = embed.encode(textbook_text)
    novelty = float(max(0.0, 1.0 - float(np.dot(textbook_emb, resp))))
    # Specificity: response length normalized.
    n_words = len(text.split())
    specificity = float(min(1.0, n_words / 80.0))
    axes = {
        "frame_coverage": frame_coverage,
        "novelty": novelty,
        "specificity": specificity,
    }
    composite = float(np.mean(list(axes.values())))
    return ItemScore(item["id"], "sci_creativity", "", text, axes, composite)
