"""Scientific-creativity surface lint for the showcase pages.

Three lightweight heuristics, parallel in spirit to ``english_meter`` but
calibrated for the v0.4 sci_creativity domain:

* ``hedge_density`` — fraction of sentences carrying explicit
  uncertainty / hedge words ("perhaps", "may", "consistent with",
  "appears to"). The honest-claims story in §11 of the paper cites this
  as a *direction* metric: cascade revisions should *raise* hedge
  density, not lower it, because the revision brief explicitly asks for
  uncertainty acknowledgement.
* ``mechanism_word_count`` — count of mechanism-flavoured nouns/verbs
  ("because", "due to", "mechanism", "feedback", "regulates", "drives").
  A sketchy sci_creativity surface tends to read like a bullet list of
  facts; the revision pass should pull in mechanism prose.
* ``citation_or_attribution_count`` — count of inline attributions
  ("Friston (2010)", "as Hopfield showed", "the standard textbook
  account", etc.) — proxy for whether the surface engages with prior
  work or not.

These are not fact-checkers. They are surface-level lints for the
showcase rendering only — the paper is explicit that none of them
substitutes for human judgement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

__all__ = [
    "hedge_density",
    "mechanism_word_count",
    "citation_or_attribution_count",
    "lint_summary",
]


_HEDGE_PATTERNS: tuple[str, ...] = (
    r"\bperhaps\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bappears? to\b",
    r"\bsuggests?\b",
    r"\bconsistent with\b",
    r"\bplausibly\b",
    r"\btentatively\b",
    r"\bunder this view\b",
    r"\bone interpretation\b",
    r"\bwe do not (yet )?know\b",
    r"\buncertain\b",
)
_MECHANISM_PATTERNS: tuple[str, ...] = (
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bmechanism\b",
    r"\bfeedback\b",
    r"\bregulat(?:es?|ion)\b",
    r"\bcaus(?:es?|ed|ing|ation)\b",
    r"\bdrives?\b",
    r"\binhibits?\b",
    r"\bcouples?\b",
    r"\bhomeostas[ie]s\b",
    r"\bgradient\b",
    r"\bphase change\b",
)
_CITATION_PATTERNS: tuple[str, ...] = (
    r"\b[A-Z][a-z]+ \([12][0-9]{3}\)",
    r"\b(?:as|per)\s+[A-Z][a-z]+\b",
    r"\baccording to\b",
    r"\b(?:the )?textbook account\b",
    r"\b(?:the )?standard view\b",
)


def _count_hits(text: str, patterns: tuple[str, ...]) -> int:
    n = 0
    for pat in patterns:
        n += len(re.findall(pat, text, flags=re.IGNORECASE))
    return n


def _sentence_count(text: str) -> int:
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return max(1, len(sents))


def hedge_density(text: str) -> dict[str, float]:
    n_hits = _count_hits(text, _HEDGE_PATTERNS)
    n_sents = _sentence_count(text)
    return {
        "density": round(n_hits / n_sents, 4),
        "n_hedges": n_hits,
        "n_sentences": n_sents,
    }


def mechanism_word_count(text: str) -> dict[str, int]:
    return {"n_mechanism_terms": _count_hits(text, _MECHANISM_PATTERNS)}


def citation_or_attribution_count(text: str) -> dict[str, int]:
    return {"n_citations_or_attributions": _count_hits(text, _CITATION_PATTERNS)}


def lint_summary(text: str) -> dict:
    out: dict[str, object] = {}
    out.update(hedge_density(text))
    out.update(mechanism_word_count(text))
    out.update(citation_or_attribution_count(text))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scientific-creativity surface lint")
    p.add_argument("--text", required=True)
    args = p.parse_args(argv)
    print(json.dumps(lint_summary(args.text), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
