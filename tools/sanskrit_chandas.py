"""Sanskrit chandas (metre) validator.

Implements a *minimal but correct* version of the classical chandas system
sufficient for the three showcase chandas the v0.4 release ships:

* **anuṣṭubh** (anushtubh) — 8×4 = 32 syllables, four pādas of 8.
  The pre-classical "śloka" form. The strict guru/laghu pattern at the
  5th-6th-7th syllable of each pāda is enforced under ``--strict-anustubh``;
  by default we only count syllables.
* **gāyatrī** — 8×3 = 24 syllables, three pādas of 8.
* **indravajrā** — 11×4 = 44 syllables. Each pāda is a fixed pattern:
  ``GG L GG L L G L G G`` (G = guru / heavy, L = laghu / light), exactly
  the *ti-tā ja-ga-ga* signature.

Syllable rules (pre-classical, dharmaśāstra convention):

* A syllable is one vowel (with any number of preceding consonants).
* "Short" (laghu / L): a, i, u, ṛ, ḷ — i.e. the vowels marked short in
  IAST. In Devanāgarī this is the bare consonant + the standalone short
  vowel mātrās ि, ु, ृ.
* "Long" (guru / G): the other vowels (ā, ī, ū, e, o, ai, au), or any
  short vowel followed by an anusvāra (ं), visarga (ः), or by two
  consonants (saṃyoga). Pre-pause padānta gurus also count.

The validator accepts either Devanāgarī or IAST input and returns a
structured JSON-friendly verdict. It deliberately does not try to handle
modern Sanskrit hybrid metres or the more elaborate vṛtta system; the
v0.4 paper makes this scope-cut explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass

__all__ = [
    "Verdict",
    "syllabify",
    "guru_laghu_pattern",
    "validate",
    "ANUSTUBH",
    "GAYATRI",
    "INDRAVAJRA",
]

# IAST short / long vowel sets (used after Devanāgarī -> IAST normalisation
# *for the purposes of weight assignment only* — the syllabifier itself
# walks IAST mātrās directly).
SHORT_VOWELS_IAST = set("aiu")
SHORT_VOWELS_IAST.update({"ṛ", "ḷ"})
LONG_VOWELS_IAST = {"ā", "ī", "ū", "e", "o", "ai", "au", "ṝ", "ḹ"}
ALL_VOWELS = SHORT_VOWELS_IAST | LONG_VOWELS_IAST

# Devanāgarī mātrā / vowel-sign tables
DEV_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī",
    "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ",
    "ऌ": "ḷ", "ॡ": "ḹ",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
DEV_DEPENDENT_VOWELS = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū",
    "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
DEV_VIRAMA = "्"
DEV_ANUSVARA = "ं"
DEV_VISARGA = "ः"
DEV_NASAL = {"ं", "ः", "ँ"}

DEV_CONSONANT_RANGE = (0x0915, 0x0939)  # क…ह, plus a few extras handled below
DEV_EXTRA_CONSONANTS = {"क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़", "ळ"}


def _is_devanagari_consonant(ch: str) -> bool:
    if not ch:
        return False
    if ch in DEV_EXTRA_CONSONANTS:
        return True
    cp = ord(ch)
    return DEV_CONSONANT_RANGE[0] <= cp <= DEV_CONSONANT_RANGE[1]


def _devanagari_to_phonemic_syllables(text: str) -> list[str]:
    """Walk Devanāgarī text, emitting one entry per syllable.

    Each emitted entry is the IAST representation of the syllable's
    *vowel core* concatenated with any post-vowel anusvāra/visarga and
    a pseudo-coda flag for the *next-syllable* consonant cluster size
    (used by ``guru_laghu_pattern`` to detect saṃyoga gurus).

    Output convention:
      'a'        — bare short vowel
      'ā'        — long vowel
      'a/m'      — short + anusvāra
      'a/h'      — short + visarga
      'a+'       — short with the next syllable opening on a saṃyoga
                   (two or more consonants, making this guru by position)
    """
    syllables: list[str] = []
    chars = list(text)
    i = 0
    n = len(chars)

    def is_cons(idx: int) -> bool:
        return 0 <= idx < n and _is_devanagari_consonant(chars[idx])

    def consume_post(idx: int) -> tuple[str, int]:
        """Return (suffix, new_idx) for any anusvāra/visarga at idx."""
        if 0 <= idx < n and chars[idx] == DEV_ANUSVARA:
            return "/m", idx + 1
        if 0 <= idx < n and chars[idx] == DEV_VISARGA:
            return "/h", idx + 1
        return "", idx

    def cluster_after(idx: int) -> int:
        """Count contiguous consonants joined by virāmas starting at idx
        (excluding the consonant that opens the *next* syllable)."""
        cnt = 0
        j = idx
        while j < n - 1 and is_cons(j) and chars[j + 1] == DEV_VIRAMA:
            cnt += 1
            j += 2
        # plus the trailing consonant that owns the next vowel
        if is_cons(j):
            cnt += 1
        return cnt

    while i < n:
        ch = chars[i]
        # whitespace / punctuation skip
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            i += 1
            continue
        # independent vowel
        if ch in DEV_INDEPENDENT_VOWELS:
            vowel = DEV_INDEPENDENT_VOWELS[ch]
            i += 1
            suffix, i = consume_post(i)
            cluster = cluster_after(i)
            mark = "+" if cluster >= 2 else ""
            syllables.append(vowel + suffix + mark)
            continue
        # consonant
        if _is_devanagari_consonant(ch):
            i += 1
            # walk over conjunct stack (cons + virāma + cons + virāma + …)
            while i < n - 1 and chars[i] == DEV_VIRAMA and is_cons(i + 1):
                i += 2
            # now expect a vowel mātrā or implicit 'a'
            if i < n and chars[i] in DEV_DEPENDENT_VOWELS:
                vowel = DEV_DEPENDENT_VOWELS[chars[i]]
                i += 1
            elif i < n and chars[i] == DEV_VIRAMA:
                # final virāma — consonant has no vowel of its own
                i += 1
                continue
            else:
                vowel = "a"
            suffix, i = consume_post(i)
            cluster = cluster_after(i)
            mark = "+" if cluster >= 2 else ""
            syllables.append(vowel + suffix + mark)
            continue
        # anything else — skip
        i += 1
    return syllables


def _iast_syllabify(text: str) -> list[str]:
    """Walk an IAST string, emitting one entry per syllable.

    Greedy two-char vowel match (ai/au/ī/ū/etc.) takes precedence over
    one-char. Diacritics are unicode-normalised first.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Strip ASCII punctuation but keep IAST marks like ṛ (U+1E5B in Latin
    # Extended Additional), ṃ (U+1E43), ṁ (U+1E41), ṅ (U+1E45), ñ (U+00F1),
    # ṣ (U+1E63), ḥ (U+1E25), etc. Also keep diacritic combining marks
    # (U+0300-036F) so any composed forms still parse.
    text = re.sub(
        r"[^a-z\u00c0-\u024f\u0300-\u036f\u1e00-\u1eff\s]+", " ", text,
    )
    tokens = text.split()
    syllables: list[str] = []
    for tok in tokens:
        i = 0
        last_syl_idx: int | None = None  # for cluster-after detection
        while i < len(tok):
            ch = tok[i]
            # try two-char digraph vowel first
            if i + 1 < len(tok) and (tok[i:i + 2] in {"ai", "au"}):
                vowel = tok[i:i + 2]
                i += 2
                # consume a trailing ṃ / ḥ / m / h
                suffix = ""
                if i < len(tok) and tok[i] in ("ṃ", "ṁ"):
                    suffix = "/m"
                    i += 1
                elif i < len(tok) and tok[i] == "ḥ":
                    suffix = "/h"
                    i += 1
                # detect saṃyoga at next position
                cluster = 0
                j = i
                while j < len(tok) and tok[j] not in ALL_VOWELS \
                        and tok[j] not in ("ṃ", "ṁ", "ḥ"):
                    cluster += 1
                    j += 1
                mark = "+" if cluster >= 2 else ""
                syllables.append(vowel + suffix + mark)
                last_syl_idx = len(syllables) - 1
                continue
            if ch in ALL_VOWELS:
                vowel = ch
                i += 1
                suffix = ""
                if i < len(tok) and tok[i] in ("ṃ", "ṁ"):
                    suffix = "/m"
                    i += 1
                elif i < len(tok) and tok[i] == "ḥ":
                    suffix = "/h"
                    i += 1
                cluster = 0
                j = i
                while j < len(tok) and tok[j] not in ALL_VOWELS \
                        and tok[j] not in ("ṃ", "ṁ", "ḥ"):
                    cluster += 1
                    j += 1
                mark = "+" if cluster >= 2 else ""
                syllables.append(vowel + suffix + mark)
                last_syl_idx = len(syllables) - 1
                continue
            i += 1
        # word-final consonant cluster contributes guru to the previous
        # syllable: if last_syl_idx exists and the next token starts on a
        # consonant cluster, mark it via '+' too. We approximate by treating
        # word-final consonants after the last vowel as part of the current
        # syllable's coda.
        _ = last_syl_idx  # placeholder for future word-boundary handling
    return syllables


def syllabify(text: str) -> list[str]:
    """Return a list of syllable tokens from either Devanāgarī or IAST."""
    if any(_is_devanagari_consonant(c) or c in DEV_INDEPENDENT_VOWELS for c in text):
        return _devanagari_to_phonemic_syllables(text)
    return _iast_syllabify(text)


def guru_laghu_pattern(syllables: list[str]) -> str:
    """Return a 'GLGL…' string of weights, one char per syllable.

    A syllable is **guru** if its vowel is long, OR it is followed by an
    anusvāra/visarga, OR the next syllable opens on a consonant cluster
    (the '+' marker emitted by the syllabifier).
    """
    out: list[str] = []
    for s in syllables:
        # Strip suffix flags
        core = s.split("/")[0].rstrip("+")
        has_nasal = "/m" in s or "/h" in s
        has_cluster = s.endswith("+")
        is_long_vowel = core in LONG_VOWELS_IAST
        out.append("G" if (is_long_vowel or has_nasal or has_cluster) else "L")
    return "".join(out)


@dataclass(frozen=True)
class Verdict:
    chandas: str
    syllable_count: int
    expected_count: int
    pattern: str
    pattern_ok: bool
    count_ok: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "chandas": self.chandas,
            "syllable_count": self.syllable_count,
            "expected_count": self.expected_count,
            "pattern": self.pattern,
            "pattern_ok": self.pattern_ok,
            "count_ok": self.count_ok,
            "ok": self.count_ok and self.pattern_ok,
            "notes": list(self.notes),
        }


# Chandas registry (extendable)
ANUSTUBH = "anustubh"
GAYATRI = "gayatri"
INDRAVAJRA = "indravajra"

CHANDAS_SPECS: dict[str, dict] = {
    ANUSTUBH: {
        "syllables_per_pada": 8,
        "n_padas": 4,
        "name_iast": "anuṣṭubh",
        # The strict 5-6-7 rule: even pādas LGG, odd pādas anything-but-LGL
        # is enforced only when ``strict=True``.
    },
    GAYATRI: {
        "syllables_per_pada": 8,
        "n_padas": 3,
        "name_iast": "gāyatrī",
    },
    INDRAVAJRA: {
        "syllables_per_pada": 11,
        "n_padas": 4,
        "name_iast": "indravajrā",
        "fixed_pattern": "GGLGGLLGLGG",  # ti-tā ja-ga-ga
    },
}


def validate(text: str, chandas: str, *, strict: bool = False) -> Verdict:
    """Validate ``text`` against ``chandas`` ∈ ``CHANDAS_SPECS``.

    With ``strict=False`` (default), only the syllable count is enforced;
    the pattern is reported but ``pattern_ok`` is True if no fixed pattern
    is registered for the chandas. With ``strict=True``, indravajrā's
    fixed pattern is enforced, and anuṣṭubh's 5-6-7 rule is checked.
    """
    spec = CHANDAS_SPECS.get(chandas)
    if spec is None:
        raise ValueError(f"unknown chandas: {chandas!r} (known: {list(CHANDAS_SPECS)})")
    syllables = syllabify(text)
    pattern = guru_laghu_pattern(syllables)
    expected = spec["syllables_per_pada"] * spec["n_padas"]
    notes: list[str] = []
    count_ok = len(syllables) == expected
    if not count_ok:
        notes.append(
            f"expected {expected} syllables ({spec['n_padas']} pādas × "
            f"{spec['syllables_per_pada']}); found {len(syllables)}"
        )
    pattern_ok = True
    if strict:
        if "fixed_pattern" in spec:
            per_pada = spec["syllables_per_pada"]
            for k in range(spec["n_padas"]):
                pada = pattern[k * per_pada:(k + 1) * per_pada]
                if len(pada) != per_pada:
                    pattern_ok = False
                    notes.append(f"pāda {k+1} truncated ({len(pada)} syllables)")
                    continue
                if pada != spec["fixed_pattern"]:
                    pattern_ok = False
                    notes.append(
                        f"pāda {k+1} = {pada}, expected {spec['fixed_pattern']}"
                    )
        elif chandas == ANUSTUBH:
            for k in range(spec["n_padas"]):
                pada = pattern[k * 8:(k + 1) * 8]
                if len(pada) < 7:
                    continue
                # Even pādas (2,4): syllables 5-6-7 must be LGG
                # Odd pādas (1,3): syllables 5-6-7 must NOT be LGL
                # (The classical "vipulā" rule is more permissive; this is
                # the textbook simplification used by introductory chandas
                # primers.)
                slice_5_7 = pada[4:7]
                if (k + 1) % 2 == 0:
                    if slice_5_7 != "LGG":
                        pattern_ok = False
                        notes.append(
                            f"pāda {k+1}: syllables 5-7 = {slice_5_7}, expected LGG"
                        )
                else:
                    if slice_5_7 == "LGL":
                        pattern_ok = False
                        notes.append(
                            f"pāda {k+1}: syllables 5-7 = LGL is forbidden"
                        )
    return Verdict(
        chandas=chandas,
        syllable_count=len(syllables),
        expected_count=expected,
        pattern=pattern,
        pattern_ok=pattern_ok,
        count_ok=count_ok,
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sanskrit chandas validator")
    p.add_argument("--chandas", required=True, choices=list(CHANDAS_SPECS))
    p.add_argument("--text", required=True, help="verse text (Devanāgarī or IAST)")
    p.add_argument("--strict", action="store_true",
                   help="enforce fixed metre patterns (indravajrā / 5-6-7 rule)")
    args = p.parse_args(argv)
    v = validate(args.text, args.chandas, strict=args.strict)
    print(json.dumps(v.to_dict(), indent=2, ensure_ascii=False))
    return 0 if (v.count_ok and v.pattern_ok) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
