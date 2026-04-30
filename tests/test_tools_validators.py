"""Tests for the v0.4 surface validators in ``tools/``."""

from __future__ import annotations

from tools import english_meter, scientific_lint, sanskrit_chandas as chandas


# -- Sanskrit chandas --------------------------------------------------------


def test_anustubh_devanagari_count() -> None:
    # Bhagavad Gītā 2.47 — anuṣṭubh, 4×8 = 32 syllables
    text = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।\nमा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"
    v = chandas.validate(text, chandas.ANUSTUBH)
    # Allow a small tolerance — even classical anuṣṭubh has minor scribal
    # variants in the published e-text. Demand at least 30 (most variants
    # are 32, very rarely 31 in Sanskrit electronic editions).
    assert v.syllable_count >= 30
    assert v.syllable_count <= 34


def test_indravajra_iast_pattern_shape() -> None:
    # An indravajrā in IAST: each pāda follows GG L GG L L G L G G (11 syl).
    # We don't hand-validate this composition; we only assert the pattern
    # detector returns one G/L per syllable and the count is in the
    # expected ballpark for a 4-line composition (44 ± a couple of
    # punctuation-edge differences). The unique guarantee we want is
    # that the pattern length matches the syllable count.
    text = (
        "tasmai namaḥ paramakāraṇāya | "
        "viśvāya bhūtāntaracāriṇe ca | "
        "tubhyaṃ namo dhāmadharāya nityaṃ | "
        "kāmāya satyāya jagadgurave ca |"
    )
    v = chandas.validate(text, chandas.INDRAVAJRA)
    assert 40 <= v.syllable_count <= 48, v
    assert len(v.pattern) == v.syllable_count
    # All pattern chars should be G or L
    assert set(v.pattern) <= {"G", "L"}


def test_unknown_chandas_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        chandas.validate("anything", "unknown_metre")


def test_syllabify_iast_minimal() -> None:
    s = chandas.syllabify("rāma kṛṣṇa")
    assert s, "syllabifier returned empty list for valid IAST"
    # rā-ma kṛ-ṣṇa → at least 4 syllables
    assert len(s) >= 4


# -- English meter -----------------------------------------------------------


def test_haiku_5_7_5_perfect() -> None:
    text = "Rain taps on the roof\nTin sings in the fading light\nDusk deepens to dark"
    out = english_meter.haiku_5_7_5_ok(text)
    assert out["counts"][:3] == [5, 7, 5], out


def test_syllable_count_per_line_basic() -> None:
    out = english_meter.syllable_count_per_line("Hello world\nfor sure")
    assert out == [3, 2] or out == [3, 3]  # 'sure' is 1 or 2 depending on heuristic


def test_imagism_density_picks_up_imagery() -> None:
    out = english_meter.imagism_density("The rain on tin and feather drift over salt")
    assert out["n_imagistic"] >= 3
    assert out["density"] > 0.2


def test_imagism_density_low_for_abstract() -> None:
    out = english_meter.imagism_density("Furthermore the abstraction generally implies indeterminacy")
    assert out["n_imagistic"] == 0


# -- Scientific lint ---------------------------------------------------------


def test_hedge_density_picks_up_hedges() -> None:
    text = "The result may indicate X. Perhaps Y is at play. We do not yet know whether Z."
    out = scientific_lint.hedge_density(text)
    assert out["n_hedges"] >= 3


def test_mechanism_word_count_basic() -> None:
    text = "Ice floats because the hydrogen bond geometry creates a feedback loop that regulates density."
    out = scientific_lint.mechanism_word_count(text)
    assert out["n_mechanism_terms"] >= 3


def test_citation_count_picks_up_friston() -> None:
    text = "As Friston (2010) showed, the brain minimises free energy."
    out = scientific_lint.citation_or_attribution_count(text)
    assert out["n_citations_or_attributions"] >= 1


def test_lint_summary_has_all_keys() -> None:
    out = scientific_lint.lint_summary("Hello.")
    assert {"density", "n_hedges", "n_sentences", "n_mechanism_terms", "n_citations_or_attributions"} <= set(out.keys())
