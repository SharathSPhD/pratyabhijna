# `docs/figures/PROMPTS.md` — image generation prompts for v0.4.2

This document is the canonical prompt registry for the v0.4.2 visual layer. It
is shipped with the v0.4.2 release so that any maintainer can regenerate the
hero image, replace placeholder boxes on the site, or produce a stylised
photo-style alternate of any flowchart without re-deriving the design intent
from scratch.

## How to use this file

1. Pick a figure section below.
2. Pick a generator. Recommendations per figure are noted in each section.
3. Use the prompt verbatim; tighten dimensions / palette knobs to match the
  target slot.
4. Save the resulting PNG to the documented path under
  `docs/site/public/figures/v0.4/...`. The `PlaceholderFigure` component on
   the site picks the file up automatically — once the file exists at the
   `src` path the placeholder box is replaced with the real image; until then
   the styled placeholder renders.

### Recommended generators


| Use case                                      | Recommended                     | Rationale                                               |
| --------------------------------------------- | ------------------------------- | ------------------------------------------------------- |
| Devanāgarī fidelity (hero)                    | Nano Banana / Gemini ImageGen 3 | Strong CJK/Devanāgarī typography; honours kerning.      |
| Stylised photo-style alternates of flowcharts | DALL-E 3 / Midjourney v6        | Strong on abstract geometric and gradient compositions. |
| Diagrammatic / schematic alternates           | Recraft / Ideogram              | Stronger on text-in-image and clean line work.          |


### Canonicality


| Figure                              | Canonical source                                            | Notes                                                                  |
| ----------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| F1 — 5-śakti cascade                | TikZ in `paper/sections/03_pratyabhijna_background.tex`     | Site uses the PNG export.                                              |
| F2 — Active-inference loop          | TikZ in `paper/sections/04_active_inference_background.tex` | Site uses the PNG export.                                              |
| F3 — Commit-policy multiplexer      | TikZ in `paper/sections/07_methods.tex`                     | Site uses the PNG export.                                              |
| F4 — Phase 7 pilot pipeline         | TikZ in `paper/sections/07_methods.tex`                     | Site uses the PNG export.                                              |
| F5 — Hypothesis dependency tree     | TikZ in `paper/sections/07_methods.tex`                     | Site uses the PNG export.                                              |
| C1 — Per-axis effect-size bar chart | matplotlib in `benchmarks/figures.py`                       | Already on disk at `paper/figures/v0.4/fig_v04_axes_breakdown.png`.    |
| C2 — Power vs realised effect       | matplotlib in `benchmarks/figures.py`                       | Already on disk at `paper/figures/v0.4/fig_v04_power_vs_realised.png`. |
| Hero                                | AI-generated (this file)                                    | No paper analogue — site-only.                                         |


The TikZ flowcharts are the **canonical** source for the paper. The site PNGs
are *exports* of those TikZ figures. Run `pdftocairo -png -r 200 paper/main.pdf` (or use `dvisvgm` for a vector path) to refresh the exports
into `docs/site/public/figures/v0.4/flowcharts/` after a paper rebuild.

The matplotlib charts (C1, C2) are also canonical — `benchmarks/figures.py`
emits them to `paper/figures/v0.4/`, and `scripts/prepare_site_data.py` copies
them into `docs/site/public/figures/v0.4/` for the site.

The hero image has **no paper analogue** — it exists only on the site, and
the prompt below is its canonical source.

## Hero image

**Target path:** `docs/site/public/figures/v0.4/hero.png`

**Dimensions:** 1600 × 680 px (21:9 aspect ratio).

**Recommended generator:** Nano Banana / Gemini ImageGen 3 for Devanāgarī
typography fidelity. DALL-E 3 if Devanāgarī is rendered as a stylised
flat-art element rather than legible script.

**Source verse (recommended):** Utpaladeva, *Īśvarapratyabhijñākārikā* I.1.1
(the opening invocation of the Recognition school):

> कथञ्चिदासाद्य महेश्वरस्य दास्यं जनस्याप्युपकारमिच्छन् ।
> समस्तसम्पत्समवाप्तिहेतुं तत्प्रत्यभिज्ञामुपपादयामि ॥

> *kathaṃcid āsādya maheśvarasya dāsyaṃ janasyāpyupakāramicchan |*
> *samastasampatsamavāptihetuṃ tatpratyabhijñām upapādayāmi ||*

Translation (approx.): "Having somehow attained service to the Great Lord and
desiring the welfare of all, I expound that recognition of him which is the
cause of the attainment of all prosperity."

This verse is the **root** invocation of the entire Pratyabhijñā tradition; it
is the most thematically apt opening for a Pratyabhijñā Creative Engine
project page. The verse is in the public domain (10th–11th c.).

### Alternate verses (if the recommended pick does not render cleanly)

- **Alternate A** — Abhinavagupta, *Tantrāloka* I.1 opening invocation
(`vimalakalāśrayābhinavasṛṣṭimahā jananī...`). Same lineage, denser
Sanskrit. Use this when the hero benefits from a longer verse for visual
weight.
- **Alternate B** — the project's existing `sanskrit_gayatri.curated_text`
(the Gāyatrī verse). More universally recognised across South Asian
audiences. Use this when accessibility to non-Śaiva readers matters more
than school-fidelity.

### Composition

- **Background:** an abstract recognition / cascade motif. Suggested visual
language: a warm-to-cool radial gradient (saffron / amber bleeding into
deep indigo / midnight blue), with a faint geometric overlay suggesting a
cascade of nested arcs (the five-śakti cascade as a radial composition —
cit at the centre, expanding outward through ānanda, icchā, jñāna, kriyā,
with vimarśa as the closing inward arc).
- **Typography:** Devanāgarī rendered cleanly with proper kerning (śloka
visarga, anusvāra, halant joins must all render correctly). Choose a
modern Devanāgarī typeface with strong horizontal śiraḥrekhā (header line)
— e.g., Sakal Bharati, Mukti, or a high-quality variant of Mangal. The
verse should be the visual centrepiece, occupying roughly the central
third of the image.
- **Negative space:** keep the right third clear of typography (the site
navigation may overlay there on small screens).
- **Palette:** the project's existing dark-mode-aware Tailwind palette
(warm accent: amber-500 / saffron; cool background: slate-900 / indigo-950;
neutral foreground: zinc-100). Hero should read in both light and dark
mode without re-rendering.

### Prompt — Nano Banana / Gemini ImageGen 3

> Render a 1600×680 px (21:9) hero banner image for an academic project page
> on Pratyabhijñā (Kashmir Śaivism, recognition philosophy) and AI / language
> models. Composition: a warm-to-cool radial gradient background (saffron and
> amber at the centre, fading outward to deep indigo and midnight blue), with
> a faint geometric overlay of five concentric nested arcs suggesting a
> cascade. Centred in the middle third, render the following Sanskrit śloka
> in clean, properly-kerned Devanāgarī typography with a strong horizontal
> śiraḥrekhā: "कथञ्चिदासाद्य महेश्वरस्य दास्यं जनस्याप्युपकारमिच्छन् ।
> समस्तसम्पत्समवाप्तिहेतुं तत्प्रत्यभिज्ञामुपपादयामि ॥". The verse must be
> legible at full resolution; visarga, anusvāra, and halant joins must render
> correctly. Keep the right third of the image clear of typography for
> navigation overlay. Mood: contemplative, scholarly, classical-Indian aesthetic
> in conversation with abstract scientific composition. No human figures, no
> photographic elements. The cascade arcs should evoke recognition (a wave of
> awareness folding back on itself) rather than waterfall or motion.

### Prompt — DALL-E 3 (stylised flat-art alternate)

> A 1600×680 wide academic hero banner. Background: warm-to-cool radial
> gradient — saffron and amber at the centre fading to deep indigo at the
> edges. Foreground centre: an abstract Sanskrit-style devanagari calligraphy
> motif (stylised, not necessarily a literal verse) suggesting the
> Pratyabhijñā recognition tradition. Five faint concentric arcs around the
> calligraphy suggest a cascade. Right third left clear for overlay. Mood:
> classical Indian scholarship in dialogue with modern abstract composition.
> No human figures. No photographic realism.

## Figure prompts

The TikZ source for F1–F5 is the **canonical** definition. The site picks up
the PNG exports from `docs/site/public/figures/v0.4/flowcharts/F<N>_*.png`.
Below are the prompts to use if you want to generate a **stylised AI alternate**
of any flowchart for marketing / blog use — the alternates are not the site's
primary surface.

### F1 — 5-śakti cascade

- **Target path (TikZ export):** `docs/site/public/figures/v0.4/flowcharts/F1_panchashakti_cascade.png`
- **Dimensions:** 1280 × 720 px (16:9).
- **Canonical source:** TikZ in `paper/sections/03_pratyabhijna_background.tex`.

**Stylised alternate prompt (optional, AI-generated):**

> A clean, schematic flowchart in a warm classical-Indian palette
> (saffron / amber / indigo). Top-to-bottom flow: prompt input → a node
> labelled "cit (K candidates)" → "ānanda (novelty pulse)" → "icchā
> (best-of-K)" → "apohana (exclusion / BMR)" → "jñāna (selection + ΔF)" →
> "kriyā (render)" → "vimarśa (reflexive read)" → "commit-policy multiplexer"
> → "committed surface". A curved backedge from vimarśa to icchā labelled
> "BMR loopback (ADR-003 F-budget gate)". Subtle Indic motif (concentric
> arcs) in the background. Sans-serif technical typography for node labels;
> Devanāgarī for the operator names (cit = चित्, ānanda = आनन्द, etc.).

### F2 — Active inference / BMR loop

- **Target path (TikZ export):** `docs/site/public/figures/v0.4/flowcharts/F2_active_inference_loop.png`
- **Dimensions:** 1280 × 720 px (16:9).
- **Canonical source:** TikZ in `paper/sections/04_active_inference_background.tex`.

**Stylised alternate prompt (optional):**

> A schematic loop diagram in a warm classical-Indian palette. Nodes (left to
> right, then loop): "prior over surfaces q(z)" → "cascade-as-likelihood
> (composite scoring)" → "posterior" → "vimarśa-as-BMR pruning" → "ADR-003
> free-energy budget gate" → "commit decision". A backedge from "commit
> decision" to "prior" closes the loop. Annotate the BMR step with the symbol
> "F = E_q[log q(z) − log p(z, o)]". Background is the same warm-to-cool
> gradient as the hero. Clean technical typography.

### F3 — Commit-policy multiplexer

- **Target path (TikZ export):** `docs/site/public/figures/v0.4/flowcharts/F3_commit_policy_multiplexer.png`
- **Dimensions:** 1280 × 720 px (16:9).
- **Canonical source:** TikZ in `paper/sections/07_methods.tex`.

**Stylised alternate prompt (optional):**

> A multiplexer schematic. On the left, three input streams: "draft",
> "shadow_revision", "judge_pair". They all feed into a central
> multiplexer-style chooser that has five outputs (one per policy):
> "always_draft", "always_revise", "event_gated", "learned_gate", "oracle".
> Each policy outputs to a single "committed surface" node on the right. The
> "learned_gate" path is highlighted (warm saffron) to indicate the H8c best
> non-oracle policy. Clean schematic style; no decorative elements.

### F4 — Phase 7 pilot pipeline

- **Target path (TikZ export):** `docs/site/public/figures/v0.4/flowcharts/F4_phase7_pipeline.png`
- **Dimensions:** 1280 × 720 px (16:9).
- **Canonical source:** TikZ in `paper/sections/07_methods.tex`.

**Stylised alternate prompt (optional):**

> A linear pipeline schematic. Left to right: "prompt set" → "four-arm random
> split" (which fans out into four parallel rails labelled T_full, T_no_bmr,
> T_no_revision, T_no_apohana) → "cascade scorer (Haiku-4.5)" → "judge
> (Sonnet-4.5)" → "stats (paired permutation, BCa, Holm)" → "fixed-effects
> pool (ADR-005)". Below the pipeline, an annotation box: "ADR-006
> substrate-deviation event: managed Anthropic-API substrate". Cool palette
> for the pipeline rails, warm accent on the ADR-006 box.

### F5 — Hypothesis dependency tree

- **Target path (TikZ export):** `docs/site/public/figures/v0.4/flowcharts/F5_hypothesis_tree.png`
- **Dimensions:** 1280 × 720 px (16:9).
- **Canonical source:** TikZ in `paper/sections/07_methods.tex`.

**Stylised alternate prompt (optional):**

> A tree diagram, top-to-bottom. Root node: "Pre-registered hypotheses
> (v0.4)". First level (cascade-vs-bare): four sibling nodes "H1 (AUT)", "H2
> (Poetry-interp)", "H3 (Poetry-gen)", "H4 (Sci-creativity)". Second level
> (pooled): a single child "H5 (fixed-effects pool)" connecting up to all four
> H1-H4 nodes. Third level (mechanism): three sibling nodes "H8a (revision >
> draft)", "H8b (gate calibration F1)", "H8c (commit-policy leaderboard)".
> Fourth level (sensitivity): "H9 (judge-vs-proxy ρ)". Highlight H8a in warm
> saffron to mark it as the load-bearing positive finding. Clean typography;
> no decorative motifs.

### C1 — Per-axis effect-size bar chart

- **Target path (matplotlib export):** `docs/site/public/figures/v0.4/fig_v04_axes_breakdown.png`
- **Canonical source:** matplotlib in `benchmarks/figures.py` →
`_figure_v04_axes_breakdown`.
- **No AI alternate** — this is a quantitative chart; only the matplotlib
output is appropriate.

### C2 — Power vs realised effect

- **Target path (matplotlib export):** `docs/site/public/figures/v0.4/fig_v04_power_vs_realised.png`
- **Canonical source:** matplotlib in `benchmarks/figures.py` →
`_figure_v04_power_vs_realised`.
- **No AI alternate** — this is a quantitative chart; only the matplotlib
output is appropriate.

## Notes

- **Accessibility:** every PNG that lands at one of the documented paths must
be paired with a meaningful `alt` attribute on the consumer site. The
`PlaceholderFigure.astro` component requires a non-empty `alt` prop and
falls back to the placeholder when the PNG is missing.
- **Light/dark mode:** the site uses a single PNG for both modes. Hero and
flowchart images should be designed to read in both — avoid pure white or
pure black backgrounds; use the project's neutral palette
(slate-900 / zinc-100) as the contrast anchors.
- **Devanāgarī typography QA:** before committing the hero PNG, paste the
rendered verse into a Devanāgarī validator (e.g.,
[https://aksharamukha.appspot.com/converter](https://aksharamukha.appspot.com/converter)) to confirm the script renders
correctly. Common failure modes: missing visarga, broken halant joins,
wrong order of vowel marks.
- **Versioning:** if the hero verse is replaced, document the swap in the
next release notes and update the `Source verse (recommended)` block at
the top of this file.

