# PCE v0.4.1 Post-Amend Adversarial Review

Date: 2026-04-30  
Reviewed branch: `main`  
Reviewed HEAD: `7985871590378ede21c4bb0c6140c423736c6563`  
Reviewed tag: `v0.4.0` -> `7985871590378ede21c4bb0c6140c423736c6563`  
Review mode: adversarial, post-amend release audit  
Prior review: `docs/reviews/2026-04-30-adversarial-v0.4-review.md`  
Amend plan: `v0.4.0_adversarial_patch_+_paper_ieee_rewrite_91b1dcae.plan.md`

## Executive Verdict

The v0.4.1 amend fixed many of the concrete v0.4 release defects: the tag now points at the amended commit, the live site builds with a frozen lockfile, base-path navigation is fixed, the main paper source has the University-of-York author block and reproducibility manifest, the judge rows have unique `formatted_prompt_sha256` values, and the Phase-8 report records 21/21 PASS.

The build is still not clean under an adversarial release bar. The strongest remaining issues are contradictions between the amended artefacts and the public archival story: `paper/v0.4/*.tex` is stale while release notes advertise it as the frozen archive; release notes and the site still describe Sanskrit outputs as curated/validated even though the current artefacts are live cascade traces with validators failing by large margins; existing showcase tests fail while Phase 8 remains green; and several statistics/token/provenance labels overstate what the data actually records.

Short version: the amend is materially better than the original v0.4.0 release, but it needs a v0.4.2 hardening pass before the tag/release can be treated as a clean archival record.

## Scope And Evidence

Primary artefacts inspected:

- `audit/v0.4/phase8_gate_report_v0_4_1.json`
- `benchmarks/results_v0.4/*.json`
- `benchmarks/results_v0.4/judge.jsonl`
- `benchmarks/showcase_v0.4/*`
- `docs/RELEASE_NOTES_v0.4.md`
- `docs/RUN_LOCAL.md`
- `docs/site/**`
- `paper/main.tex`
- `paper/sections/*.tex`
- `paper/v0.4/**/*.tex`
- `paper/references.bib`
- `scripts/phase8_gate_stack.py`
- `scripts/showcase_specs.toml`
- `tests/test_*`

Verification probes run:

```bash
git status --short
git rev-parse HEAD
git rev-parse v0.4.0
gh release view v0.4.0 --json tagName,targetCommitish,publishedAt,url
git ls-remote --tags origin v0.4.0
git ls-remote --heads origin v0.4.0-backup
.venv/bin/python -m pytest tests/test_cli_doc_examples.py tests/test_judge_audit_v04.py tests/test_pce_config_precedence.py tests/test_plugin_manifests.py -q
.venv/bin/python -m pytest tests/test_v0_4_showcase.py -q
pnpm --dir docs/site install --frozen-lockfile
pnpm --dir docs/site build
cd paper && tectonic main.tex
curl -sI https://sharathsphd.github.io/pratyabhijna/presentation/
curl -sI https://sharathsphd.github.io/pratyabhijna/discussion/
curl -sI https://sharathsphd.github.io/pratyabhijna/paper/main.pdf
```

## P0 Findings

### 1. Frozen `paper/v0.4` TeX Sources Contradict The Amended Paper And Release Notes

Severity: P0  
Primary paths:

- `paper/v0.4/main.tex`
- `paper/v0.4/sections/10c_showcase_examples.tex`
- `docs/RELEASE_NOTES_v0.4.md`
- `paper/main.tex`
- `paper/sections/10c_showcase_examples.tex`

Evidence:

`docs/RELEASE_NOTES_v0.4.md` advertises the frozen archive as `paper/v0.4/main.pdf` plus `paper/v0.4/sections/*.tex`. But the frozen TeX tree is not the amended paper source. `paper/v0.4/main.tex` still uses `authblk`, has `Sharath Sathish` as `Independent Researcher`, has no University-of-York author block, no highlighted reproducibility manifest box, no plain-language summary, and still uses the old bold enumerated abstract structure. In contrast, `paper/main.tex` has the amended title block and manifest.

The frozen `paper/v0.4/sections/10c_showcase_examples.tex` also still says the Sanskrit items are "maintainer-curated reference compositions" and "validated" by `tools/sanskrit_chandas.py`, while the live `paper/sections/10c_showcase_examples.tex` says the three Sanskrit items are live cascade outputs regenerated for v0.4.1 with validator status informational only.

Why it matters:

The release notes tell readers that `paper/v0.4/sections/*.tex` is part of the frozen archive. A reader auditing the archival source will see the pre-amend paper, not the paper described by the release. This is an archival-integrity problem, not a cosmetic mismatch.

Required fix:

- Either copy/sync the amended TeX sources, appendices, and bibliography into `paper/v0.4/` whenever `paper/v0.4/main.pdf` is refreshed, or stop advertising `paper/v0.4/sections/*.tex` as part of the canonical frozen archive.
- Add a check that `paper/v0.4/main.pdf` and `paper/v0.4/*.tex` are generated from the same source revision, or explicitly document that only `paper/v0.4/main.pdf` is canonical.

Confidence: high.

## P1 Findings

### 2. Existing Showcase Tests Fail While Phase 8 Reports 21/21 PASS

Severity: P1  
Primary paths:

- `tests/test_v0_4_showcase.py`
- `benchmarks/showcase_v0.4/sanskrit_anustubh/validator.json`
- `benchmarks/showcase_v0.4/sanskrit_anustubh/trace.json`
- `scripts/phase8_gate_stack.py`
- `audit/v0.4/phase8_gate_report_v0_4_1.json`

Evidence:

Running the existing showcase test module fails:

```text
FAILED tests/test_v0_4_showcase.py::test_sanskrit_chandas_count_within_tolerance
AssertionError: sanskrit_anustubh chandas count off by >2
syllable_count: 185, expected_count: 32

FAILED tests/test_v0_4_showcase.py::test_curated_reference_traces_present
AssertionError: 'live_cascade_v0_4_1' != 'curated_reference'
```

At the same time, `audit/v0.4/phase8_gate_report_v0_4_1.json` reports 21/21 PASS. Phase 8 does not run `tests/test_v0_4_showcase.py`, and `gate_chandas_validators_pass()` is deliberately reporting-only.

Why it matters:

The test suite and release gate now encode different truths. Phase 8 says the build is release-clean while an existing domain-specific showcase test says the shipped Sanskrit artefacts violate the old semantics. If the semantics intentionally changed, the tests must change and be wired into the gate; if they did not, the artefacts are wrong.

Required fix:

- Update `tests/test_v0_4_showcase.py` for live-cascade Sanskrit semantics, or regenerate Sanskrit outputs/validators to satisfy the old test.
- Wire the showcase test module, or its replacement, into Phase 8 or CI.
- Ensure Phase 8 cannot report a clean release while committed tests for the same artefacts fail.

Confidence: high.

### 3. The Plan-Required Showcase Semantics Test Is Missing

Severity: P1  
Primary paths:

- `tests/test_showcase_file_semantics.py`
- `benchmarks/showcase_v0.4/*/revised.txt`
- `benchmarks/showcase_v0.4/*/shadow_revision.txt`
- `benchmarks/showcase_v0.4/*/committed.txt`

Evidence:

The amend plan required a new `tests/test_showcase_file_semantics.py` asserting that `revised.txt` equals the actual `shadow_revision` surface and `committed.txt` equals the committed surface. That file does not exist. Running:

```bash
.venv/bin/python -m pytest tests/test_showcase_file_semantics.py -q
```

fails immediately with:

```text
ERROR: file or directory not found: tests/test_showcase_file_semantics.py
```

An ad hoc probe over the current showcase directories found the invariant currently holds, but the promised regression test is absent.

Why it matters:

`revised.txt` semantics were one of the explicit review fixes. The current artefacts happen to be consistent, but the release has no committed test protecting that behavior.

Required fix:

- Add `tests/test_showcase_file_semantics.py` or merge equivalent assertions into `tests/test_v0_4_showcase.py`.
- Run it in Phase 8 and CI.

Confidence: high.

### 4. Release Notes Still Describe Pre-Amend Sanskrit, CLI, And Config Semantics

Severity: P1  
Primary paths:

- `docs/RELEASE_NOTES_v0.4.md`
- `src/pce/config.py`
- `README.md`
- `paper/sections/10c_showcase_examples.tex`
- `benchmarks/showcase_v0.4/*/trace.json`

Evidence:

`docs/RELEASE_NOTES_v0.4.md` still says:

- `pce` has a `showcase generate` subcommand.
- Config precedence is `defaults -> ~/.config/pce/config.toml -> repo pce.toml -> env vars -> CLI flags`.
- Sanskrit demos are "curated reference verses validated by tools/sanskrit_chandas.py" and v0.5 will swap in cascade-generated outputs.

The current amended tree says otherwise:

- README and `src/pce/cli.py` use `pce showcase --regenerate SLUG`.
- `src/pce/config.py` documents `defaults -> repo TOML -> user TOML -> env -> overrides`.
- Current Sanskrit traces use `source = "live_cascade_v0_4_1"`.

Why it matters:

The GitHub release was re-published from this release notes file. Release consumers will read stale pre-amend instructions and the old Sanskrit authenticity story even though the tag points at amended artefacts.

Required fix:

- Add an explicit "v0.4.1 amend" section to `docs/RELEASE_NOTES_v0.4.md`.
- Replace `showcase generate` with `showcase --regenerate SLUG`.
- Correct config precedence to repo TOML before user TOML.
- Replace "curated reference verses validated" with live-cascade v0.4.1 outputs and informational validator status.

Confidence: high.

### 5. Live Paper Still Contains A Stale Sanskrit Claim In The Discussion

Severity: P1  
Primary paths:

- `paper/sections/10_discussion.tex`
- `paper/sections/10c_showcase_examples.tex`
- `benchmarks/showcase_v0.4/sanskrit_anustubh/trace.json`

Evidence:

The live amended `paper/sections/10c_showcase_examples.tex` correctly says the Sanskrit items are live cascade outputs with informational validator reports. But `paper/sections/10_discussion.tex` still says:

```text
the Sanskrit demos are maintainer-curated reference compositions, not live cascade outputs,
because the v0.4 cascade scorer is not yet chandas-aware
```

That directly contradicts the current trace source (`live_cascade_v0_4_1`) and the showcase section.

Why it matters:

The paper speaks in two voices about one of the central authenticity fixes. This undermines the amend's core claim that the Sanskrit showcase is now live-rerun and honestly reported.

Required fix:

- Rewrite the §10 discussion paragraph to match §10c: Sanskrit items are live cascade outputs, the validator is informational, and chandas-aware scoring remains v0.5 work.

Confidence: high.

### 6. H8a Release Notes Table Mislabels The Confidence Interval

Severity: P1  
Primary paths:

- `docs/RELEASE_NOTES_v0.4.md`
- `benchmarks/results_v0.4/stats.json`
- `paper/sections/09_results.tex`
- `docs/site/src/components/HypothesisCard.astro`

Evidence:

`docs/RELEASE_NOTES_v0.4.md` reports:

```text
H8a.v4 ... g = 0.649, BCa 95 % CI [0.031, 0.095]
```

Those bounds are the BCa CI on the paired mean delta (`estimate ≈ 0.058`) in `stats.json`, not a CI on Hedges' `g = 0.649`. The paper body is more careful in places, but the release note table and site card layout pair `g` and `ci` in a way that reads as "CI for g".

Why it matters:

This is a statistical-labeling bug on the strongest positive claim in the release. A CI with the wrong estimand is worse than omitting the CI.

Required fix:

- Relabel the interval everywhere as "BCa CI on paired mean Δ".
- If the UI component is generic, split effect size (`g`) from paired-delta CI or create a dedicated H8a layout.
- Add a small check that release-note/site labels match the source estimand from `stats.json`.

Confidence: high.

## P2 Findings

### 7. Chandas Gate Name And Report Semantics Are Misleading

Severity: P2  
Primary paths:

- `scripts/phase8_gate_stack.py`
- `audit/v0.4/phase8_gate_report_v0_4_1.json`
- `benchmarks/showcase_v0.4/sanskrit_* /validator.json`

Evidence:

All three current Sanskrit validators have `ok: false`:

```text
sanskrit_anustubh: expected 32 syllables, found 185
sanskrit_gayatri: expected 24 syllables, found 214
sanskrit_indravajra: expected 44 syllables, found 348
```

`gate_chandas_validators_pass()` nevertheless returns:

```json
{
  "name": "verify_chandas_validators_pass",
  "passed": true,
  "details": "sanskrit_indravajra:chandas_review[...] ..."
}
```

The plan explicitly demoted chandas to reporting-only, so the non-blocking behavior is intentional. The problem is naming: a gate called `verify_chandas_validators_pass` passes even when zero chandas validators pass.

Why it matters:

Automated consumers and future maintainers will read "validators_pass" as conformance, not "validator files exist and were recorded."

Required fix:

- Rename the gate to `verify_chandas_validators_recorded` or `verify_chandas_validator_reports_present`.
- Include structured fields like `release_blocking: false` and `validator_ok_count: 0`.

Confidence: high.

### 8. Sanskrit "Release-Quality Showcase" Artefacts Are Markdown/Commentary, Not Clean Verse Surfaces

Severity: P2  
Primary paths:

- `benchmarks/showcase_v0.4/sanskrit_anustubh/committed.txt`
- `benchmarks/showcase_v0.4/sanskrit_gayatri/committed.txt`
- `benchmarks/showcase_v0.4/sanskrit_indravajra/committed.txt`
- `paper/sections/10c_showcase_examples.tex`
- `docs/site/src/pages/showcase/index.astro`

Evidence:

`sanskrit_anustubh/committed.txt` begins:

```text
# Vimarśa-Anuṣṭubh

Using the **mirror** as the universal frame:

**Ātma-vimarśa-samvidaḥ**
Cid-ārṇava-samāśritaḥ
...
## Syllable count:
8 + 8 + 8 + 8 = **32 syllables** ✓
```

The validator is run against the whole markdown/commentary block, which explains the 185-syllable count. Meanwhile `paper/sections/10c_showcase_examples.tex` calls the release "nine release-quality showcase outputs," and the site showcase index still says Sanskrit demos use curated reference verses validated by the validator.

Why it matters:

The artefact is an honest live model output, but it is not a clean Sanskrit verse surface. Calling it release-quality Sanskrit chandas without separating "model answer" from "verse block" overstates what shipped.

Required fix:

- Either extract and validate only the verse block, or relabel Sanskrit showcase entries as "raw live cascade traces, not chandas-conformant verse surfaces."
- Surface the validator failure and caveat prominently on each Sanskrit page.
- Remove "release-quality" wording for Sanskrit until the surface is clean.

Confidence: high.

### 9. Phase 8 Audits Existing Artefacts But Does Not Rebuild The Paper Or Site

Severity: P2  
Primary paths:

- `scripts/phase8_gate_stack.py`
- `.github/workflows/pages.yml`

Evidence:

`gate_paper_builds()` checks whether `paper/main.pdf` and `paper/v0.4/main.pdf` exist and are larger than 100 KB. It does not run `tectonic`. `gate_site_builds()` inspects `docs/site/dist`; it does not run `pnpm build`. GitHub Pages builds the site, but CI does not run Python tests or `scripts/phase8_gate_stack.py`.

Why it matters:

The 21/21 PASS report is a local artefact audit, not a reproducible build gate. A stale `dist/` or PDF can satisfy Phase 8 even if the source no longer builds.

Required fix:

- Rename Phase 8 report language to "artefact audit" or make the gate run `tectonic`, `pnpm build`, and focused pytest modules.
- Add a GitHub Actions job for Phase 8 and pytest, separate from Pages deploy.

Confidence: high.

### 10. Judge Recovery Is Missing Row-Level Provenance And Has Implausible Token Counts

Severity: P2  
Primary paths:

- `benchmarks/results_v0.4/judge.jsonl`
- `benchmarks/results_v0.4/judge_agreement.json`
- `scripts/recover_judge_formatted_sha.py`
- `scripts/judge_subset.py`

Evidence:

`judge.jsonl` has 23 rows, and all 23 have unique `formatted_prompt_sha256`. But:

```text
input_tokens_unique = [9]
has_recovery_provenance = 0
```

The plan required per-row `recovery_provenance: "post_hoc_v0_4_1"`. The aggregate `judge_agreement.json` records recovery summary metadata, but each judge row does not say whether its formatted hash was original or reconstructed.

Why it matters:

The judge bridge is described as replay-auditable. The hash helps, but the row-level audit still cannot distinguish original metadata from post-hoc reconstruction, and `input_tokens = 9` is not a credible estimate for long paired prompts.

Required fix:

- Add `recovery_provenance` per row, or explicitly document that provenance is only aggregate-level.
- Recompute `input_tokens` from formatted prompt length or from provider usage.
- Extend `tests/test_judge_audit_v04.py` to assert row-level provenance and a sane token floor.

Confidence: high.

### 11. Config Precedence Is Still Wrong On The Plugin Page

Severity: P2  
Primary paths:

- `docs/site/src/pages/plugin.astro`
- `src/pce/config.py`
- `docs/RUN_LOCAL.md`

Evidence:

`docs/site/src/pages/plugin.astro` says:

```text
~/.config/pce/config.toml > repo pce.toml > environment variables > CLI flags
```

`src/pce/config.py` and `docs/RUN_LOCAL.md` say the actual order is:

```text
defaults -> repo pce.toml -> user TOML -> env -> CLI overrides
```

Why it matters:

Configuration precedence is part of the portability contract. The plugin page currently tells users the user config has lower/greater precedence than repo config ambiguously and in the wrong order relative to the implementation.

Required fix:

- Update plugin page wording to "defaults < repo TOML < user TOML < env < CLI flags" or "later layers override earlier ones."
- Add the plugin page to the config precedence test or documentation smoke test.

Confidence: high.

### 12. Paper Build Succeeds But Emits Serious Layout Warnings

Severity: P2  
Primary paths:

- `paper/main.tex`
- `paper/appendices/G_audit_trail.tex`
- `paper/main.pdf`

Evidence:

`tectonic paper/main.tex` exits 0, but the build emits:

```text
warning: Annotation out of page boundary.
warning: Maybe incorrect paper size specified.
```

It also emits many large overfull boxes, including several above 100pt and one above 200pt in bibliography/audit-trail areas.

Why it matters:

The plan target is journal-quality IEEE-style presentation. A clickable annotation outside the page boundary and severe overfull hboxes are PDF-quality defects even if TeX exits 0.

Required fix:

- Use breakable URL/path commands (`\url`, `\path`, or `\nolinkurl`) in the author block and reproducibility manifest.
- Shorten long monospaced paths in the audit appendix or add manual breakpoints.
- Consider failing the paper build if "Annotation out of page boundary" appears.

Confidence: medium-high.

### 13. H5 Is Called "BCa" In Some Paper Text Even Though It Uses A Wald CI

Severity: P2  
Primary paths:

- `paper/main.tex`
- `paper/sections/07_methods.tex`
- `benchmarks/results_v0.4/stats.json`

Evidence:

`paper/main.tex` describes the H5 fixed-effects pool as:

```text
g=+0.14 (BCa 95% CI [-0.26,+0.54])
```

`paper/sections/07_methods.tex` also says H5 is reported as `g, BCa 95% CI, and per-domain weights`, but the statistical protocol later says H5 uses a fixed-effects inverse-variance pool with a 95% Wald CI. `stats.json` records `method = fixed_effects_inverse_variance` and `ci_95`, not a BCa bootstrap interval for the H5 pool.

Why it matters:

This is another estimand/method-label mismatch in the headline statistical story.

Required fix:

- Strip "BCa" from H5 wording wherever the interval is the fixed-effects Wald interval.
- Reserve "BCa" for paired mean delta intervals where the bootstrap was actually used.

Confidence: high.

### 14. Site H8b Card Shows A Meaningless Zero-Width CI

Severity: P2  
Primary paths:

- `docs/site/src/pages/results.astro`
- `docs/site/src/pages/hypotheses.astro`
- `docs/site/src/components/HypothesisCard.astro`

Evidence:

The H8b card passes `ci={[0, 0]}` while representing an F1 contrast:

```astro
g={h8b.learned_gate.f1 - h8b.event_gated.f1}
ci={[0, 0]}
p={null}
```

The built site renders a confidence interval of `[0.00, 0.00]`, which looks exact rather than "not applicable."

Why it matters:

This creates false precision for a classifier metric that is not a Hedges' g contrast and has no reported CI.

Required fix:

- Add a non-effect-size card variant for classifier metrics.
- Omit CI/p fields for H8b unless an actual bootstrap interval is computed.

Confidence: high.

## P3 Findings

### 15. `scripts/showcase_specs.toml` Header Still Describes The Old Curated Sanskrit Mode

Severity: P3  
Primary path:

- `scripts/showcase_specs.toml`

Evidence:

The file header says Sanskrit entries are "curated reference compositions, validated by tools/sanskrit_chandas.py" and that Sanskrit entries "embed the curated reference verse directly." The individual entries now say v0.4.1 ships live cascade output and validator status is informational.

Why it matters:

The executable spec tells two stories. Future maintainers may run the wrong mode or misunderstand why `source = "curated_reference"` remains in the TOML while the current artefacts are live-rerun.

Required fix:

- Rewrite the header to say `source = "curated_reference"` is the curate-mode fallback, while v0.4.1 release artefacts were generated with `--mode live`.

Confidence: high.

### 16. Showcase Index Still Hardcodes Old Sanskrit Copy

Severity: P3  
Primary path:

- `docs/site/src/pages/showcase/index.astro`

Evidence:

The showcase index says:

```text
Sanskrit demos use curated reference verses validated by tools.sanskrit_chandas
```

That contradicts the current live trace source and the amended §10c paper text.

Why it matters:

The page-level copy is what most readers will see before opening individual traces. It revives the old authenticity story after the amend changed it.

Required fix:

- Replace with "Sanskrit demos are live v0.4.1 cascade outputs; the chandas validator result is informational because v0.4 has no chandas-aware scorer."

Confidence: high.

### 17. Internal Link Checker Does Not Validate Relative Links

Severity: P3  
Primary paths:

- `docs/site/scripts/check_internal_links.mjs`
- `docs/site/src/pages/plugin.astro`

Evidence:

The link checker validates root-absolute links but ignores relative links. `docs/site/src/pages/plugin.astro` contains `href="../showcase"`, which likely resolves correctly now but would not be caught by the checker if broken.

Why it matters:

The original Pages bug came from links that were correct locally but wrong under the GitHub Pages base. Relative links are still outside the crawler's safety net.

Required fix:

- Resolve relative `href`s against each HTML file path in `check_internal_links.mjs`, or require site pages to use `withBase('/...')` for internal routes.

Confidence: medium.

### 18. Untracked Old Paper Build Artefacts Remain On Disk

Severity: P3  
Primary paths:

- `paper/v0.2/main.log`
- `paper/v0.2/main.blg`
- `paper/v0.3/main.log`
- `paper/v0.3/main.blg`

Evidence:

`git ls-files` shows old v0.1-v0.3 archive paths are gone, but ignored build artefacts remain on disk under `paper/v0.2/` and `paper/v0.3/`.

Why it matters:

This does not affect the committed release, but it makes the working tree look less clean than the cleanup story implies and can confuse local audits that scan the filesystem rather than git.

Required fix:

- Delete ignored `paper/v0.2/` and `paper/v0.3/` build artefacts locally, or document that cleanup means "not tracked by git."

Confidence: medium.

## Positive Findings

- `HEAD`, local `v0.4.0`, and remote `refs/tags/v0.4.0` all resolve to `7985871590378ede21c4bb0c6140c423736c6563`.
- `v0.4.0-backup` exists locally and was not pushed to `origin`.
- `pnpm --dir docs/site install --frozen-lockfile` succeeds.
- `pnpm --dir docs/site build` succeeds and the link crawler reports: `OK: scanned 23 HTML files; all internal links resolve.`
- Live Pages probes return 200 for `/pratyabhijna/presentation/`, `/pratyabhijna/discussion/`, and `/pratyabhijna/paper/main.pdf`.
- `/pratyabhijna/presentation/` now meta-refreshes to `/pratyabhijna/`, not the GitHub user root.
- The available focused tests pass: `tests/test_cli_doc_examples.py`, `tests/test_judge_audit_v04.py`, `tests/test_pce_config_precedence.py`, and `tests/test_plugin_manifests.py` report 47 passed / 1 skipped.
- The current showcase file semantics invariant holds: `revised.txt` equals the trace shadow revision and `committed.txt` equals the trace committed surface for all nine showcase slugs.
- `paper/references.bib` now correctly reports 48 total entries, 46 verified, and 2 `not_verified_no_handle` entries, matching `audit/v0.4/lit_verification_summary.json`.
- `docs/RUN_LOCAL.md` now documents the correct config precedence order.

## Recommended Fix Order

1. Sync or de-scope `paper/v0.4/*.tex` so the frozen archive stops contradicting the amended paper.
2. Update `docs/RELEASE_NOTES_v0.4.md`, `paper/sections/10_discussion.tex`, `docs/site/src/pages/showcase/index.astro`, and `scripts/showcase_specs.toml` to one consistent Sanskrit story: live v0.4.1 cascade traces, validator informational, not curated validated verses.
3. Update or replace `tests/test_v0_4_showcase.py`, add the missing showcase semantics test, and wire showcase tests into Phase 8/CI.
4. Fix H8a and H5 interval labels so every CI says which estimand it belongs to.
5. Add row-level judge recovery provenance and realistic token estimates; extend judge tests beyond hash presence.
6. Rename the chandas gate to reflect "reports recorded" rather than "validators pass."
7. Add CI coverage for Phase 8 and pytest, not only Pages deploy.
8. Clean up PDF layout warnings, especially the out-of-page annotation.

## Final Assessment

The v0.4.1 amend substantially improves the release and closes many visible v0.4 defects. It is not a fake release, and the core benchmark artefacts are mostly traceable. But the public archive still has dual truths: live paper source vs stale `paper/v0.4` source, live Sanskrit traces vs curated/validated release prose, green Phase-8 report vs failing existing showcase tests, and statistics labels that collapse different estimands into one UI/release-note field.

The release should be treated as patched but not fully hardened. A v0.4.2 pass should focus less on new content and more on making every public artefact, test, gate, and frozen archive tell the same story.
