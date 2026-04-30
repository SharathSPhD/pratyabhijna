# PCE v0.4 Adversarial Review - Release Integrity, Paper, Pages, Benchmarks

Date: 2026-04-30  
Reviewed branch: `main`  
Reviewed HEAD at start of review: `e574898`  
Review mode: adversarial, evidence-first, post-release audit  
Live release: `v0.4.0`  
Live site checked: <https://sharathsphd.github.io/pratyabhijna/>

## Executive Verdict

PCE v0.4 is a substantial release: the Phase 7 English/science showcase traces are not obviously fabricated, the paper and site are wired to real v0.4 artefacts, and the standalone CLI has a real parser and JSON surfaces. But the revision is **not release-clean** under an adversarial bar.

The strongest failures are not philosophical. They are concrete integrity issues:

- one shipped Sanskrit showcase fails its own validator, while the Phase 8 gate still reports success;
- the live GitHub Pages site contains project-base link failures;
- public CLI examples in README, RUN_LOCAL, and the plugin page do not parse against the real CLI;
- the paper conflates the Haiku cascade cost ledger with the separate Sonnet judge ledger;
- the showcase claim needs sharper language: six items are Phase 7 cascade-derived, three Sanskrit items are curated references, and one curated Sanskrit item currently fails validation.

Short version: **the v0.4 mechanism-study claims are mostly traceable, but the release has enough audit, documentation, and live-site defects that it should be patched before being treated as a stable archival release.**

## Scope and Evidence Reviewed

Primary repo files inspected:

- `benchmarks/results_v0.4/*.json`
- `benchmarks/results_v0.4/judge.jsonl`
- `benchmarks/results_v0.4/judge_agreement.json`
- `benchmarks/results_v0.4/STATUS.md`
- `benchmarks/showcase_v0.4/*`
- `scripts/generate_v0_4_showcase.py`
- `scripts/showcase_specs.toml`
- `scripts/phase8_gate_stack.py`
- `tools/sanskrit_chandas.py`
- `tools/english_meter.py`
- `tools/scientific_lint.py`
- `src/pce/cli.py`
- `src/pce/config.py`
- `pyproject.toml`
- `README.md`
- `RUN_LOCAL.md`
- `docs/RELEASE_NOTES_v0.4.md`
- `docs/site/**`
- `.github/workflows/pages.yml`
- `paper/main.tex`
- `paper/sections/*.tex`
- `paper/references.bib`
- `audit/v0.4/*`
- `plugin/.cursor-plugin/plugin.json`
- `plugin/.claude-plugin/plugin.json`
- `plugin/.mcp.json`
- `plugin/mcp/server.py`
- `plugin/README.md`

Runtime probes performed:

```bash
python3 -m pce --help
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce config show
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce smoke --dry-run
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce showcase
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce smoke
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce cascade --prompt ... --K 4 --seed 4242 --dry-run
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce judge-pair --a draft.txt --b revised.txt --prompt ... --dry-run
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce showcase generate
curl -I https://sharathsphd.github.io/pratyabhijna/presentation
curl https://sharathsphd.github.io/pratyabhijna/presentation/
curl https://sharathsphd.github.io/pratyabhijna/results/
```

## P0 Findings

### 1. Phase 8 chandas gate passes a failing Sanskrit validator

Severity: P0  
Primary paths:

- `benchmarks/showcase_v0.4/sanskrit_indravajra/validator.json`
- `scripts/phase8_gate_stack.py`
- `scripts/showcase_specs.toml`

Evidence:

`benchmarks/showcase_v0.4/sanskrit_indravajra/validator.json` reports:

```json
{
  "chandas": "indravajra",
  "syllable_count": 45,
  "expected_count": 44,
  "count_ok": false,
  "ok": false,
  "notes": [
    "expected 44 syllables (4 padas x 11); found 45"
  ]
}
```

But `gate_chandas_validators_pass()` in `scripts/phase8_gate_stack.py` only fails missing JSON, bad JSON, or `ok == false` with no notes. A validator with `"ok": false` and a note is rendered as `review`, not as failure. This let the Phase 8 gate stack report `18/18 PASS` while shipping a Sanskrit showcase item that fails the project's own validator.

Why it matters:

The release explicitly claims the showcase includes validated Sanskrit chandas. A gate that records failure as "review" is not a gate. This is the single cleanest release-integrity bug in v0.4.

Required fix:

- Make `gate_chandas_validators_pass()` fail on any `validator.json` with `ok != true`.
- Replace or repair the `sanskrit_indravajra` curated text so the validator passes.
- Re-run `scripts/generate_v0_4_showcase.py`, `scripts/phase8_gate_stack.py`, and the site build.
- Update the release notes if the fixed output changes.

Confidence: high.

### 2. Live GitHub Pages has broken project-base navigation

Severity: P0  
Primary paths:

- `docs/site/astro.config.mjs`
- `docs/site/src/pages/results.astro`
- live `/pratyabhijna/presentation/`
- live `/pratyabhijna/results/`

Evidence:

`docs/site/astro.config.mjs` configures:

```js
redirects: {
  '/presentation': '/',
}
```

On GitHub project Pages, that redirect emits an HTML refresh to `/`, meaning `https://sharathsphd.github.io/`, not `https://sharathsphd.github.io/pratyabhijna/`.

Live probe:

```text
https://sharathsphd.github.io/pratyabhijna/presentation/
<meta http-equiv="refresh" content="0;url=/">
canonical: https://sharathsphd.github.io/
```

`docs/site/src/pages/results.astro` also contains:

```html
<a href="/discussion">discussion page</a>
```

Live probe:

```text
https://sharathsphd.github.io/discussion -> 404
https://sharathsphd.github.io/pratyabhijna/discussion -> 301/200 path
```

Why it matters:

The site is a core Phase 8 deliverable. Project Pages must treat `/pratyabhijna` as the deployment base. Bare root links make the public surface look broken even when the generated static files exist.

Required fix:

- Replace `/presentation -> /` with a base-aware target, or implement a small explicit redirect page that links/refreshes to `/pratyabhijna/`.
- Replace `href="/discussion"` with the existing `withBase('/discussion')` helper or an equivalent base-aware link.
- Add a build-time link crawl that fails on internal `href="/..."` links unless explicitly whitelisted.

Confidence: high.

## P1 Findings

### 3. Public CLI documentation does not match the real parser

Severity: P1  
Primary paths:

- `README.md`
- `RUN_LOCAL.md`
- `docs/site/src/pages/plugin.astro`
- `docs/site/src/pages/methods.astro`
- `src/pce/cli.py`

Evidence:

The docs show:

```bash
pce cascade --prompt "..." --K 4 --seed 4242
pce judge-pair --a draft.txt --b revised.txt --prompt "Which is more inventive?"
pce showcase generate
```

The actual parser in `src/pce/cli.py` supports:

```bash
pce cascade --prompt "..." --k 4 --seed 4242
pce judge-pair --domain poetry_gen --item-id p06 \
  --treatment-text revised.txt --control-text draft.txt
pce showcase --regenerate sanskrit_anustubh
```

Runtime probes against the documented examples:

```text
pce cascade ... --K 4
=> error: unrecognized arguments: --K 4

pce judge-pair --a draft.txt --b revised.txt --prompt ...
=> error: required: --domain, --item-id, --treatment-text, --control-text

pce showcase generate
=> error: unrecognized arguments: generate
```

Why it matters:

Phase 8's portability claim depends on the standalone CLI being usable by a human or agent following the shipped docs. The documented commands are currently false.

Required fix:

- Update all public examples to match `argparse`.
- Or add backward-compatible aliases (`--K`, `judge-pair --a/--b/--prompt`, and `showcase generate`) if the docs are the desired UX.
- Add tests that parse every README/RUN_LOCAL/plugin-page command snippet.

Confidence: high.

### 4. "Standalone CLI, not using venv" is underspecified and currently fails in common shells

Severity: P1  
Primary paths:

- `src/pce/cli.py`
- `pyproject.toml`
- `README.md`
- `RUN_LOCAL.md`

Evidence:

Non-venv probes:

```text
/usr/bin/python3 -m pce --help
=> No module named pce

/opt/homebrew/bin/python3.12 -m pce --help
=> No module named pce

PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce config show
=> OK

PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce smoke --dry-run
=> OK

PYTHONPATH=src /opt/homebrew/bin/python3.12 -m pce smoke
=> ModuleNotFoundError: No module named 'numpy'
```

Why it matters:

The CLI is IDE-independent, but it is not dependency-independent and not importable until installed or run with `PYTHONPATH=src`. The phrase "standalone" should mean "no Cursor/Claude Code host required," not "can run from an arbitrary system Python."

Required fix:

- State explicitly that `pip install -e .` or `uv pip install -e .` is required before `python -m pce`.
- Add a "no venv" install recipe, for example using a global/user install or `uv tool` if appropriate.
- Catch missing dependency import errors at the CLI boundary and print install guidance rather than raw tracebacks.

Confidence: high.

### 5. Paper cost ledger conflates Haiku cascade cost with Sonnet judge cost

Severity: P1  
Primary paths:

- `paper/main.tex`
- `paper/sections/09_results.tex`
- `paper/sections/10b_honest_ai_claims.tex`
- `README.md`
- `docs/RELEASE_NOTES_v0.4.md`
- `audit/v0.4/cost_ledger_merged.json`
- `benchmarks/results_v0.4/judge_agreement.json`
- `benchmarks/results_v0.4/judge.jsonl`

Evidence:

`audit/v0.4/cost_ledger_merged.json` is Haiku-only:

```json
{
  "total_usd": 12.725062000000001,
  "n_calls": 1277,
  "by_model": {
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {
      "total_usd": 12.725062000000001,
      "n_calls": 1277
    }
  }
}
```

The judge cost is separate:

```json
{
  "n": 23,
  "total_cost_usd": 0.483636
}
```

`benchmarks/results_v0.4/judge.jsonl` sums to `$0.483636` across 23 rows. README/release notes report `$13.21`, which is approximately `$12.73 + $0.48`. The paper, however, says `$12.73 over 1,277 Bedrock calls (Haiku scorer + Sonnet judge)`, which implies Sonnet judge cost is included in the `$12.73` and `1,277` call ledger.

Why it matters:

The paper should not make readers infer which cost boundary is being used. Cost/accounting is part of the v0.4 reproducibility claim.

Required fix:

- Split the wording:
  - Haiku cascade ledger: `$12.73`, `1,277` calls.
  - Sonnet judge ledger: `$0.48`, `23` judge calls/rows.
  - Combined pilot spend: `$13.21`.
- Update `paper/main.tex`, `paper/sections/09_results.tex`, `paper/sections/10b_honest_ai_claims.tex`, README, release notes, and site data labels consistently.

Confidence: high.

### 6. Pages dependency install is not reproducible

Severity: P1  
Primary paths:

- `.github/workflows/pages.yml`
- `.gitignore`
- `docs/site/package.json`
- `docs/site/pnpm-lock.yaml`

Evidence:

The Pages workflow runs:

```yaml
pnpm install --frozen-lockfile=false
```

`docs/site/pnpm-lock.yaml` exists locally, but is not tracked by git. `.gitignore` ignores `docs/site/pnpm-lock.yaml`.

Why it matters:

Every push to `main` can resolve a different Astro/Tailwind transitive tree. That weakens the "frozen v0.4 site" claim and makes future Pages failures non-reproducible.

Required fix:

- Track `docs/site/pnpm-lock.yaml`.
- Remove the `.gitignore` entry for it.
- Change CI to `pnpm install --frozen-lockfile`.

Confidence: high.

### 7. Judge audit metadata is weak for prompt replay

Severity: P1  
Primary paths:

- `benchmarks/results_v0.4/judge.jsonl`
- `benchmarks/results_v0.4/judge_agreement.json`
- `scripts/judge_subset.py`

Evidence:

Audit probe:

```text
n_rows = 23
unique_prompt_sha = 1
unique_input_tokens = [9]
total_cost = 0.483636
```

The constant `prompt_sha256` is the frozen template hash, not a pair-specific formatted prompt hash. `input_tokens = 9` for every row is implausibly small for long paired judgement prompts and appears to come from fallback usage handling.

Why it matters:

The H9 judge/scorer disagreement can still be a real result, but third-party replay is weaker than the current "frozen prompt sha" framing suggests. A reader cannot verify the exact per-item judge prompt from the recorded hash alone.

Required fix:

- Record both `prompt_template_sha256` and `formatted_prompt_sha256`.
- Record token estimates based on the actual formatted prompt when CLI usage is missing.
- Update docs to say the current v0.4 audit hash is template-level.

Confidence: high.

## P2 Findings

### 8. Showcase authenticity is mixed: six cascade-derived, three curated references

Severity: P2  
Primary paths:

- `scripts/showcase_specs.toml`
- `benchmarks/showcase_v0.4/*/prompt.json`
- `benchmarks/showcase_v0.4/*/trace.json`
- `paper/sections/10c_showcase_examples.tex`
- `docs/site/src/pages/showcase/*`

Evidence:

`scripts/showcase_specs.toml` defines:

- `english_*`: `source = "phase7_cascade"`
- `science_*`: `source = "phase7_cascade"`
- `sanskrit_*`: `source = "curated_reference"`

Trace probe confirmed:

```text
english_* model = claude-haiku via Bedrock (Phase 7), source = phase7_cascade
science_* model = claude-haiku via Bedrock (Phase 7), source = phase7_cascade
sanskrit_* model = n/a (curated reference; awaiting v0.5 chandas-aware cascade)
```

Spot-check:

`benchmarks/showcase_v0.4/english_imagist_haiku/trace.json` matches `benchmarks/results_v0.4/poetry_gen.json` item `p06`.

Why it matters:

The showcase is not broadly fabricated. But the phrase "9 generated creative outputs" is too strong unless it is immediately qualified: six are curated from real Phase 7 cascade outputs; three Sanskrit entries are maintainer-authored curated references.

Required fix:

- Rename public copy to "9 showcase outputs" or "6 cascade outputs + 3 curated Sanskrit references."
- Keep the Sanskrit disclosure visible on the site cards, not only in JSON/prose.
- Do not describe Sanskrit items as "generated by the cascade" until v0.5 adds chandas-aware generation.

Confidence: high.

### 9. `revised.txt` is a misleading alias

Severity: P2  
Primary paths:

- `scripts/generate_v0_4_showcase.py`
- `benchmarks/showcase_v0.4/*/revised.txt`
- `benchmarks/showcase_v0.4/*/shadow_revision.txt`

Evidence:

`scripts/generate_v0_4_showcase.py` writes:

```python
(out_dir / "revised.txt").write_text(
    trace.get("committed") or trace.get("revised") or trace.get("shadow_revision") or ""
)
```

For event-gated draft commits, `revised.txt` duplicates the committed draft. The actual revision pass is in `shadow_revision.txt`.

Why it matters:

Any downstream auditor expecting `revised.txt` to represent the revision surface will silently read the wrong file.

Required fix:

- Make `revised.txt` equal the actual revision/shadow-revision surface.
- If backward compatibility is needed, create `committed.txt` as the committed surface and document it as such.
- Add a test that `shadow_revision.txt` and `revised.txt` semantics cannot drift.

Confidence: high.

### 10. STATUS.md contradicts the committed result files

Severity: P2  
Primary path:

- `benchmarks/results_v0.4/STATUS.md`

Evidence:

`STATUS.md` says each domain has `items_with_rows | 1` and `complete_items (4/4 arms) | 0`, while the corresponding JSON files contain many item rows, for example `poetry_gen.json` has 20 row keys.

Why it matters:

The status file is an audit artifact and should not contradict the data it claims to summarize. This is especially confusing because `STATUS.md` is in the same result directory as `stats.json` and `judge_agreement.json`.

Required fix:

- Regenerate `STATUS.md` from the actual result JSON schema.
- Explain `judge_rc: 3` if it is expected due to partial quota/cap behavior.

Confidence: high.

### 11. Citation verification is useful but overstated

Severity: P2  
Primary paths:

- `paper/references.bib`
- `audit/v0.4/lit_verification_summary.json`
- `audit/v0.4/lit_verification.jsonl`

Evidence:

`audit/v0.4/lit_verification_summary.json` reports 48 total entries, 46 verified, and 2 `not_verified_no_handle`. The bibliography header still says "22 entries auto-verified," which is stale. The audit also records title collisions and at least one year mismatch.

Why it matters:

The bibliography work is stronger than hand-curated citations, but the paper/docs should present it as automated plausibility verification, not proof of every field, venue, and duplicate relationship.

Required fix:

- Update the `references.bib` header.
- Explicitly list the two offline-only entries.
- Decide whether duplicate POEMetric entries are intentional or need consolidation.

Confidence: medium-high.

### 12. Config precedence documentation contradicts implementation

Severity: P2  
Primary paths:

- `src/pce/config.py`
- `RUN_LOCAL.md`
- `docs/site/src/pages/plugin.astro`

Evidence:

Implementation in `PCEConfig.load()` layers:

```text
defaults -> repo pce.toml -> user TOML -> env -> CLI overrides
```

The module docstring presents the order in the opposite conceptual direction, and `RUN_LOCAL.md` says user TOML then repo TOML with repo overriding user.

Why it matters:

Config precedence is a core portability claim. Agents and users need to know whether project config can override personal config.

Required fix:

- Pick one precedence contract.
- Update `src/pce/config.py` docstring, `RUN_LOCAL.md`, README, and site plugin page to match.
- Add one documentation-linked test for precedence.

Confidence: high.

## P3 Findings

### 13. Plugin portability is weaker than manifest wording implies

Severity: P3  
Primary paths:

- `plugin/.cursor-plugin/plugin.json`
- `plugin/.claude-plugin/plugin.json`
- `plugin/.mcp.json`
- `plugin/hooks/hooks.json`
- `plugin/README.md`

Evidence:

The two `plugin.json` files are metadata only. Actual MCP startup depends on `plugin/.mcp.json`, `uv`, `${CLAUDE_PLUGIN_ROOT}/..`, repo-relative paths, and shell hooks. Cursor portability therefore depends on full repo layout and tooling, not only the Cursor manifest.

Required fix:

- Document the real portability contract: full repo clone, `uv`, `claude` CLI, and host support for the plugin root variable.
- Consider validating `.mcp.json` and hook paths in `tests/test_plugin_manifests.py`.

Confidence: high.

### 14. Plugin docs still contain stale v0.3 / SDK language

Severity: P3  
Primary paths:

- `plugin/mcp/server.py`
- `plugin/README.md`

Evidence:

`plugin/mcp/server.py` still documents `PCE_USE_SDK=1` as preferring the Anthropic SDK path, even though ADR-007 says that path was removed. `plugin/README.md` says 15 MCP tools, while README/manifests describe 19+.

Required fix:

- Remove the SDK preference text.
- Update tool counts and command names.

Confidence: high.

### 15. Site accessibility needs a pass

Severity: P3  
Primary paths:

- `docs/site/src/layouts/BaseLayout.astro`
- `docs/site/src/components/DiffView.astro`
- `docs/site/src/components/ChandasMeterDisplay.astro`
- `docs/site/src/components/JudgeScorerScatter.astro`

Evidence:

Reviewer pass found:

- header and sidebar both use `<nav>` without distinct `aria-label`s;
- no skip link before repeated sidebar navigation;
- diff colors rely mainly on background color;
- chandas pattern glyphs use `title` but no screen-reader summary.

Required fix:

- Add labelled nav landmarks and a skip link.
- Add text cues to diff rows.
- Add a compact textual summary for chandas patterns and scatter data.

Confidence: medium.

## Positive Findings

The review also found several things working correctly:

- The six English/science showcase traces are consistent with being curated from Phase 7 result JSON, not freshly invented prose.
- The Sanskrit entries are disclosed as `curated_reference` in the spec and generated prompt metadata.
- `pce config show`, `pce smoke --dry-run`, `pce showcase`, valid `pce judge-pair --dry-run`, and valid `pce cascade --dry-run` work under `PYTHONPATH=src` with Python 3.12.
- The live site serves index, results, hypotheses, showcase pages, references, plugin page, and `/paper/main.pdf`.
- The paper is more honest than v0.3 about H1-H5 null/inconclusive findings and H9 judge/scorer disagreement.

## Fix Order

1. Fix the Indravajra showcase and make the chandas gate fail on `ok != true`.
2. Fix GitHub Pages base-aware redirects and the `/discussion` link.
3. Make README/RUN_LOCAL/site CLI examples parse against the actual CLI, or add compatibility aliases.
4. Correct paper/site/release cost wording by separating Haiku cascade ledger, Sonnet judge ledger, and combined spend.
5. Track `docs/site/pnpm-lock.yaml` and use `pnpm install --frozen-lockfile` in CI.
6. Update judge audit metadata semantics: template hash vs formatted prompt hash, and realistic token estimates.
7. Regenerate or correct `benchmarks/results_v0.4/STATUS.md`.
8. Tighten showcase wording: six cascade-derived outputs plus three curated Sanskrit references.
9. Repair config precedence docs and stale plugin SDK/tool-count docs.
10. Add a small link crawl, CLI docs parser test, and accessibility smoke pass.

## Final Assessment

PCE v0.4 should not be described as fabricated or fake. The main benchmark artefacts and most showcase traces are real enough to audit, and the paper does engage with negative and mixed results.

But v0.4 should also not be treated as a clean archival release until the issues above are patched. The most serious defect is the validator/gate mismatch: a release gate passed while a shipped showcase item failed. The second most visible defect is the Pages base-path bug. The third is operator trust: shipped CLI instructions fail when copied verbatim.

The release can be salvaged cleanly with a v0.4.1 patch that fixes the artefacts, docs, Pages routing, and cost/accounting prose without changing the underlying benchmark conclusions.
