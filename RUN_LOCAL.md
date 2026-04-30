# RUN_LOCAL.md — running PCE locally

PCE v0.4 ships as a portable plugin with three first-class install paths. The cascade module is identical in all three; only the host integration changes.

## Prerequisites

* Python ≥ 3.12 (`uv` strongly recommended).
* The Anthropic [`claude` CLI](https://docs.anthropic.com/en/docs/claude-code) on `PATH`, signed in via OAuth. PCE uses `claude --print` exclusively — no SDK calls.
* Optional, for the showcase site: Node ≥ 20 + pnpm ≥ 10.
* Optional, for the paper PDF: a TeX engine ([`tectonic`](https://tectonic-typesetting.github.io/) is the easiest install on macOS).

## 1. Standalone `pce` CLI

The lowest-friction install path. No host IDE required.

```bash
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv venv && uv pip install -e .
source .venv/bin/activate                 # or: rehash; pce --version

pce smoke                                 # checks claude on PATH, OAuth, and cascade module
pce config show                           # prints resolved PCEConfig
pce cascade --prompt "Write a contemporary haiku on attention drift." --K 4 --seed 4242
pce judge-pair --a draft.txt --b revised.txt --prompt "Which is more inventive?"
pce showcase generate                     # rebuild the 9-demo showcase under benchmarks/showcase_v0.4/
```

Available subcommands:

| command | role |
|---|---|
| `pce smoke` | end-to-end check: CLI on PATH + cascade imports + one frozen prompt |
| `pce cascade <prompt>` | run a single cascade with `--K`, `--seed`, `--model`, `--judge-model` |
| `pce judge-pair --a A --b B --prompt P` | invoke the Sonnet judge bridge and emit a verdict JSON |
| `pce config show` | print the resolved configuration |
| `pce showcase generate` | drive `scripts/generate_v0_4_showcase.py` |

## 2. Cursor plugin

```bash
# from a clone of the repo:
cursor --install-plugin .

# verify install
cursor plugin list
```

The Cursor manifest at `plugin/.cursor-plugin/plugin.json` mirrors the Claude Code manifest — same MCP tools, same slash commands, same hooks. After install, the `/pce-compose`, `/pce-interpret`, `/pce-aut`, `/pce-bbh`, and `/pce-trace` commands appear in the Cursor command palette.

## 3. Claude Code plugin

```bash
claude plugin install https://github.com/SharathSPhD/pratyabhijna
# or, for a local clone:
ln -s "$(pwd)" "$HOME/.claude/plugins/pce"
```

The Claude Code manifest at `plugin/.claude-plugin/plugin.json` is the canonical reference; the Cursor manifest is generated to match.

## Configuring the cascade and judge models

Defaults: `cascade_model = "haiku"`, `judge_model = "sonnet"`. Resolution order (later overrides earlier):

1. Built-in defaults (in `src/pce/config.py`).
2. `~/.config/pce/config.toml` (user-level).
3. `pce.toml` at the repository root (project-level).
4. Environment variables: `PCE_CASCADE_MODEL`, `PCE_JUDGE_MODEL`, `PCE_CLI_BIN`, `PCE_TIMEOUT_S`, `PCE_COST_CAP_USD`.
5. Legacy back-compat: `PCE_HAIKU_MODEL` (deprecated, maps to `cascade_model`).
6. CLI flags: `--model` and `--judge-model` on `pce cascade` and friends.

Example `~/.config/pce/config.toml`:

```toml
[pce]
cascade_model = "sonnet"            # use Sonnet-4.5 for the cascade
judge_model   = "opus"              # use Opus for the LLM-judge (when CLI exposes it)
cli_bin       = "claude"            # binary on PATH
timeout_s     = 240
cost_cap_usd  = 50.0
```

PCE accepts any Anthropic CLI-addressable model name: bare aliases (`haiku`, `sonnet`, `opus`) and full Bedrock IDs (`global.anthropic.claude-haiku-4-5-20251001-v1:0`, etc.). PCE does not validate the model string — that's the CLI's job.

## Substrate boundary — OAuth CLI only (ADR-007)

The Anthropic Python SDK code path was removed in Phase 8. PCE has a single supported substrate: `claude --print` over the OAuth-bound CLI. Legacy users who set `PCE_USE_SDK=1` will see a clear deprecation error with a one-line remediation path. The Phase 7 mechanism pilot used Bedrock through the same CLI's profile selector; this is documented as a deliberate substrate-deviation event in §7 of the paper (ADR-006).

## Building the paper

```bash
cd paper
tectonic -X compile main.tex          # produces paper/main.pdf
# or, with a full TeX Live:
latexmk -pdf main.tex
```

Frozen archives live at `paper/v0.1/`, `paper/v0.2/`, `paper/v0.3/`, and `paper/v0.4/`. The current `paper/main.tex` is the v0.4 paper; `paper/v0.4/` is the snapshot at release.

## Building the Astro v0.4 site

```bash
python scripts/prepare_site_data.py    # materialises stats / showcase / figures into docs/site/public/data/
cd docs/site
pnpm install
pnpm build                             # writes docs/site/dist/
pnpm preview                           # serves dist/ locally on http://localhost:4321/pratyabhijna/
```

The site reads `benchmarks/results_v0.4/stats.json` (via `prepare_site_data.py`'s materialised assets), the per-demo showcase trace bundles, and the verified bibliography. The GitHub Actions workflow in `.github/workflows/pages.yml` does the same on push to `main`.

## Troubleshooting

* `pce smoke` fails on `claude --version` → install / sign-in to the Claude CLI; verify with `which claude`.
* `pce cascade` returns empty → check `PCE_HAIKU_CLI` and `PCE_HAIKU_MODEL` (or the new `PCE_CASCADE_MODEL` / `PCE_CLI_BIN`) and run `pce config show` to see the resolved chain.
* The Astro build complains about JSON imports → make sure you ran `python scripts/prepare_site_data.py` first.
* The paper build fails on a missing v0.4 figure → run `python -m benchmarks.figures --version v0.4` then retry the LaTeX build.
* Legacy users seeing `PCE_USE_SDK is no longer supported` → switch to the OAuth CLI; the SDK path was removed in ADR-007 and there is no flag to re-enable it.
