# RUN_LOCAL.md — running PCE locally

PCE v0.4 ships as a portable plugin with three first-class install paths. The cascade module is identical in all three; only the host integration changes.

## Prerequisites

* Python ≥ 3.11 (PCE's `pyproject.toml` declares `>=3.11,<3.14`; `uv` strongly recommended).
* The Anthropic [`claude` CLI](https://docs.anthropic.com/en/docs/claude-code) on `PATH`, signed in via OAuth. PCE uses `claude --print` exclusively — no SDK calls.
* Optional, for the showcase site: Node ≥ 20 + pnpm ≥ 10.
* Optional, for the paper PDF: a TeX engine ([`tectonic`](https://tectonic-typesetting.github.io/) is the easiest install on macOS).

## 1. Standalone `pce` CLI

The lowest-friction install path. No host IDE required.

The CLI is a real Python package, not a vendored script: an editable install with
`pip install -e .` (or `uv pip install -e .`) is required so the `pce` console
script and the `numpy` / `sentence-transformers` runtime dependencies are
importable. Running `python -m pce --help` directly out of `src/` without an
install will fail with `No module named pce`.

```bash
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv venv && uv pip install -e .            # required: registers `pce` and pulls deps
source .venv/bin/activate                 # or use `uv run pce ...` without activating

pce smoke                                 # checks claude on PATH, OAuth, and cascade module
pce config show                           # prints resolved PCEConfig
pce cascade --prompt "Write a contemporary haiku on attention drift." \
            --constraint "imagistic specificity" --k 4 --seed 4242
pce judge-pair --domain poetry_gen --item-id p07 \
               --treatment-text out/treatment.txt --control-text out/control.txt
pce showcase                              # list the 9-demo on-disk index
pce showcase --regenerate sanskrit_anustubh   # regenerate one slug (curate mode)
```

Subcommand reference (mirrors `pce.cli.build_parser`):

| command | role |
|---|---|
| `pce smoke [--dry-run]` | end-to-end check: CLI on PATH + cascade imports + one frozen prompt |
| `pce cascade --prompt P [--constraint C] [--k K] [--seed S] [--out PATH]` | run a single cascade pass and print a JSON summary |
| `pce judge-pair --domain DOM --item-id ID --treatment-text PATH --control-text PATH` | invoke the Sonnet judge bridge and emit a verdict JSON |
| `pce config show` | print the resolved configuration |
| `pce showcase` | print the on-disk showcase index under `benchmarks/showcase_v0.4/` |
| `pce showcase --regenerate <slug\|all>` | drive `scripts/generate_v0_4_showcase.py` |

Global flags accepted by every subcommand: `--model`, `--judge-model`, `--cli-bin`, `--timeout-s`, `--config`, `--dry-run`.

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

Defaults: `cascade_model = "haiku"`, `judge_model = "sonnet"`. Resolution order (later layers override earlier ones — this matches `PCEConfig.load()` in [`src/pce/config.py`](../src/pce/config.py)):

1. Built-in defaults (the `PCEConfig` dataclass field defaults).
2. Repo-level TOML at `./pce.toml` (project-pinned).
3. User-level TOML at `~/.config/pce/config.toml` (or `$XDG_CONFIG_HOME/pce/config.toml`).
4. Environment variables: `PCE_MODEL` / `PCE_CASCADE_MODEL`, `PCE_JUDGE_MODEL`, `PCE_CLI`, `PCE_TIMEOUT_S`, `PCE_COST_CAP_USD`. Legacy back-compat aliases (`PCE_HAIKU_MODEL`, `PCE_HAIKU_CLI`, `PCE_HAIKU_TIMEOUT_S`, …) still work; the new names take precedence when both are set.
5. CLI flags: `--model`, `--judge-model`, `--cli-bin`, `--timeout-s`, `--config`. CLI flags win over everything else.

`PCE_USE_SDK=1` is honoured only insofar as it emits a `DeprecationWarning`; the Anthropic Python SDK code path was removed in ADR-007.

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

The current `paper/main.tex` is the v0.4 paper; `paper/v0.4/` holds the frozen snapshot served from the live site. Pre-v0.4 archives are no longer tracked in this repo (they were removed in the v0.4.1 cleanup pass; see git history if you need them).

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
* `pce cascade` returns empty → check `PCE_CLI` (path to the CLI binary) and `PCE_CASCADE_MODEL` (or the legacy `PCE_HAIKU_CLI` / `PCE_HAIKU_MODEL`) and run `pce config show` to see the resolved chain.
* `python -m pce: No module named pce` → run `uv pip install -e .` (or `pip install -e .`) inside the repo. The `pce` package is not vendored as a script.
* `ModuleNotFoundError: No module named 'numpy'` after a non-venv install attempt → the runtime needs the package's declared dependencies. Use `uv pip install -e .` from a working venv, or accept that PCE is not pip-less.
* The Astro build complains about JSON imports → make sure you ran `python scripts/prepare_site_data.py` first.
* The paper build fails on a missing v0.4 figure → run `python -m benchmarks.figures --version v0.4` then retry the LaTeX build.
* Legacy users seeing `PCE_USE_SDK is no longer supported` → switch to the OAuth CLI; the SDK path was removed in ADR-007 and there is no flag to re-enable it.
