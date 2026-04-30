# PCE v0.4 — Phase 7 powered pilot on AWS Bedrock

> **Audience.** Two readers. (1) The human operator who opens a fresh
> Claude Code Desktop session that is already authenticated against
> Anthropic's models on **AWS Bedrock**. (2) The Claude Code agent that
> runs inside that session and executes the pipeline below. The agent
> has no prior context on this repo — everything it needs is in this
> file plus the repo it clones.
>
> **What this pilot is.** The Phase-7 powered run of the *Pratyabhijñā
> Creative Engine* (PCE) v0.4 mechanism study: 4 base arms × 4 commit
> policies (multiplexed post-hoc, no extra LM calls) × n=20 items × 4
> domains, plus a Sonnet-as-judge subset and the v0.4 stats pipeline.
> See `paper/v0.4/SPEC_v0.4.md` for the scientific framing.
>
> **Why Bedrock.** The macOS author's run hit OAuth subscription quota
> after 4½ items at $2.41 spent. The driver halts cleanly on
> `HaikuRateLimitError` so the partial state on the
> `v0.4-mechanism-study` branch is safe to resume; this Bedrock run
> picks up where the macOS run stopped.
>
> **Estimated runtime.** ~2–4 hours wall clock with `--max-parallel 4`
> on Bedrock (one process per domain). Estimated Bedrock cost: <$30
> at current Haiku-4.5 + Sonnet-4.5 pricing.

---

## 0. Prerequisites on the Bedrock host

These must be true before the agent starts. The human operator should
verify them; the agent will re-verify in the pre-flight step.

### 0.1 OS + tooling

- macOS, Linux, or WSL2.
- Python 3.13 (the repo is pinned to 3.13 via `.python-version` and
  `pyproject.toml`).
- `git`, `curl`, `jq` available on `PATH`.
- `claude` CLI (Claude Code) installed and on `PATH`. Verify:
  ```bash
  claude --version
  ```

### 0.2 AWS Bedrock access

- AWS account with Bedrock access in the region holding the model
  catalog you want to use. The script defaults to **global
  cross-region inference profiles**:
  - Haiku: `global.anthropic.claude-haiku-4-5-20251001-v1:0`
  - Sonnet: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Both models must be **enabled in your Bedrock Model Access** console.
- AWS auth resolved via the standard chain. Pick one:
  - `AWS_PROFILE` pointing at a profile in `~/.aws/config` (the clean
    substrate symlinks `~/.aws` into the inner subprocess on v0.4 —
    you do not need to copy anything).
  - Static keys: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
    (+ `AWS_SESSION_TOKEN` if STS).
  - Bedrock API key: `AWS_BEARER_TOKEN_BEDROCK`.

### 0.3 Tell the `claude` CLI to use Bedrock

Set in the shell that will spawn the orchestrator:

```bash
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1                                                   # or your bedrock region
export AWS_PROFILE=my-bedrock-profile                                         # if using profile auth
export ANTHROPIC_MODEL=global.anthropic.claude-sonnet-4-5-20250929-v1:0       # default model
export ANTHROPIC_SMALL_FAST_MODEL=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

Alternatively, if your Claude Code Desktop already has Bedrock enabled
via "Use Claude with third-party platforms" (the doc you linked: *Use
Claude Cowork with third-party platforms*), the parent CLI will inherit
those env vars automatically. The orchestrator script warns if
`CLAUDE_CODE_USE_BEDROCK` is unset.

Smoke test (should return a few words and exit 0):

```bash
claude --print --model global.anthropic.claude-haiku-4-5-20251001-v1:0 --output-format json "Reply with the single word OK." | jq -r '.result'
```

If that prints `OK`, Bedrock is wired correctly.

---

## 1. Clone + bootstrap (one-shot)

The Claude Code agent should run this verbatim.

```bash
# 1. Clone the v0.4 branch into a fresh dir.
git clone --branch v0.4-mechanism-study --single-branch \
    https://github.com/SharathSPhD/pratyabhijna.git pratyabhijna
cd pratyabhijna

# 2. Confirm we have the resume artefacts from the macOS run.
ls benchmarks/results_v0.4/         # expect: poetry_gen.json (partial)
cat audit/cost_ledger.json          # expect: total_usd 0.0 (reset before push)

# 3. Set up Python via uv (preferred — pinned to 3.13, exact lock).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen                    # creates .venv with the pinned env
source .venv/bin/activate

# Fallback if uv is unavailable:
#   python3.13 -m venv .venv
#   source .venv/bin/activate
#   pip install --upgrade pip
#   pip install -e ".[dev]"
```

Sanity-check the install:

```bash
python -c "import pce, benchmarks; print('pce', pce.__file__)"
python -m pytest tests/test_haiku_rate_limit_error.py -q
```

The pytest call should pass — it exercises the same typed-error path
the orchestrator relies on to halt cleanly.

---

## 2. Pre-flight (no Bedrock calls)

Confirm the environment is wired without spending anything:

```bash
python scripts/run_v0_4_bedrock.py --skip-pilot --skip-judge --skip-stats
```

This runs only the pre-flight + ledger merge. Expected output:

```
[orch ...] claude --version: 2.x.x ...
[orch ...] haiku model = global.anthropic.claude-haiku-4-5-20251001-v1:0
[orch ...] sonnet model = global.anthropic.claude-sonnet-4-5-20250929-v1:0
[orch ...] merged ledger -> .../audit/v0.4/cost_ledger_merged.json
```

If you see a `WARN: CLAUDE_CODE_USE_BEDROCK is not set to '1'`, fix
your shell env before continuing.

---

## 3. Run the pipeline

### 3.1 Default: 4 parallel domain workers, all phases

```bash
python scripts/run_v0_4_bedrock.py --git-push 2>&1 | tee logs/orchestrator.log
```

What happens:

1. **Phase 7-A** spawns 4 subprocesses (`benchmarks.driver`), one per
   domain (`poetry_gen`, `poetry_interp`, `aut`, `sci_creativity`),
   each with isolated audit sinks under `audit/v0.4/<domain>` so the
   workers never race on the cost ledger or integrity log.
2. The orchestrator polls every 30 s, prints a per-domain status
   table, and rewrites `benchmarks/results_v0.4/STATUS.md` so a human
   or a babysitting agent can `cat STATUS.md` at any time.
3. Each worker resumes from existing rows (the macOS partial
   `poetry_gen.json` is filled in via `--retry-failed`).
4. **Phase 7-B**: once all four workers have exited, the orchestrator
   merges per-domain ledgers, runs `scripts/judge_subset.py` against
   the Bedrock Sonnet model (32 stratified items by default), and
   writes `benchmarks/results_v0.4/judge.jsonl` +
   `judge_agreement.json`.
5. **Phase 7-C**: runs `python -m benchmarks.stats --version v0.4`
   which populates `benchmarks/results_v0.4/stats.json` with the
   v0.4 hypothesis set (H1.v4–H9.v4, fixed-effects H5).
6. With `--git-push`, commits `benchmarks/results_v0.4/` and
   `audit/v0.4/` to the `v0.4-mechanism-study` branch and pushes to
   `origin`. The local author then completes Phase 8 (paper rewrite,
   tag, merge PR).

### 3.2 Useful variants

```bash
# Throttle to 2 simultaneous workers (lower Bedrock parallel quota).
python scripts/run_v0_4_bedrock.py --max-parallel 2

# Pilot only (no judge, no stats), for stepping through manually.
python scripts/run_v0_4_bedrock.py --skip-judge --skip-stats

# Judge + stats only (assumes rows already exist).
python scripts/run_v0_4_bedrock.py --skip-pilot

# Specific Bedrock model IDs (override defaults).
python scripts/run_v0_4_bedrock.py \
  --haiku-model  global.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --sonnet-model global.anthropic.claude-sonnet-4-5-20250929-v1:0

# Smaller pilot (debug only — DO NOT use for the real run).
python scripts/run_v0_4_bedrock.py --n-per-domain 2
```

### 3.3 Babysit mode for the Claude Code agent

The Claude Code agent driving this session should orchestrate as
follows. **Use parallel `Task` subagents**, one per domain log, so the
agent can react to per-worker events without serializing on a single
log file:

1. Spawn the orchestrator in the background:
   ```bash
   nohup python scripts/run_v0_4_bedrock.py --git-push \
     > logs/orchestrator.log 2>&1 &
   echo $! > logs/orchestrator.pid
   ```
2. Spawn one `Task` subagent per domain with this brief: *"Tail
   `logs/v0_4_pilot.<domain>.bedrock.log` every 5 minutes. If the file
   contains `HALT: HaikuRateLimitError`, escalate to the parent.
   Otherwise, every 5 minutes, emit a one-line status: domain, items
   complete (4/4 arms), cost so far. Stop when the log shows a final
   summary line containing 'haiku cost'."* — those subagents run in
   parallel with each other and with the orchestrator.
3. Spawn one `Task` subagent that watches
   `benchmarks/results_v0.4/STATUS.md` every 5 minutes and reports its
   final state.
4. When all subagents converge ("DONE"), the parent agent runs
   `git status` to confirm `--git-push` succeeded, and reports the
   final `stats.json` summary back to the human.

---

## 4. What lands in the repo (Phase 7 deliverables)

After a successful run, these paths exist on the
`v0.4-mechanism-study` branch:

```
benchmarks/results_v0.4/
  poetry_gen.json            # 20 items × 4 base arms × multiplexed commit policies
  poetry_interp.json
  aut.json
  sci_creativity.json
  judge.jsonl                # one row per Sonnet judge call (~32 rows)
  judge_agreement.json       # H9.v4 metrics
  stats.json                 # H1.v4–H9.v4 (incl. fixed-effects H5)
  STATUS.md                  # human-readable summary

audit/v0.4/
  cost_ledger_<domain>.json  # per-domain Haiku ledger (4 files)
  cost_ledger_merged.json    # summed
  cost_snapshot_<domain>.json
  integrity_probes_<domain>.jsonl
  integrity_probes_merged.jsonl

logs/
  orchestrator.log
  v0_4_pilot.<domain>.bedrock.log   # per-domain stdout/stderr (4 files)
```

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `claude: command not found` | CLI not on PATH. | Install Claude Code per the desktop docs; restart shell. |
| `WARN: CLAUDE_CODE_USE_BEDROCK is not set` and run halts on a 429. | OAuth quota path was used. | `export CLAUDE_CODE_USE_BEDROCK=1`; re-run. |
| `An error occurred (AccessDeniedException) ...` in worker log. | Bedrock model access not enabled. | Enable both `claude-haiku-4-5` and `claude-sonnet-4-5` in the Bedrock Model Access console. |
| `An error occurred (ValidationException) ... model not found` | Wrong model ID for your region. | Pass explicit `--haiku-model` / `--sonnet-model`. |
| Workers exit with `HaikuRateLimitError` even on Bedrock. | Quota throttle from the AWS account, not Anthropic. | Re-run with `--retry-failed --max-parallel 2`. |
| `stats.json` has `"placeholder": true` markers. | Pilot did not produce enough rows. | Inspect `STATUS.md`; re-run `--skip-judge --skip-stats` then re-run stats. |
| Git push rejected. | Branch protection. | The orchestrator commits with a synthetic email; pull+rebase or push with the operator's credentials. |

---

## 6. Pickup hand-off when finished

Once the pipeline exits 0 and the orchestrator pushes, the local author
will pull the branch and execute **Phase 8** in the macOS session:

1. Re-write `paper/v0.4/active_inference.md` against the fresh
   `stats.json` and `judge_agreement.json` (FE causal chain,
   `cit_temperature` → best-of-K story, Hopfield warm-start status).
2. Update HTML/PDF outputs and the GitHub Pages dashboard.
3. Reconcile fixed/random for H5 in the discussion.
4. Tag `v0.4.0` and merge the PR.

The Bedrock agent does NOT need to do any of that — its scope ends at
the green pipeline + pushed result tree.

---

## 7. One-line abstract for the agent

> Clone `v0.4-mechanism-study`, set up a uv venv, export your AWS +
> Bedrock env, then run `python scripts/run_v0_4_bedrock.py --git-push`
> with parallel `Task` subagents tailing `logs/v0_4_pilot.<domain>.bedrock.log`.
> Stop when the orchestrator prints "DONE" and `git status` is clean.
