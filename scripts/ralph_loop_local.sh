#!/usr/bin/env bash
# Ralph-loop local release-clean check.
#
# This is the single command a maintainer must run before committing,
# tagging, or pushing a release. It is intentionally a thin shell
# script (not a Makefile target, not a CI workflow) because the project
# treats ralph-loop discipline as a *local* contract: a green run here
# is the precondition for any v0.4.x release artefact moving to GitHub.
#
# Stages (in order):
#   1. pytest        - the full Python test suite.
#   2. site data     - regenerate stats_v0.4.json and friends from the
#                      committed benchmark JSON so the Astro site reflects
#                      current artefacts.
#   3. flowchart png - rebuild the F1-F5 TikZ flowcharts as standalone PNGs
#                      and copy them into both paper/figures/v0.4/flowcharts/
#                      and docs/site/public/figures/v0.4/flowcharts/ so the
#                      site placeholders auto-swap with real flowchart art.
#   4. site          - install with --frozen-lockfile and run pnpm build
#                      so the audit gate can crawl the rendered HTML.
#   5. paper         - rebuild paper/main.pdf with tectonic and snapshot
#                      it to paper/v0.4/main.pdf so the frozen PDF stays
#                      in sync with the live TeX sources.
#   6. phase 8       - run the artefact audit gate stack and fail loudly
#                      on any gate that did not pass.
#
# A non-zero exit at any stage stops the loop. Don't tag/release if
# this script is red.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[ralph-loop] .venv missing; run: uv venv && uv pip install -e .[dev]" >&2
  exit 1
fi

echo "[ralph-loop] (1/6) pytest"
"$PY" -m pytest -q

echo "[ralph-loop] (2/6) prepare_site_data.py"
"$PY" scripts/prepare_site_data.py

echo "[ralph-loop] (3/6) build_paper_flowchart_pngs.py"
"$PY" scripts/build_paper_flowchart_pngs.py

echo "[ralph-loop] (4/6) docs/site build"
pnpm --dir docs/site install --frozen-lockfile
pnpm --dir docs/site build

echo "[ralph-loop] (5/6) tectonic paper"
( cd paper && tectonic --keep-logs main.tex )
cp -f paper/main.pdf paper/v0.4/main.pdf

echo "[ralph-loop] (6/6) phase 8 artefact audit"
"$PY" scripts/phase8_gate_stack.py

echo "[ralph-loop] OK — all stages green"
