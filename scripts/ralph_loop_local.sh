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
#   1. pytest    - the full Python test suite, including the new
#                  v0.4.2 hardening tests.
#   2. site data - regenerate stats_v0.4.json and friends from the
#                  committed benchmark JSON so the Astro site reflects
#                  current artefacts.
#   3. site      - install with --frozen-lockfile and run pnpm build
#                  so the audit gate can crawl the rendered HTML.
#   4. paper     - rebuild paper/main.pdf with tectonic and snapshot
#                  it to paper/v0.4/main.pdf so the frozen PDF stays
#                  in sync with the live TeX sources.
#   5. phase 8   - run the artefact audit gate stack and fail loudly
#                  on any gate that did not pass.
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

echo "[ralph-loop] (1/5) pytest"
"$PY" -m pytest -q

echo "[ralph-loop] (2/5) prepare_site_data.py"
"$PY" scripts/prepare_site_data.py

echo "[ralph-loop] (3/5) docs/site build"
pnpm --dir docs/site install --frozen-lockfile
pnpm --dir docs/site build

echo "[ralph-loop] (4/5) tectonic paper"
( cd paper && tectonic main.tex )
cp -f paper/main.pdf paper/v0.4/main.pdf

echo "[ralph-loop] (5/5) phase 8 artefact audit"
"$PY" scripts/phase8_gate_stack.py

echo "[ralph-loop] OK — all stages green"
