.PHONY: help install test lint type smoke bench stats figures autoreport paper present clean all benchmark.pilot stats.pilot

help:
	@echo "Pratyabhijna Creative Engine — Make targets"
	@echo "  make install     uv sync --extra dev"
	@echo "  make test        fast pytest (skips slow + real_model)"
	@echo "  make lint        ruff check"
	@echo "  make type        mypy --strict (src/pce + scripts)"
	@echo "  make smoke       plugin manifest + in-process MCP smoke"
	@echo "  make bench       run the full 3-arm benchmark to benchmarks/results"
	@echo "  make stats       statistics over benchmarks/results -> stats.json"
	@echo "  make figures     matplotlib figures into paper/figures + presentation/figures"
	@echo "  make autoreport  fill paper/autoreport.tex + main.tex placeholders"
	@echo "  make paper       run latexmk in paper/ (requires latex toolchain)"
	@echo "  make present     open presentation/index.html"
	@echo "  make all         smoke -> bench -> stats -> figures -> autoreport -> paper"

install:
	uv sync --extra dev

test:
	uv run pytest -q -m 'not slow and not real_model'

lint:
	uv run ruff check .

type:
	uv run mypy --strict

smoke:
	uv run python scripts/verify_plugin.py

bench:
	uv run python benchmarks/driver.py \
	    --n-poetry-gen 12 --n-poetry-interp 10 \
	    --n-aut 8 --n-sci-creativity 8 \
	    --max-tokens 120 --K 4 \
	    --out-dir benchmarks/results

stats:
	uv run python benchmarks/stats.py

benchmark.pilot:
	uv run python benchmarks/driver.py --pilot \
	    --K 3 --max-tokens 150 \
	    --cost-cap-usd 18.0 \
	    --out-dir benchmarks/results_v2

stats.pilot:
	uv run python benchmarks/stats.py \
	    --results-dir benchmarks/results_v2 \
	    --out benchmarks/results_v2/stats.json \
	    --treatment haiku_cascade --control haiku_bare

figures:
	uv run python benchmarks/figures.py

autoreport:
	uv run python benchmarks/autoreport.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

present:
	@echo "Open: file://$(PWD)/presentation/index.html"

clean:
	rm -rf benchmarks/_synth benchmarks/results paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/main.pdf

all: smoke bench stats figures autoreport
	@echo "Pipeline complete. Run 'make paper' to compile the PDF if a TeX toolchain is installed."
