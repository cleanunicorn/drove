.PHONY: install test lint fmt typecheck completions

install:
	uv tool install . --force

completions: install
	vllama completions install

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src/
