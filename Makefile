.PHONY: help install run debug clean lint lint-strict

export HF_HOME=/tmp/hf_home
export UV_CACHE_DIR=/tmp/uv_cache_dir
export UV_PROJECT_ENVIRONMENT=/tmp/uv_venv

# parameters =    --functions_definition data/input/functions_definition.json \
# 				--input data/input/function_calling_tests.json \
# 				--output data/output/function_calls.json

help:
	@echo "Available targets:"
	@echo "  make install      Install project dependencies"
	@echo "  make run          Run the application"
	@echo "  make runs         Run the application with no parameters"
	@echo "  make debug        Run the application with pdb"
	@echo "  make clean        Remove cache and temporary files"
	@echo "  make lint         Run flake8 and mypy with required flags"

install:
	uv sync
	@echo
	@echo "To activate the virtual environment"
	@echo "  run:  source /tmp/uv_venv/bin/activate"

run:
	uv run python -m  src

runs:
	uv run python -m  src

debug:
	uv run python -m pdb src
# 	uv run python -m pdb src $(parameters)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf uv.lock

lint:
	flake8 src
	mypy src    --warn-return-any \
	            --warn-unused-ignores \
	            --ignore-missing-imports \
	            --disallow-untyped-defs \
	            --check-untyped-defs
