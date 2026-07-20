.PHONY: help install run debug clean lint lint-strict moulinette moulinette2

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

run2:
	uv run python -m  src

moulinette:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output.json data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
moulinette2:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output.json data/datasets_public/public/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

moulinette_p:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output.json data/datasets_private/private/AnsweredQuestions/dataset_docs_private.json --k 10 --max_context_length 2000

all: run moulinette
all2: run2 moulinette2
all3: run moulinette_p

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
