PUBLIC_DOC_PATH      = "data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
PUBLIC_CODE_PATH     = "data/datasets_public/public/UnansweredQuestions/dataset_code_public.json"
PRIVATE_DOC_PATH     = "data/datasets_private/private/UnansweredQuestions/dataset_docs_private.json"
PRIVATE_CODE_PATH    = "data/datasets_private/private/UnansweredQuestions/dataset_code_private.json"

PUBLIC_ANSWERD_DOC   = "data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
PUBLIC_ANSWERD_CODE  = "data/datasets_public/public/AnsweredQuestions/dataset_code_public.json"
PRIVATE_ANSWERD_DOC  = "data/datasets_private/private/AnsweredQuestions/dataset_docs_private.json"
PRIVATE_ANSWERD_CODE = "data/datasets_private/private/AnsweredQuestions/dataset_code_private.json"

OUTPUT_FILE          = "data/output.json"
M_PARAMETERS         = --k 10 --max_context_length 2000
MOULINETTE           = ./data/moulinette/moulinette-ubuntu evaluate_student_search_results
P_PARAMETERS         = --save_directory=$(OUTPUT_FILE) --k 10 --max_chunk_size 2000

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Setup:"
	@echo "  install              Install project dependencies"
	@echo "  download             Download datasets, vLLM, and moulinette"
	@echo
	@echo "Run:"
	@echo "  run                  Run application with default parameters"
	@echo "  run_public_doc       Run on public documentation dataset"
	@echo "  run_public_code      Run on public code dataset"
	@echo "  run_private_doc      Run on private documentation dataset"
	@echo "  run_private_code     Run on private code dataset"
	@echo
	@echo "Evaluation:"
	@echo "  moulinette_public_doc     Evaluate public documentation results"
	@echo "  moulinette_public_code    Evaluate public code results"
	@echo "  moulinette_private_doc    Evaluate private documentation results"
	@echo "  moulinette_private_code   Evaluate private code results"
	@echo
	@echo "Combined run + evaluation:"
	@echo "  public_doc           Run + evaluate public documentation"
	@echo "  public_code          Run + evaluate public code"
	@echo "  private_doc          Run + evaluate private documentation"
	@echo "  private_code         Run + evaluate private code"
	@echo
	@echo "Development:"
	@echo "  debug                Run application with pdb"
	@echo "  lint                 Run flake8 and mypy checks"
	@echo "  clean                Remove caches and generated files"

export HF_HOME=/tmp/hf_home
export UV_CACHE_DIR=/tmp/uv_cache_dir
export UV_PROJECT_ENVIRONMENT=/tmp/uv_venv

install:
	uv sync
	@echo
	@echo "To activate the virtual environment"
	@echo "  run:  source /tmp/uv_venv/bin/activate"

run:
	uv run python -m src

run_public_doc:
	uv run python -m src --dataset_path=$(PUBLIC_DOC_PATH) $(P_PARAMETERS)
run_public_code:
	uv run python -m src --dataset_path=$(PUBLIC_CODE_PATH) $(P_PARAMETERS)
run_private_doc:
	uv run python -m src --dataset_path=$(PRIVATE_DOC_PATH) $(P_PARAMETERS)
run_private_code:
	uv run python -m src --dataset_path=$(PRIVATE_CODE_PATH) $(P_PARAMETERS)

moulinette:
	$(MOULINETTE)

moulinette_public_doc:
	$(MOULINETTE) $(OUTPUT_FILE) $(PUBLIC_ANSWERD_DOC) $(M_PARAMETERS)
moulinette_public_code:
	$(MOULINETTE) $(OUTPUT_FILE) $(PUBLIC_ANSWERD_CODE) $(M_PARAMETERS)
moulinette_private_doc:
	$(MOULINETTE) $(OUTPUT_FILE) $(PRIVATE_ANSWERD_DOC) $(M_PARAMETERS)
moulinette_private_code:
	$(MOULINETTE) $(OUTPUT_FILE) $(PRIVATE_ANSWERD_CODE) $(M_PARAMETERS)

public_doc: run_public_doc moulinette_public_doc
public_code: run_public_code moulinette_public_code
private_doc: run_private_doc moulinette_private_doc
private_code: run_private_code moulinette_private_code

download:
	mkdir -p data
	curl -L -o datasets_private.zip https://cdn.intra.42.fr/document/document/54697/datasets_private.zip
	unzip -o datasets_private.zip -d data/datasets_private
	rm -rf datasets_private.zip

	curl -L -o datasets_public.zip https://cdn.intra.42.fr/document/document/55367/datasets_public.zip
	unzip -o datasets_public.zip -d data
	rm -rf datasets_public.zip

	curl -L -o vllm-0.10.1.zip https://cdn.intra.42.fr/document/document/55369/vllm-0.10.1.zip
	unzip -o vllm-0.10.1.zip -d data/raw
	rm -rf vllm-0.10.1.zip
	
	curl -L -o moulinette.zip https://cdn.intra.42.fr/document/document/55370/moulinette.zip
	unzip -o moulinette.zip -d data/moulinette
	rm -rf moulinette.zip

debug:
	uv run python -m pdb src

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

.PHONY: \
help \
install \
run \
run_public_doc \
run_public_code \
run_private_doc \
run_private_code \
moulinette \
moulinette_public_doc \
moulinette_public_code \
moulinette_private_doc \
moulinette_private_code \
public_doc \
public_code \
private_doc \
private_code \
download \
debug \
clean \
lint