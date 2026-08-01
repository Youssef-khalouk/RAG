PUBLIC_DOC_PATH      = "data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
PUBLIC_CODE_PATH     = "data/datasets_public/public/UnansweredQuestions/dataset_code_public.json"
PRIVATE_DOC_PATH     = "data/datasets_private/private/UnansweredQuestions/dataset_docs_private.json"
PRIVATE_CODE_PATH    = "data/datasets_private/private/UnansweredQuestions/dataset_code_private.json"

PUBLIC_ANSWERD_DOC   = "data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
PUBLIC_ANSWERD_CODE  = "data/datasets_public/public/AnsweredQuestions/dataset_code_public.json"
PRIVATE_ANSWERD_DOC  = "data/datasets_private/private/AnsweredQuestions/dataset_docs_private.json"
PRIVATE_ANSWERD_CODE = "data/datasets_private/private/AnsweredQuestions/dataset_code_private.json"

MOULINETTE           = ./data/moulinette/moulinette-ubuntu evaluate_student_search_results
DATASET_OUTPUT       = data/dataset_output.json
ANSWERD_DATASET		 = data/answer_dataset_results.json
K                   ?= 10
MAX_CHUNK_SIZE      ?= 2000
M_PARAMETERS         = --k $(K) --max_context_length $(MAX_CHUNK_SIZE)
QUERY               ?= What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?

ifeq ($(OS),Windows_NT)
    # Windows
	export HF_HOME=C:/rag_env/hf_home
	export UV_CACHE_DIR=C:/rag_env/uv_cache_dir
	export UV_PROJECT_ENVIRONMENT=C:/rag_env/uv_venv

else
    ifneq ($(wildcard /mnt/c/RAG),)
        # WSL
		export HF_HOME=$(HOME)/.cache/hf_home
		export UV_CACHE_DIR=$(HOME)/.cache/uv_cache_dir
		export UV_PROJECT_ENVIRONMENT=$(HOME)/.cache/uv_venv

    else
        # Linux cluster
		export HF_HOME=/tmp/hf_home
		export UV_CACHE_DIR=/tmp/uv_cache_dir
		export UV_PROJECT_ENVIRONMENT=/tmp/uv_venv
    endif
endif

install:
	uv sync
	@echo
	@echo "To activate the virtual environment"
	@echo "  run:  source /tmp/uv_venv/bin/activate"

run:
	uv run python -m src

index:
	uv run python -m src index  --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search --k $(K) --query "$(QUERY)"

search_content:
	uv run python -m src search_content --k $(K) --query "$(QUERY)"

search_dataset_public_doc:
	uv run python -m src search_dataset  --save_directory $(DATASET_OUTPUT) --k $(K) --dataset_path=$(PUBLIC_DOC_PATH)
search_dataset_public_code:
	uv run python -m src search_dataset --save_directory $(DATASET_OUTPUT) --k $(K) --dataset_path=$(PUBLIC_CODE_PATH)
search_dataset_private_doc:
	uv run python -m src search_dataset --save_directory $(DATASET_OUTPUT) --k $(K) --dataset_path=$(PRIVATE_DOC_PATH)
search_dataset_private_code:
	uv run python -m src search_dataset --save_directory $(DATASET_OUTPUT) --k $(K) --dataset_path=$(PRIVATE_CODE_PATH)


answer:
	uv run python -m src answer --query "$(QUERY)" --k $(K)

answer_dataset:
	uv run python -m src answer_dataset --student_search_results_path $(DATASET_OUTPUT) --save_directory $(ANSWERD_DATASET)

evaluate_public_doc:
	uv run python -m src evaluate $(DATASET_OUTPUT) $(PUBLIC_ANSWERD_DOC)
evaluate_public_code:
	uv run python -m src evaluate $(DATASET_OUTPUT) $(PUBLIC_ANSWERD_CODE)
evaluate_private_doc:
	uv run python -m src evaluate $(DATASET_OUTPUT) $(PRIVATE_ANSWERD_DOC)
evaluate_private_code:
	uv run python -m src evaluate $(DATASET_OUTPUT) $(PRIVATE_ANSWERD_CODE)


moulinette_public_doc:
	$(MOULINETTE) $(DATASET_OUTPUT) $(PUBLIC_ANSWERD_DOC) $(M_PARAMETERS)
moulinette_public_code:
	$(MOULINETTE) $(DATASET_OUTPUT) $(PUBLIC_ANSWERD_CODE) $(M_PARAMETERS)
moulinette_private_doc:
	$(MOULINETTE) $(DATASET_OUTPUT) $(PRIVATE_ANSWERD_DOC) $(M_PARAMETERS)
moulinette_private_code:
	$(MOULINETTE) $(DATASET_OUTPUT) $(PRIVATE_ANSWERD_CODE) $(M_PARAMETERS)

server:
	uv run uvicorn src.api:app --host 127.0.0.1 --port 8000

download:
	mkdir -p data
	curl -L -o datasets_private.zip https://cdn.intra.42.fr/document/document/54697/datasets_private.zip
	curl -L -o datasets_public.zip https://cdn.intra.42.fr/document/document/55367/datasets_public.zip
	curl -L -o vllm-0.10.1.zip https://cdn.intra.42.fr/document/document/55369/vllm-0.10.1.zip
	curl -L -o moulinette.zip https://cdn.intra.42.fr/document/document/55370/moulinette.zip

	if [ "$(OS)" = "Windows_NT" ]; then \
		powershell -Command "Expand-Archive -Path datasets_private.zip -DestinationPath data/datasets_private -Force"; \
		powershell -Command "Expand-Archive -Path datasets_public.zip -DestinationPath data -Force"; \
		powershell -Command "Expand-Archive -Path vllm-0.10.1.zip -DestinationPath data/raw -Force"; \
		powershell -Command "Expand-Archive -Path moulinette.zip -DestinationPath data/moulinette -Force"; \
	else \
		unzip -o datasets_private.zip -d data/datasets_private; \
		unzip -o datasets_public.zip -d data; \
		unzip -o vllm-0.10.1.zip -d data/raw; \
		unzip -o moulinette.zip -d data/moulinette; \
	fi

	rm -rf datasets_private.zip
	rm -rf datasets_public.zip
	rm -rf vllm-0.10.1.zip
	rm -rf moulinette.zip

debug:
	uv run python -m pdb src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean_cache:
	rm -rf data/processed
	rm -rf $(DATASET_OUTPUT)
	rm -rf $(ANSWERD_DATASET)

lint:
	flake8 src
	mypy src    --warn-return-any \
	            --warn-unused-ignores \
	            --ignore-missing-imports \
	            --disallow-untyped-defs \
	            --check-untyped-defs

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Setup:"
	@echo "  install                      Install project dependencies"
	@echo "  download                     Download datasets, vLLM, and moulinette"
	@echo
	@echo "Indexing:"
	@echo "  index                        Build the index from data/raw/"
	@echo
	@echo "Search:"
	@echo "  search                       Run a single query search"
	@echo "  search_content                Run a single content search"
	@echo "  search_dataset_public_doc    Search over the public docs dataset"
	@echo "  search_dataset_public_code   Search over the public code dataset"
	@echo "  search_dataset_private_doc   Search over the private docs dataset"
	@echo "  search_dataset_private_code  Search over the private code dataset"
	@echo
	@echo "Answer generation:"
	@echo "  answer                       Answer a single query"
	@echo "  answer_dataset               Generate answers from the last search_dataset output"
	@echo
	@echo "Evaluation (moulinette):"
	@echo "  moulinette_public_doc        Evaluate public documentation results"
	@echo "  moulinette_public_code       Evaluate public code results"
	@echo "  moulinette_private_doc       Evaluate private documentation results"
	@echo "  moulinette_private_code      Evaluate private code results"
	@echo
	@echo "Development:"
	@echo "  run                          Run the application"
	@echo "  debug                        Run application with pdb"
	@echo "  lint                         Run flake8 and mypy checks"
	@echo "  clean                        Remove caches (__pycache__, .mypy_cache, etc.)"
	@echo "  clean_cache                  Remove generated index and output files"

.PHONY: \
	help \
	install \
	download \
	index \
	run \
	search \
	search_content \
	search_dataset_public_doc \
	search_dataset_public_code \
	search_dataset_private_doc \
	search_dataset_private_code \
	answer \
	answer_dataset \
	moulinette \
	moulinette_public_doc \
	moulinette_public_code \
	moulinette_private_doc \
	moulinette_private_code \
	debug \
	clean \
	clean_cache \
	lint