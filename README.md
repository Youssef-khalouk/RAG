*This project has been created as part of the 42 curriculum by ykhalouk.

# RAG Against the Machine — A Retrieval-Augmented Generation (RAG) System

## Description

This project is a command-line Retrieval-Augmented Generation (RAG) system built to index a
corpus of source material — here, the documentation and codebase of **vLLM 0.10.1**
— and answer natural-language questions about it.

The project is split into three cooperating stages:

1. **Indexing** — scan a directory, chunk every text (`.md`, `.txt`) and code (`.py`)
   file, and persist the chunks to disk.
2. **Retrieval** — build a hybrid lexical + semantic search index over those chunks
   and, given a query, return the most relevant ones.
3. **Generation** — feed the retrieved chunks as context to a local language model
   (Qwen3-0.6B) so it can produce a grounded answer instead of hallucinating one.

The goal of the project is to understand — hands-on, without relying on a managed
RAG framework — how each piece of a retrieval pipeline works: document chunking,
lexical ranking (BM25), dense embeddings, hybrid fusion, and finally answer
generation, along with how to measure the quality of retrieval using recall metrics.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running the
  project (the Makefile drives everything through `uv run` / `uv sync`)
- Packages (managed via `uv`/`pyproject.toml`): `fire`, `tqdm`, `pydantic`,
  `rank_bm25`, `sentence-transformers`, `langchain-text-splitters`,
  `transformers`, `torch`, `numpy`, `fastapi`, `uvicorn` (for the optional API server)

### Setup

```bash
make install
```

This runs `uv sync` and prints the command to activate the resulting virtual
environment (`source /tmp/uv_venv/bin/activate`, path varies by OS — see the
Makefile's `HF_HOME` / `UV_*` variables).

Then download the datasets and the raw vLLM 0.10.1 corpus:

```bash
make download
```

This pulls `datasets_private.zip`, `datasets_public.zip`, `vllm-0.10.1.zip`, and
`moulinette.zip` from the 42 intranet CDN and extracts them into `data/`.

### Building the index

```bash
make index
```

Equivalent to `uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)`
(`MAX_CHUNK_SIZE` defaults to `2000`, override with `make index MAX_CHUNK_SIZE=1500`).
This scans `data/raw/vllm-0.10.1`, chunks every changed file, builds the BM25 and
embedding indexes, and caches everything under `data/processed/`. Re-running the
command only reprocesses files that changed since the last run.

### Searching

```bash
make search QUERY="How does vLLM handle continuous batching?" K=10
make search_content QUERY="What is PagedAttention?" K=5
```

`search` prints the matching file paths and character ranges; `search_content`
also prints the retrieved chunk text. `K` and `QUERY` are Makefile variables with
defaults (`K=10`, `QUERY` defaults to a sample vLLM question — see the Makefile).

### Asking a question

```bash
make answer QUERY="What scheduling policy does vLLM use?" K=10
```

This retrieves context and asks the local LLM to answer using only that context.

### Batch evaluation workflow (public / private, docs / code)

```bash
make search_dataset_public_doc      # or _public_code / _private_doc / _private_code
make answer_dataset_public_doc      # or the matching _public_code / _private_doc / _private_code
make evaluate_public_doc            # or the matching _public_code / _private_doc / _private_code
```

`search_dataset_*` runs retrieval over one of the four `UnansweredQuestions`
datasets (public/private × docs/code) and saves results to
`data/output/search_results/UnansweredQuestions`. `answer_dataset_*` generates
answers for the corresponding saved search-results file into
`data/output/search_results_and_answer/UnansweredQuestions`. `evaluate_*` compares
retrieved sources against the matching `AnsweredQuestions` ground-truth dataset to
compute Recall@1/3/5/10.

### Official grading (moulinette)

```bash
make moulinette_public_doc      # or _public_code / _private_doc / _private_code
```

Runs the project's official evaluation binary
(`data/moulinette/moulinette-ubuntu evaluate_student_search_results`) against the
search results, using the same `K` / `MAX_CHUNK_SIZE` Makefile variables.

### Running directly with `python -m`

Every `make` target above is a thin wrapper around the underlying CLI, which can
also be called directly:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search --k 10 --query "How does vLLM handle continuous batching?"
uv run python -m src search_content --k 5 --query "What is PagedAttention?"
uv run python -m src answer --query "What scheduling policy does vLLM use?" --k 10
uv run python -m src search_dataset --dataset_path=<path> --k 10 --save_directory <dir>
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>
uv run python -m src evaluate <search_results_dir> <answered_dataset_path>
```

### Makefile command reference

| Target | Description |
|---|---|
| `make install` | Install project dependencies via `uv sync` |
| `make download` | Download datasets, the vLLM 0.10.1 corpus, and moulinette from the 42 CDN |
| `make index` | Build the retrieval index from `data/raw/` (`MAX_CHUNK_SIZE` variable) |
| `make search` | Run a single query search and print matching source locations (`QUERY`, `K` variables) |
| `make search_content` | Run a single query search and print the retrieved chunk content |
| `make search_dataset_public_doc` / `_public_code` / `_private_doc` / `_private_code` | Run retrieval over the corresponding public/private, docs/code dataset |
| `make answer` | Answer a single query with the local LLM (`QUERY`, `K` variables) |
| `make answer_dataset_public_doc` / `_public_code` / `_private_doc` / `_private_code` | Generate answers for the corresponding saved search-results file |
| `make evaluate_public_doc` / `_public_code` / `_private_doc` / `_private_code` | Compute Recall@1/3/5/10 against the corresponding ground-truth dataset |
| `make moulinette_public_doc` / `_public_code` / `_private_doc` / `_private_code` | Run the official 42 moulinette evaluation binary |
| `make run` | Run the application (`uv run python -m src`) |
| `make server` | Start the FastAPI server (`uvicorn src.api:app`) on `127.0.0.1:8000` |
| `make debug` | Run the application under `pdb` |
| `make lint` | Run `flake8` and `mypy` checks on `src` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, `.pytest_cache`, and `.pyc` files |
| `make clean_cache` | Remove generated index (`data/processed`) and output directories |
| `make help` | Print a summary of all available targets |

## System Architecture

The pipeline is composed of four cooperating modules, orchestrated by a `fire`-based
CLI (`__main__.py`):

```
                ┌────────────┐
   raw files → │  UploadDir  │ → chunked text/code documents (JSON)
                └────────────┘
                       │
                       ▼
                ┌────────────────┐
                │ RetrievalEngine │ → BM25 index + dense embeddings (pickled cache)
                └────────────────┘
                       │  query(question)
                       ▼
                ┌────────────┐
                │     LLM     │ → grounded answer (Qwen3-0.6B, cached)
                └────────────┘
```

- **`UploadDir`** walks the raw data directory, chunks every `.py`/`.md`/`.txt` file
  with the appropriate `langchain_text_splitters.RecursiveCharacterTextSplitter`,
  records each chunk's character offsets, and persists the result as
  `data/processed/doc_documents.json` and `data/processed/code_documents.json`.
  It tracks each file's last-modified time so only changed files are reprocessed.

- **`RetrievalEngine`** consumes those chunk dictionaries and builds three BM25
  indexes (`text`, `code`, `both`) with `rank_bm25.BM25Okapi`, plus dense embeddings
  for each chunk using `sentence-transformers` (`paraphrase-MiniLM-L3-v2`). All
  indexes/embeddings are cached to `data/processed/*.pkl` so subsequent runs load
  instantly instead of recomputing.

- **`LLM`** wraps a local causal language model (`Qwen/Qwen3-0.6B` via
  `transformers`). It assembles a context string from the retrieved chunks (capped
  at `max_context_characters`), builds a chat prompt instructing the model to answer
  only from the given context (or say "I don't know"), and generates a response.
  Answers are cached (keyed by question + context) to avoid recomputation.

- **CLI layer (`__main__.py`)** exposes `index`, `search`, `search_content`,
  `search_dataset`, `answer`, `answer_dataset`, and `evaluate` commands via
  [`fire`](https://github.com/google/python-fire), each validating its inputs
  (`k`, `max_chunk_size`, non-empty queries) before delegating to the modules above.

Validation and I/O contracts between stages (dataset format, search-results format,
answer format) are enforced with `pydantic` models (`RagDataset`, `MinimalSource`,
`StudentSearchResults`, `MinimalAnswer`, etc.), so intermediate JSON files produced
by one command can be safely consumed by the next.

## Chunking Strategy

Chunking is handled by `UploadDir` using three specialized
`RecursiveCharacterTextSplitter` instances, one per file type:

| File type | Splitter | Separators |
|---|---|---|
| `.txt` | `splitter_txt` | `[".\n", "\n", " ", ""]` |
| `.md`  | `splitter_md`  | `["\n# ", ".\n", "\n", " ", ""]` (splits on headings first) |
| `.py`  | `splitter_code` (`Language.PYTHON`) | Python-aware separators (functions, classes, blocks) |

Design choices:

- **No overlap** (`chunk_overlap=0`) between chunks was chosen to keep the index
  compact and avoid duplicated content being retrieved twice for the same source
  span.
- **Configurable chunk size** (`max_chunk_size`, bounded between 200 and 2000
  characters) lets us trade off between finer-grained, more precise retrieval
  (small chunks) and richer context per chunk (larger chunks).
- **Start-index tracking** (`add_start_index=True`) records the exact character
  offsets of each chunk in its source file. This is what allows `evaluate` to
  compare retrieved chunks against ground-truth spans, and lets `answer_dataset`
  re-slice the original file to recover exact chunk text without storing it twice.
- **Markdown-aware splitting**: `.md` files split on `"\n# "` first, so top-level
  sections stay intact as long as they fit in the chunk budget, before falling back
  to sentence/paragraph/word-level splitting.
- **Path-aware embeddings**: for every chunk, a small `path_tokens` string derived
  from the file path (its stem, with and without underscores) is prepended before
  computing the embedding, so the file's identity subtly influences the semantic
  representation of its chunks.

## Retrieval Method

`RetrievalEngine.query()` performs **hybrid retrieval**, combining two
complementary ranking signals:

1. **Lexical ranking (BM25)** — `rank_bm25.BM25Okapi`, run over tokenized,
   preprocessed text (`Process.preprocess_doc` / `preprocess_code`). BM25 is strong
   at matching exact keywords, identifiers, and rare terms (e.g. function names,
   error codes) that dense embeddings can under-weight.
2. **Semantic ranking (dense embeddings)** — cosine similarity between the query
   embedding and every chunk embedding, computed with
   `sentence-transformers/paraphrase-MiniLM-L3-v2`. This captures paraphrases and
   conceptual matches that don't share exact wording with the query.

Separate indexes are maintained for **text-only**, **code-only**, and **both**
combined, selectable via a `type_flag` (`doc` / `code` / default = both), and the
CLI automatically infers this from the dataset filename in `search_dataset`.

**Fusion strategy**: for a requested `top_k`, the engine takes the top `top_k // 2`
results from BM25 and fills the remaining slots from the embedding results,
deduplicating chunks that appear in both lists by their `(file_path, chunk_index)`
key. This is a simple **rank-interleaving** fusion (rather than score normalization
/ RRF), chosen for its simplicity and because BM25 and embedding-cosine scores are
not on a comparable scale.

Both `query_bm25` and `query_embeddings` (and `query` itself) are memoized with
`functools.lru_cache`, so repeated queries during evaluation runs are fast.

## Performance Analysis

Retrieval quality is measured with `evaluate`, which computes **Recall@k** for
k ∈ {1, 3, 5, 10} over a dataset of questions with known ground-truth source spans.

A retrieved chunk counts as a hit if it overlaps the ground-truth span with an
**Intersection-over-Union (IoU) ≥ 0.05** — a deliberately loose threshold, since a
chunk only needs to substantially cover part of the answer's source location to be
useful to the downstream LLM.

```
📈 Recall@1:  XX%
📈 Recall@3:  XX%
📈 Recall@5:  XX%
📈 Recall@10: XX%
```

*(Replace the placeholders above with the actual scores obtained by running
`make evaluate_public_doc` (or the matching `_public_code` / `_private_doc` /
`_private_code` target, or `uv run python -m src evaluate <search_results_dir>
<dataset_path>` directly) on your evaluation set, and add any discussion of how
chunk size, `k`, or the BM25/embedding mix affected these numbers.)*

Observed trends worth noting in this kind of hybrid setup:
- Recall improves monotonically with `k`, as expected, but plateaus once the
  correct chunk is reliably in the top BM25 or embedding results.
- BM25 tends to dominate for code-heavy questions containing exact identifiers;
  embeddings help most on conceptual/paraphrased questions.
- Smaller `max_chunk_size` values increase the number of chunks (and index build
  time) but can improve `IoU` overlap precision against narrow ground-truth spans.

## Design Decisions

- **Hybrid retrieval over single-method retrieval**: combining BM25 and embeddings
  hedges against each method's weaknesses (BM25 misses paraphrases, embeddings miss
  exact identifiers).
- **File-based caching at every stage**: chunked documents (`data/processed/*.json`),
  BM25 indexes and embeddings (`data/processed/*.pkl`), and even LLM answers
  (`llm_cache.pkl`) are all persisted, so re-running commands is fast and
  incremental — only changed files or uncached (question, context) pairs trigger
  real work.
- **Small local LLM (Qwen3-0.6B)**: chosen to keep the system runnable on
  commodity/CPU hardware without external API costs, at the cost of answer quality
  compared to larger hosted models.
- **`pydantic` models for all JSON I/O**: guarantees that data passed between the
  `search_dataset` → `answer_dataset` → `evaluate` pipeline is well-formed and
  fails fast (with a clear error) on malformed input rather than crashing deep in
  the pipeline.
- **`fire` for the CLI**: keeps the command surface declarative — each Python
  function becomes a CLI command automatically, with argument parsing handled for
  us.
- **Change detection via file mtimes**: avoids reprocessing the entire corpus on
  every `index` call, which matters once the corpus (a full codebase + docs) grows
  large.

## Challenges Faced

- **Aligning chunk offsets across stages**: since chunks are only referenced by
  `(file_path, start_index, end_index)` rather than storing full text everywhere,
  keeping these offsets exactly consistent between indexing, retrieval, and
  evaluation (and slicing the original file correctly, off-by-one on the start
  index) required care.
- **Mixing text and code corpora**: text and code need different tokenization and
  splitting strategies, which led to maintaining fully separate BM25
  indexes/splitters for each while still supporting a combined "both" index.
- **Fusing two differently-scaled ranking signals**: BM25 scores and cosine
  similarities aren't directly comparable, so a rank-based interleaving approach
  was used instead of trying to normalize and sum raw scores.
- **Cache invalidation**: detecting when the underlying corpus or chunk size has
  changed (so caches must be rebuilt) versus when they haven't (so cached indexes
  can be reused) required tracking both file modification times and the configured
  chunk size (`k`) inside the persisted JSON documents.
- **Running a local LLM efficiently**: without a GPU, generation is slow, so
  answers are cached by `(question, context)` to avoid ever re-generating the same
  answer twice.

## Example Usage

```bash
# 0. Install dependencies and download the vLLM corpus + datasets
make install
make download

# 1. Build the index from the raw vLLM corpus
make index

# 2. Search for relevant chunks
make search QUERY="How is KV cache managed?"

# 3. Ask a direct question
make answer QUERY="What is continuous batching in vLLM?"

# 4. Run retrieval + generation + evaluation over the public docs dataset
make search_dataset_public_doc
make answer_dataset_public_doc
make evaluate_public_doc

# Equivalent, called directly without make:
uv run python -m src index
uv run python -m src search --query "How is KV cache managed?"
uv run python -m src answer --query "What is continuous batching in vLLM?"
```

## Resources

- [BM25 (Okapi) — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [rank_bm25 documentation](https://github.com/dorianbrown/rank_bm25)
- [Sentence-Transformers documentation](https://www.sbert.net/)

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [vLLM documentation](https://docs.vllm.ai/)

- [Chatgpt and Claude]()


### Use of AI

AI assistant was used during this project as follows:
 
- **Concept explanations**: explaining how underlying techniques work (e.g. BM25
  scoring, cosine similarity over embeddings).
- **Debugging support**: explaining unfamiliar or unexpected error messages and
  stack traces.
- **Documentation**: helping with structuring this README.
- **Performance**: asking AI for suggestions on how to speed up parts of the
  code (e.g. caching, batching, avoiding repeated recomputation).
