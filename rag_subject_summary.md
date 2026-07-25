# RAG Against the Machine — CLI, Edge Cases & Scoring Summary

## CLI commands & parameters (Chapter VI.6)

Your CLI must be built with **Python Fire**, invoked as:

```
uv run python -m src <command> [options]
```

Six commands are required:

| Command | Required parameters | Purpose |
|---|---|---|
| `index` | `--max_chunk_size <int>` (default 2000) | Ingest `data/raw/` and build the index under `data/processed/` |
| `search` | `<query> --k <int>` | Return top-k sources for a single query |
| `search_dataset` | `--dataset_path <path> --k <int> --save_directory <dir>` | Run search over a whole dataset, write a `StudentSearchResults` JSON |
| `answer` | `<query> --k <int>` | Answer a single query using retrieved context |
| `answer_dataset` | `--student_search_results_path <path> --save_directory <dir>` | Generate answers for a dataset, write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate` | `--student_search_results_path <path> --dataset_path <path>` | Report your **own** recall@k for iteration (not the official grading tool) |

**Key constraint:** all input/output paths must be CLI arguments, **never hardcoded** — the evaluator will point them at its own datasets/output directories.

---

## Edge cases you must handle gracefully (no crashes)

The subject explicitly states the CLI "is tested with such edge cases and must never crash with an unhandled traceback":

- Empty query
- Nonsensical query
- `k=0`
- Missing files
- Malformed JSON

Combine this with the **General Rules** (V.1):

- Use `try-except` around anything that can fail.
- Prefer **context managers** for files/connections so resources are always released.
- Add type hints (via `typing`, checked with `mypy`) + docstrings (PEP 257) everywhere.
- All functions must pass `mypy` without errors; code must respect `flake8`.

---

## Scoring vs. Resources — two separate topics

### Scoring / Evaluation (Chapter VII)

- Your own `evaluate` command is just for **your iteration** — it computes recall@k against a ground-truth dataset so you can tune your retriever.
- The **official** score at defense time is computed by the provided `moulinette` executable:

  ```
  ./moulinette evaluate_student_search_results \
      <student_results.json> <ground_truth.json> \
      --k 10 --max_context_length 2000
  ```

  **Your solution must never import or call the moulinette.**
- **Metric:** recall@k = share of a question's correct sources found in your top-k results, where "correct" means:
  - same `file_path` (**exact match**, compared verbatim)
  - character-range overlap with IoU ≥ 0.05 (low bar — doesn't need to match exactly)
- **Thresholds:**
  - ≥ 80% recall@5 on **docs** questions
  - ≥ 50% recall@5 on **code** questions
  - Indexing ≤ 5 minutes for the whole corpus
  - Retrieval ≤ 90 seconds for 200 questions

### Resources — two different meanings in the subject

1. **README "Resources" section** (Chapter VIII): must list classic references (documentation, articles, tutorials) related to RAG, plus a description of how AI was used — specifying which tasks and which parts of the project.

2. **System resources** (V.1, General Rules): file handles, network/model connections, etc. must be properly managed — use context managers (`with ...`) so nothing leaks, especially since you're loading a model (Qwen3-0.6B) and reading many files during indexing.

*(If "resources" meant something else, e.g. the bonus part or `data/raw` source files, let me know and I can expand that section.)*
