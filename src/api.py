from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .bm25searcher import BM25Searcher
from .LLM import LLM
from .validaters import (MinimalSource,
                         MinimalSearchResults,
                         MinimalAnswer,
                         ChunksResults,
                         ChunkInfo)


app = FastAPI(
    title="RAG against the machine",
    description=("Local HTTP API for querying the index"
                 " and generating answers."),
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Loaded once at startup
_searcher = BM25Searcher()
_searcher.load_bm25_cache()
LLM._init_model()


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.get("/search_content", response_model=ChunksResults)
def search_content(
    query: str = Query(..., description="the search conent"),
    k: int = Query(10, description="Number of top results to return")
        ) -> ChunksResults:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive.")
    try:
        documents = _searcher.query(query, top_k=k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    chunks = []
    for d in documents:
        chunks.append(ChunkInfo(file_path=d[1], text_range=d[3], text=d[0]))
    return ChunksResults(search_results=chunks, k=k)


@app.get("/search", response_model=MinimalSearchResults)
def search(
    query: str = Query(..., description="The search query"),
    k: int = Query(10, description="Number of top results to return"),
) -> MinimalSearchResults:
    """Return the top-k retrieved sources for a single query."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive.")
    try:
        documents = _searcher.query(query, top_k=k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    retrieved_sources = []
    for d in documents:
        doc = _searcher.get_document(d[1], d[2])
        retrieved_sources.append(
            MinimalSource(
                file_path=doc["file_path"],
                first_character_index=doc["first_character_index"],
                last_character_index=doc["last_character_index"],
            )
        )
    return MinimalSearchResults(
        question_id="http-query",
        question=query,
        retrieved_sources=retrieved_sources,
    )


@app.get("/answer", response_model=MinimalAnswer)
def answer(
    query: str = Query(..., description="The question to answer"),
    k: int = Query(10,
                   description="Number of top sources to retrieve as context"),
        ) -> MinimalAnswer:
    """Retrieve sources for a query and generate a grounded answer."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if k <= 0:
        raise HTTPException(
            status_code=400, detail="k must be a positive integer.")

    try:
        documents = _searcher.query(query, top_k=k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    retrieved_sources = []
    for d in documents:
        doc = _searcher.get_document(d[1], d[2])
        retrieved_sources.append(
            MinimalSource(
                file_path=doc["file_path"],
                first_character_index=doc["first_character_index"],
                last_character_index=doc["last_character_index"],
            )
        )
    try:
        context = LLM.get_context(documents)
        generated_answer = LLM.ask(query, context=context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"Answer generation failed: {exc}")

    return MinimalAnswer(
        question_id="http-query",
        question=query,
        retrieved_sources=retrieved_sources,
        answer=generated_answer,
    )
