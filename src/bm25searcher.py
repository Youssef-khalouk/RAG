from rank_bm25 import BM25Okapi
import string
from typing import Any
import sys


class BM25Searcher:
    def __init__(self):
        self.documents: list[dict] = None
        self.chunks: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        self.bm25: Any = None
        self.top_k = 5

    def set_top_k(self, size: int) -> None:
        if size <= 0:
            print("top_k size should be more then 0!")
            sys.exit(1)
        self.top_k = size

    # am not using this one for now tell i see whats going on ->
    def _preprocess(self, text: str) -> str:
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.lower()

    def set_documents(self, documents: list[dict]) -> None:
        self.documents = documents
        self.chunks = []
        for d in documents:
            self.chunks.append(d["text"])
            # if d["path"].endswith(".txt") or d["path"].endswith(".md"):
            #     self.chunks.append(self._preprocess(d["text"]))
            # else:
            #     self.chunks.append(d["text"])

        self._tokenized_docs = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_docs)

    def query(self, query: str) -> list[dict]:
        if self.documents is None:
            print("there is no documents yet to call query!")
            sys.exit(1)

        # scores = self.bm25.get_scores(self._preprocess(query).split())
        scores = self.bm25.get_scores(query.split())

        # Get indexes of highest scores
        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.top_k]

        results = []
        for index in top_indexes:
            if scores[index] <= 0:
                continue
            results.append(self.documents[index])
        return results
