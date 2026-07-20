from rank_bm25 import BM25Okapi
import string
from typing import Any
import sys


class BM25Searcher:
    def __init__(self):
        self.text_documents: list[dict] = None
        self.code_documents: list[dict] = None
        self.chunks: list[str] = []
        self._tokenized_text_docs: list[list[str]] = []
        self._tokenized_code_docs: list[list[str]] = []
        self.bm25_text: Any = None
        self.bm25_code: Any = None
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

    def set_text_documents(self, documents: list[dict]) -> None:
        self.text_documents = documents
        self.chunks = []
        for d in documents:
            self.chunks.append(d["text"])
        self._tokenized_text_docs = [self._preprocess(chunk).split() for chunk in self.chunks]
        self.bm25_text = BM25Okapi(self._tokenized_text_docs)

    def set_code_documents(self, documents: list[dict]) -> None:
        self.code_documents = documents
        self.chunks = []
        for d in documents:
            self.chunks.append(d["text"])
        self._tokenized_code_docs = [chunk.split() for chunk in self.chunks]
        self.bm25_code = BM25Okapi(self._tokenized_code_docs)

    def query(self, query: str, type_flag: str = "text") -> list[dict]:
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents yet to call query!")
            sys.exit(1)
        scores = self.bm25_text.get_scores(self._preprocess(query).split())
        # Get indexes of highest scores
        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.top_k]

        text_results = []
        for index in top_indexes:
            if scores[index] <= 0:
                continue
            text_results.append(self.text_documents[index])

        scores = self.bm25_code.get_scores(query.split())
        # Get indexes of highest scores
        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.top_k]

        code_results = []
        for index in top_indexes:
            if scores[index] <= 0:
                continue
            code_results.append(self.code_documents[index])

        if type_flag == "text":
            return text_results
        else:
            return code_results
