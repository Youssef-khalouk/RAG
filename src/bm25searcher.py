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
        self._translation1 = str.maketrans({"-": " ", ".": " ", "'": " "})
        self._translation2 = str.maketrans("", "", string.punctuation)
        self._translation1_code = str.maketrans(
            {"_": " ", ".": " ", '"': ' ',
             "'": " ", ":": " ", "(": " ", ")": " "})
        self._translation2_code = str.maketrans("", "", string.punctuation)

    def set_top_k(self, size: int) -> None:
        if size <= 0:
            print("top_k size should be more then 0!")
            sys.exit(1)
        self.top_k = size

    # am not using this one for now tell i see whats going on ->
    def _preprocess_doc(self, text: str) -> list[str]:
        return (text.lower()
                .translate(self._translation1)
                .translate(self._translation2)
                .split())

    def _preprocess_code(self, text: str) -> str:
        return (text.lower()
                .translate(self._translation1_code)
                .translate(self._translation2_code)
                .split())

    def set_text_documents(self, documents: list[dict]) -> None:
        self.text_documents = documents
        self.chunks = []
        for d in documents:
            self.chunks.append(d["text"])
        self._tokenized_text_docs = [
            self._preprocess_doc(chunk) for chunk in self.chunks]
        self.bm25_text = BM25Okapi(self._tokenized_text_docs)

    def set_code_documents(self, documents: list[dict]) -> None:
        self.code_documents = documents
        self.chunks = []
        for d in documents:
            self.chunks.append(d["text"])
        self._tokenized_code_docs = [
            self._preprocess_code(chunk) for chunk in self.chunks]
        self.bm25_code = BM25Okapi(self._tokenized_code_docs)

    def query(self, query: str, type_flag: str = "doc") -> list[dict]:
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents yet to call query!")
            sys.exit(1)

        if type_flag == "doc":
            scores = self.bm25_text.get_scores(self._preprocess_doc(query))
            documents = self.text_documents
        else:
            scores = self.bm25_code.get_scores(self._preprocess_code(query))
            documents = self.code_documents

        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)[:self.top_k]

        code_results = []
        for index in top_indexes:
            code_results.append(documents[index])
        return code_results
