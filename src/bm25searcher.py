from rank_bm25 import BM25Okapi
from .process import Process
from typing import Any
import sys


class BM25Searcher:
    def __init__(self):
        self.text_documents: dict = None
        self.code_documents: dict = None
        self.doc_chunks: list[str] = []
        self.code_chunks: list[str] = []
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

    def get_document(self, path: str, chunk: int) -> dict | None:
        doc = self.text_documents.get(path, None)
        if doc is None:
            doc = self.code_documents.get(path, None)
        if doc is None:
            return None
        chunk_doc = doc["chunks"][chunk]
        d = {}
        d["file_path"] = path.replace("\\", "/")
        d["first_character_index"] = chunk_doc["start_index"]
        d["last_character_index"] = chunk_doc["end_index"]
        return d

    def set_text_documents(self, documents: list[dict]) -> None:
        self.text_documents = documents
        self.doc_chunks = []
        for k, v in documents.items():
            for chunk in v["chunks"]:
                self.doc_chunks.append([chunk["processed_text"], k, chunk["chunk"]])
        self._tokenized_text_docs = [chunk[0].split() for chunk in self.doc_chunks]
        self.bm25_text = BM25Okapi(self._tokenized_text_docs)

    def set_code_documents(self, documents: list[dict]) -> None:
        self.code_documents = documents
        self.code_chunks = []
        for k, v in documents.items():
            for chunk in v["chunks"]:
                self.code_chunks.append([chunk["processed_text"], k, chunk["chunk"]])
        self._tokenized_code_docs = [chunk[0].split() for chunk in self.code_chunks]
        self.bm25_code = BM25Okapi(self._tokenized_code_docs)

    def query(self, query: str, type_flag: str = "doc") -> list[dict]:
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents yet to call query!")
            sys.exit(1)

        if type_flag == "doc":
            scores = self.bm25_text.get_scores(Process.preprocess_doc(query))
            documents = self.doc_chunks
        else:
            scores = self.bm25_code.get_scores(Process.preprocess_code(query))
            documents = self.code_chunks

        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)[:self.top_k]

        code_results = []
        for index in top_indexes:
            code_results.append(documents[index])
        return code_results
