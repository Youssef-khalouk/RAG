from rank_bm25 import BM25Okapi
from .process import Process
from typing import Any
import sys
import pickle
from pathlib import Path
import os


class BM25Searcher:
    def __init__(self):
        self.text_documents: dict = None
        self.code_documents: dict = None
        self.doc_chunks: list[str] = []
        self.code_chunks: list[str] = []
        self.both_chunks: list[str] = []
        self._tokenized_text_docs: list[list[str]] = []
        self._tokenized_code_docs: list[list[str]] = []
        self.bm25_text: Any = None
        self.bm25_code: Any = None
        self.bm25_both: Any = None
        self.top_k: int = 10

    def _save_bm25_cache(self) -> None:
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/bm25_text_cache.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25_text,
                "text_documents": self.text_documents,
                "doc_chunks": self.doc_chunks
                }, f)
        with open("data/processed/bm25_code_cache.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25_code,
                "code_documents": self.code_documents,
                "code_chunks": self.code_chunks
                }, f)
        with open("data/processed/bm25_both_cache.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25_both,
                "both_chunks": self.both_chunks
                }, f)

        print("bm25 cache saved successfully.")

    def set_top_k(self, size: int) -> None:
        if size <= 0:
            print("top_k size should be more then 0!")
            sys.exit(1)
        self.top_k = size

    def get_document_content(self, path: str, chunk: int) -> list[dict]:
        doc = self.text_documents.get(path, None)
        if doc is None:
            doc = self.code_documents.get(path, None)
        if doc is None:
            return {}
        chunk_doc = doc["chunks"][chunk]
        d = {}
        d["file_path"] = path.replace("\\", "/")
        d["first_character_index"] = chunk_doc["start_index"]
        d["last_character_index"] = chunk_doc["end_index"]
        d["text"] = chunk_doc["text"]
        return d

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

    def check_documents(self,
                        documents_text: list[dict],
                        documents_code: list[dict],
                        is_files_changed: bool) -> None:
        """this method does rebuild BM25 cache if its not exists."""
        # this function used when i need to check the cached file is exist
        # if not rebuild cache
        if not is_files_changed and self._is_cached_files_exist():
            return
        self.set_documents(documents_text, documents_code, is_files_changed)

    def create_bm25_cache_and_save(self,
                                   documents_text: list[dict],
                                   documents_code: list[dict],
                                   is_files_changed: bool) -> None:
        if not is_files_changed and self._is_cached_files_exist():
            return

        self.text_documents = documents_text
        self.code_documents = documents_code
        self.code_chunks = []
        for k, v in documents_code.items():
            if k == "k":
                continue
            for chunk in v["chunks"]:
                self.code_chunks.append(
                    [chunk["processed_text"], k, chunk["chunk"]])
        self.doc_chunks = []
        for k, v in documents_text.items():
            if k == "k":
                continue
            for chunk in v["chunks"]:
                self.doc_chunks.append(
                    [chunk["processed_text"], k, chunk["chunk"]])
        self.both_chunks = self.doc_chunks + self.code_chunks

        self._tokenized_text_docs = [
            chunk[0].split() for chunk in self.doc_chunks]
        self.bm25_text = BM25Okapi(self._tokenized_text_docs)
        self._tokenized_code_docs = [
            chunk[0].split() for chunk in self.code_chunks]
        self.bm25_code = BM25Okapi(self._tokenized_code_docs)
        self.bm25_both = BM25Okapi(self._tokenized_text_docs +
                                   self._tokenized_code_docs)
        self._save_bm25_cache()

    def _is_cached_files_exist(self) -> bool:
            return (Path("data/processed/bm25_text_cache.pkl").exists() and
                    Path("data/processed/bm25_code_cache.pkl").exists() and
                    Path("data/processed/bm25_both_cache.pkl").exists())

    def _remove_cache_files(self)-> None:
        """remove the cache if its curapted or something went wrong."""
        try:
            Path("data/processed/bm25_text_cache.pkl").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/bm25_code_cache.pkl").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/bm25_code_cache.pkl").unlink()
        except Exception:
            pass

    def load_bm25_cache(self) -> None:
        if not self._is_cached_files_exist():
            print("bm25 cache dosn't exists! run index to create cache.")
            exit(1)

        with open("data/processed/bm25_text_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_text = cached["bm25"]
                self.text_documents = cached["text_documents"]
                self.doc_chunks = cached["doc_chunks"]
            except Exception:
                print("Error: the cache is corupted"
                      "run index to create new cache.")
                self._remove_cache_files()
                exit(1)

        with open("data/processed/bm25_code_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_code = cached["bm25"]
                self.code_documents = cached["code_documents"]
                self.code_chunks = cached["code_chunks"]
            except Exception:
                print("Error: the cache is corupted"
                      "run index to create new cache.")
                self._remove_cache_files()
                exit(1)
        with open("data/processed/bm25_both_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_both = cached["bm25"]
                self.both_chunks = cached["both_chunks"]
            except Exception:
                print("Error: the cache is corupted"
                      "run index to create new cache.")
                self._remove_cache_files()
                exit(1)

    def query(self, query: str, type_flag: str = "") -> list[dict]:
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents yet to call query!")
            sys.exit(1)

        if type_flag == "doc":
            scores = self.bm25_text.get_scores(
                Process.preprocess_doc(query).split())
            documents = self.doc_chunks
        elif type_flag == "code":
            scores = self.bm25_code.get_scores(
                Process.preprocess_code(query).split())
            documents = self.code_chunks
        else:
            # get scores from both documents
            scores = self.bm25_both.get_scores(
                Process.preprocess_doc(query).split())
            documents = self.both_chunks

        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)[:self.top_k]

        code_results = []
        for index in top_indexes:
            code_results.append(documents[index])
        return code_results
