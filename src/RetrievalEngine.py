"""Retrieval utilities for indexing, caching, and querying document and code chunks."""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .process import Process
from typing import Any
import sys
import pickle
from pathlib import Path
import os
import json
from functools import lru_cache
from tqdm import tqdm

import torch
import numpy as np


class RetrievalEngine:
    """Build and query BM25 and embedding-based retrieval indexes for documents."""

    def __init__(self):
        """Initialize retrieval state, caches, and the sentence-transformer model."""
        self.text_documents: dict = None
        self.code_documents: dict = None
        self.doc_chunks: list[str] = []
        self.code_chunks: list[str] = []
        self.both_chunks: list[str] = []
        self.bm25_text: Any = None
        self.bm25_code: Any = None
        self.bm25_both: Any = None
        self.top_k: int = 10
        self.model: Any = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.doc_embeddings: Any = None
        self.code_embeddings: Any = None
        self.both_embeddings: Any = None

    def _save_bm25_cache(self) -> None:
        """Persist BM25 indexes and embeddings to the processed-data cache directory."""
        pkls = {
            "data/processed/bm25_text_cache.pkl":
            {
                "bm25": self.bm25_text,
                "doc_chunks": self.doc_chunks
            },
            "data/processed/bm25_code_cache.pkl":
            {
                "bm25": self.bm25_code,
                "code_chunks": self.code_chunks,
            },
            "data/processed/bm25_both_cache.pkl":
            {
                "bm25": self.bm25_both,
                "both_chunks": self.both_chunks
            },
            "data/processed/doc_embeddings_cache.pkl":
            {
                "embeddings": self.doc_embeddings,
            },
            "data/processed/code_embeddings_cache.pkl":
            {
                "embeddings": self.code_embeddings,
            }
        }
        os.makedirs("data/processed", exist_ok=True)
        for path, content in tqdm(pkls.items(),
                                  desc="Cacheing BM25",
                                  unit="file"):
            with open(path, "wb") as file:
                pickle.dump(content, file)

    def set_top_k(self, size: int) -> None:
        """Set the number of top results returned by query methods."""
        if size <= 0:
            print("top_k size should be more then 0!")
            sys.exit(1)
        self.top_k = size

    def get_document_content(self, path: str, chunk: int) -> list[dict]:
        """Return the content and metadata for a specific chunk from a document or code file."""
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents!")
            exit(1)
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
        """Return metadata for a specific chunk without including its text content."""
        if self.text_documents is None or self.code_documents is None:
            print("there is no documents!")
            exit(1)
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
        """Ensure retrieval indexes exist, rebuilding them when needed."""
        # this function used when i need to check the cached file is exist
        # if not rebuild cache
        if not is_files_changed and self._is_cached_files_exist():
            return
        self.set_documents(documents_text, documents_code, is_files_changed)

    def create_bm25_cache_and_save(self,
                                   documents_text: list[dict],
                                   documents_code: list[dict],
                                   is_files_changed: bool) -> None:
        """Build BM25 indexes and embeddings from documents and persist them to disk."""
        if not is_files_changed and self._is_cached_files_exist():
            return

        self.text_documents = documents_text
        self.code_documents = documents_code

        tokenized_text = []
        tokenized_code = []
        self.code_chunks = []
        self.doc_chunks = []
        self.doc_embeddings = []
        self.code_embeddings = []
        self.both_embeddings = []
        doc_embeddings_chunks = []
        code_embeddings_chunks = []

        for k, v in documents_text.items():
            if k == "k":
                continue
            for chunk in v["chunks"]:
                good_text = (
                    chunk["text"]
                    .removeprefix(".\n\n")
                    .removeprefix(".\n")
                    .removeprefix(".")
                )
                text = chunk["path_tokens"] + chunk["text"]
                doc_embeddings_chunks.append(text)
                tokenized_text.append(Process.preprocess_doc(text).split())
                pos = f"[{chunk['start_index']}, {chunk['end_index']}]:"
                text = k + f" {pos}\n" + good_text
                self.doc_chunks.append((text, k, chunk["chunk"], pos))

        for k, v in documents_code.items():
            if k == "k":
                continue
            for chunk in v["chunks"]:
                text = chunk["path_tokens"] + chunk["text"]
                code_embeddings_chunks.append(text)
                tokenized_code.append(Process.preprocess_code(text).split())
                pos = f"[{chunk['start_index']}, {chunk['end_index']}]:"
                text = k + f" {pos}\n" + chunk["text"]
                self.code_chunks.append((text, k, chunk["chunk"], pos))

        self.both_chunks = self.doc_chunks + self.code_chunks

        self.bm25_text = BM25Okapi(tokenized_text)
        self.bm25_code = BM25Okapi(tokenized_code)
        self.bm25_both = BM25Okapi(tokenized_text + tokenized_code)
        with torch.inference_mode():
            self.doc_embeddings = self.model.encode(
                doc_embeddings_chunks, batch_size=256,
                show_progress_bar=True, convert_to_numpy=True
            )
        with torch.inference_mode():
            self.code_embeddings = self.model.encode(
                code_embeddings_chunks, batch_size=256,
                show_progress_bar=True, convert_to_numpy=True
            )
        self.both_embeddings = np.concatenate([self.doc_embeddings,
                                               self.code_embeddings])
        self._save_bm25_cache()

    def _is_cached_files_exist(self) -> bool:
        """Return whether the required cache files are present on disk."""
        return (Path("data/processed/bm25_text_cache.pkl").exists() and
                Path("data/processed/bm25_code_cache.pkl").exists() and
                Path("data/processed/bm25_both_cache.pkl").exists() and
                Path("data/processed/doc_documents.json").exists() and
                Path("data/processed/code_documents.json").exists() and
                Path("data/processed/doc_embeddings_cache.pkl").exists() and
                Path("data/processed/code_embeddings_cache.pkl").exists())

    def _remove_cache_files(self) -> None:
        """Remove retrieval cache files from disk."""
        try:
            Path("data/processed/doc_documents.json").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/code_documents.json").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/bm25_text_cache.pkl").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/bm25_code_cache.pkl").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/bm25_both_cache.pkl").unlink()
        except Exception:
            pass
        try:
            Path("data/processed/doc_embeddings_cache.pkl").unlink()
        except Exception:
            pass

    def load_cache(self) -> None:
        """Load BM25 indexes, embeddings, and document metadata from cache."""
        if not self._is_cached_files_exist():
            print("bm25 cache dosn't exists!"
                  "\nrun 'make index' to create cache.")
            exit(1)

        def _cache_curapted() -> None:
            """Handle corrupted cache files by deleting them and exiting."""
            print("Error: the cache is corupted, "
                  "run 'make index' to create new cache.")
            self._remove_cache_files()
            exit(1)

        with open("data/processed/doc_documents.json", "r") as file:
            try:
                self.text_documents = json.load(file)
            except Exception:
                _cache_curapted()
        with open("data/processed/code_documents.json", "r") as file:
            try:
                self.code_documents = json.load(file)
            except Exception:
                _cache_curapted()

        with open("data/processed/bm25_text_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_text = cached["bm25"]
                self.doc_chunks = cached["doc_chunks"]
            except Exception:
                _cache_curapted()
        with open("data/processed/bm25_code_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_code = cached["bm25"]
                self.code_chunks = cached["code_chunks"]
            except Exception:
                _cache_curapted()
        with open("data/processed/bm25_both_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.bm25_both = cached["bm25"]
                self.both_chunks = cached["both_chunks"]
            except Exception:
                _cache_curapted()
        with open("data/processed/doc_embeddings_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.doc_embeddings = cached["embeddings"]
            except Exception:
                _cache_curapted()
        with open("data/processed/code_embeddings_cache.pkl", "rb") as file:
            try:
                cached = pickle.load(file)
                self.code_embeddings = cached["embeddings"]
            except Exception:
                _cache_curapted()
        self.both_embeddings = np.concatenate([self.doc_embeddings,
                                               self.code_embeddings])

    @lru_cache
    def query_embeddings(self, query: str, type_flag: str = "",
                         top_k: int = None) -> list[dict]:
        """Return the top matching chunks using semantic embeddings."""
        if self.doc_embeddings is None or self.code_embeddings is None:
            print("embeddings cache did not load yet, "
                  "call load_cache() before you query.")
            exit(1)
        if top_k is not None:
            self.set_top_k(top_k)

        if type_flag == "doc":
            vectors = self.doc_embeddings
            chunks = self.doc_chunks
        elif type_flag == "code":
            vectors = self.code_embeddings
            chunks = self.code_chunks
        else:
            vectors = self.both_embeddings
            chunks = self.both_chunks

        with torch.inference_mode():
            query_vector = self.model.encode(
                [query], convert_to_numpy=True
            )[0]

        query_norm = query_vector / np.linalg.norm(query_vector)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = vectors_norm @ query_norm

        top_indexes = np.argsort(scores)[::-1][:self.top_k]
        return [chunks[i] for i in top_indexes]

    @lru_cache
    def query_bm25(self, query: str, type_flag: str = "",
                   top_k: int = None) -> list[dict]:
        """Return the top matching chunks using BM25 lexical ranking."""
        if not self.bm25_text or not self.bm25_code or not self.bm25_both:
            print("the cache did not upload yet,"
                  " call load_cache() before you query.")
            exit(1)
        if top_k is not None:
            self.set_top_k(top_k)

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

        query_results = []
        for index in top_indexes:
            query_results.append(documents[index])
        return query_results

    @lru_cache
    def query(self, query: str, type_flag: str = "",
              top_k: int = None) -> list[dict]:
        """Combine BM25 and embedding results into a deduplicated ranked list."""
        if top_k is not None:
            self.set_top_k(top_k)

        half = max(1, self.top_k // 2)
        bm25_results = self.query_bm25(query, type_flag, top_k)
        embedding_results = self.query_embeddings(query, type_flag, top_k)

        seen = set()
        combined = []

        def add_unique(chunks, limit):
            added = 0
            for chunk in chunks:
                key = (chunk[1], chunk[2])
                if key in seen:
                    continue
                seen.add(key)
                combined.append(chunk)
                added += 1
                if added >= limit:
                    break

        add_unique(bm25_results, half)
        add_unique(embedding_results, self.top_k - half)

        return combined[:self.top_k]
