"""
Utilities for scanning a directory, chunking text and code files,
and persisting them.
"""

from pathlib import Path
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
import os
from datetime import datetime
from tqdm import tqdm


class UploadDir:
    """
    Index files from a directory into chunked text and code document stores.
    """

    def __init__(self, directory: str = "", chunk_size: int = 2000):
        """
        Initialize the uploader, chunking splitters, and document containers.
        """
        self.directory: str = directory
        self.files_path: list[str] = []
        self.text_documents: dict = {}
        self.code_documents: dict = {}
        self._chunk_size = chunk_size
        self._doc_chunk_overlap = 0
        self._code_chunk_overlap = 0

        # this two splitters for spliting docs
        self.splitter_txt = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._doc_chunk_overlap,
            separators=[".\n", "\n", " ", ""],
            add_start_index=True
        )
        self.splitter_md = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._doc_chunk_overlap,
            separators=["\n# ", ".\n", "\n", " ", ""],
            add_start_index=True
        )
        # this one for python code
        self.splitter_code = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_chunk_overlap,
            add_start_index=True,
        )

    def set_chunk_size(self, size: int) -> None:
        """Set the chunk size used for future file chunking."""
        if size <= 0:
            print("chunk size should be more then 0!")
            sys.exit(1)
        self._chunk_size = size

    def get_text_documents(self) -> dict:
        """Return the indexed text-document store."""
        return self.text_documents

    def get_code_documents(self) -> dict:
        """Return the indexed code-document store."""
        return self.code_documents

    def _chunk_file_and_save(self, path: str, text: str) -> None:
        """
        Split a file into chunks and store the result in
        the appropriate document map.
        """
        if path.endswith(".md"):
            docs = self.splitter_md.create_documents([text])
            documents = self.text_documents
        elif path.endswith(".txt"):
            docs = self.splitter_txt.create_documents([text])
            documents = self.text_documents
        elif path.endswith(".py"):
            docs = self.splitter_code.create_documents([text])
            documents = self.code_documents
        else:
            print(f"Unknown file to save: '{path}'.")
            return
        p = Path(path)
        path_info = f"Path: {path.replace('/', ' ')}\n{p.stem}\n {p.stem}\n"
        path_info += f" {p.stem.replace('_', ' ')}\n{p.stem.replace('_', ' ')}"
        last_updated_time = str(datetime.fromtimestamp(p.stat().st_mtime))
        documents[path] = {
            "last_updated_time": last_updated_time,
            "chunks": []
        }
        chunks = []
        for i, doc in enumerate(docs):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            document = {
                "chunk": i,
                "path_tokens": path_info,
                "text": doc.page_content,
                "start_index": start,
                "end_index": end
            }
            chunks.append(document)
        documents[path]["chunks"] = chunks

    def _open_json_files(self, use_file_chunk_size: bool = False) -> None:
        """Load previously persisted document indexes from disk."""
        if Path("data/processed/doc_documents.json").exists():
            with open("data/processed/doc_documents.json", "r") as file:
                try:
                    self.text_documents = json.load(file)
                except Exception:
                    self.text_documents = {}
        if Path("data/processed/code_documents.json").exists():
            with open("data/processed/code_documents.json", "r") as file:
                try:
                    self.code_documents = json.load(file)
                except Exception:
                    self.code_documents = {}
        # after loading the json files, we need to set the chunk size
        # to the one in the json files if it exists
        if use_file_chunk_size:
            k = self.text_documents.get("k", None)
            if k is None:
                k = self.code_documents.get("k", None)
            if k:
                self._chunk_size = k

    def _load_files(self, paths: list[str] = []) -> None:
        """Chunk and index the provided files in order."""
        # open the files and save the content as chunks
        if paths == []:
            print("All files alrdy indexd.")
            return
        for path in tqdm(paths, desc="indexing", unit="file"):
            with open(path, "r", encoding="utf-8") as file:
                self._chunk_file_and_save(path, file.read())

    def _save_documents(self) -> None:
        """
        Persist the indexed text and code documents to disk as JSON files.
        """
        os.makedirs("data/processed", exist_ok=True)
        self.text_documents["k"] = self._chunk_size
        self.code_documents["k"] = self._chunk_size
        with open("data/processed/doc_documents.json", "w") as file:
            json.dump(self.text_documents, file, indent=4)
        with open("data/processed/code_documents.json", "w") as file:
            json.dump(self.code_documents, file, indent=4)

    def upload(self, use_file_chunk_size: bool = False) -> bool:
        """
        Scan a directory, process changed files, and save any updated indexes.
        """
        if not Path(self.directory).exists():
            print(f"Error: directory '{self.directory}' dons't exist?")
            exit(1)

        # load the json files if they exist
        self._open_json_files(use_file_chunk_size)
        # if the chunk size has changed, we need to reprocess all files
        if self.text_documents.get("k", -1) != self._chunk_size:
            self.text_documents = {}
        if self.code_documents.get("k", -1) != self._chunk_size:
            self.code_documents = {}
        changed_files = []

        def get_dir_content(path: str) -> None:
            directory = Path(path)
            for item in directory.iterdir():
                if item.is_dir():
                    get_dir_content(str(item))
                elif (item.suffix in [".py", ".md", ".txt"]):
                    self.files_path.append(str(item))
                    l_time = str(datetime.fromtimestamp(item.stat().st_mtime))
                    if item.suffix == ".py":
                        doc = self.code_documents.get(str(item), None)
                    else:
                        doc = self.text_documents.get(str(item), None)
                    if (doc is None or doc["last_updated_time"] != l_time):
                        changed_files.append(str(item))
        get_dir_content(self.directory)

        self._load_files(changed_files)
        if changed_files != []:
            # save the documents if any file was processed
            self._save_documents()
            return True
        return False
