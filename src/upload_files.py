from pathlib import Path
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
import os
from datetime import datetime
from .process import Process

class UploadDir:
    def __init__(self, directory: str = ""):
        self.directory: str = directory
        self.files_path: list[str] = []
        self.text_documents: dict = {}
        self.code_documents: dict = {}
        self._chunk_size = 2000
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
        if size <= 0:
            print("chunk size should be more then 0!")
            sys.exit(1)
        self._chunk_size = size

    def get_text_documents(self) -> list[dict]:
        return self.text_documents

    def get_code_documents(self) -> list[dict]:
        return self.code_documents

    def _chunk_file_and_save(self, path: str, text: str) -> None:
        if path.endswith(".md"):
            docs = self.splitter_md.create_documents([text])
        elif path.endswith(".txt"):
            docs = self.splitter_txt.create_documents([text])
        elif path.endswith(".py"):
            docs = self.splitter_code.create_documents([text])
        else:
            print(f"Unknown file to save: '{path}'.")
        p = Path(path)
        path_info = f"Path: {path.replace('/', ' ')}\n{p.stem}\n {p.stem}\n"
        path_info += f" {p.stem.replace("_", " ")}\n{p.stem.replace("_", " ")}"

        if path.endswith(".py"):
            self.code_documents[path] = {
                "last_updated_time": str(datetime.fromtimestamp(p.stat().st_mtime)),
                "chunks": []
            }
        else:
            self.text_documents[path] = {
                "last_updated_time": str(datetime.fromtimestamp(p.stat().st_mtime)),
                "chunks": []
            }

        for i, doc in enumerate(docs):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            document = {
                "chunk": i,
                "text": f"{path_info}\n{doc.page_content}",
                "processed_text": Process.preprocess_doc(f"{path_info}\n{doc.page_content}"),
                "start_index": start,
                "end_index": end
            }
            if path.endswith(".py"):
                self.code_documents[path]["chunks"].append(document)
            else:
                self.text_documents[path]["chunks"].append(document)

    def upload(self) -> None:
        if self.directory == "":
            return

        def get_dir_content(path: str) -> None:
            directory = Path(path)
            for item in directory.iterdir():
                if item.is_dir():
                    get_dir_content(str(item))
                elif (item.suffix in [".py", ".md", ".txt"]):
                    self.files_path.append(str(item))
        get_dir_content(self.directory)

        # open all the files and save the content as chunks
        for file_path in self.files_path:
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    self._chunk_file_and_save(file_path, file.read())
                except Exception as error:
                    print(error)
                    continue
        
        # save the documents
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/doc_documents.json", "w") as file:
            json.dump(self.text_documents, file, indent=4)
            print("text documents saved seccessfuly.")
        with open("data/processed/code_documents.json", "w") as file:
            json.dump(self.code_documents, file, indent=4)
            print("code documents saved seccessfuly.")