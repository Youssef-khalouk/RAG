from pathlib import Path
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


class UploadDir:
    def __init__(self, directory: str = ""):
        self.directory: str = directory
        self.files_path: list[str] = []
        self.text_documents: list[dict] = []
        self.code_documents: list[dict] = []
        self._chunk_size = 2000
        self._text_chunk_overlap = 300
        self._code_chunk_overlap = 0

    def set_chunk_size(self, size: int) -> None:
        if size <= 0:
            print("chunk size should be more then 0!")
            sys.exit(1)
        self._chunk_size = size

    def get_text_documents(self) -> list[dict]:
        return self.text_documents

    def get_code_documents(self) -> list[dict]:
        return self.code_documents

    def _chunk_code_file(self, path: str, text: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_chunk_overlap,
            add_start_index=True
        )
        docs = splitter.create_documents([text])
        # chunks = splitter.split_text(text)

        new_path = Path(path)
        path_info = f"{path} {new_path.name} {new_path.stem}"
        for i, doc in enumerate(docs):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            self.code_documents.append({
                "path": path,
                "chunk": i,
                "text": f"{path_info}\n{doc.page_content}",
                "index": (start, end)
            })

    def _chunk_text_file(self, path: str, text: str) -> None:
        separators = [".\n", "\n", " ", ""]
        if path.endswith(".md"):
            separators = ["\n# ", "\n## ", "\n### ", ".\n", "\n", " ", ""]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._text_chunk_overlap,
            separators=separators,
            add_start_index=True
        )
        docs = splitter.create_documents([text])

        new_path = Path(path)
        path_info = (
            ""
            f"{path.replace("\\", " ").replace("/", " ")}\n"
            f"{new_path.stem}\n"
            f" {new_path.stem}\n"
            f" {new_path.stem.replace("_", " ").replace("-", " ")}\n"
        )
        # print(path_info)
        for i, doc in enumerate(docs):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            self.text_documents.append({
                "path": path,
                "chunk": i,
                "text": f"{path_info}\n{doc.page_content}",
                "index": (start, end)
            })

    def _save_file_content(self, path: str, text: str) -> None:
        if path.endswith(".py"):
            self._chunk_code_file(path, text)
        elif path.endswith(".md") or path.endswith(".txt"):
            self._chunk_text_file(path, text)

    def upload(self) -> None:
        if self.directory == "":
            return

        def get_dir_content(path: str) -> None:
            directory = Path(path)
            for item in directory.iterdir():
                if item.is_dir():
                    get_dir_content(str(item))
                elif (item.suffix in [".py", ".md", ".txt"]):
                    # if not str(item).endswith("CMakeLists.txt"):
                    self.files_path.append(str(item))
                    # else:
                    #     print(str(item))

        get_dir_content(self.directory)

        for file_path in self.files_path:
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    self._save_file_content(file_path, file.read())
                except Exception as error:
                    print(error)
                    continue
