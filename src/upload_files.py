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
        self._chunk_overlap = 150

    def set_chunk_size(self, size: int) -> None:
        if size <= 0:
            print("chunk size should be more then 0!")
            sys.exit(1)
        self._chunk_size = size

    def get_text_documents(self) -> list[dict]:
        return self.text_documents

    def get_code_documents(self) -> list[dict]:
        return self.code_documents

    def _cut_chunk(self, text: str, s_index: int,
                   e_index: int, path: str) -> int:
        if e_index >= len(text):
            return len(text) - 1

        def cut_in(char: str, index: int) -> int | None:
            ln = len(char)
            index -= ln
            middle = s_index + (e_index - s_index) // 2
            # middle = s_index
            while index > middle:
                if text[index:index + ln] == char:
                    return index
                index -= 1
            return None

        chars = ["\n", " "]
        if path.endswith(".txt") or path.endswith(".md") or path.endswith("LICENSE"):
            chars.insert(0, ".")
            chars.insert(0, ".\n")
            if path.endswith(".md"):
                chars.insert(0, "#")
                chars.insert(0, "##")
                chars.insert(0, "###")
        if path.endswith(".py"):
            chars.insert(0, "def ")
            chars.insert(0, "class ")
        for char in chars:
            index = cut_in(char, e_index)
            if index is not None:
                return index
        return s_index + self._chunk_size

    def _save_python_file(self, path: str, text: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            add_start_index=True
        )
        docs = splitter.create_documents([text])
        for i, doc in enumerate(docs):
            document = {}
            document["path"] = path
            document["chunk"] = i
            new_path = Path(path)
            # adding path content to the shunks
            p = path + " " + new_path.name + " " + new_path.stem
            document["text"] = p + "\n" + doc.page_content
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            document["index"] = (start, end)
            self.code_documents.append(document)

    def _save_file_content(self, path: str, text: str) -> None:
        if (
                path.endswith(".py") or
                path.endswith(".sh") or
                path.endswith(".cuh") or
                path.endswith(".cu") or
                path.endswith(".hpp") or
                path.endswith(".yml") or
                path.endswith("CMakeLists.txt")
                ):
            self._save_python_file(path, text)
            return
        index = 0
        chunk = 1
        while (index != len(text)-1):
            document = {}
            document["path"] = path
            document["chunk"] = chunk
            end_index = index + self._chunk_size
            end_index = self._cut_chunk(text, index, end_index, path)
            new_path = Path(path)
            # adding path content to the shunks
            p = path + " " + new_path.name + " " + new_path.stem
            document["text"] = p + "\n" + text[index:end_index]
            document["index"] = (index, end_index - 1)
            self.text_documents.append(document)

            index = end_index
            chunk += 1

    def upload(self) -> None:
        if self.directory == "":
            return

        def get_dir_content(path: str) -> None:
            directory = Path(path)
            for item in directory.iterdir():
                if item.is_dir():
                    get_dir_content(str(item))
                else:
                    self.files_path.append(str(item))

        get_dir_content(self.directory)

        for file_path in self.files_path:
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    self._save_file_content(file_path, file.read())
                except Exception:
                    continue
