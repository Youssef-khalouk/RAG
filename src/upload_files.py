from pathlib import Path
# from typing import Any
import sys
import re


class UploadDir:
    def __init__(self, directory: str = ""):
        self.directory: str = directory
        self.files_path: list[str] = []
        self.documents: list[dict] = []
        self._chunk_size = 2000

    def set_chunk_size(self, size: int) -> None:
        if size <= 0:
            print("chunk size should be more then 0!")
            sys.exit(1)
        self._chunk_size = size

    def get_documents(self) -> list[dict]:
        return self.documents

    def _cut_chunk(self, text: str, s_index: int,
                   e_index: int, path: str) -> int:
        if e_index >= len(text):
            return len(text) - 1

        def cut_in(char: str, index: int) -> int | None:
            while text[index] != char:
                index -= 1
                if index <= s_index:
                    return None
            return index + 1

        chars = ["\n", " "]
        if path.endswith(".txt") or path.endswith(".md"):
            chars.insert(0, ".")
        for char in chars:
            index = cut_in(char, e_index)
            if index is not None:
                return index
        return s_index + self._chunk_size

    def _save_file_content(self, path: str, lines: list) -> None:
        text = "".join(lines)
        index = 0
        chunk = 1
        while (index != len(text)-1):
            document = {}
            document["path"] = path
            document["chunk"] = chunk

            end_index = index + self._chunk_size
            end_index = self._cut_chunk(text, index, end_index, path)
            p = re.split(r"[\\/_.]", path)
            p = " ".join(p)
            new_path = Path(path)
            # print(p)
            # print(new_path.stem)
            document["text"] = p + " " + new_path.stem + "\n" + text[index:end_index]
            document["index"] = (index, end_index - 1)
            self.documents.append(document)

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
                    self._save_file_content(file_path, file.readlines())
                except Exception:
                    continue
