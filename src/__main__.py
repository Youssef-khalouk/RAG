from .upload_files import UploadDir
from .bm25searcher import BM25Searcher
from pathlib import Path
import fire
import json
from .print_data import print_data


def index(
        max_chunk_size: int = 2000,
        k: int = 10,
        save_directory: str = "data/output.json") -> None:

    updir = UploadDir("data/raw/vllm-0.10.1", max_chunk_size)
    # updir.set_chunk_size(max_chunk_size)
    is_any_file_changed = updir.upload()

    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.check_documents(updir.get_text_documents(), updir.get_code_documents(), is_any_file_changed)


def search(query: str, k: int = 10) -> None:
    updir = UploadDir("data/raw/vllm-0.10.1")
    is_any_file_changed = updir.upload(use_file_chunk_size=True)

    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.set_documents(updir.get_text_documents(), updir.get_code_documents(), is_any_file_changed)
    documents = searcher.query(query)
    for d in documents:
        doc = searcher.get_document(d[1], d[2])
        print(
            f"\n\nfile_path: {doc['file_path']}, "
            f"\nfirst_character_index: {doc['first_character_index']}, "
            f"\nlast_character_index: {doc['last_character_index']}"
        )

def search_content(query: str, k: int = 10) -> None:
    updir = UploadDir("data/raw/vllm-0.10.1")
    is_any_file_changed = updir.upload(use_file_chunk_size=True)

    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.set_documents(updir.get_text_documents(), updir.get_code_documents(), is_any_file_changed)
    documents = searcher.query(query)
    print_data(query, documents, searcher)


def search_dataset(dataset_path: str,
                   k: int = 10,
                   save_directory: str = "data/output.json") -> None:

    updir = UploadDir("data/raw/vllm-0.10.1")
    is_any_file_changed = updir.upload(use_file_chunk_size=True)

    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.set_documents(updir.get_text_documents(), updir.get_code_documents(), is_any_file_changed)

    with open(dataset_path, "r") as file:
        dictionary = json.load(file)

    path = Path(dataset_path)
    type = "both"
    if "code" in path.name:
        type = "code"
    elif "doc" in path.name:
        type = "doc"

    array = []
    for q in dictionary["rag_questions"]:
        dic = {}
        retrieved_sources = []
        dic["question_id"] = q["question_id"]
        dic["question"] = q["question"]
        documents = searcher.query(q["question"], type)
        for d in documents:
            retrieved_sources.append(searcher.get_document(d[1], d[2]))
        dic["retrieved_sources"] = retrieved_sources
        array.append(dic)
    my_dict = {}
    my_dict["search_results"] = array
    my_dict["k"] = k
    with open(save_directory, "w") as file:
        json.dump(my_dict, file, indent=4)
        print(f"json saved seccessfuly to {save_directory}.")

def answer(query: str, k: int = 10) -> None:
    pass

def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
    pass

def evaluate() -> None:
    pass

if __name__ == "__main__":

    fire.Fire({
        "index": index,
        "search": search,
        "search_content": search_content,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })
