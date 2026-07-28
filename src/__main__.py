from numpy import array
from .upload_files import UploadDir
from .bm25searcher import BM25Searcher
from pathlib import Path
import fire
from .print_data import print_data
from .validaters import (RagDataset,
                         MinimalSource,
                         MinimalSearchResults,
                         StudentSearchResults,
                         MinimalAnswer,
                         StudentSearchResultsAndAnswer)


def index(max_chunk_size: int = 2000) -> None:
    updir = UploadDir("data/raw/vllm-0.10.1", max_chunk_size)
    is_any_file_changed = updir.upload()

    searcher = BM25Searcher()
    searcher.create_bm25_cache_and_save(
        updir.get_text_documents(),
        updir.get_code_documents(),
        is_any_file_changed
    )


def search(query: str, k: int = 10) -> None:
    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()
    documents = searcher.query(query)
    for d in documents:
        doc = searcher.get_document(d[1], d[2])
        print(
            f"{doc['file_path']} "
            f"[{doc['first_character_index']}, {doc['last_character_index']}]"
        )


def search_content(query: str, k: int = 10) -> None:
    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()
    documents = searcher.query(query)
    print_data(query, documents, searcher)


def search_dataset(dataset_path: str,
                   k: int = 10,
                   save_directory: str = "data/output.json") -> None:
    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()

    with open(dataset_path, "r", encoding="utf-8") as file:
        try:
            dictionary = RagDataset.model_validate_json(file.read())
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
            return

    path = Path(dataset_path)
    type = "both"
    if "code" in path.name:
        type = "code"
    elif "doc" in path.name:
        type = "doc"

    search_results: list[MinimalSearchResults] = []
    for q in dictionary.rag_questions:
        documents = searcher.query(q.question, type)
        retrieved_sources: list[MinimalSource] = []
        for d in documents:
            document = searcher.get_document(d[1], d[2])
            retrieved_sources.append(
                MinimalSource(
                    file_path=document["file_path"],
                    first_character_index=document["first_character_index"],
                    last_character_index=document["last_character_index"],
                )
            )
        search_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=retrieved_sources,
            )
        )
    output = StudentSearchResults(search_results=search_results, k=k)

    with open(save_directory, "w", encoding="utf-8") as file:
        file.write(output.model_dump_json(indent=4))
        print(f"json saved successfully to '{save_directory}'.")


def answer(query: str, k: int = 10) -> None:
    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()
    documents = searcher.query(query)

    # here i need to ask the llm model and print the answer.


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:

    if not Path(student_search_results_path).exists():
        print(f"Error: file '{student_search_results_path}' not found.")
        exit(1)

    with open(student_search_results_path, "r", encoding="utf-8") as file:
        try:
            search_results = StudentSearchResults.model_validate_json(
                                                            file.read())
        except Exception as e:
            print(f"Error: {e}")
            exit(1)

    files_chunks = {}
    for s_result in search_results.search_results:
        content = []
        for chunk_info in s_result.retrieved_sources:
            with open(chunk_info.file_path, "r", encoding="utf-8") as file:
                text = file.read()
                text_chunk = text[chunk_info.first_character_index:
                                  chunk_info.last_character_index]
            content.append({
                "file_path": chunk_info.file_path,
                "text": text_chunk
            })
        files_chunks[s_result.question_id] = content

    answer_results: list[MinimalAnswer] = []
    for q in search_results.search_results:

        answer = "This is a placeholder answer."
        # asking the llm model should be here
        # and the answer in the variable 'answer'

        answer_result = MinimalAnswer(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=q.retrieved_sources,
            answer=answer,
        )
        answer_results.append(answer_result)

    output = StudentSearchResultsAndAnswer(
        search_results=answer_results, k=search_results.k)
    with open(save_directory, "w", encoding="utf-8") as file:
        file.write(output.model_dump_json(indent=4))


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
