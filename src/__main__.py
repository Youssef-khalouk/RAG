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
from tqdm import tqdm


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
    chunks = searcher.query(query)
    print_data(query, chunks, searcher)


def search_dataset(dataset_path: str,
                   k: int = 10,
                   save_directory: str = "data/output.json") -> None:

    if not Path(dataset_path).exists():
        print(f"Error: dataset_path '{dataset_path}' dosn't exist.")
        exit(1)
    with open(dataset_path, "r", encoding="utf-8") as file:
        try:
            dictionary = RagDataset.model_validate_json(file.read())
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
            return

    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()

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
    from .LLM import LLM
    searcher = BM25Searcher()
    searcher.set_top_k(k)
    searcher.load_bm25_cache()

    chunks = searcher.query(query)
    context = LLM.get_context(chunks)
    the_answer = LLM.ask(question=query, context=context)
    print(f"Question: {query}\nAnswer: {the_answer}")


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    from .LLM import LLM
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
    LLM._init_model()  # initalize the LLM model before we start
    files_chunks = {}
    print("Loading chunks:")
    for s_result in tqdm(search_results.search_results,
                         desc="Uploading Questions data", unit="Question"):
        content = []
        for chunk in s_result.retrieved_sources:
            with open(chunk.file_path, "r", encoding="utf-8") as file:
                text = file.read()
                text_chunk = text[chunk.first_character_index + 1:
                                  chunk.last_character_index]
            text_pos = (f" [{chunk.first_character_index},"
                        f" {chunk.last_character_index}]:")
            text_path_info = chunk.file_path + text_pos
            content.append({
                "text": text_path_info + "\n" + text_chunk.strip()
            })
        files_chunks[s_result.question_id] = content

    answer_results: list[MinimalAnswer] = []
    print("Start generating answer_dataset_results.json:")
    for q in tqdm(search_results.search_results,
                  desc="Processing", unit="Question"):
        # asking the llm model should be here
        chunks = files_chunks.get(q.question_id, None)
        if chunks is None:
            print(f"this question '{q.question}', "
                  "is not in the student_search_results.")
            continue
        context = LLM.get_context(chunks)
        answer = LLM.ask(q.question, context=context)

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


def evaluate(student_search_results_path: str, dataset_path: str) -> None:

    if not Path(student_search_results_path).exists():
        print(f"Error: file '{student_search_results_path}' not found.")
        exit(1)
    if not Path(dataset_path).exists():
        print(f"Error: dataset_path '{dataset_path}' dosn't exist.")
        exit(1)

    with open(student_search_results_path, "r", encoding="utf-8") as file:
        try:
            search_results = StudentSearchResults.model_validate_json(
                                                            file.read())
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
    with open(dataset_path, "r", encoding="utf-8") as file:
        try:
            dictionary = RagDataset.model_validate_json(file.read())
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
            return

    def is_it_in(source) -> int:
        dataset_source = item.sources[0]
        if source.file_path == dataset_source.file_path:
            source_s = source.first_character_index
            source_e = source.last_character_index
            dataset_s = dataset_source.first_character_index
            dataset_e = dataset_source.last_character_index

            intersection = max(0, min(source_e, dataset_e)
                               - max(source_s, dataset_s))
            union = max(source_e, dataset_e) - min(source_s, dataset_s)
            iou = intersection / union if union > 0 else 0
            # this means the student result should take
            # at list 5% of the dataset interval
            if iou >= 0.05:
                return 1
        return 0

    results = search_results.search_results
    dataset = dictionary.rag_questions
    recall1 = recall3 = recall5 = recall10 = 0

    for doc in results:
        item = next((x for x in dataset if x.question_id == doc.question_id),
                    None)
        if item is None:
            print(f"the question '{doc.question_id}' "
                  "did not found in dataset.")
            exit(1)
        recall1 += any(is_it_in(s) for s in doc.retrieved_sources[:1])
        recall3 += any(is_it_in(s) for s in doc.retrieved_sources[:3])
        recall5 += any(is_it_in(s) for s in doc.retrieved_sources[:5])
        recall10 += any(is_it_in(s) for s in doc.retrieved_sources[:10])

    recall1 = recall1/len(dataset)
    recall3 = recall3/len(dataset)
    recall5 = recall5/len(dataset)
    recall10 = recall10/len(dataset)

    print(f"\nTotal number of questions with sources: {len(dataset)}")
    print(f"Total number of questions with student sources: {len(results)}\n")

    print("🎯 Evaluation Results\n=======================================")
    print(f"📊 Questions evaluated: {len(dataset)}")
    print(f"📈 Recall@1: {recall1} ({recall1 * 100:.1f}%)")
    print(f"📈 Recall@3: {recall3} ({recall3 * 100:.1f}%)")
    print(f"📈 Recall@5: {recall5} ({recall5 * 100:.1f}%)")
    print(f"📈 Recall@10: {recall10} ({recall10 * 100:.1f}%)")


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
