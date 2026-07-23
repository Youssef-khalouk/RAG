from .upload_files import UploadDir
from .bm25searcher import BM25Searcher
import json
from colorama import Fore, Style
from pathlib import Path
import argparse

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "aren't", "as", "at",

    "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by",

    "can", "can't", "cannot", "could", "couldn't",

    "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during",

    "each", "must", "passed", "run", "What", "step", "How",

    "few", "for", "from", "further",

    "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's",

    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself",

    "just",

    "let's",

    "me", "more", "most", "mustn't", "my", "myself",

    "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own",

    "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such",

    "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too",

    "under", "until", "up",

    "very",

    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "will", "with", "won't", "would", "wouldn't",

    "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# path = "data/datasets_private/private/UnansweredQuestions/dataset_docs_private.json"
# path = "data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
# path = "data/datasets_private/private/UnansweredQuestions/dataset_code_private.json"

# with open(path, "r", encoding="utf-8") as file:
#     json_data = json.load(file)


def print_data(data, documents):
    for q in data["rag_questions"]:
        question = q["question"]
        for d in documents:
            print(f"{Fore.YELLOW}\n\nQuestion: {question}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}path: {d['path']} [chunk: {d['chunk']}]{Style.RESET_ALL}")
            text = d["text"]
            for q in question.split(" "):
                if q not in STOP_WORDS:
                    text = text.replace(q, f"{Fore.YELLOW}{q}{Fore.WHITE}")
            print(f"{Fore.WHITE}\ntext: {text}{Style.RESET_ALL}")
        # break


def save_data(data):
    pass


def positive_int(value: str) -> int:
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid integer"
        )

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        )

    return value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--max_chunk_size", default=2000, type=positive_int)
    parser.add_argument("--k", default=10, type=positive_int)
    parser.add_argument("--save_directory", default="data/output.json")
    args = parser.parse_args()

    if args.dataset_path == "":
        print("place set dataset path before runing the programe.")
        exit(1)

    json_data = []
    with open(args.dataset_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    updir = UploadDir("data/raw/vllm-0.10.1")
    updir.set_chunk_size(args.max_chunk_size)
    updir.upload()
    searcher = BM25Searcher()
    searcher.set_top_k(args.k)
    searcher.set_text_documents(updir.get_text_documents())
    searcher.set_code_documents(updir.get_code_documents())

    path = Path(args.dataset_path)
    type = "doc"
    if "code" in path.name:
        type = "code"

    array = []
    for q in json_data["rag_questions"]:
        dic = {}
        ar = []
        dic["question_id"] = q["question_id"]
        dic["question"] = q["question"]
        documents = searcher.query(q["question"], type)
        for d in documents:
            dd = {}
            dd["file_path"] = d["path"].replace("\\", "/")
            dd["first_character_index"] = d["index"][0]
            dd["last_character_index"] = d["index"][1]
            ar.append(dd)
        dic["retrieved_sources"] = ar
        array.append(dic)
    my_dict = {}
    my_dict["search_results"] = array
    my_dict["k"] = 10
    with open("data/output.json", "w") as file:
        json.dump(my_dict, file, indent=4)
        print("json saved seccessfuly.")

    # print_data(json_data, documents)
