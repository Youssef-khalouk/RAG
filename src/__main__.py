from .upload_files import UploadDir
from .bm25searcher import BM25Searcher
import json
from colorama import Fore, Style

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

json_data = []

path = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
with open(path, "r", encoding="utf-8") as file:
    json_data = json.load(file)


if __name__ == "__main__":

    updir = UploadDir("data")
    updir.set_chunk_size(2000)
    updir.upload()

    searcher = BM25Searcher()
    searcher.set_documents(updir.get_documents())
    searcher.set_top_k(10)

    array = []


    for q in json_data["rag_questions"]:
        dic = {}
        dic["question_id"] = q["question_id"]
        dic["question"] = q["question"]
        ar = []
        question = q["question"]
        documents = searcher.query(question)
        for d in documents:
            dd = {}
            dd["file_path"] = d["path"].replace("\\", "/")
            dd["first_character_index"] = d["index"][0]
            dd["last_character_index"] = d["index"][1]
            ar.append(dd)

            # print(f"{Fore.YELLOW}\n\nQuestion: {question}{Style.RESET_ALL}")
            # print(f"{Fore.BLUE}path: {d['path']} [chunk: {d['chunk']}]{Style.RESET_ALL}")
            # text = d["text"]
            # for q in question.split(" "):
            #     if q not in STOP_WORDS:
            #         text = text.replace(q, f"{Fore.YELLOW}{q}{Fore.WHITE}") 

            # print(f"{Fore.WHITE}\ntext: {text}{Style.RESET_ALL}")
        dic["retrieved_sources"] = ar
        array.append(dic)
        # break

    my_dict = {}
    my_dict["search_results"] = array
    my_dict["k"] = 10

    with open("output.json", "w") as file:
        json.dump(my_dict, file, indent= 4)


# a = {

#     "search_results": [
#         {
#             "question_id": "q1",
#             "question": "How to configure OpenAI server?",
#             "retrieved_sources":
#             [
#                 {
#                     "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
#                     "first_character_index": 9867,
#                     "last_character_index": 10100
#                 },
#                 {
#                     "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
#                     "first_character_index": 267,
#                     "last_character_index": 400
#                 }
#             ]
#         }
#     ],
#     "k": 10
# }