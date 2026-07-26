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


def print_data(query, documents, searcher=None) -> None:
    for d in documents:
        doc = searcher.get_document_content(d[1], d[2])
        print(f"{Fore.YELLOW}\n\nQuestion: {query}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}path: {doc['file_path']} [chunk: {doc['first_character_index']}]{Style.RESET_ALL}")
        text = doc["text"].replace("\n", "\n\t")
        for q in query.split(" "):
            if q.lower() not in STOP_WORDS:
                text = text.replace(q, f"{Fore.YELLOW}{q}{Fore.WHITE}")
        print(f"{Fore.GREEN}text:\n\t{Fore.WHITE}{text}{Style.RESET_ALL}")
