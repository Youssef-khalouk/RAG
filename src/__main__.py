from rank_bm25 import BM25Okapi

documents = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "cats are wonderful pets"
]

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

query = "cat".split()

scores = bm25.get_scores(query)

print(scores)
