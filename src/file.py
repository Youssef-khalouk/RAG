import ast

with open("bm25searcher.py", "r") as file:
    source = file.read()

# tree = ast.parse(source)

# for node in tree.body:
#     # print(type(node).__name__)
#     code = ast.get_source_segment(source, node)
#     print(code)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20
)

chunks = splitter.split_text(source)

for ch in chunks:
    print("####################################################################")
    print(ch)