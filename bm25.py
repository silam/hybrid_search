from pathlib import Path
import pandas as pd
import bm25s



DATA_DIR = Path(__file__).parent / "data" / "fiqa" / "fiqa"
INDEX_DIR = DATA_DIR / "indexes" / "bm25"


corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus['_id'].tolist()
doc_texts = corpus['text'].tolist()

print(f"Index Directory: {INDEX_DIR}")
print(f"Number of Documents:       {len(doc_texts)} ")

# Tokenize the document texts using BM25's tokenizer with English stopwords
# The resulting `tokens` variable will be a list of tokenized documents, ready for BM25 indexing.   

tokens = bm25s.tokenize(doc_texts, stopwords="en")

print(tokens.ids[:1])
print(tokens.vocab.items())

retriever = bm25s.BM25()
print(f"Retriever created with {len(retriever.tokens.ids)} documents.")
retriever.index(tokens)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(INDEX_DIR)
print(f"Index saved to {INDEX_DIR}")
(INDEX_DIR/"doc_ids.txt").write_text("\n".join(doc_ids))
print(f"Document IDs saved to {INDEX_DIR/'doc_ids.txt'}")

