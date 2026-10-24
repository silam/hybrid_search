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


def search_bm25(query: str, k: int=10) -> list[tuple[str, float]]:

    # Tokenize the query and retrieve the top_k results from the BM25 index.
    query_tokens = bm25s.tokenize([query], stopwords="en")

    # Ensure the query tokens are in the correct format for retrieval.
    # Retrieve the top_k documents based on the query tokens.
    # `indices` contains the document indices, and `scores` contains the corresponding BM25 scores.
    # Combine the document IDs with their corresponding scores to form the final results list.
    indices, scores = retriever.retrieve(query_tokens, k=k)
    results = [(doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]
    return results


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    k = 10
    results = search_bm25(query, k=k)
    for i, (doc_id, score) in enumerate(results):
        text = corpus.loc[corpus['_id'] == doc_id, 'text'].iloc[0]
        print(f"Document ID: {doc_id}, Score: {score}")
        print(f"Text: {text}\n")