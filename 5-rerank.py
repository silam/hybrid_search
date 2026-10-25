import os
import cohere
from dotenv import load_dotenv
from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus


load_dotenv()
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"

bm25 = BM25Retriever()
dense = DenseRetriever()

# Load the corpus and index it by document ID for quick access
corpus_by_id = load_corpus().set_index("_id")

def search_reranked(
    query: str,
    k: int = 10,
    candidates: int = 50
) -> list[tuple[str, float]]:     

    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidates)  
    
    candidate_ids = [doc_id for doc_id, _ in candidates]
    candidate_texts = [corpus_by_id.loc[doc_id, "text"] for doc_id in candidate_ids]

    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=candidate_texts,
        top_n=k
    )

    return [(candidate_ids[r.index], r.relevance_score) for r in response.results]


def show(label: str, results: list[tuple[str, float]]) -> None:

    for i,(doc_id, score) in enumerate(results[:5], start=1):
        print(f"{label} {i}: {doc_id} (score: {score})")
        text = corpus_by_id.loc[doc_id, "text"]
        print(f"Text: {text}\n")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Hybrid search for query: {query}")
    print(f"hybrid_candidates ",  hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])

    print(f"Reranked results for query", search_reranked(query, k=10, candidates=50))

"""
Hybrid search for query: Where should I park my rainy-day fund?
                                                     
hybrid_candidates  [('32833', 0.03225806451612903), ('497993', 0.030776515151515152), ('376148', 0.027629397679130595), ('7625', 0.027205882352941177), ('345428', 0.02690100430416069)]
                                                     
Reranked results for query [('376148', 0.82310104), ('580025', 0.80415994), ('8088', 0.77294225), ('285812', 0.72299665), ('497993', 0.69064254), ('261983', 0.68729424), ('345428', 0.6856128), ('21420', 0.6563059), ('538023', 0.6456563), ('527939', 0.6438669)]
"""