from collections import defaultdict
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

K_RRF = 60

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = K_RRF) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)

    ranked_docs = sorted(scores.items(), key=lambda x: -x[1])
    return ranked_docs


bm25 = BM25Retriever()
dense = DenseRetriever()
corpus = load_corpus()


def search_hybrid(query: str, k: int = 10, candidates: int = K_RRF) -> list[tuple[str, float]]:
    bm25_results = [doc_id for doc_id, _ in bm25.search(query, k=candidates)]
    dense_results = [doc_id for doc_id, _ in dense.search(query, k=candidates)]
    return reciprocal_rank_fusion([bm25_results, dense_results], k=k)   



def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"{label}:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank}. {doc_id} (score: {score:.4f})")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    results = search_hybrid(query)
    show("BM25 only", bm25.search(query, k=10))
    show("Dense only", dense.search(query, k=10))    
    show("Hybrid Search RRF Results", search_hybrid(query, k=10))