"""
Polished version of the Cohere cross-encoder rerank built in 5-rerank.py.
6-evaluate.py imports from here so the eval script can focus on the metric.
"""

import os
import cohere
from fastapi import requests
import pandas as pd
from dotenv import load_dotenv
from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


"""
OpenRouter API configuration for reranking.
"""
RERANK_MODEL = "rerank-v4.0-fast"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/rerank"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}



def rerank_with_openrouter(
         query: str,
            candidate_ids: list[str],
            corpus_by_id: pd.DataFrame,
            k: int = 10,
        ) -> list[tuple[str, float]]:

        documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]

        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": k,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        reranked_indices = [int(i) for i in response.json()["top_n"]]
        return [(candidate_ids[i], float(response.json()["scores"][i])) for i in reranked_indices]



def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]
    response = co.rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=k)
    return [(candidate_ids[r.index], r.relevance_score) for r in response.results]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_cohere(query, candidate_ids, corpus_by_id, k=k)