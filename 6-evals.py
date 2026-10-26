import math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus, DATA_DIR
from utils.fusion import hybrid_candidates
from utils.reranker import rerank_with_cohere


RERANK_SAMPLE_SIZE = 50
SEED = 42


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:

    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(idx + 2)
        for idx, doc_id in enumerate(predicted_ids[:k])

    )

    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")


qrels: dict[str, dict[str, int]] = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])


"""
	_id	title	text
6000	4641		Where should I park my rainy-day / emergency f...
6001	5503		Tax considerations for selling a property belo...
6002	7803		Can the Delta be used to calculate the option ..
"""
queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()



"""
	_id	title	text
6635	3394		What is the easiest way to back-test index fun...
6220	5505		Can I deduct interest and fees on a loan for q...
6593	6792		Where to find the full book of outstanding bid...
6429	6909		Why do stocks priced above $2.00 on the ASX so..
"""
sample = queries_with_qrels.sample(n=RERANK_SAMPLE_SIZE, random_state=SEED)
print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")

bm25 = BM25Retriever()
dense = DenseRetriever()
corpus_by_id = load_corpus().set_index("_id")

def hybrid_topk(query: str, candidate_k: int = 10) -> list[str]:
    return [
        d for d, _ in hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    ]


def reranked_topk(query: str, candidate_k: int = 10) -> list[str]:
    candidates = hybrid_topk(query, candidate_k=candidate_k)
    return [
        d for d, _ in rerank_with_openrouter(query, candidates, corpus_by_id, k=candidate_k)
    ] 


results: dict[str, list[float]] = defaultdict(list)
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating queries"):
    query_id = str(row["_id"])
    query_text = row["text"]
    relevant = qrels[query_id]

    results["BM25"].append(
        ndcg_at_k([d for d, _ in bm25.search(query_text, k=10)], relevant)
    )
    results["Dense"].append(
        ndcg_at_k([d for d, _ in dense.search(query_text, k=10)], relevant)
    )
    results["Hybrid (RRF)"].append(ndcg_at_k(hybrid_topk(query_text)[:10], relevant))
    results["Hybrid + Rerank"].append(ndcg_at_k(reranked_topk(query_text), relevant))


if __name__ == "__main__":
    for method, scores in results.items():
        print(f"{method}: {np.mean(scores):.4f}")