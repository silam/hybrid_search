from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "fiqa" / "fiqa"

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")


print(f"Corpus: {corpus['text'].iloc[0] } ")
print(f"Queries: {queries['text'].iloc[0] } ")
print(f"Qrels: {qrels.iloc[0] } ")

# step 2: filter the queries to only include the test queries
# use Set to get unique test query IDs
test_query_ids = set(qrels['query-id'].astype(str)) # no duplicates

test_queries = queries[queries['_id'].astype(str).isin(test_query_ids)]
print(f"Test Queries: {test_queries['text'].iloc[0] } ")
print(f"Number of Test Queries: {len(test_queries)}")

query = test_queries.iloc[0]
print(f"First Test Query: {query['text']}")



relevant_docs = qrels[qrels['query-id'].astype(str) == str(query['_id'])]
print(f"Relevant Docs: {list(relevant_docs['corpus-id']) } ")