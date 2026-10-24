import os
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "text-embedding-3-small"

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

def embed_batch(texts: list[str]) -> np.ndarray:
    """ Embed a list of texts into their corresponding vector representations using the OpenAI API. 

    Args:
        texts (list[str]): A list of text strings to be embedded.

    Returns:
        array of 1536 dimensions representing the embeddings of the input texts.
        np.ndarray: A 2D numpy array where each row corresponds to the embedding of a text.
    """
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def build_index(doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
    """ Build an index for a list of document texts by embedding them.

    Args:
        doc_texts (list[str]): A list of document text strings to be embedded.

    Returns:
        np.ndarray: A 2D numpy array where each row corresponds to the embedding of a document text.
    """
    embeddings = []
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding batches"):
        batch = doc_texts[i:i + batch_size]
        embeddings.append(embed_batch(batch)) 
    return np.vstack(embeddings)    



corpus = pd.read_parquet(DATA_DIR / "corpus.parquet") 
doc_ids = corpus["_id"].tolist()

doc_texts = [t.strip() or "[Empty document]" for t in corpus["text"].tolist()]