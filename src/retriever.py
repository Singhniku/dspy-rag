"""The retriever: given a question, return the most relevant chunks.

This is deliberately a plain, transparent class rather than a DSPy built-in
retriever. Keeping retrieval in our own code makes the pipeline easy to reason
about and lets DSPy focus on what it's best at: optimizing the *generation*
prompt. The retriever embeds the question with the SAME model used at index
time, then asks ChromaDB for the nearest chunk vectors (cosine similarity).
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from . import config


class ChromaRetriever:
    def __init__(self):
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(
            name=config.COLLECTION_NAME, embedding_function=embed_fn
        )

    def __call__(self, question: str, k: int = config.TOP_K) -> list[str]:
        """Return the top-k chunk texts most relevant to the question."""
        result = self.collection.query(query_texts=[question], n_results=k)
        # query returns a batch (one row per query text); we sent one query.
        return result["documents"][0]
