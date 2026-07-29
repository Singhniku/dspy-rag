"""Milestone 2 — Build the vector index (the 'R' setup in RAG).

Pipeline: read .md docs -> split into overlapping chunks -> embed each chunk
with a local sentence-transformer -> store vectors + text in ChromaDB (on disk).

Run once (or whenever the docs change):
    python -m src.ingest
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from . import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows.

    Why overlap? A fact can straddle a chunk boundary. Overlap means the
    sentence at the end of one chunk also appears at the start of the next,
    so we never slice a fact in half and lose it from every chunk.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  # step forward, but leave `overlap` behind
    return chunks


def load_documents() -> list[tuple[str, str]]:
    """Return (source_filename, full_text) for every .md file in data/."""
    docs = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_index() -> int:
    """Chunk, embed, and persist all documents. Returns the chunk count."""
    # One embedding function, used consistently for indexing AND querying.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )

    # PersistentClient writes to disk so the index survives across runs.
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Start clean each build so re-runs are idempotent.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME, embedding_function=embed_fn
    )

    ids, texts, metadatas = [], [], []
    for source, text in load_documents():
        for i, chunk in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{source}::chunk{i}")
            texts.append(chunk)
            metadatas.append({"source": source})

    # ChromaDB embeds every text with embed_fn under the hood and stores it.
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB at {config.CHROMA_DIR}")
    print(f"Collection: {config.COLLECTION_NAME}  |  embed model: {config.EMBED_MODEL}")
