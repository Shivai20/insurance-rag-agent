"""Retriever: loads the persisted Chroma index and exposes a retriever.

Interview note: the retriever is the only thing the graph knows about. Swapping
Chroma -> pgvector / Pinecone / FAISS happens here and nowhere else.
"""
import os
from functools import lru_cache

from langchain_chroma import Chroma

from app import config
from app.llm import get_embeddings


@lru_cache(maxsize=1)
def get_vectordb() -> Chroma:
    if not os.path.exists(config.CHROMA_DIR):
        raise RuntimeError(
            "Vector index not found. Run `python -m app.ingest` first."
        )
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def retrieve(query: str, k: int = config.TOP_K):
    """Return top-k relevant chunks for a query."""
    return get_vectordb().similarity_search(query, k=k)
