"""Ingestion pipeline: markdown KB -> chunks -> embeddings -> Chroma.

This is the 'knowledge fabric'. Kept deliberately simple (plain markdown) so it
runs anywhere with no OCR/system deps. The pattern (load -> split -> embed ->
persist with metadata) is identical for any source.

Run once:  python -m app.ingest
"""
import glob
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app import config
from app.llm import get_embeddings
from app.observability import log_event


def _load_kb_documents() -> list[Document]:
    docs = []
    for path in sorted(glob.glob(os.path.join(config.KB_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        # Metadata travels with every chunk -> enables citations + filtered retrieval.
        docs.append(Document(page_content=text, metadata={"source": source}))
    return docs


def build_index() -> Chroma:
    """(Re)build the vector index from the KB and persist it to disk."""
    raw_docs = _load_kb_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],  # respect markdown structure
    )
    chunks = splitter.split_documents(raw_docs)
    log_event("ingest_chunks", n_docs=len(raw_docs), n_chunks=len(chunks))

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    log_event("ingest_done", persist_dir=config.CHROMA_DIR)
    return vectordb


if __name__ == "__main__":
    build_index()
    print(f"Index built and persisted to {config.CHROMA_DIR}")
