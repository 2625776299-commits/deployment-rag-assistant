from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection


def open_collection(path: Path, name: str) -> Collection:
    """Open a local cosine collection; embeddings are supplied by our code."""
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("Existing collection must use cosine distance")
    return collection


def save_chunks(
    collection: Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Save corresponding chunks and vectors, updating existing chunk IDs."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")
    if not chunks:
        return

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings,
    )


def search_chunks(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Return nearest chunks with original content, metadata and distance."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for index, chunk_id in enumerate(result["ids"][0]):
        matches.append(
            {
                "chunk_id": chunk_id,
                "content": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "distance": result["distances"][0][index],
            }
        )
    return matches

