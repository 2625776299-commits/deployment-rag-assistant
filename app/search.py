import argparse
import hashlib
import json
import sys

from app.build_index import CHROMA_PATH, CHUNKS_PATH, MANIFEST_PATH
from app.embedding_runtime import BASE_URL, MODEL, DIMENSIONS, create_embedding_client, embed_texts
from app.vector_store import open_collection, search_chunks


def retrieve_chunks(
    question: str,
    top_k: int = 3,
    *,
    usage_report: dict | None = None,
) -> list[dict]:
    """Retrieve evidence for either the search CLI or the answer pipeline."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not MANIFEST_PATH.exists():
        raise ValueError("Index missing; run python -m app.build_index first")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["embedding_model"] != MODEL or manifest["embedding_base_url"] != BASE_URL:
        raise ValueError("Embedding configuration changed; rebuild the index first")
    if DIMENSIONS is not None and manifest['embedding_dimension'] != DIMENSIONS:
        raise ValueError('Embedding dimensions changed; rebuild the index first')
    if manifest["corpus_sha256"] != hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest():
        raise ValueError("Source chunks changed; rebuild the index first")

    collection = open_collection(CHROMA_PATH, manifest["collection_name"])
    if collection.count() != manifest["chunk_count"]:
        raise ValueError("Stored index is incomplete; rebuild it first")
    with create_embedding_client() as client:
        query_vector = embed_texts(client, [question], usage_report=usage_report)[0]
    if len(query_vector) != manifest["embedding_dimension"]:
        raise ValueError("Query embedding dimension differs from the index")
    return search_chunks(collection, query_vector, top_k=top_k)


def main() -> None:
    # Preserve Chinese text and typographic punctuation in Windows tool output.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Retrieve source chunks for a question")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    matches = retrieve_chunks(args.question, top_k=args.top_k)

    for rank, match in enumerate(matches, start=1):
        print(f"\n{rank}. {match['chunk_id']} | cosine distance: {match['distance']:.4f}")
        print(f"Source section: {match['metadata']['section_id']}")
        print(f"URL: {match['metadata']['source_url']}")
        print(match["content"])


if __name__ == "__main__":
    main()
