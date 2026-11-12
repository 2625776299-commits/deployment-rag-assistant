import hashlib
import json
import os

from app.embedding_runtime import BASE_URL, MODEL, DIMENSIONS, create_embedding_client, embed_texts
from app.loader import PROJECT_ROOT
from app.vector_store import open_collection, save_chunks


CHUNKS_PATH = PROJECT_ROOT / os.getenv('RAG_CHUNKS_PATH', 'data/chunks.json')
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
MANIFEST_PATH = PROJECT_ROOT / os.getenv('RAG_INDEX_MANIFEST', 'data/index_manifest.json')


def main() -> None:
    raw_chunks = CHUNKS_PATH.read_bytes()
    chunks = json.loads(raw_chunks.decode("utf-8"))
    if not chunks:
        raise ValueError("No chunks available; run python -m app.prepare_chunks first")

    corpus_hash = hashlib.sha256(raw_chunks).hexdigest()
    identity = f"{BASE_URL}|{MODEL}|{DIMENSIONS}|{corpus_hash}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    collection_name = f"deploy-rag-{suffix}"
    collection = open_collection(CHROMA_PATH, collection_name)

    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous.get("collection_name") == collection_name and collection.count() == len(chunks):
            print(f"Index already ready: {collection.count()} chunks; no embedding request sent")
            return

    texts = [chunk["content"] for chunk in chunks]
    with create_embedding_client() as client:
        vectors = embed_texts(client, texts)
    save_chunks(collection, chunks, vectors)
    if collection.count() != len(chunks):
        raise ValueError("Stored chunk count does not match the source chunks")

    manifest = {
        "collection_name": collection_name,
        "embedding_model": MODEL,
        "embedding_base_url": BASE_URL,
        "embedding_dimension": len(vectors[0]),
        "corpus_sha256": corpus_hash,
        "chunk_count": len(chunks),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed chunks: {collection.count()}")
    print(f"Collection: {collection_name}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
