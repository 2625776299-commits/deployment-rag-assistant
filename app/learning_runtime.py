"""Explicit learning profile; importing this module sends no requests.

Both retrieval routes use the same frozen structured corpus. Exact cosine here
is not a Chroma/HNSW deployment. Each run keeps its own attempt records.
"""

from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from tempfile import mkdtemp

from app.hybrid_retrieval import bm25_search, cosine_search, rrf_fuse
from app.main import create_app
from app.requirement_evaluation import evaluate_requirements


from dotenv import load_dotenv
PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / '.env', override=False)
LEGACY_SNAPSHOT = PROJECT / "data/chunking_experiments/strategy_20261018T065249212222Z"
SNAPSHOT = PROJECT / os.getenv('RAG_SNAPSHOT_PATH', str(LEGACY_SNAPSHOT))


class RunAccounting:
    """Record attempts before I/O, including attempts which later fail; no quota."""

    def __init__(self, path: Path, learning_date: str):
        self.path, self.lock = path, Lock()
        self.data = {"learning_date": learning_date,
                     "attempts": {"generation": 0, "embedding": 0}, "events": []}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)

    def reserve(self, kind: str, stage: str) -> int:
        with self.lock:
            used = self.data["attempts"][kind]
            self.data["attempts"][kind] = used + 1
            self.data["events"].append({"kind": kind, "stage": stage, "attempt": used + 1})
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            return used + 1


class HybridRetriever:
    """Five candidates per route -> RRF top five -> coverage evaluator.

At most five new chunks per extra retrieval keeps the existing controller's
merged pool within ten. Final answer evidence is selected later (at most three).
These are initial learning-profile limits, not experimentally optimal values.
"""

    def __init__(self, chunks: list[dict], vectors_by_text: dict, embed_query):
        self.chunks = deepcopy(chunks)
        self.vectors = deepcopy(vectors_by_text)
        self.embed_query = embed_query
        self.lock = Lock()
        self.trace = []
        if not self.chunks or len({c["chunk_id"] for c in self.chunks}) != len(self.chunks):
            raise ValueError("Require a nonempty corpus with unique chunk IDs")
        if any(c["content"] not in self.vectors for c in self.chunks):
            raise ValueError("Missing corpus vectors; do not embed the corpus implicitly")

    def __call__(self, question: str) -> list[dict]:
        with self.lock:
            cached = question in self.vectors
            if not cached:
                self.vectors[question] = self.embed_query(question)
            vector = cosine_search(self.chunks, self.vectors, self.vectors[question], top_k=5)
            lexical = bm25_search(self.chunks, question, top_k=5)
            fused = rrf_fuse({"vector": vector, "bm25": lexical}, top_k=5)
            self.trace.append({"query": question, "query_vector_cached": cached,
                "vector_ids": [c["chunk_id"] for c in vector],
                "bm25_ids": [c["chunk_id"] for c in lexical],
                "fused_ids": [c["chunk_id"] for c in fused]})
            return deepcopy(fused)


def load_snapshot(snapshot: Path = SNAPSHOT) -> tuple[list[dict], dict, dict]:
    def read(name):
        return json.loads((snapshot / name).read_text(encoding="utf-8-sig"))

    manifest = read("manifest.json")
    if manifest["status"] != "complete" or not manifest["inputs_and_code_unchanged"]:
        raise ValueError("Require a completed frozen corpus run")
    for filename, expected in manifest["input_and_code_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Protected source or baseline code changed")
    chunks = read("structured_context0_chunks.json")
    cache = read("embedding_vectors.json")
    if (len(cache["texts"]) != len(cache["vectors"])
            or len(set(cache["texts"])) != len(cache["texts"])
            or cache["usage"]["model"] != manifest["embedding_model"]
            or any(len(v) != manifest["embedding_dimension"] for v in cache["vectors"])):
        raise ValueError("Cached vector model, dimensions, or alignment differ")
    return chunks, dict(zip(cache["texts"], cache["vectors"])), manifest


def create_learning_app(*, learning_date: str, clarification_findings: bool = False,
                        initial_scope=None, scope_policy=None):
    """Configure real callbacks explicitly with a fresh run directory each time.

    No daily call cap, automatic translation/reranking, or provider retries.
    """
    # Lazy imports preserve the ordinary app's offline/import-only behavior.
    from app.ask import create_generation_client, generate_answer
    from app.compare_reranking import request_order
    from app.embedding_runtime import BASE_URL, MODEL, DIMENSIONS, create_embedding_client, embed_texts

    # The date also forms a filename; accept only an ISO calendar date.
    from datetime import date
    if date.fromisoformat(learning_date).isoformat() != learning_date:
        raise ValueError("Use an ISO learning date")
    chunks, vectors, manifest = load_snapshot()
    if MODEL != manifest["embedding_model"] or BASE_URL != manifest["embedding_base_url"]:
        raise ValueError("Embedding configuration differs from the frozen cache")
    if DIMENSIONS is not None and DIMENSIONS != manifest['embedding_dimension']:
        raise ValueError('Embedding dimensions differ from the frozen cache')
    from app.retrieval_contract import SourceScope
    if initial_scope is not None and not isinstance(initial_scope, SourceScope):
        raise ValueError('Invalid initial scope')
    if scope_policy is not None and (initial_scope is None or not callable(scope_policy)):
        raise ValueError('Scope policy requires explicit scoped mode and must be callable')
    if initial_scope is not None:
        documents = {c['metadata']['document_id'] for c in chunks}
        if (set(initial_scope.preferred_document_ids) | set(initial_scope.excluded_document_ids)) - documents:
            raise ValueError('Scope references unknown documents in the selected snapshot')
    client, model = create_generation_client()
    root = PROJECT / "data/http_experiments"
    try:
        root.mkdir(parents=True, exist_ok=True)
        output = Path(mkdtemp(prefix="live_" + learning_date + "_", dir=root))
        accounting = RunAccounting(output / "attempts.json", learning_date)
    except Exception:
        client.close()
        raise

    def embed_query(question):
        attempt = accounting.reserve("embedding", "query_embedding")
        usage = {}
        with create_embedding_client() as embedding_client:
            vector = embed_texts(embedding_client, [question], usage_report=usage)[0]
        if len(vector) != manifest["embedding_dimension"]:
            raise ValueError("New query vector dimension differs")
        (output / f"embedding_{attempt}_usage.json").write_text(
            json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
        return vector

    def evaluate_text(messages):
        attempt = accounting.reserve("generation", "evaluate")
        return request_order(client, model, messages, output / f"model_{attempt}_evaluate.json")

    def generate(question, evidence):
        attempt = accounting.reserve("generation", "answer")
        return generate_answer(client, model, question, evidence,
                               raw_output_path=output / f"model_{attempt}_answer.json")

    try:
        scoped_options = {}
        if initial_scope is not None:
            from app.scoped_retrieval import ScopedHybridRetriever
            retrieve = ScopedHybridRetriever(chunks, vectors, embed_query)
            scoped_options = {'request_retrieval': retrieve.search, 'initial_scope': initial_scope,
                              'scope_policy': scope_policy}
        else:
            retrieve = HybridRetriever(chunks, vectors, embed_query)
        app = create_app(retrieve=retrieve,
                         evaluate=partial(evaluate_requirements, generate=evaluate_text,
                                          repair_unselected_once=True,
                                          include_user_reply=clarification_findings),
                         generate=generate, **scoped_options)
    except Exception:
        client.close()
        raise
    app.state.learning_runtime = {"retriever": retrieve, "accounting": accounting,
        "output": output, "client": client, "model": model,
        "vector_snapshot": str(SNAPSHOT),
        "embedding_model": manifest['embedding_model'],
        "embedding_dimensions": manifest.get('embedding_dimension'),
        "profile": "structured-context0/" + ('scoped-' if initial_scope is not None else '') + "exact-cosine+raw-bm25/rrf5/coverage-final3",
        "retrieval_contract": 'request-v1' if initial_scope is not None else 'query-string',
        "initial_scope": (dict(preferred_document_ids=initial_scope.preferred_document_ids,
                               excluded_document_ids=initial_scope.excluded_document_ids)
                          if initial_scope is not None else None),
        "clarification_findings": clarification_findings}
    return app
