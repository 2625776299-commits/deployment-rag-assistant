"""A validated artifact build and request-local RAG dependencies.

These artifacts use exact cosine + BM25, not an external ANN index. A build's
embedding identity is independent of the generation model used to read evidence.
"""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from app.retrieval_contract import RetrievalRequest
from app.scoped_retrieval import ScopedHybridRetriever


@dataclass(frozen=True)
class BuildManifest:
    build_id: str
    artifact_directory: str  # Relative to the configured artifact root.
    chunks_sha256: str
    vectors_sha256: str
    chunk_count: int
    embedding_model: str
    embedding_base_url: str
    embedding_dimension: int
    retrieval_kind: str = 'exact-cosine-bm25-v1'

    def __post_init__(self):
        for name in ('build_id', 'artifact_directory', 'embedding_model', 'embedding_base_url'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('invalid_' + name)
        for name in ('chunks_sha256', 'vectors_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid_' + name)
        if any(type(v) is not int or v < 1 for v in (self.chunk_count, self.embedding_dimension)):
            raise ValueError('invalid_build_sizes')
        if self.retrieval_kind != 'exact-cosine-bm25-v1':
            raise ValueError('unsupported_retrieval_kind')

    def to_dict(self):
        return asdict(self)


def load_build(manifest: BuildManifest, artifact_root: Path) -> tuple[list[dict], dict]:
    """Read exact bytes once, then verify identity, hashes and vector alignment."""
    root = artifact_root.resolve()
    relative = Path(manifest.artifact_directory)
    directory = (root / relative).resolve()
    if relative.is_absolute() or directory == root or not directory.is_relative_to(root):
        raise ValueError('artifact_directory_outside_root')
    raw = {}
    for name, expected in (('chunks.json', manifest.chunks_sha256), ('vectors.json', manifest.vectors_sha256)):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('artifact_file_outside_build')
        body = path.read_bytes()
        if sha256(body).hexdigest() != expected:
            raise ValueError('artifact_hash_mismatch:' + name)
        raw[name] = json.loads(body)
    chunks, vector_file = raw['chunks.json'], raw['vectors.json']
    if not isinstance(chunks, list) or len(chunks) != manifest.chunk_count:
        raise ValueError('chunk_count_mismatch')
    identity = ('embedding_model', 'embedding_base_url', 'embedding_dimension')
    if not isinstance(vector_file, dict) or any(vector_file.get(k) != getattr(manifest, k) for k in identity):
        raise ValueError('embedding_identity_mismatch')
    by_id = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError('invalid_chunk')
        key, text, meta = chunk.get('chunk_id'), chunk.get('content'), chunk.get('metadata')
        if (not isinstance(key, str) or not key.strip() or key in by_id
                or not isinstance(text, str) or not text.strip() or not isinstance(meta, dict)
                or any(not isinstance(meta.get(k), str) or not meta[k].strip()
                       for k in ('document_id', 'section_id', 'source_url'))):
            raise ValueError('invalid_chunk_identity_or_source')
        if meta.get('build_id', manifest.build_id) != manifest.build_id:
            raise ValueError('chunk_build_mismatch')
        meta['build_id'] = manifest.build_id
        by_id[key] = chunk
    rows = vector_file.get('vectors')
    if not isinstance(rows, list) or len(rows) != len(chunks):
        raise ValueError('vector_count_mismatch')
    seen, vectors = set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid_vector_row')
        key, values = row.get('chunk_id'), row.get('embedding')
        if not isinstance(key, str) or key not in by_id or key in seen:
            raise ValueError('vector_chunk_identity_mismatch')
        text = by_id[key]['content']
        if row.get('content_sha256') != sha256(text.encode('utf-8')).hexdigest():
            raise ValueError('vector_content_mismatch')
        if (not isinstance(values, list) or len(values) != manifest.embedding_dimension
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not any(values)):
            raise ValueError('invalid_vector_dimension_or_values')
        if text in vectors and vectors[text] != values:
            raise ValueError('same_text_has_different_vectors')
        seen.add(key)
        vectors[text] = values
    return chunks, vectors


@dataclass(frozen=True)
class PinnedRag:
    """One instance per request. No stage reads the active catalog pointer again."""
    build: BuildManifest
    retriever: ScopedHybridRetriever
    evaluator: Callable
    generator: Callable

    def check_evidence(self, chunks):
        canonical = {c['chunk_id']: c for c in self.retriever.chunks}
        for chunk in chunks:
            original = canonical.get(chunk.get('chunk_id'))
            if (original is None or chunk.get('metadata', {}).get('build_id') != self.build.build_id
                    or chunk.get('content') != original['content']
                    or chunk.get('metadata') != original['metadata']):
                raise ValueError('evidence_does_not_belong_to_pinned_build')

    def search(self, request: RetrievalRequest):
        if request.build_id != self.build.build_id:
            raise ValueError('retrieval_build_mismatch')
        chunks = self.retriever.search(request)
        self.check_evidence(chunks)
        return chunks

    def retrieve(self, query):
        return self.search(RetrievalRequest(query, build_id=self.build.build_id))

    def evaluate(self, question, candidates):
        self.check_evidence(candidates)
        return self.evaluator(question, deepcopy(candidates), self.build)

    def generate(self, question, evidence):
        self.check_evidence(evidence)
        return self.generator(question, deepcopy(evidence), self.build)


def build_request_binder(catalog, artifact_root: Path, *, bind_embedding: Callable,
                         evaluate: Callable, generate: Callable, route_k=5, result_k=5):
    """Return the zero-argument factory called once at each /ask request boundary.

    bind_embedding receives the pinned manifest and must bind that model/endpoint,
    rather than resolve a mutable 'current embedding model' during a later query.
    Evaluation/generation callbacks receive (question, evidence, manifest).
    """
    def bind(build_id=None):
        # Recovery uses the saved ID, even if another build has become active.
        manifest = catalog.read_active() if build_id is None else catalog.read_build(build_id)
        chunks, vectors = load_build(manifest, artifact_root)
        embedding = bind_embedding(manifest)
        def checked_embedding(query):
            values = embedding(query)
            if (not isinstance(values, list) or len(values) != manifest.embedding_dimension
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or not any(values)):
                raise ValueError('query_embedding_dimension_or_values_mismatch')
            return values
        retriever = ScopedHybridRetriever(chunks, vectors, checked_embedding,
                                          route_k=route_k, result_k=result_k)
        return PinnedRag(manifest, retriever, evaluate, generate)
    return bind
