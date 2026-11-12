"""Opt-in scoped cosine/BM25 retrieval, with the ordinary chunk-list output.

Hard exclusions apply before route top-k. Document preference breaks equal RRF
scores only in this first implementation: it cannot outweigh stronger relevance
or guarantee a particular source appears. No translation or provider defaults.
"""
from copy import deepcopy
from threading import Lock

from app.hybrid_retrieval import bm25_search, cosine_search, rrf_fuse
from app.retrieval_contract import RetrievalRequest, SourceScope


class ScopedHybridRetriever:
    def __init__(self, chunks, vectors_by_text, embed_query, *, route_k=5, result_k=5):
        if (not chunks or len({c['chunk_id'] for c in chunks}) != len(chunks)
                or any(not isinstance(c.get('metadata', {}).get('document_id'), str) for c in chunks)):
            raise ValueError('require_unique_chunks_with_document_ids')
        if any(c['content'] not in vectors_by_text for c in chunks):
            raise ValueError('missing_corpus_vectors')
        if any(type(k) is not int or k < 1 for k in (route_k, result_k)):
            raise ValueError('retrieval_sizes_must_be_positive')
        self.chunks = deepcopy(chunks)
        self.vectors = deepcopy(vectors_by_text)
        self.embed_query = embed_query
        self.route_k, self.result_k = route_k, result_k
        self.document_ids = {c['metadata']['document_id'] for c in chunks}
        self.trace = []
        self.lock = Lock()

    def __call__(self, query: str) -> list[dict]:
        """Compatibility adapter; scoped flows call search with an explicit request."""
        return self.search(RetrievalRequest(query))

    def validate_scope(self, scope: SourceScope) -> None:
        if not isinstance(scope, SourceScope):
            raise ValueError('invalid_source_scope')
        if (set(scope.preferred_document_ids) | set(scope.excluded_document_ids)) - self.document_ids:
            raise ValueError('scope_references_unknown_documents')

    def search(self, request: RetrievalRequest) -> list[dict]:
        if not isinstance(request, RetrievalRequest):
            raise TypeError('require_retrieval_request')
        with self.lock:
            scope = request.scope
            self.validate_scope(scope)
            excluded_ids = set(request.exclude_chunk_ids)
            excluded_docs = set(scope.excluded_document_ids)
            eligible = [c for c in self.chunks if c['chunk_id'] not in excluded_ids
                        and c['metadata']['document_id'] not in excluded_docs]
            result_limit = (self.result_k if request.max_results is None
                            else min(self.result_k, request.max_results))
            record = {'request': request.to_dict(), 'eligible_ids': [c['chunk_id'] for c in eligible],
                      'vector_ids': [], 'bm25_ids': [], 'result_ids': [],
                      'preference_policy': 'equal_rrf_score_tiebreak_only', 'embedding_requested': False,
                      'result_limit': result_limit}
            if not eligible:
                self.trace.append(record)
                return []
            if request.query not in self.vectors:
                record['embedding_requested'] = True
                self.vectors[request.query] = self.embed_query(request.query)
            vector = cosine_search(eligible, self.vectors, self.vectors[request.query], top_k=self.route_k)
            lexical = bm25_search(eligible, request.query, top_k=self.route_k)
            union_size = len({c['chunk_id'] for c in vector + lexical})
            fused = rrf_fuse({'vector': vector, 'bm25': lexical}, top_k=max(1, union_size))
            preferred = set(scope.preferred_document_ids)
            fused.sort(key=lambda c: (-c['rrf_score'], c['metadata']['document_id'] not in preferred, c['chunk_id']))
            result = fused[:result_limit]
            record.update(vector_ids=[c['chunk_id'] for c in vector],
                          bm25_ids=[c['chunk_id'] for c in lexical], result_ids=[c['chunk_id'] for c in result])
            self.trace.append(record)
            return deepcopy(result)
