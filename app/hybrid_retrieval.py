"""Small offline BM25/cosine/RRF baseline; no external service dependencies.

The tokenizer preserves technical identifiers and uses character bigrams for
Chinese. It does not translate queries, stem English words or infer semantics.
BM25 uses log(1 + (N-df+0.5)/(df+0.5)) IDF and the standard saturated TF form.
This is a transparent baseline, not bit-for-bit Elasticsearch compatibility.
"""

import math
import re
from collections import Counter


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*|[0-9]+(?:\.[0-9]+)*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN.finditer(text.lower()):
        term = match.group()
        if "\u4e00" <= term[0] <= "\u9fff":
            if len(term) > 1:
                tokens.extend(term[i:i + 2] for i in range(len(term) - 1))
            else:
                tokens.append(term)
        else:
            tokens.append(term)
    return tokens


def bm25_search(chunks: list[dict], query: str, top_k: int = 5, *, k1: float = 1.2, b: float = 0.75) -> list[dict]:
    if top_k <= 0 or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Require top_k > 0, k1 > 0 and 0 <= b <= 1")
    docs = [Counter(tokenize(chunk["content"])) for chunk in chunks]
    lengths = [sum(doc.values()) for doc in docs]
    if not docs or not sum(lengths):
        return []
    average = sum(lengths) / len(docs)
    query_terms = list(dict.fromkeys(tokenize(query)))
    frequencies = {term: sum(term in doc for doc in docs) for term in query_terms}
    results = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        matched = [term for term in query_terms if term in doc]
        score = 0.0
        for term in matched:
            df, tf = frequencies[term], doc[term]
            idf = math.log1p((len(docs) - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        # Zero-score documents are not lexical hits and must not gain RRF votes.
        if score > 0:
            results.append({**chunk, "score": score, "matched_terms": matched})
    return sorted(results, key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]


def cosine_search(chunks: list[dict], vectors_by_text: dict[str, list[float]], query_vector: list[float], top_k: int = 5) -> list[dict]:
    if top_k <= 0 or not query_vector or not all(math.isfinite(x) for x in query_vector):
        raise ValueError("Require positive top_k and a finite, nonempty query vector")
    query_norm = math.sqrt(math.fsum(x * x for x in query_vector))
    if not query_norm:
        raise ValueError("Query vector cannot be zero")
    results = []
    for chunk in chunks:
        vector = vectors_by_text[chunk["content"]]
        if len(vector) != len(query_vector) or not all(math.isfinite(x) for x in vector):
            raise ValueError("Corpus vectors must have matching dimensions and finite values")
        norm = math.sqrt(math.fsum(x * x for x in vector))
        if not norm:
            raise ValueError("Corpus vector cannot be zero")
        score = math.fsum(x * y for x, y in zip(query_vector, vector)) / (query_norm * norm)
        results.append({**chunk, "score": score, "distance": 1 - score})
    return sorted(results, key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]


def rrf_fuse(rankings: dict[str, list[dict]], top_k: int = 3, *, rank_constant: int = 60) -> list[dict]:
    """Fuse the UNION by chunk_id; retain one body and every route's rank/score."""
    if top_k <= 0 or rank_constant < 1:
        raise ValueError("Require top_k > 0 and rank_constant >= 1")
    merged = {}
    for route, results in rankings.items():
        seen = set()
        for rank, item in enumerate(results, 1):
            key = item["chunk_id"]
            if key in seen:
                raise ValueError("A route cannot vote more than once for the same chunk")
            seen.add(key)
            if key not in merged:
                merged[key] = {"chunk_id": key, "content": item["content"], "metadata": item["metadata"],
                               "rrf_score": 0.0, "routes": {}}
            record = merged[key]
            if (record["content"], record["metadata"]) != (item["content"], item["metadata"]):
                raise ValueError("The same chunk_id has inconsistent evidence across routes")
            record["rrf_score"] += 1 / (rank_constant + rank)
            record["routes"][route] = {"rank": rank, "score": item["score"]}
    # ID is only a deterministic tie-breaker, not another relevance signal.
    return sorted(merged.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))[:top_k]
