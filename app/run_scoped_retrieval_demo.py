"""Offline architecture walkthrough: scripted gaps, real scoped search/control.

Text, source URLs and two-dimensional vectors below are hand-written fixtures,
not added production documents or a semantic-quality experiment.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import mkdtemp

from app.bounded_retrieval import run_bounded_retrieval
from app.retrieval_contract import SourceScope
from app.scoped_retrieval import ScopedHybridRetriever


QUERY = 'connection pool handling'


def fixture():
    def chunk(key, document, text):
        return {'chunk_id': key, 'content': text, 'metadata': {
            'document_id': document, 'section_id': key, 'source_url': 'https://example.invalid/fixture/' + document}}
    chunks = [
        chunk('D0', 'docker-networking', 'Old connections close after the database container is replaced.'),
        chunk('S1', 'sqlalchemy-pooling', 'connection pool handling: check stale connections when checking out.'),
        chunk('D1', 'docker-networking', 'Applications detect broken connections and reconnect to the service name.'),
        chunk('P1', 'psycopg-pooling', 'connection pool handling using psycopg APIs.'),
    ]
    vectors = {c['content']: v for c, v in zip(chunks, ([1., 0.], [1., 0.], [.9, .1], [1., 0.]))}
    vectors[QUERY] = [1., 0.]
    return chunks, vectors


def run_demo():
    chunks, vectors = fixture()
    original = deepcopy(chunks)
    def no_embedding(_):
        raise AssertionError('Offline fixture must not request embeddings')
    search = ScopedHybridRetriever(chunks, vectors, no_embedding, result_k=1)
    evaluated = []

    def evaluate(question, pool):
        ids = [c['chunk_id'] for c in pool]
        evaluated.append(ids)
        gap = ['SQLAlchemy checkout handling'] if 'S1' not in ids else ['Docker application reconnect handling']
        complete = 'S1' in ids and 'D1' in ids
        return {'status': 'proposed', 'proposal': {'action': 'answer' if complete else 'retrieve',
                'search_query': None if complete else QUERY, 'missing_knowledge': [] if complete else gap},
                'selected_evidence': pool if complete else [], 'coverage_verified': False}

    def policy(feedback):
        preferred = 'docker-networking' if any('Docker' in gap for gap in feedback.missing_knowledge) else 'sqlalchemy-pooling'
        return SourceScope(preferred_document_ids=(preferred,), excluded_document_ids=('psycopg-pooling',))

    def no_legacy_lookup(_):
        raise AssertionError('Explicit request path must be used')
    result = run_bounded_retrieval('Fixture: database container replaced, SQLAlchemy connection fails',
        chunks[:1], evaluate, no_legacy_lookup, request_retrieval=search.search,
        scope_policy=policy, max_retrievals=2, max_evaluations=3)
    assert evaluated == [['D0'], ['D0', 'S1'], ['D0', 'S1', 'D1']]
    assert result['status'] == 'answer_proposed'
    assert chunks == original
    return {'scope': 'Offline interface/control demonstration with scripted evaluation and fixture vectors',
            'live_model_calls': 0, 'embedding_calls': 0, 'real_corpus_modified': False,
            'http_entry_enabled': False, 'semantic_quality_verified': False,
            'preference_policy': 'Equal RRF scores only; not a hard source whitelist or proven relevance improvement',
            'evaluated_pools': evaluated, 'retrieval_trace': search.trace, 'controller_result': result}


def main():
    root = Path(__file__).resolve().parents[1] / 'data/architecture_demos'
    root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='scoped_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=root))
    report = run_demo()
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'status': report['controller_result']['status'],
                      'evaluated_pools': report['evaluated_pools'], 'model_calls': 0, 'embedding_calls': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
