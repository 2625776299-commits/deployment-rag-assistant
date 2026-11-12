"""Isolated PostgreSQL + ASGI publication demo; scripted models, fixture vectors.

Requires an explicitly provided disposable PostgreSQL instance. Never reads .env.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import httpx

from app.build_catalog import PostgresBuildCatalog
from app.build_runtime import BuildManifest, build_request_binder
from app.main import create_app


QUESTION = 'old connections close'
SUPPLEMENT = 'application reconnect'


def write_fixture_build(root: Path, build_id: str) -> BuildManifest:
    directory = root / build_id
    directory.mkdir(parents=True, exist_ok=False)
    chunks, rows = [], []
    for key, text, vector in (('P1:1', QUESTION, [1., 0.]), ('P2:1', SUPPLEMENT, [0., 1.])):
        text += ' ' + build_id
        chunks.append({'chunk_id': key, 'content': text, 'metadata': {
            'document_id': 'fixture-network', 'section_id': key.split(':')[0],
            'source_url': 'https://example.invalid/fixture'}})
        rows.append({'chunk_id': key, 'content_sha256': sha256(text.encode()).hexdigest(), 'embedding': vector})
    identity = {'embedding_model': 'fixture-2d', 'embedding_base_url': 'https://example.invalid', 'embedding_dimension': 2}
    bodies = {'chunks.json': chunks, 'vectors.json': {**identity, 'vectors': rows}}
    hashes = {}
    for name, data in bodies.items():
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        (directory / name).write_bytes(body)
        hashes[name] = sha256(body).hexdigest()
    return BuildManifest(build_id, build_id, hashes['chunks.json'], hashes['vectors.json'], 2, **identity)


def fixture_embedding(manifest):
    if manifest.embedding_model != 'fixture-2d':
        raise ValueError('fixture_embedding_identity_mismatch')
    return lambda query: [1., 0.] if query == QUESTION else [0., 1.]


async def verify_requests(catalog, root, publish_v2):
    events, bound_requests = [], []
    published = False
    def evaluate(question, pool, build):
        nonlocal published
        events.append({'stage': 'evaluate', 'build_id': build.build_id,
                       'evidence_builds': sorted({c['metadata']['build_id'] for c in pool})})
        if not published:
            publish_v2()
            published = True
            events.append({'stage': 'publish', 'build_id': 'demo-v2'})
        supplement = len(pool) == 1
        return {'status': 'proposed', 'proposal': {
            'action': 'retrieve' if supplement else 'answer',
            'search_query': SUPPLEMENT if supplement else None,
            'missing_knowledge': ['reconnect method'] if supplement else []},
            'selected_evidence': [] if supplement else deepcopy(pool)}
    def generate(question, evidence, build):
        events.append({'stage': 'generate', 'build_id': build.build_id,
                       'evidence_builds': sorted({c['metadata']['build_id'] for c in evidence})})
        return {'status': 'answer', 'answer': build.build_id + ': connections close and reconnect. [P1:1] [P2:1]',
                'citation_ids': ['P1:1', 'P2:1'], 'missing_information': []}
    def scope_policy(feedback):
        events.append({'stage': 'controller_feedback', 'build_id': feedback.build_id})
        return feedback.previous_scope
    bind = build_request_binder(catalog, root, bind_embedding=fixture_embedding,
                               evaluate=evaluate, generate=generate, route_k=1, result_k=1)
    def observed_bind():
        result = bind()
        bound_requests.append(result)
        return result
    app = create_app(bind_request=observed_bind, scope_policy=scope_policy)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as http:
        responses = [await http.post('/ask', json={'question': QUESTION}) for _ in range(2)]
    for bound, response in zip(bound_requests, responses):
        assert response.status_code == 200 and response.json()['status'] == 'answer', response.text
        assert bound.build.build_id in response.json()['message']
        assert len(bound.retriever.trace) == 2
        assert {step['request']['build_id'] for step in bound.retriever.trace} == {bound.build.build_id}
    assert [b.build.build_id for b in bound_requests] == ['demo-v1', 'demo-v2']
    assert all(event['evidence_builds'] == [event['build_id']] for event in events if 'evidence_builds' in event)
    return {'events': events, 'requests': [
        {'build_id': bound.build.build_id, 'http_status': response.status_code,
         'response': response.json(), 'retrieval_trace': bound.retriever.trace}
        for bound, response in zip(bound_requests, responses)]}


def main():
    import asyncio
    import psycopg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / 'artifacts'
    root.mkdir()
    password = Path(os.environ['RAG_DEMO_PASSWORD_FILE']).read_text().strip()
    connections = []
    def connect():
        connection = psycopg.connect(host='127.0.0.1', port=args.port, user='postgres',
                                     password=password, dbname='postgres', connect_timeout=10)
        connections.append(connection)
        return connection
    catalog = PostgresBuildCatalog(connect, schema=args.schema)
    catalog.initialize()
    v1, v2, bad = [write_fixture_build(root, name) for name in ('demo-v1', 'demo-v2', 'demo-bad')]
    for manifest in (v1, v2, bad):
        catalog.register(manifest)
    catalog.validate_ready(v1.build_id, root)
    catalog.publish(v1.build_id, expected_active=None)
    def rejected(call, message):
        try:
            call()
        except ValueError as error:
            assert message in str(error), str(error)
            return str(error)
        raise AssertionError('expected rejection: ' + message)
    unready = rejected(lambda: catalog.publish(v2.build_id, expected_active=v1.build_id), 'only_ready')
    (root / bad.artifact_directory / 'vectors.json').write_text('{}', encoding='utf-8')
    corrupt = rejected(lambda: catalog.validate_ready(bad.build_id, root), 'artifact_hash_mismatch')
    assert catalog.read_active().build_id == v1.build_id
    catalog.validate_ready(v2.build_id, root)
    request_reads = 0
    class ObservedCatalog:
        def read_active(self):
            nonlocal request_reads
            request_reads += 1
            return catalog.read_active()
    report = asyncio.run(verify_requests(ObservedCatalog(), root,
        lambda: catalog.publish(v2.build_id, expected_active=v1.build_id)))
    assert request_reads == 2
    stale = rejected(lambda: catalog.publish(v1.build_id, expected_active=v1.build_id), 'active_build_changed')
    assert catalog.read_active().build_id == v2.build_id
    conflict = rejected(lambda: catalog.register(replace(v1, chunk_count=3)), 'different_manifest')
    assert all(connection.closed for connection in connections)
    report.update(scope='real isolated PostgreSQL; ASGI /ask; scripted evaluation/generation; 2D fixture vectors',
                  public_corpus_changed=False, provider_requests=0, request_active_pointer_reads=request_reads,
                  all_database_connections_closed=True,
                  rejected_publications={'unready': unready, 'corrupt': corrupt, 'stale_publisher': stale,
                                         'reused_build_id': conflict})
    (args.output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'request_builds': [r['build_id'] for r in report['requests']],
                      'request_active_pointer_reads': request_reads, 'provider_requests': 0,
                      'all_database_connections_closed': True, 'report': str(args.output / 'verification.json')}))


if __name__ == '__main__':
    main()
