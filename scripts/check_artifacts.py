"""Offline, standard-library-only integrity check of the published snapshot."""
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'data/corpus_releases/deploy_faults_20261102_v1/release_openai-embedding-20261105T113923Z_3caf7a.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check():
    release = load(RELEASE)
    build = release['candidate']
    directory = RELEASE.parent / release['artifact_root'] / build['artifact_directory']
    for name, key in [('chunks.json', 'chunks_sha256'), ('vectors.json', 'vectors_sha256')]:
        require(sha256((directory / name).read_bytes()).hexdigest() == build[key], 'hash mismatch: ' + name)
    chunks, vectors = load(directory / 'chunks.json'), load(directory / 'vectors.json')
    require(len(chunks) == build['chunk_count'], 'chunk count mismatch')
    by_id = {chunk['chunk_id']: chunk for chunk in chunks}
    require(len(by_id) == len(chunks), 'duplicate chunk IDs')
    for key in ('embedding_model', 'embedding_dimension', 'embedding_base_url'):
        require(vectors[key] == build[key], 'embedding identity mismatch: ' + key)
    rows = vectors['vectors']
    require(len(rows) == len(chunks), 'vector count mismatch')
    require({row['chunk_id'] for row in rows} == set(by_id), 'vector/chunk IDs differ')
    for row in rows:
        chunk = by_id[row['chunk_id']]
        require(row['content_sha256'] == sha256(chunk['content'].encode('utf-8')).hexdigest(), 'content hash mismatch')
        require(len(row['embedding']) == build['embedding_dimension'], 'vector dimension mismatch')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in row['embedding']), 'invalid vector value')
        require(chunk['metadata']['source_url'].startswith('https://'), 'missing source URL')
    provenance = load(ROOT / 'evidence/source-provenance.json')
    for entry in provenance['files']:
        target = (ROOT / entry['path']).resolve()
        require(target.is_relative_to(ROOT), 'provenance path outside repository')
        require(sha256(target.read_bytes()).hexdigest() == entry['sha256'], 'exported file changed: ' + entry['path'])
    result = {'status': 'passed', 'build_id': build['build_id'], 'chunks': len(chunks),
              'dimensions': build['embedding_dimension'], 'source_files_checked': len(provenance['files']),
              'model_calls': 0, 'database_connections': 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    check()
