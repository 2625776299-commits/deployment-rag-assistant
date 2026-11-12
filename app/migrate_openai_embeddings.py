"""Re-embed a frozen corpus without overwriting or publishing its old build.

Build performs the explicitly requested embedding calls. Activate only registers
validated artifacts, publishes with compare-and-swap, and updates non-secret
local settings; stop the old local service before activation.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values, set_key

from app.build_runtime import BuildManifest, load_build
from app.embedding_runtime import EmbeddingConfig, create_embedding_client, embed_texts

PROJECT = Path(__file__).resolve().parents[1]
TARGET = EmbeddingConfig('openai', 'text-embedding-3-large', 'https://api.openai.com/v1', 3072)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(source_release):
    source_release = source_release.resolve()
    source = json.loads(source_release.read_text(encoding='utf-8'))
    old = BuildManifest(**source['candidate'])
    root = source_release.parent / source['artifact_root']
    old_chunks, _ = load_build(old, root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    name = 'openai-embedding-' + stamp
    target = root / name
    target.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'building', 'source_release': str(source_release.relative_to(PROJECT)),
        'source_manifest': old.to_dict(), 'source_release_sha256': digest(source_release),
        'target_model': TARGET.model, 'target_dimensions': TARGET.dimensions,
        'source_text_count': len(old_chunks), 'generation_calls': 0,
    }
    write(target / 'migration.json', report)
    try:
        texts = list(dict.fromkeys(c['content'] for c in old_chunks))
        usage = {}
        try:
            with create_embedding_client(config=TARGET) as client:
                vectors = embed_texts(client, texts, config=TARGET, usage_report=usage)
        finally:
            write(target / 'embedding_usage.json', usage)
        by_text = dict(zip(texts, vectors, strict=True))
        build_id = 'corpus-v4-openai-' + stamp
        chunks = deepcopy(old_chunks)
        for chunk in chunks:
            chunk['metadata']['build_id'] = build_id
        write(target / 'chunks.json', chunks)
        write(target / 'vectors.json', {
            'embedding_model': TARGET.model, 'embedding_base_url': TARGET.base_url,
            'embedding_dimension': TARGET.dimensions,
            'vectors': [{'chunk_id': c['chunk_id'],
                'content_sha256': sha256(c['content'].encode('utf-8')).hexdigest(),
                'embedding': by_text[c['content']]} for c in chunks],
        })
        manifest = BuildManifest(build_id, name, digest(target / 'chunks.json'),
            digest(target / 'vectors.json'), len(chunks), TARGET.model, TARGET.base_url, TARGET.dimensions)
        verified, _ = load_build(manifest, root)
        load_build(old, root)
        if digest(source_release) != report['source_release_sha256']:
            raise ValueError('source_release_changed')
        for before, after in zip(old_chunks, verified, strict=True):
            expected = deepcopy(before)
            expected['metadata']['build_id'] = build_id
            if expected != after:
                raise ValueError('source_text_or_metadata_changed')
        release_path = source_release.parent / ('release_' + name + '.json')
        release = {'status': 'artifacts_validated', 'artifact_root': source['artifact_root'],
            'candidate': manifest.to_dict(), 'baseline': old.to_dict(),
            'catalog_publication': 'not_published', 'migration_report': str((target / 'migration.json').relative_to(source_release.parent)),
            'all_document_vectors_regenerated': True, 'reused_old_document_vectors': 0,
            'generation_model_calls': 0}
        write(release_path, release)
        report.update(status='artifacts_validated', target_manifest=manifest.to_dict(),
            release_path=str(release_path.relative_to(PROJECT)),
            unchanged_source_chunks=True, old_build_still_loadable=True,
            unique_texts=len(texts), embedding_requests=len(usage.get('requests', [])),
            reported_embedding_tokens=usage.get('total_tokens'))
        write(target / 'migration.json', report)
        return report
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        write(target / 'migration.json', report)
        raise


def activate(release_path):
    import psycopg
    from app.build_catalog import PostgresBuildCatalog
    release_path = release_path.resolve()
    release = json.loads(release_path.read_text(encoding='utf-8'))
    target = BuildManifest(**release['candidate'])
    if EmbeddingConfig.from_build(target) != TARGET:
        raise ValueError('activation_requires_validated_target_embedding_identity')
    previous = BuildManifest(**release['baseline'])
    root = release_path.parent / release['artifact_root']
    load_build(target, root)
    load_build(previous, root)
    local = dotenv_values(PROJECT / '.env')
    dsn = local.get('RAG_CONVERSATION_DATABASE_URL')
    if not dsn:
        raise ValueError('require_local_conversation_database_configuration')
    catalog = PostgresBuildCatalog(lambda: psycopg.connect(dsn, connect_timeout=5),
                                   schema='rag_build_conversation_learning')
    active = catalog.read_active()
    if active not in (previous, target):
        raise ValueError('active_build_changed_since_migration')
    if active == previous:
        catalog.register(target)
        catalog.validate_ready(target.build_id, root)
        catalog.publish(target.build_id, expected_active=previous.build_id)
    settings = {'EMBEDDING_PROVIDER': TARGET.provider, 'EMBEDDING_BASE_URL': TARGET.base_url,
        'EMBEDDING_MODEL': TARGET.model, 'EMBEDDING_DIMENSIONS': str(TARGET.dimensions),
        'RAG_CONVERSATION_RELEASE_PATH': release_path.relative_to(PROJECT).as_posix()}
    for key, value in settings.items():
        set_key(PROJECT / '.env', key, value, quote_mode='never')
    result = {'status': 'published_and_local_defaults_updated', 'active_build_id': catalog.read_active().build_id,
        'release_path': str(release_path.relative_to(PROJECT)), 'settings': settings,
        'old_build_still_ready': catalog.read_build(previous.build_id) == previous,
        'service_restart_required': True, 'generation_calls': 0}
    write(release_path.with_suffix('.activation.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'activate'))
    parser.add_argument('--source-release', type=Path,
        default=PROJECT / 'data/corpus_releases/deploy_faults_20261102_v1/release.json')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.phase == 'activate' and args.release is None:
        parser.error('activate requires --release')
    try:
        result = build(args.source_release) if args.phase == 'build' else activate(args.release)
    except Exception as error:
        # Avoid printing SDK/connection exceptions that may contain credentials.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
