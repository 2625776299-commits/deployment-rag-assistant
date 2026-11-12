"""PostgreSQL authority for immutable build manifests and an atomic active pointer.

Only release bookkeeping lives here. Document ingestion/catalog CRUD and external
ANN index management are separate work; this does not retrofit the frozen corpus.
"""
from pathlib import Path

from psycopg import sql
from psycopg.types.json import Jsonb

from app.build_runtime import BuildManifest, load_build
from app.retrieval_contract import RetrievalRequest
from app.scoped_retrieval import ScopedHybridRetriever


class PostgresBuildCatalog:
    def __init__(self, connect, *, schema: str):
        if not isinstance(schema, str) or not schema.startswith('rag_build_') or not schema.replace('_', '').isalnum():
            raise ValueError('require_dedicated_rag_build_schema')
        self.connect, self.schema = connect, schema

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    def initialize(self):
        with self.connect() as connection:
            connection.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            connection.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                build_id TEXT PRIMARY KEY, manifest JSONB NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'ready')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(), validated_at TIMESTAMPTZ
            )''').format(self._table('builds')))
            connection.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                name TEXT PRIMARY KEY, build_id TEXT NOT NULL REFERENCES {}(build_id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )''').format(self._table('active_build'), self._table('builds')))
            connection.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                build_id TEXT NOT NULL REFERENCES {}(build_id), document_id TEXT NOT NULL,
                document JSONB NOT NULL, PRIMARY KEY (build_id, document_id)
            )''').format(self._table('documents'), self._table('builds')))

    def register(self, manifest: BuildManifest):
        with self.connect() as connection:
            connection.execute(sql.SQL('''INSERT INTO {} (build_id, manifest, state)
                VALUES (%s, %s, 'pending') ON CONFLICT (build_id) DO NOTHING''').format(self._table('builds')),
                (manifest.build_id, Jsonb(manifest.to_dict())))
            row = connection.execute(sql.SQL('SELECT manifest FROM {} WHERE build_id = %s').format(self._table('builds')),
                                     (manifest.build_id,)).fetchone()
            if row[0] != manifest.to_dict():
                raise ValueError('build_id_already_has_different_manifest')

    def register_documents(self, build_id: str, documents: list[dict]):
        ids = [d.get('document_id') for d in documents]
        if not ids or any(not isinstance(key, str) or not key for key in ids) or len(ids) != len(set(ids)):
            raise ValueError('require_unique_document_ids')
        with self.connect() as connection:
            row = connection.execute(sql.SQL('SELECT state FROM {} WHERE build_id = %s FOR UPDATE')
                .format(self._table('builds')), (build_id,)).fetchone()
            if row is None or row[0] != 'pending':
                raise ValueError('document_registration_requires_pending_build')
            for document in documents:
                connection.execute(sql.SQL('''INSERT INTO {} (build_id, document_id, document) VALUES (%s, %s, %s)
                    ON CONFLICT (build_id, document_id) DO NOTHING''').format(self._table('documents')),
                    (build_id, document['document_id'], Jsonb(document)))
                existing = connection.execute(sql.SQL('SELECT document FROM {} WHERE build_id = %s AND document_id = %s')
                    .format(self._table('documents')), (build_id, document['document_id'])).fetchone()
                if existing[0] != document:
                    raise ValueError('document_revision_conflict')

    def read_documents(self, build_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(sql.SQL('SELECT document FROM {} WHERE build_id = %s ORDER BY document_id')
                .format(self._table('documents')), (build_id,)).fetchall()
        return [row[0] for row in rows]

    def validate_ready(self, build_id: str, artifact_root: Path, *, require_documents=False):
        with self.connect() as connection:
            row = connection.execute(sql.SQL('SELECT manifest FROM {} WHERE build_id = %s').format(self._table('builds')),
                                     (build_id,)).fetchone()
        if row is None:
            raise ValueError('build_not_found')
        manifest = BuildManifest(**row[0])
        chunks, vectors = load_build(manifest, artifact_root)
        if require_documents:
            documents = {d['document_id']: d for d in self.read_documents(build_id)}
            if set(documents) != {c['metadata']['document_id'] for c in chunks}:
                raise ValueError('catalog_and_chunk_documents_differ')
            for chunk in chunks:
                meta = chunk['metadata']
                document = documents[meta['document_id']]
                for key in ('technology', 'applicable_version', 'source_release'):
                    if key in meta and meta[key] != document.get(key):
                        raise ValueError('catalog_projection_mismatch:' + key)
        # Open and exercise the actual exact-cosine/BM25 adapter without a provider.
        def no_provider(_):
            raise AssertionError('validation_must_not_call_embedding_provider')
        retriever = ScopedHybridRetriever(chunks, vectors, no_provider)
        if not retriever.search(RetrievalRequest(chunks[0]['content'], build_id=build_id)):
            raise ValueError('build_search_probe_failed')
        # Artifact work happens outside DB transactions; request binding rechecks hashes.
        with self.connect() as connection:
            connection.execute(sql.SQL("UPDATE {} SET state = 'ready', validated_at = now() WHERE build_id = %s")
                               .format(self._table('builds')), (build_id,))

    def publish(self, build_id: str, *, expected_active: str | None):
        """Compare-and-swap: a stale publisher cannot overwrite a newer release."""
        with self.connect() as connection:
            row = connection.execute(sql.SQL('SELECT state FROM {} WHERE build_id = %s FOR SHARE')
                                     .format(self._table('builds')), (build_id,)).fetchone()
            if row is None or row[0] != 'ready':
                raise ValueError('only_ready_build_can_be_published')
            if expected_active is None:
                cursor = connection.execute(sql.SQL('''INSERT INTO {} (name, build_id) VALUES ('default', %s)
                    ON CONFLICT (name) DO NOTHING''').format(self._table('active_build')), (build_id,))
            else:
                cursor = connection.execute(sql.SQL('''UPDATE {} SET build_id = %s, updated_at = now()
                    WHERE name = 'default' AND build_id = %s''').format(self._table('active_build')),
                    (build_id, expected_active))
            if cursor.rowcount != 1:
                raise ValueError('active_build_changed')

    def read_active(self) -> BuildManifest:
        with self.connect() as connection:
            row = connection.execute(sql.SQL('''SELECT b.manifest FROM {} a JOIN {} b ON b.build_id = a.build_id
                WHERE a.name = 'default' AND b.state = 'ready' ''')
                .format(self._table('active_build'), self._table('builds'))).fetchone()
        if row is None:
            raise ValueError('no_ready_active_build')
        return BuildManifest(**row[0])

    def read_build(self, build_id: str) -> BuildManifest:
        """Resolve the exact immutable ready build saved with a logical run."""
        with self.connect() as connection:
            row = connection.execute(sql.SQL('SELECT manifest FROM {} WHERE build_id=%s AND state=\'ready\'')
                .format(self._table('builds')), (build_id,)).fetchone()
        if row is None:
            raise ValueError('build_not_ready_or_not_found')
        return BuildManifest(**row[0])
