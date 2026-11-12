import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as S
from unittest.mock import Mock, patch
from app import embedding_runtime as embedding


class EmbeddingMigrationTests(unittest.TestCase):
    def test_activation_rejects_old_provider_before_reading_artifacts_or_database(self):
        from app import migrate_openai_embeddings as migration
        manifest = {'build_id':'old', 'artifact_directory':'old',
            'chunks_sha256':'0'*64, 'vectors_sha256':'0'*64, 'chunk_count':1,
            'embedding_model':'text-embedding-v4',
            'embedding_base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'embedding_dimension':1024}
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'release.json'
            path.write_text(json.dumps({'candidate':manifest}),encoding='utf-8')
            with patch.object(migration,'load_build') as load:
                with self.assertRaisesRegex(ValueError,'activation_requires_validated_target'):
                    migration.activate(path)
                load.assert_not_called()

    def test_pinned_openai_and_qwen_configs_do_not_follow_active_globals(self):
        configs = [
            embedding.EmbeddingConfig('openai', 'text-embedding-3-large', 'https://api.openai.com/v1', 3072),
            embedding.EmbeddingConfig('dashscope', 'text-embedding-v4', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 1024),
        ]
        for config in configs:
            with self.subTest(provider=config.provider):
                build = S(embedding_model=config.model, embedding_base_url=config.base_url,
                          embedding_dimension=config.dimensions)
                pinned = embedding.EmbeddingConfig.from_build(build)
                client = Mock()
                client.embeddings.create.return_value = S(model=config.model,
                    usage=S(total_tokens=7), data=[S(index=0, embedding=[1.] * config.dimensions)])
                with patch.multiple(embedding, PROVIDER='unrelated', MODEL='wrong', DIMENSIONS=2):
                    vectors = embedding.embed_texts(client, ['query'], config=pinned)
                self.assertEqual(len(vectors[0]), config.dimensions)
                self.assertEqual(client.embeddings.create.call_args.kwargs['model'], config.model)
                self.assertEqual(client.embeddings.create.call_args.kwargs['dimensions'], config.dimensions)

    def test_official_credentials_require_explicit_same_endpoint_key(self):
        config = embedding.EmbeddingConfig('openai', 'text-embedding-3-large', 'https://api.openai.com/v1', 3072)
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'unrelated-ambient-key'}, clear=True):
            with patch.object(embedding, 'OpenAI') as factory:
                with self.assertRaises(ValueError):
                    embedding.create_embedding_client(config=config)
                factory.assert_not_called()
        with patch.dict('os.environ', {'GENERATION_API_KEY': 'local-fixture-key',
                'GENERATION_BASE_URL': 'https://api.openai.com/v1'}, clear=True):
            with patch.object(embedding, 'OpenAI') as factory:
                embedding.create_embedding_client(config=config)
                self.assertEqual(factory.call_args.kwargs['api_key'], 'local-fixture-key')
                self.assertEqual(factory.call_args.kwargs['max_retries'], 0)
        foreign = embedding.EmbeddingConfig('openai', config.model, 'https://untrusted.example/v1', 3072)
        with patch.object(embedding, 'OpenAI') as factory:
            with self.assertRaises(ValueError):
                embedding.create_embedding_client(config=foreign)
            factory.assert_not_called()

    def test_official_response_must_match_pinned_dimension_and_input_rows(self):
        config = embedding.EmbeddingConfig('openai', 'text-embedding-3-large', 'https://api.openai.com/v1', 3072)
        client = Mock()
        client.embeddings.create.return_value = S(model=config.model, usage=None,
            data=[S(index=0, embedding=[1.] * 1024)])
        with self.assertRaises(ValueError):
            embedding.embed_texts(client, ['query'], config=config)
        client.embeddings.create.return_value.data = [S(index=1, embedding=[1.] * 3072)]
        with self.assertRaises(ValueError):
            embedding.embed_texts(client, ['query'], config=config)

    def test_batches_preserve_order_and_record_each_request(self):
        def reply(**kwargs):
            return S(model='text-embedding-v4',usage=S(total_tokens=len(kwargs['input'])),
                     data=[S(index=i,embedding=[float(t),1.])
                           for i,t in reversed(list(enumerate(kwargs['input'])))])
        client=Mock();client.embeddings.create.side_effect=reply
        usage={}
        with patch.multiple(embedding,PROVIDER='dashscope',MODEL='text-embedding-v4',DIMENSIONS=2):
            # Use a supported dimension, padding vectors is unrelated to batching.
            with patch.object(embedding,'DIMENSIONS',64):
                def padded(**kwargs):
                    value=reply(**kwargs)
                    for item in value.data:item.embedding += [0.]*62
                    return value
                client.embeddings.create.side_effect=padded
                result=embedding.embed_texts(client,[str(i) for i in range(21)],usage_report=usage)
        self.assertEqual([v[0] for v in result],list(range(21)))
        self.assertEqual([e['count'] for e in usage['requests']],[10,10,1])
        self.assertEqual(usage['total_tokens'],21)

    def test_wrong_model_or_dimensions_are_rejected(self):
        client=Mock()
        client.embeddings.create.return_value=S(model='wrong-model',usage=None,data=[S(index=0,embedding=[1.]*64)])
        with patch.multiple(embedding,PROVIDER='dashscope',MODEL='text-embedding-v4',DIMENSIONS=64):
            with self.assertRaises(ValueError):embedding.embed_texts(client,['q'])
            client.embeddings.create.return_value.model='text-embedding-v4'
            client.embeddings.create.return_value.data[0].embedding=[1.]*63
            with self.assertRaises(ValueError):embedding.embed_texts(client,['q'])

    def test_shipped_release_is_loadable_without_developer_paths(self):
        from app.build_runtime import BuildManifest, load_build
        project = Path(__file__).resolve().parents[1]
        release_path = project / (
            'data/corpus_releases/deploy_faults_20261102_v1/'
            'release_openai-embedding-20261105T113923Z_3caf7a.json')
        release = json.loads(release_path.read_text(encoding='utf-8'))
        manifest = BuildManifest(**release['candidate'])
        chunks, vectors = load_build(manifest, release_path.parent / release['artifact_root'])
        self.assertEqual(len(chunks), manifest.chunk_count)
        self.assertEqual(manifest.embedding_model, 'text-embedding-3-large')
        self.assertEqual(manifest.embedding_dimension, 3072)
        self.assertTrue(all(chunk['content'] in vectors for chunk in chunks))


if __name__=='__main__':unittest.main()
