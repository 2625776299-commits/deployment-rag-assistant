"""Publication boundary, artifact identity and request isolation regression tests."""
import asyncio
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from app.build_runtime import build_request_binder, load_build
from app.main import create_app
from app.retrieval_contract import RetrievalRequest
from app.run_build_pinning_demo import fixture_embedding, verify_requests, write_fixture_build


class BuildPinningTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'data/architecture_demos'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='pin_test_', dir=root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        assert self.root.resolve().is_relative_to(root.resolve())
        self.v1 = write_fixture_build(self.root, 'demo-v1')
        self.v2 = write_fixture_build(self.root, 'demo-v2')

    def binder(self, manifest=None):
        selected = manifest or self.v1
        class Catalog:
            def read_active(self):
                return selected
        return build_request_binder(Catalog(), self.root, bind_embedding=fixture_embedding,
                                   evaluate=lambda *args: None, generate=lambda *args: None)

    def test_in_flight_publish_does_not_change_supplement_evaluation_or_generation(self):
        active, reads = self.v1, []
        class Catalog:
            def read_active(inner):
                reads.append(active.build_id)
                return active
        def publish():
            nonlocal active
            active = self.v2
        report = asyncio.run(verify_requests(Catalog(), self.root, publish))
        self.assertEqual(reads, ['demo-v1', 'demo-v2'])
        self.assertEqual([e['build_id'] for e in report['events'] if e['stage'] == 'controller_feedback'], reads)

    def test_context_is_immutable_and_wrong_build_retrieval_is_rejected(self):
        bound = self.binder()()
        with self.assertRaises(FrozenInstanceError):
            bound.build.build_id = 'demo-v2'
        for build_id in ('demo-v2', None):
            with self.assertRaisesRegex(ValueError, 'retrieval_build_mismatch'):
                bound.search(RetrievalRequest('query', build_id=build_id))

    def test_recovery_resolves_saved_build_without_reading_new_active_pointer(self):
        selected, calls = self.v1, []
        class Catalog:
            def read_active(inner):
                raise AssertionError('recovery_read_active_pointer')
            def read_build(inner, build_id):
                calls.append(build_id)
                if build_id != selected.build_id:
                    raise ValueError('build_not_ready_or_not_found')
                return selected
        binder = build_request_binder(Catalog(), self.root, bind_embedding=fixture_embedding,
                                      evaluate=lambda *a: None, generate=lambda *a: None)
        bound = binder(self.v1.build_id)
        self.assertEqual((calls, bound.build.build_id), ([self.v1.build_id], self.v1.build_id))
        with self.assertRaisesRegex(ValueError, 'not_found'):
            binder('missing-build')

    def test_other_build_evidence_and_forged_same_id_are_rejected_before_models(self):
        bound = self.binder()()
        chunks, _ = load_build(self.v2, self.root)
        for call in (bound.evaluate, bound.generate):
            with self.assertRaisesRegex(ValueError, 'pinned_build'):
                call('question', chunks)
        chunks, _ = load_build(self.v1, self.root)
        chunks[0]['content'] = 'changed content with the same id and build label'
        with self.assertRaisesRegex(ValueError, 'pinned_build'):
            bound.generate('question', chunks)

    def test_artifact_corruption_fails_before_embedding_factory(self):
        (self.root / self.v1.artifact_directory / 'chunks.json').write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'artifact_hash_mismatch'):
            self.binder()()

    def rewrite_vectors(self, edit):
        path = self.root / self.v1.artifact_directory / 'vectors.json'
        data = json.loads(path.read_bytes())
        edit(data)
        body = json.dumps(data).encode()
        path.write_bytes(body)
        return replace(self.v1, vectors_sha256=sha256(body).hexdigest())

    def test_equal_dimensions_do_not_allow_different_embedding_model(self):
        manifest = self.rewrite_vectors(lambda data: data.update(embedding_model='another-2d-model'))
        with self.assertRaisesRegex(ValueError, 'embedding_identity_mismatch'):
            load_build(manifest, self.root)

    def test_equal_counts_do_not_allow_duplicate_vector_ids_or_wrong_bodies(self):
        original = (self.root / self.v1.artifact_directory / 'vectors.json').read_bytes()
        for mutation, message in (
            (lambda data: data['vectors'][1].update(chunk_id='P1:1'), 'vector_chunk_identity_mismatch'),
            (lambda data: data['vectors'][1].update(content_sha256='0' * 64), 'vector_content_mismatch'),
            (lambda data: data['vectors'][1].update(embedding=[1.]), 'invalid_vector_dimension'),
        ):
            with self.subTest(message=message):
                (self.root / self.v1.artifact_directory / 'vectors.json').write_bytes(original)
                manifest = self.rewrite_vectors(mutation)
                with self.assertRaisesRegex(ValueError, message):
                    load_build(manifest, self.root)

    def test_artifact_path_cannot_escape_root(self):
        with self.assertRaisesRegex(ValueError, 'outside_root'):
            load_build(replace(self.v1, artifact_directory='../outside'), self.root)

    def test_no_active_build_fails_without_calling_models(self):
        def unavailable():
            raise ValueError('no_ready_active_build')
        app = create_app(bind_request=unavailable)
        async def request():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as client:
                return await client.post('/ask', json={'question': 'why?'})
        with self.assertLogs('app.main', level='ERROR'):
            response = asyncio.run(request())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()['status'], 'failed')

    def test_public_request_cannot_select_build(self):
        calls = []
        app = create_app(bind_request=lambda: calls.append('unexpected'))
        async def request():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as client:
                return await client.post('/ask', json={'question': 'why?', 'build_id': 'demo-v2'})
        response = asyncio.run(request())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
