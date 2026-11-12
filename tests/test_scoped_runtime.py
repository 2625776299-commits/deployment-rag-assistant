"""Runtime assembly + ASGI/service wiring with scripted evaluation/generation."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from app.learning_runtime import PROJECT, HybridRetriever, create_learning_app
from app.main import create_app
from app.retrieval_contract import RetrievalRequest, SourceScope
from app.scoped_retrieval import ScopedHybridRetriever


QUESTION = 'old connections close'
EXTRA_QUERY = 'pool checkout'


def fixture():
    def chunk(key, doc, content):
        return {'chunk_id': key, 'content': content, 'metadata': {'document_id': doc,
            'section_id': key, 'source_url': 'https://example.invalid/fixture/' + doc}}
    chunks = [chunk('D0:0', 'docker', QUESTION), chunk('S1:0', 'sqlalchemy', EXTRA_QUERY),
              chunk('P1:0', 'psycopg', 'pool connection')]
    vectors = {QUESTION: [0., 1.], EXTRA_QUERY: [1., 0.], 'pool connection': [1., 0.]}
    return chunks, vectors, {'embedding_model': 'fixture', 'embedding_base_url': 'https://fixture', 'embedding_dimension': 2}


def proposed(pool, *, retrieve=False):
    return {'status': 'proposed', 'proposal': {'action': 'retrieve' if retrieve else 'answer',
        'search_query': EXTRA_QUERY if retrieve else None,
        'missing_knowledge': ['pool checkout'] if retrieve else []},
        'selected_evidence': [] if retrieve else deepcopy(pool)}


def generated(*_, **__):
    return {'status': 'answer', 'answer': 'Old connections close; check the pool. [D0:0] [S1:0]',
            'citation_ids': ['D0:0', 'S1:0'], 'missing_information': []}


@contextmanager
def runtime_fixture():
    root = PROJECT / 'data/architecture_demos'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='runtime_test_', dir=root) as directory:
        # Confirm the automatic cleanup target remains in this test workspace.
        assert Path(directory).resolve().is_relative_to(root.resolve())
        with ExitStack() as stack:
            snapshot = fixture()
            client = Mock()
            stack.enter_context(patch('app.learning_runtime.PROJECT', Path(directory)))
            stack.enter_context(patch('app.learning_runtime.load_snapshot', return_value=snapshot))
            make_client = stack.enter_context(patch('app.ask.create_generation_client', return_value=(client, 'fixture-model')))
            embedding = stack.enter_context(patch('app.embedding_runtime.create_embedding_client'))
            stack.enter_context(patch('app.embedding_runtime.MODEL', 'fixture'))
            stack.enter_context(patch('app.embedding_runtime.BASE_URL', 'https://fixture'))
            stack.enter_context(patch('app.embedding_runtime.DIMENSIONS', 2))
            raw_evaluation = stack.enter_context(patch('app.compare_reranking.request_order'))
            evaluation = stack.enter_context(patch('app.learning_runtime.evaluate_requirements',
                                                   side_effect=lambda question, pool, **kwargs: proposed(pool)))
            generation = stack.enter_context(patch('app.ask.generate_answer', side_effect=generated))
            yield client, make_client, embedding, raw_evaluation, evaluation, generation


class ScopedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_selects_scoped_instance_and_real_asgi_receives_ordinary_question(self):
        with runtime_fixture() as (client, make_client, embedding, raw_evaluation, evaluation, generation):
            scope = SourceScope(('sqlalchemy',), ('psycopg',))
            app = create_learning_app(learning_date='2026-10-30', initial_scope=scope)
            state = app.state.learning_runtime
            self.assertIsInstance(state['retriever'], ScopedHybridRetriever)
            self.assertEqual(state['retrieval_contract'], 'request-v1')
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as http:
                response = await http.post('/ask', json={'question': QUESTION})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'answer')
            trace = state['retriever'].trace
            self.assertEqual(trace[0]['request']['scope']['excluded_document_ids'], ('psycopg',))
            self.assertNotIn('P1:0', trace[0]['eligible_ids'])
            self.assertEqual(set(response.json()), {'request_id', 'status', 'message', 'required_information', 'sources'})
            embedding.assert_not_called()
            raw_evaluation.assert_not_called()
            generation.assert_called_once()
            client.close()

    async def test_request_contract_reaches_both_initial_and_supplementary_retrieval(self):
        chunks, vectors, _ = fixture()
        retriever = ScopedHybridRetriever(chunks, vectors, self.fail, route_k=1, result_k=1)
        evaluated, feedbacks = [], []
        def evaluate(question, pool):
            evaluated.append([c['chunk_id'] for c in pool])
            return proposed(pool, retrieve=len(evaluated) == 1)
        def policy(feedback):
            feedbacks.append(feedback)
            return SourceScope(('sqlalchemy',), ('psycopg',))
        app = create_app(retrieve=self.fail, request_retrieval=retriever.search,
                         initial_scope=SourceScope(excluded_document_ids=('psycopg',)),
                         scope_policy=policy, evaluate=evaluate, generate=generated)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as http:
            response = await http.post('/ask', json={'question': QUESTION})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'answer')
        self.assertEqual(evaluated, [['D0:0'], ['D0:0', 'S1:0']])
        self.assertEqual(retriever.trace[1]['request']['exclude_chunk_ids'], ('D0:0',))
        self.assertEqual(feedbacks[0].missing_knowledge, ('pool checkout',))
        self.assertEqual(feedbacks[0].previous_scope.excluded_document_ids, ('psycopg',))

    async def test_http_body_cannot_set_internal_scope_or_loop_policy(self):
        retrieve = Mock()
        app = create_app(retrieve=retrieve, evaluate=Mock(), generate=Mock())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture') as http:
            response = await http.post('/ask', json={'question': QUESTION, 'initial_scope': {'excluded_document_ids': []}})
        self.assertEqual(response.status_code, 422)
        retrieve.assert_not_called()

    async def test_unknown_scope_fails_before_creating_provider_client(self):
        with runtime_fixture() as (_, make_client, embedding, *_):
            with self.assertRaisesRegex(ValueError, 'unknown documents'):
                create_learning_app(learning_date='2026-10-30', initial_scope=SourceScope(('not-in-snapshot',)))
            make_client.assert_not_called()
            embedding.assert_not_called()

    async def test_old_runtime_is_preserved_without_opt_in(self):
        with runtime_fixture() as (client, *_):
            app = create_learning_app(learning_date='2026-10-30')
            state = app.state.learning_runtime
            self.assertIsInstance(state['retriever'], HybridRetriever)
            self.assertEqual(state['retrieval_contract'], 'query-string')
            self.assertIsNone(state['initial_scope'])
            client.close()

    async def test_bad_scoped_configuration_is_rejected_during_assembly(self):
        with self.assertRaisesRegex(ValueError, 'Scope requires'):
            create_app(retrieve=Mock(), evaluate=Mock(), generate=Mock(), initial_scope=SourceScope())
        with self.assertRaisesRegex(ValueError, 'configured RAG'):
            create_app(request_retrieval=Mock())


if __name__ == '__main__':
    unittest.main()
