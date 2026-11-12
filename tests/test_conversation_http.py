"""Opt-in HTTP -> real PostgreSQL -> real RAG; scripted model replies only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from threading import Event
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql

from app.conversation_api import bearer_authenticator
from app.conversation_rag import run_prepared_snapshot
from app.conversation_service import ConversationExecution, ConversationService
from app.conversation_store import PostgresConversationStore
from app.main import create_app
from app.provider_errors import ModelProviderError


PASSAGE = {'chunk_id': 'HTTP:1',
    'content': '宿主机应用连接容器数据库时使用宿主机地址和发布端口；容器间连接使用服务名和容器端口。',
    'metadata': {'document_id': 'http-fixture', 'section_id': 'HTTP', 'source_url': 'https://example.invalid/fixture'}}
FIRST = '应用连接Compose数据库失败，连接地址应该怎么填？'
SUPPLEMENT = '更正：应用在宿主机上，数据库在Compose中，映射到宿主机的端口是5433。'


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'), 'requires isolated PostgreSQL probe')
class ConversationHttpTests(unittest.TestCase):
    def setUp(self):
        self.schema = 'rag_conversation_http_' + uuid4().hex
        self.connections, self.calls, self.snapshots, self.task_inputs = [], [], [], []
        self.timeout_evaluate = False
        self.block_version = None
        self.started, self.release = Event(), Event()
        self.store = PostgresConversationStore(self.connect, schema=self.schema)
        self.store.initialize()
        self.service = self.make_service()
        self.auth = bearer_authenticator({'fixture-u1': 'U1', 'fixture-u2': 'U2'})
        self.client = TestClient(create_app(conversation_service=self.service, authenticate=self.auth))
        self.headers = {'Authorization': 'Bearer fixture-u1'}
        self.cid = self.client.post('/conversations', headers=self.headers).json()['conversation_id']
        self.addCleanup(self.cleanup)

    def connect(self):
        conn = psycopg.connect(host='127.0.0.1', port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres', user='rag_conversation_probe', connect_timeout=5)
        self.connections.append(conn)
        return conn

    def cleanup(self):
        self.release.set()
        self.client.close()
        with self.connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(c.closed for c in self.connections))

    def check_closed(self):
        self.assertTrue(all(c.closed for c in self.connections))

    def task_model(self, messages):
        self.check_closed()
        p = json.loads(messages[1]['content'])
        self.task_inputs.append(p)
        self.calls.append('task')
        user = p['current_user_message']
        text, mid = user['content'], user['message_id']
        tasks, pending = p['existing_tasks'], p['pending_task_clarifications']
        if '不确定哪一项' in text:
            return json.dumps({'selection': {'operation': 'clarify', 'candidate_task_ids': [t['task_id'] for t in tasks],
                'source_quote': text, 'question': '你指原生产环境，还是另一个测试环境？'}, 'turn_request': None})
        task = next((t for t in tasks if t['task_id'] == p['current_task_id']), None)
        if '第一个任务' in text:
            task = next(t for t in tasks if t['task_id'] == pending[0]['candidate_task_ids'][0])
            selection = {'operation': 'resolve_task', 'task_id': task['task_id'],
                         'clarification_version': pending[0]['input_version'], 'source_quote': text}
        elif not tasks or '另一个测试环境' in text:
            selection = {'operation': 'create', 'goal': '排查另一个测试环境连接问题' if tasks else '排查生产环境数据库连接失败',
                         'source_quote': text}
            task = None
        else:
            if '回到生产环境' in text:
                task = next(t for t in tasks if t['goal'] == '排查生产环境数据库连接失败')
            selection = {'operation': 'select', 'task_id': task['task_id'], 'source_quote': text}
        request = ({'mode': 'continue', 'from_input_version': task['latest_input_version']} if task else
                   {'mode': 'question', 'question': text, 'source': {'message_id': mid, 'quote': text}})
        return json.dumps({'selection': selection, 'turn_request': request})

    def context_model(self, messages):
        self.check_closed()
        self.calls.append('context')
        p = json.loads(messages[1]['content'])
        user = p.get('resolved_original_message', p['current_user_message'])
        updates = []
        for registered in p['registered_targets']:
            target = registered['target']
            field, entity = target['field'], target['entity_id']
            value = ('host' if entity == 'application' and field == 'runtime_location' and '应用在宿主机' in user['content'] else
                     'container' if entity == 'application' and field == 'runtime_location' and '应用在容器' in user['content'] else
                     'compose' if entity == 'database' and field == 'runtime_location' and 'Compose' in user['content'] else
                     '5433' if field == 'published_host_port' and '5433' in user['content'] else None)
            if value is None:
                continue
            previous = next((f for f in p['current_user_stated_facts'] if f['target'] == target), None)
            updates.append({'operation': 'replace' if previous else 'add', 'target': target,
                'previous_fact_id': previous['fact_id'] if previous else None, 'value': value,
                'source': {'message_id': user['message_id'], 'quote': user['content']}})
        return json.dumps({'updates': updates, 'analysis_request': {'mode': 'current',
            'source': {'message_id': p['current_user_message']['message_id'],
                       'quote': p['current_user_message']['content']}, 'confirmation_from_input_version': None}})

    def evaluate(self, messages):
        self.check_closed()
        self.calls.append('evaluate')
        if self.timeout_evaluate:
            raise TimeoutError('private provider details')
        p = json.loads(messages[1]['content'])
        enough = any(f['target']['field'] == 'published_host_port' for f in p['conversation_context']['user_stated_facts'])
        question = p['question']
        return json.dumps({'requirements': [{'requirement_id': 'R1', 'requirement': '说明本次连接地址', 'question_quote': question}],
            'assessments': [{'requirement_id': 'R1', 'coverage': 'covered' if enough else 'partial',
                'evidence': [{'chunk_id': PASSAGE['chunk_id'], 'quote': PASSAGE['content']}],
                'gap': None if enough else '缺少应用位置和发布端口'}],
            'action': 'answer' if enough else 'clarify', 'selected_chunk_ids': [PASSAGE['chunk_id']],
            'missing_knowledge': [], 'missing_user_information': [] if enough else ['应用的运行位置和数据库发布到宿主机的端口'],
            'search_query': None, 'reason': 'Scripted fixture, not a semantic quality claim.',
            'known_findings': [], 'pending_impacts': []})

    def answer(self, messages):
        self.check_closed()
        self.calls.append('answer')
        return json.dumps({'status': 'answer', 'answer': '应用可使用宿主机地址及发布端口5433连接数据库。[HTTP:1]',
                           'citation_ids': [PASSAGE['chunk_id']], 'missing_information': []})

    def make_service(self):
        def bind():
            self.calls.append('bind')
            def execute(snapshot):
                self.snapshots.append(deepcopy(snapshot))
                if snapshot['turn']['input_version'] == self.block_version:
                    self.started.set()
                    if not self.release.wait(12):
                        raise TimeoutError('test_wait_timeout')
                return run_prepared_snapshot(snapshot, retrieve=lambda _: [PASSAGE],
                    evaluate_raw=self.evaluate, generate_raw=self.answer)
            return ConversationExecution('fixture-build', execute)
        return ConversationService(self.store, generate_task=self.task_model, generate_context=self.context_model,
                                   bind_execution=bind)

    def send(self, mid, content, *, client=None, headers=None):
        return (client or self.client).post(f'/conversations/{self.cid}/messages',
            json={'message_id': mid, 'content': content}, headers=headers or self.headers)

    def test_full_clarification_supplement_answer_and_public_history(self):
        first = self.send('M1', FIRST)
        self.assertEqual((first.status_code, first.json()['status']), (200, 'needs_user'))
        result = self.send('M2', SUPPLEMENT)
        self.assertEqual((result.status_code, result.json()['status']), (200, 'answer'))
        self.assertTrue(result.json()['is_current'])
        self.assertEqual(result.headers['X-Request-ID'], result.json()['request_id'])
        self.assertEqual(len(result.json()['sources']), 1)
        self.assertEqual(self.snapshots[0]['task']['task_id'], self.snapshots[1]['task']['task_id'])
        self.assertEqual(self.snapshots[1]['turn']['question'], FIRST)
        history = self.client.get(f'/conversations/{self.cid}/messages', headers=self.headers).json()
        self.assertEqual([t['status'] for t in history['turns']], ['needs_user', 'answer'])
        self.assertEqual([t['is_current'] for t in history['turns']], [False, True])
        for private in ('registered_targets', 'raw_text', 'input_snapshot', 'fact_id', 'owner_id'):
            self.assertNotIn(private, json.dumps(history))

    def test_authentication_ownership_and_client_cannot_supply_server_fields(self):
        before = list(self.calls)
        self.assertEqual(self.client.post('/conversations').status_code, 401)
        other = {'Authorization': 'Bearer fixture-u2'}
        self.assertEqual(self.send('M1', FIRST, headers=other).status_code, 404)
        self.assertEqual(self.client.get(f'/conversations/{self.cid}/messages', headers=other).status_code, 404)
        for extra in ('owner_id', 'task_id', 'context', 'build_id'):
            result = self.client.post(f'/conversations/{self.cid}/messages', headers=self.headers,
                json={'message_id': 'M1', 'content': FIRST, extra: 'forged'})
            self.assertEqual(result.status_code, 422)
        self.assertEqual(self.calls, before)
        self.assertEqual(self.store.read_history('U1', self.cid)['messages'], [])

    def test_raw_code_whitespace_is_preserved_and_blank_messages_rejected(self):
        text = '\n  Compose配置：\n    ports:\n      - "5433:5432"\n'
        self.send('M1', text)
        self.assertEqual(self.task_inputs[0]['current_user_message']['content'], text)
        self.assertEqual(self.store.read_history('U1', self.cid)['messages'][0]['content'], text)
        before = list(self.calls)
        for payload in ({'message_id': 'M2', 'content': '  \n '}, {'message_id': ' ', 'content': FIRST}):
            response = self.client.post(f'/conversations/{self.cid}/messages', json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, before)

    def test_duplicate_message_replays_without_models_and_conflicting_content_rejected(self):
        self.send('M1', FIRST)
        calls = list(self.calls)
        replay = self.send('M1', FIRST)
        self.assertEqual(replay.status_code, 200)
        self.assertTrue(replay.json()['replayed'])
        self.assertEqual(self.calls, calls)
        self.assertEqual(self.send('M1', 'different').status_code, 409)
        self.assertEqual(len(self.store.read_history('U1', self.cid)['messages']), 1)

    def test_failed_answer_keeps_prepared_context_and_next_message_continues(self):
        self.send('M1', FIRST)
        self.timeout_evaluate = True
        failed = self.send('M2', SUPPLEMENT)
        self.assertEqual((failed.status_code, failed.json()['status']), (503, 'failed'))
        self.assertNotIn('private provider', failed.text)
        state = self.store.read_preparation('U1', self.cid)['state']
        self.assertIn('5433', [f.value for f in state.current_facts])
        calls = list(self.calls)
        self.assertEqual(self.send('M2', SUPPLEMENT).status_code, 503)
        self.assertEqual(self.calls, calls)
        self.timeout_evaluate = False
        result = self.send('M3', '继续上一问')
        self.assertEqual(result.json()['status'], 'answer')
        self.assertEqual(self.snapshots[-1]['turn']['question'], FIRST)

    def test_provider_balance_failure_is_preserved_at_each_model_stage(self):
        for stage, target, attribute in (
            ('task', self.service, 'generate_task'), ('context', self.service, 'generate_context'),
            ('evaluate', self, 'evaluate'), ('answer', self, 'answer'),
        ):
            with self.subTest(stage=stage):
                self.cid = self.store.create('U1')
                error = ModelProviderError('provider_balance_insufficient', upstream_status=402)
                with patch.object(target, attribute, side_effect=error) as failed:
                    response = self.send('M1', FIRST + SUPPLEMENT)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['error_code'], 'provider_balance_insufficient')
                self.assertFalse(response.json()['retryable'])
                self.assertEqual(failed.call_count, 1)
                history = self.store.read_history('U1', self.cid)
                self.assertEqual(len(history['messages']), 1)
                self.assertEqual(len(history['contexts']), 0 if stage in ('task','context') else 1)
                if stage in ('evaluate','answer'):
                    failure = history['runs'][0]['result']['error']
                    self.assertEqual(failure['upstream_status'], 402)
                    self.assertEqual(failure['code'], 'provider_balance_insufficient')

    def test_provider_connection_failure_is_retryable_and_does_not_become_clarification(self):
        error = ModelProviderError('provider_connection_failed')
        with patch.object(self, 'evaluate', side_effect=error) as failed:
            response = self.send('M1', FIRST)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['error_code'], 'provider_connection_failed')
        self.assertTrue(response.json()['retryable'])
        self.assertEqual(response.json()['required_information'], [])
        self.assertEqual(failed.call_count, 1)

    def test_task_scopes_persist_across_new_service_and_do_not_leak_into_rag(self):
        self.send('M1', FIRST)
        self.send('M2', SUPPLEMENT)
        old = self.snapshots[-1]
        self.send('M3', '另一个测试环境，应用在容器里，应该怎么连接数据库？')
        new = self.snapshots[-1]
        self.assertNotEqual(old['task']['task_id'], new['task']['task_id'])
        self.assertNotIn('5433', [f['value'] for f in new['context']['current_facts']])
        self.assertNotIn('host', [f['value'] for f in new['context']['current_facts']])
        restarted = TestClient(create_app(conversation_service=self.make_service(), authenticate=self.auth))
        try:
            result = self.send('M4', '回到生产环境继续', client=restarted)
            self.assertEqual(result.json()['status'], 'answer')
            self.assertEqual(self.snapshots[-1]['task']['task_id'], old['task']['task_id'])
            self.assertIn('5433', [f['value'] for f in self.snapshots[-1]['context']['current_facts']])
        finally:
            restarted.close()

    def test_task_ambiguity_is_replied_without_rag_and_resolution_uses_original_content(self):
        self.send('M1', FIRST)
        self.send('M2', '另一个测试环境，应该怎么连接数据库？')
        before = len(self.snapshots)
        self.send('M3', '不确定哪一项；补充应用在宿主机上，端口5433。')
        self.assertEqual(len(self.snapshots), before)
        self.assertEqual(self.calls[-1], 'task')
        resolved = self.send('M4', '我指第一个任务')
        self.assertEqual(resolved.json()['status'], 'answer')
        self.assertEqual({f['source']['message_id'] for f in self.snapshots[-1]['context']['current_facts']
                          if f['target']['field'] == 'published_host_port'}, {'M3'})

    def test_duplicate_inflight_message_is_pending_and_does_not_execute_again(self):
        self.block_version = 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.send, 'M1', FIRST)
            self.assertTrue(self.started.wait(6))
            calls = list(self.calls)
            duplicate = self.send('M1', FIRST)
            self.assertEqual((duplicate.status_code, duplicate.json()['status']), (202, 'pending'))
            self.assertEqual(self.calls, calls)
            self.release.set()
            self.assertEqual(future.result(timeout=8).json()['status'], 'needs_user')
        self.assertEqual(len(self.snapshots), 1)

    def test_new_supplement_prevents_late_old_reply_from_becoming_current(self):
        self.block_version = 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.send, 'M1', FIRST)
            self.assertTrue(self.started.wait(6))
            newest = self.send('M2', SUPPLEMENT)
            self.assertEqual(newest.json()['status'], 'answer')
            self.release.set()
            old = future.result(timeout=8)
        self.assertEqual(old.json()['status'], 'superseded')
        self.assertFalse(old.json()['is_current'])
        self.assertEqual(self.store.read_current('U1', self.cid)['input_version'], 2)

    def test_prior_assistant_clarification_is_automatically_loaded(self):
        first_reply = self.send('M1', FIRST).json()['message']
        self.send('M2', SUPPLEMENT)
        prior = self.task_inputs[-1]['recent_dialogue']
        self.assertTrue(any(m['role'] == 'assistant' and m['content'] == first_reply for m in prior))

