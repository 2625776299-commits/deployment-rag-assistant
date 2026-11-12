"""Opt-in real PostgreSQL checks. Use app.run_conversation_store_probe to isolate DB."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import subprocess
import sys
from threading import Event
import time
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql

from app.conversation_store import PostgresConversationStore
from app.conversation_context import FactTarget


APP = FactTarget('app_1', 'incident_environment', 'runtime_location')
DB = FactTarget('db_1', 'incident_environment', 'runtime_location')
TARGETS = frozenset((APP, DB))
PROD_DB = FactTarget('database', 'production', 'runtime_location')
TEST_DB = FactTarget('database', 'testing', 'runtime_location')
TASK_TARGETS = frozenset((PROD_DB, TEST_DB))


def add_fact(message_id, target, value, quote):
    return {'operation': 'add', 'target': asdict(target), 'value': value, 'previous_fact_id': None,
            'source': {'message_id': message_id, 'quote': quote}}


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'), 'requires isolated PostgreSQL probe')
class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.connections = []
        self.schema = 'rag_conversation_test_' + uuid4().hex
        self.store = PostgresConversationStore(self.connect, schema=self.schema)
        self.store.initialize()
        self.cid = self.store.create('U1')
        self.addCleanup(self.cleanup)

    def connect(self, *, application_name='conversation-probe'):
        conn = psycopg.connect(host='127.0.0.1', port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres', user='rag_conversation_probe', connect_timeout=5,
            application_name=application_name)
        # This role is created only by the disposable probe, not a production default.
        self.connections.append(conn)
        return conn

    def cleanup(self):
        with self.connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(conn.closed for conn in self.connections))

    def start(self, message_id='M1', content='app in container', snapshot=None):
        state = self.store.append_message('U1', self.cid, message_id, content)
        version = state['input_version']
        run_id = self.store.start_run('U1', self.cid, version,
            snapshot if snapshot is not None else {'user_statement': content, 'source_message_id': message_id}, 'build-v1')
        return run_id

    def checkpoint(self, mid, base, proposal, store=None):
        return (store or self.store).commit_context_update('U1', self.cid, mid, base, proposal,
                                                         allowed_targets=TARGETS)

    def prepare(self, mid, base, proposal, selection, store=None, targets=TARGETS, turn_request=None):
        # Explicit fixture decisions only: production code must classify the
        # user's current question/continuation rather than use this test helper.
        if turn_request is None and selection['operation'] != 'clarify':
            task_id = selection.get('task_id')
            prior = [p for p in self.store.read_history('U1', self.cid)['preparations']
                     if task_id is not None and p['task_id'] == task_id and p['input_version'] <= base]
            if prior:
                turn_request = {'mode': 'continue', 'from_input_version': prior[-1]['input_version']}
            else:
                turn_request = {'mode': 'question', 'question': selection.get('goal', selection['source_quote']),
                                'source': {'message_id': mid, 'quote': selection['source_quote']}}
        return (store or self.store).commit_preparation('U1', self.cid, mid, base, proposal, selection,
                                                       allowed_targets=targets, turn_request=turn_request)

    def task_clarification_fixture(self):
        self.store.append_message('U1', self.cid, 'M1', '生产环境数据库断连。')
        prod = self.prepare('M1', 0, {'updates': []},
            {'operation': 'create', 'goal': '排查生产环境数据库断连', 'source_quote': '生产环境数据库断连'},
            targets=TASK_TARGETS)['task']
        self.store.append_message('U1', self.cid, 'M2', '测试环境数据库报密码错误，数据库在宿主机。')
        testing = self.prepare('M2', 1, {'updates': [add_fact('M2', TEST_DB, 'host', '数据库在宿主机')]},
            {'operation': 'create', 'goal': '排查测试环境数据库密码错误', 'source_quote': '测试环境数据库报密码错误'},
            targets=TASK_TARGETS)['task']
        self.store.append_message('U1', self.cid, 'M3', '回到之前那个数据库问题，数据库在容器里。')
        pending = {'operation': 'clarify', 'candidate_task_ids': [prod['task_id'], testing['task_id']],
                   'source_quote': '之前那个数据库问题', 'question': '你指生产环境断连，还是测试环境密码错误？'}
        self.prepare('M3', 2, {'updates': []}, pending, targets=TASK_TARGETS)
        return prod, testing, pending

    def test_recognition_and_scoped_context_commit_return_to_original_task(self):
        from app.context_proposal import RegisteredTarget
        from app.conversation_preparation import prepare_next_turn
        self.store.append_message('U1', self.cid, 'M1', '生产环境数据库连接失败，应用在容器里。')
        first = self.prepare('M1', 0, {'updates': [add_fact('M1', APP, 'container', '应用在容器里')]},
            {'operation': 'create', 'goal': '排查生产环境数据库连接失败', 'source_quote': '生产环境数据库连接失败'})
        self.store.append_message('U1', self.cid, 'M2', '镜像构建失败，构建运行在容器里。')
        builder = FactTarget('builder', 'build_environment', 'runtime_location')
        second = self.prepare('M2', 1, {'updates': [add_fact('M2', builder, 'container', '构建运行在容器里')]},
            {'operation': 'create', 'goal': '排查镜像构建失败', 'source_quote': '镜像构建失败'},
            targets=frozenset((builder,)))
        message = '先回到数据库连接那个问题，应用其实在宿主机上，数据库还是Compose里的。连接地址应该怎么填？'
        self.store.append_message('U1', self.cid, 'M3', message)
        task_id = first['task']['task_id']
        captured = []
        def generate_task(messages):
            self.assertTrue(all(conn.closed for conn in self.connections))
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['current_task_id'], second['task']['task_id'])
            self.assertEqual(len(payload['existing_tasks']), 2)
            return json.dumps({'selection': {'operation': 'select', 'task_id': task_id,
                'source_quote': '回到数据库连接那个问题'}, 'turn_request': {'mode': 'question',
                'question': '宿主机应用如何填写Compose数据库的连接地址？',
                'source': {'message_id': 'M3', 'quote': '连接地址应该怎么填？'}}})
        def generate_context(messages):
            self.assertTrue(all(conn.closed for conn in self.connections))
            payload = json.loads(messages[1]['content'])
            captured.append(payload)
            app_fact = payload['current_user_stated_facts'][0]
            return json.dumps({'updates': [
                {'operation': 'replace', 'target': asdict(APP), 'previous_fact_id': app_fact['fact_id'],
                 'value': 'host', 'source': {'message_id': 'M3', 'quote': '应用其实在宿主机上'}},
                add_fact('M3', DB, 'compose', '数据库还是Compose里的')],
                'analysis_request': {'mode': 'current', 'source': {'message_id': 'M3', 'quote': '连接地址应该怎么填？'},
                                     'confirmation_from_input_version': None}})
        def registry(decision):
            self.assertEqual(decision.selection['task_id'], task_id)
            return (RegisteredTarget(APP, '故障应用位置'), RegisteredTarget(DB, '故障数据库位置'))
        result = prepare_next_turn(self.store, 'U1', self.cid, generate_task=generate_task,
            generate_context=generate_context, resolve_registry=registry)
        self.assertEqual(result['task']['goal'], '排查生产环境数据库连接失败')
        self.assertEqual(result['turn_question'], '宿主机应用如何填写Compose数据库的连接地址？')
        values = {f.target: f.value for f in result['state'].current_facts}
        self.assertEqual(values, {APP: 'host', DB: 'compose', builder: 'container'})
        self.assertEqual([f['target'] for f in captured[0]['current_user_stated_facts']], [asdict(APP)])
        catalog = {t['task_id']: t for t in self.store.read_preparation('U1', self.cid)['task_catalog']}
        self.assertEqual(catalog[task_id]['latest_input_version'], 3)
        self.assertEqual(catalog[second['task']['task_id']]['latest_input_version'], 2)
        self.assertEqual(prepare_next_turn(self.store, 'U1', self.cid, generate_task=generate_task,
            generate_context=generate_context, resolve_registry=registry)['status'], 'nothing_to_prepare')
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.read_preparation('U2', self.cid)

    def test_recognized_ambiguity_does_not_call_context_model_or_registry(self):
        from app.conversation_preparation import prepare_next_turn
        prod, testing, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '刚才那个问题，数据库在容器里。')
        def forbidden(*args):
            self.fail('ambiguous task must not resolve scope or call context model')
        candidate = {'selection': {'operation': 'clarify',
            'candidate_task_ids': [prod['task_id'], testing['task_id']],
            'source_quote': '刚才那个问题', 'question': '指生产环境还是测试环境？'}, 'turn_request': None}
        before = self.store.read_preparation('U1', self.cid)['state'].current_facts
        result = prepare_next_turn(self.store, 'U1', self.cid, generate_task=lambda _: json.dumps(candidate),
            generate_context=forbidden, resolve_registry=forbidden)
        self.assertEqual(result['status'], 'needs_task')
        self.assertEqual(result['state'].current_facts, before)
        self.assertFalse(result['ready_for_run'])

    def test_context_model_failure_leaves_raw_message_without_half_preparation(self):
        from app.context_proposal import RegisteredTarget
        from app.conversation_preparation import prepare_next_turn
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机，为什么连接失败？')
        candidate = {'selection': {'operation': 'create', 'goal': '排查连接失败', 'source_quote': '为什么连接失败'},
            'turn_request': {'mode': 'question', 'question': '宿主机应用为什么连接失败？',
                'source': {'message_id': 'M1', 'quote': '应用在宿主机，为什么连接失败'}}}
        def timeout(messages):
            raise TimeoutError('context_timeout')
        with self.assertRaises(TimeoutError):
            prepare_next_turn(self.store, 'U1', self.cid, generate_task=lambda _: json.dumps(candidate),
                generate_context=timeout, resolve_registry=lambda _: (RegisteredTarget(APP, '应用运行位置'),))
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(len(history['messages']), 1)
        for name in ('tasks', 'preparations', 'contexts', 'runs'):
            self.assertEqual(history[name], [])

    def test_turn_focus_is_saved_separately_and_supplement_continues_same_task(self):
        broad = '解释数据库容器重建后旧连接失效的原因和完整修复方法。'
        self.store.append_message('U1', self.cid, 'M1', broad)
        first = self.prepare('M1', 0, {'updates': []},
            {'operation': 'create', 'goal': broad, 'source_quote': broad})
        task = first['task']
        focused = '先不讨论完整修复方案，只解释pool_pre_ping能否处理事务执行中的断连。'
        self.store.append_message('U1', self.cid, 'M2', focused)
        narrowed = self.prepare('M2', 1, {'updates': []},
            {'operation': 'select', 'task_id': task['task_id'], 'source_quote': focused},
            turn_request={'mode': 'question', 'question': focused, 'source': {'message_id': 'M2', 'quote': focused}})
        self.assertEqual(narrowed['task']['goal'], broad)
        self.assertEqual(narrowed['turn_question'], focused)
        self.store.append_message('U1', self.cid, 'M3', '应用在宿主机')
        supplemented = self.prepare('M3', 2, {'updates': [add_fact('M3', APP, 'host', '应用在宿主机')]},
            {'operation': 'select', 'task_id': task['task_id'], 'source_quote': '应用在宿主机'},
            turn_request={'mode': 'continue', 'from_input_version': 2})
        self.assertEqual(supplemented['turn_question'], focused)
        run = self.store.start_prepared_run('U1', self.cid, 3, 'build-v1')
        frozen = self.store.read_run_input('U1', self.cid, run)
        self.assertEqual(frozen['input_snapshot']['task']['goal'], broad)
        self.assertEqual(frozen['input_snapshot']['turn']['question'], focused)
        self.assertEqual(frozen['input_snapshot']['turn']['raw_message'], '应用在宿主机')
        self.store.append_message('U1', self.cid, 'M4', '换个问题，解释SQL JOIN')
        self.prepare('M4', 3, {'updates': []},
            {'operation': 'create', 'goal': '解释SQL JOIN', 'source_quote': '解释SQL JOIN'})
        self.store.append_message('U1', self.cid, 'M5', '回到数据库问题继续')
        resumed = self.prepare('M5', 4, {'updates': []},
            {'operation': 'select', 'task_id': task['task_id'], 'source_quote': '回到数据库问题'},
            turn_request={'mode': 'continue', 'from_input_version': 3})
        self.assertEqual(resumed['turn_question'], focused)
        self.assertEqual(self.store.read_run_input('U1', self.cid, run), frozen)
        history = self.store.read_history('U1', self.cid)
        self.assertEqual([p['turn_question'] for p in history['preparations']],
                         [broad, focused, focused, '解释SQL JOIN', focused])

    def test_turn_question_is_required_atomic_and_cannot_change_on_replay(self):
        self.store.append_message('U1', self.cid, 'M1', '解释数据库断连')
        selection = {'operation': 'create', 'goal': '解释数据库断连', 'source_quote': '解释数据库断连'}
        with self.assertRaisesRegex(ValueError, 'require_turn_request'):
            self.store.commit_preparation('U1', self.cid, 'M1', 0, {'updates': []}, selection, allowed_targets=TARGETS)
        history = self.store.read_history('U1', self.cid)
        self.assertEqual((len(history['messages']), len(history['tasks']), len(history['contexts'])), (1, 0, 0))
        request = {'mode': 'question', 'question': '解释数据库断连', 'source': {'message_id': 'M1', 'quote': '解释数据库断连'}}
        prepared = self.prepare('M1', 0, {'updates': []}, selection, turn_request=request)
        replay = self.prepare('M1', 0, {'updates': []}, selection, turn_request=request)
        self.assertTrue(replay['replayed'])
        self.assertEqual(replay['turn_question'], prepared['turn_question'])
        with self.assertRaisesRegex(ValueError, 'turn_question_conflict'):
            self.prepare('M1', 0, {'updates': []}, selection, turn_request={**request, 'question': '改变回答范围'})
        self.store.append_message('U1', self.cid, 'M2', '应用在宿主机')
        selection = {'operation': 'select', 'task_id': prepared['task']['task_id'], 'source_quote': '应用在宿主机'}
        with self.assertRaisesRegex(ValueError, 'turn_question_source_not_authorized'):
            self.prepare('M2', 1, {'updates': []}, selection, turn_request=request)
        self.assertEqual(self.store.read_preparation('U1', self.cid)['next_message'].message_id, 'M2')

    def test_question_continuation_rejects_other_task_future_and_stale_question(self):
        self.store.append_message('U1', self.cid, 'M1', '解释数据库断连')
        task = self.prepare('M1', 0, {'updates': []},
            {'operation': 'create', 'goal': '解释数据库断连', 'source_quote': '解释数据库断连'})['task']
        self.store.append_message('U1', self.cid, 'M2', '解释FastAPI')
        self.prepare('M2', 1, {'updates': []}, {'operation': 'create', 'goal': '解释FastAPI', 'source_quote': '解释FastAPI'})
        self.store.append_message('U1', self.cid, 'M3', '回到数据库问题')
        selection = {'operation': 'select', 'task_id': task['task_id'], 'source_quote': '数据库问题'}
        for version, error in ((2, 'question_continuation_must_use_latest_same_task'),
                               (3, 'invalid_question_continuation'), (True, 'invalid_question_continuation')):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, error):
                    self.prepare('M3', 2, {'updates': []}, selection,
                                 turn_request={'mode': 'continue', 'from_input_version': version})
        self.prepare('M3', 2, {'updates': []}, selection, turn_request={'mode': 'continue', 'from_input_version': 1})
        self.store.append_message('U1', self.cid, 'M4', '只解释pool_pre_ping')
        self.prepare('M4', 3, {'updates': []}, {**selection, 'source_quote': '只解释pool_pre_ping'},
            turn_request={'mode': 'question', 'question': '只解释pool_pre_ping',
                          'source': {'message_id': 'M4', 'quote': '只解释pool_pre_ping'}})
        self.store.append_message('U1', self.cid, 'M5', '继续')
        with self.assertRaisesRegex(ValueError, 'question_continuation_must_use_latest_same_task'):
            self.prepare('M5', 4, {'updates': []}, {**selection, 'source_quote': '继续'},
                         turn_request={'mode': 'continue', 'from_input_version': 1})

    def test_resolved_task_can_take_question_from_exact_deferred_message(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        request = {'mode': 'question', 'question': '生产数据库在容器里时如何排查之前的连接问题？',
                   'source': {'message_id': 'M3', 'quote': '之前那个数据库问题，数据库在容器里'}}
        result = self.prepare('M4', 3, {'updates': []}, self.resolve_selection(prod),
                              targets=TASK_TARGETS, turn_request=request)
        self.assertEqual(result['turn_question'], request['question'])
        self.assertEqual(self.store.read_preparation('U1', self.cid)['turn_request']['source']['message_id'], 'M3')

    def test_ambiguous_task_cannot_receive_question_and_old_checkpoint_cannot_guess_one(self):
        prod, _, pending = self.task_clarification_fixture()
        request = {'mode': 'question', 'question': '解释断连', 'source': {'message_id': 'M3', 'quote': '数据库问题'}}
        with self.assertRaisesRegex(ValueError, 'turn_question_conflict'):
            self.prepare('M3', 2, {'updates': []}, pending, targets=TASK_TARGETS, turn_request=request)
        # Simulate an older stored checkpoint lacking the newly introduced field.
        with self.connect() as conn:
            conn.execute(sql.SQL('UPDATE {} SET turn_question=NULL,turn_request=NULL WHERE conversation_id=%s AND input_version=2')
                .format(sql.Identifier(self.schema, 'preparations')), (self.cid,))
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境')
        selected = self.prepare('M4', 3, {'updates': []}, self.resolve_selection(prod), targets=TASK_TARGETS)
        with self.connect() as conn:
            conn.execute(sql.SQL('UPDATE {} SET turn_question=NULL,turn_request=NULL WHERE conversation_id=%s AND input_version=4')
                .format(sql.Identifier(self.schema, 'preparations')), (self.cid,))
        restored = self.store.read_preparation('U1', self.cid)
        self.assertEqual(restored['prepared_status'], 'needs_question')
        self.assertFalse(restored['ready_for_run'])
        self.assertEqual(restored['prepared_task'], selected['task'])
        with self.assertRaisesRegex(ValueError, 'turn_question_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 4, 'build-v1')

    def test_stored_turn_snapshot_drives_real_rag_code_with_scripted_models(self):
        from app.conversation_rag import run_prepared_snapshot
        self.store.append_message('U1', self.cid, 'M1', '解释数据库断连原因和完整修复方案')
        task = self.prepare('M1', 0, {'updates': []}, {'operation': 'create',
            'goal': '解释数据库断连原因和完整修复方案', 'source_quote': '解释数据库断连原因和完整修复方案'})['task']
        question = '只解释连接检查的一般作用'
        self.store.append_message('U1', self.cid, 'M2', question)
        self.prepare('M2', 1, {'updates': []}, {'operation': 'select', 'task_id': task['task_id'], 'source_quote': question},
            turn_request={'mode': 'question', 'question': question, 'source': {'message_id': 'M2', 'quote': question}})
        run = self.store.start_prepared_run('U1', self.cid, 2, 'fixture-build')
        stored = self.store.read_run_input('U1', self.cid, run)
        passage = {'chunk_id': 'TEST:1', 'content': '固定测试资料：检查连接可识别部分失效连接。',
                   'metadata': {'document_id': 'fixture', 'section_id': 'TEST', 'source_url': 'https://example.invalid/fixture'}}
        calls = {'queries': [], 'questions': [], 'backgrounds': []}
        def retrieve(query):
            calls['queries'].append(query)
            return [passage]
        def evaluate(messages):
            payload = json.loads(messages[1]['content'])
            calls['questions'].append(payload['question'])
            calls['backgrounds'].append(payload['task_background_goal'])
            return json.dumps({'requirements': [{'requirement_id': 'R1', 'requirement': question, 'question_quote': question}],
                'assessments': [{'requirement_id': 'R1', 'coverage': 'covered', 'gap': None,
                    'evidence': [{'chunk_id': 'TEST:1', 'quote': passage['content']}]}],
                'action': 'answer', 'selected_chunk_ids': ['TEST:1'], 'missing_knowledge': [], 'missing_user_information': [],
                'search_query': None, 'reason': 'Scripted wiring check.', 'known_findings': [], 'pending_impacts': []})
        def generate(messages):
            calls['questions'].append(json.loads(messages[1]['content'])['question'])
            return json.dumps({'status': 'answer', 'answer': '检查连接可识别部分失效连接。[TEST:1]',
                               'citation_ids': ['TEST:1'], 'missing_information': []})
        result = run_prepared_snapshot(stored['input_snapshot'], retrieve=retrieve, evaluate_raw=evaluate, generate_raw=generate)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(calls['queries'], [question])
        self.assertEqual(calls['questions'], [question, question])
        self.assertEqual(calls['backgrounds'], [task['goal']])
        published = self.store.finish_run('U1', self.cid, run, result)
        self.assertTrue(published['is_current'])
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.read_run_input('U2', self.cid, run)

    def resolve_selection(self, task, quote='我指生产环境'):
        return {'operation': 'resolve_task', 'task_id': task['task_id'],
                'clarification_version': 3, 'source_quote': quote}

    def test_ambiguous_task_is_checkpointed_without_updating_either_environment(self):
        prod, testing, pending = self.task_clarification_fixture()
        loaded = self.store.read_preparation('U1', self.cid)
        self.assertEqual(loaded['prepared_status'], 'needs_task')
        self.assertFalse(loaded['ready_for_run'])
        self.assertIsNone(loaded['prepared_task'])
        self.assertIsNone(loaded['next_message'])
        self.assertEqual(loaded['state'].input_version, 3)
        self.assertEqual({f.target: f.value for f in loaded['state'].current_facts}, {TEST_DB: 'host'})
        self.assertEqual(loaded['pending_task_clarifications'], [{
            'input_version': 3, 'message_id': 'M3', 'content': '回到之前那个数据库问题，数据库在容器里。',
            'candidate_task_ids': [prod['task_id'], testing['task_id']], 'question': pending['question']}])
        with self.assertRaisesRegex(ValueError, 'turn_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 3, 'build-v1')
        with self.assertRaisesRegex(ValueError, 'use_prepared_run_entry'):
            self.store.start_run('U1', self.cid, 3, {}, 'build-v1')
        with self.assertRaisesRegex(ValueError, 'ambiguous_task_cannot_update_context'):
            self.prepare('M3', 2, {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]},
                         pending, targets=TASK_TARGETS)
        # Receipt of the clarification must not be blocked by the unresolved task.
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        next_message = self.store.read_preparation('U1', self.cid)['next_message']
        self.assertEqual(next_message.message_id, 'M4')
        self.assertEqual(next_message.input_version, 4)

    def test_task_resolution_preserves_original_quote_and_separate_attribution(self):
        prod, testing, pending = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        proposal = {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]}
        result = self.prepare('M4', 3, proposal, self.resolve_selection(prod), targets=TASK_TARGETS)
        self.assertTrue(result['ready_for_run'])
        self.assertEqual(result['task'], prod)
        self.assertEqual({f.target: f.value for f in result['state'].current_facts},
                         {TEST_DB: 'host', PROD_DB: 'container'})
        fact = next(f for f in result['state'].current_facts if f.target == PROD_DB)
        self.assertEqual(fact.source.message_id, 'M3')
        self.assertEqual(fact.source.quote, '数据库在容器里')
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(history['task_resolutions'], [{'clarification_version': 3, 'resolved_input_version': 4}])
        self.assertEqual(history['preparations'][-1]['selection']['source_quote'], '我指生产环境')
        self.assertEqual(history['preparations'][-1]['task_id'], prod['task_id'])
        self.assertEqual(self.store.read_preparation('U1', self.cid)['pending_task_clarifications'], [])
        self.assertEqual(len(history['tasks']), 2)
        self.store.start_prepared_run('U1', self.cid, 4, 'build-v1')

    def test_resolution_cannot_fabricate_current_quote_or_import_other_history(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        for source_id, quote in (('M4', '数据库在容器里'), ('M2', '数据库在宿主机'), ('M3', '不存在的引文')):
            with self.subTest(source=source_id, quote=quote):
                with self.assertRaisesRegex(ValueError, 'source_quote_not_in_authorized_user_message'):
                    self.prepare('M4', 3, {'updates': [add_fact(source_id, PROD_DB, 'container', quote)]},
                                 self.resolve_selection(prod), targets=TASK_TARGETS)
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
            self.prepare('M4', 3, {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]},
                {'operation': 'select', 'task_id': prod['task_id'], 'source_quote': '我指生产环境'}, targets=TASK_TARGETS)
        self.assertEqual(self.store.read_preparation('U1', self.cid)['state'].input_version, 3)
        self.assertEqual(self.store.read_history('U1', self.cid)['task_resolutions'], [])

    def test_resolution_can_use_a_new_explicit_correction_instead_of_old_supplement(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境，但刚确认数据库实际在宿主机。')
        result = self.prepare('M4', 3, {'updates': [add_fact('M4', PROD_DB, 'host', '数据库实际在宿主机')]},
                              self.resolve_selection(prod), targets=TASK_TARGETS)
        fact = next(f for f in result['state'].current_facts if f.target == PROD_DB)
        self.assertEqual(fact.value, 'host')
        self.assertEqual(fact.source.message_id, 'M4')

    def test_unrelated_task_can_proceed_and_cannot_consume_unmatched_clarification(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '先解释FastAPI异步。')
        other = self.prepare('M4', 3, {'updates': []},
            {'operation': 'create', 'goal': '解释FastAPI异步', 'source_quote': '解释FastAPI异步'}, targets=TASK_TARGETS)
        self.assertTrue(other['ready_for_run'])
        self.assertEqual(len(self.store.read_preparation('U1', self.cid)['pending_task_clarifications']), 1)
        self.store.append_message('U1', self.cid, 'M5', '我指生产环境。')
        with self.assertRaisesRegex(ValueError, 'task_not_in_clarification_candidates'):
            self.prepare('M5', 4, {'updates': []}, self.resolve_selection(other['task']), targets=TASK_TARGETS)
        result = self.prepare('M5', 4, {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]},
                              self.resolve_selection(prod), targets=TASK_TARGETS)
        self.assertTrue(result['ready_for_run'])
        self.assertEqual(result['task'], prod)

    def test_deferred_source_cannot_overwrite_a_newer_environment_statement(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '另补充生产环境，数据库在宿主机。')
        updated = self.prepare('M4', 3, {'updates': [add_fact('M4', PROD_DB, 'host', '数据库在宿主机')]},
            {'operation': 'select', 'task_id': prod['task_id'], 'source_quote': '生产环境'}, targets=TASK_TARGETS)
        fact = next(f for f in updated['state'].current_facts if f.target == PROD_DB)
        self.store.append_message('U1', self.cid, 'M5', '我指生产环境，现在数据库仍在宿主机。')
        change = {'operation': 'replace', 'target': asdict(PROD_DB), 'previous_fact_id': fact.fact_id,
                  'value': 'container', 'source': {'message_id': 'M3', 'quote': '数据库在容器里'}}
        with self.assertRaisesRegex(ValueError, 'deferred_source_stale_for_target'):
            self.prepare('M5', 4, {'updates': [change]}, self.resolve_selection(prod), targets=TASK_TARGETS)
        change = {**change, 'value': 'host', 'source': {'message_id': 'M5', 'quote': '数据库仍在宿主机'}}
        confirmed = self.prepare('M5', 4, {'updates': [change]}, self.resolve_selection(prod), targets=TASK_TARGETS)
        self.assertTrue(confirmed['ready_for_run'])
        self.assertEqual(next(f for f in confirmed['state'].current_facts if f.target == PROD_DB).value, 'host')

    def test_resolution_failure_rolls_back_assignment_fact_and_resolution_together(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        proposal = {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]}
        @contextmanager
        def rollback_connect():
            with self.connect() as conn:
                yield conn
                raise RuntimeError('resolution_commit_failed')
        failing = PostgresConversationStore(rollback_connect, schema=self.schema)
        with self.assertRaisesRegex(RuntimeError, 'resolution_commit_failed'):
            self.prepare('M4', 3, proposal, self.resolve_selection(prod), failing, TASK_TARGETS)
        loaded = self.store.read_preparation('U1', self.cid)
        self.assertEqual(loaded['prepared_status'], 'needs_task')
        self.assertEqual(loaded['next_message'].message_id, 'M4')
        self.assertEqual({f.target: f.value for f in loaded['state'].current_facts}, {TEST_DB: 'host'})
        self.assertEqual(self.store.read_history('U1', self.cid)['task_resolutions'], [])
        with self.assertRaisesRegex(ValueError, 'turn_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 4, 'build-v1')
        self.assertTrue(self.prepare('M4', 3, proposal, self.resolve_selection(prod), targets=TASK_TARGETS)['ready_for_run'])

    def test_newer_same_value_confirmation_also_protects_against_deferred_overwrite(self):
        self.store.append_message('U1', self.cid, 'M1', '生产数据库在宿主机')
        first = self.prepare('M1', 0, {'updates': [add_fact('M1', PROD_DB, 'host', '数据库在宿主机')]},
            {'operation': 'create', 'goal': '排查生产', 'source_quote': '生产数据库'}, targets=TASK_TARGETS)
        prod, original = first['task'], first['state'].current_facts[0]
        self.store.append_message('U1', self.cid, 'M2', '排查测试数据库')
        testing = self.prepare('M2', 1, {'updates': []},
            {'operation': 'create', 'goal': '排查测试', 'source_quote': '测试数据库'}, targets=TASK_TARGETS)['task']
        self.store.append_message('U1', self.cid, 'M3', '之前那个数据库在容器里')
        self.prepare('M3', 2, {'updates': []}, {'operation': 'clarify',
            'source_quote': '之前那个数据库', 'candidate_task_ids': [prod['task_id'], testing['task_id']],
            'question': '生产还是测试？'}, targets=TASK_TARGETS)
        self.store.append_message('U1', self.cid, 'M4', '再次确认生产数据库在宿主机')
        unchanged = {'operation': 'replace', 'target': asdict(PROD_DB), 'previous_fact_id': original.fact_id,
                     'value': 'host', 'source': {'message_id': 'M4', 'quote': '数据库在宿主机'}}
        confirmed = self.prepare('M4', 3, {'updates': [unchanged]},
            {'operation': 'select', 'task_id': prod['task_id'], 'source_quote': '生产数据库'}, targets=TASK_TARGETS)
        self.assertEqual(confirmed['state'].current_facts[0], original)
        self.store.append_message('U1', self.cid, 'M5', '我指生产环境')
        stale = {**unchanged, 'value': 'container', 'source': {'message_id': 'M3', 'quote': '数据库在容器里'}}
        with self.assertRaisesRegex(ValueError, 'deferred_source_stale_for_target'):
            self.prepare('M5', 4, {'updates': [stale]}, self.resolve_selection(prod), targets=TASK_TARGETS)

    def test_concurrent_resolution_and_old_pending_replay_do_not_duplicate_or_reopen(self):
        prod, _, pending = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        proposal = {'updates': [add_fact('M3', PROD_DB, 'container', '数据库在容器里')]}
        def save(_):
            return self.prepare('M4', 3, proposal, self.resolve_selection(prod),
                                PostgresConversationStore(self.connect, schema=self.schema), TASK_TARGETS)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, range(2)))
        self.assertEqual(sorted(r['replayed'] for r in results), [False, True])
        self.assertEqual(results[0]['state'], results[1]['state'])
        self.assertEqual(len(self.store.read_history('U1', self.cid)['task_resolutions']), 1)
        replay = self.prepare('M3', 2, {'updates': []}, pending, targets=TASK_TARGETS)
        self.assertTrue(replay['replayed'])
        self.assertFalse(replay['ready_for_run'])
        loaded = self.store.read_preparation('U1', self.cid)
        self.assertEqual(loaded['prepared_task'], prod)
        self.assertEqual(loaded['pending_task_clarifications'], [])
        self.store.append_message('U1', self.cid, 'M5', '我指生产环境。')
        with self.assertRaisesRegex(ValueError, 'task_clarification_already_resolved'):
            self.prepare('M5', 4, {'updates': []}, self.resolve_selection(prod), targets=TASK_TARGETS)

    def test_pending_clarification_restores_in_new_process_and_accepts_reply(self):
        prod, _, _ = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        script = '''import json, os, sys, psycopg
from app.conversation_store import PostgresConversationStore
from app.conversation_context import FactTarget
def connect():
    return psycopg.connect(host="127.0.0.1", port=int(os.environ["RAG_CONVERSATION_TEST_PORT"]),
        dbname="postgres", user="rag_conversation_probe", connect_timeout=5)
store = PostgresConversationStore(connect, schema=sys.argv[1])
before = store.read_preparation("U1", sys.argv[2])
proposal = {"updates": [{"operation":"add", "target":{"entity_id":"database","scope_id":"production","field":"runtime_location"},
    "previous_fact_id":None, "value":"container", "source":{"message_id":"M3","quote":"数据库在容器里"}}]}
selection = {"operation":"resolve_task", "task_id":sys.argv[3], "clarification_version":3, "source_quote":"我指生产环境"}
result = store.commit_preparation("U1",sys.argv[2],"M4",3,proposal,selection,
    allowed_targets=frozenset((FactTarget("database","production","runtime_location"),)),
    turn_request={"mode":"continue","from_input_version":1})
print(json.dumps({"pid":os.getpid(),"before":before["prepared_status"],"next":before["next_message"].message_id,
    "pending_message":before["pending_task_clarifications"][0]["message_id"],"ready":result["ready_for_run"],"state":result["state"].to_dict()}))
'''
        process = subprocess.run([sys.executable, '-c', script, self.schema, self.cid, prod['task_id']],
            capture_output=True, text=True, encoding='utf-8', timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(process.returncode, 0, process.stderr)
        loaded = json.loads(process.stdout)
        self.assertNotEqual(loaded['pid'], os.getpid())
        self.assertEqual((loaded['before'], loaded['next'], loaded['pending_message']), ('needs_task', 'M4', 'M3'))
        self.assertTrue(loaded['ready'])
        self.assertEqual(self.store.read_preparation('U1', self.cid)['prepared_task'], prod)

    def test_clarification_candidate_and_scope_guards(self):
        prod, testing, pending = self.task_clarification_fixture()
        self.store.append_message('U1', self.cid, 'M4', '我指生产环境。')
        with self.assertRaisesRegex(ValueError, 'task_quote_not_in_current_user_message'):
            self.prepare('M4', 3, {'updates': []}, self.resolve_selection(prod, '我指测试环境'), targets=TASK_TARGETS)
        with self.assertRaisesRegex(ValueError, 'unregistered_target'):
            self.prepare('M4', 3, {'updates': [add_fact('M3', TEST_DB, 'container', '数据库在容器里')]},
                         self.resolve_selection(prod), targets=frozenset((PROD_DB,)))
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.read_preparation('U2', self.cid)
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.commit_preparation('U2', self.cid, 'M4', 3, {'updates': []},
                                         self.resolve_selection(prod), allowed_targets=TASK_TARGETS)
        other = self.store.create('U1')
        self.store.append_message('U1', other, 'X1', '之前那个数据库问题')
        with self.assertRaisesRegex(ValueError, 'task_not_found'):
            self.store.commit_preparation('U1', other, 'X1', 0, {'updates': []}, pending, allowed_targets=TASK_TARGETS)

    def test_atomic_preparation_returns_to_original_task_with_updated_environment(self):
        self.store.append_message('U1', self.cid, 'M1', '请解释数据库断连的原因和处理方法，数据库位置待确认。')
        first = self.prepare('M1', 0, {'updates': [
            {'operation': 'mark_pending', 'target': asdict(DB), 'previous_fact_id': None,
             'reason': 'uncertain_value', 'source': {'message_id': 'M1', 'quote': '数据库位置待确认'}},
        ]}, {'operation': 'create', 'goal': '解释数据库断连原因和处理方法', 'source_quote': '请解释数据库断连的原因和处理方法'})
        task1 = first['task']
        self.store.append_message('U1', self.cid, 'M2', '补充：应用在宿主机。')
        self.prepare('M2', 1, {'updates': [add_fact('M2', APP, 'host', '应用在宿主机')]},
                     {'operation': 'select', 'task_id': task1['task_id'], 'source_quote': '补充'})
        self.store.append_message('U1', self.cid, 'M3', '换个问题，解释FastAPI同步与异步。')
        second = self.prepare('M3', 2, {'updates': []},
            {'operation': 'create', 'goal': '解释FastAPI同步与异步', 'source_quote': '解释FastAPI同步与异步'})
        self.assertNotEqual(task1['task_id'], second['task']['task_id'])
        self.store.append_message('U1', self.cid, 'M4', '回到数据库断连问题，数据库在容器里。')
        self.assertFalse(self.store.read_preparation('U1', self.cid)['ready_for_run'])
        with self.assertRaisesRegex(ValueError, 'turn_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 4, 'build-v1')
        resumed = self.prepare('M4', 3, {'updates': [
            {'operation': 'resolve', 'target': asdict(DB), 'pending_id': first['state'].pending_facts[0].pending_id,
             'value': 'container', 'source': {'message_id': 'M4', 'quote': '数据库在容器里'}},
        ]}, {'operation': 'select', 'task_id': task1['task_id'], 'source_quote': '回到数据库断连问题'})
        self.assertEqual(resumed['task'], task1)
        self.assertTrue(resumed['ready_for_run'])
        self.assertEqual({f.target: f.value for f in resumed['state'].current_facts}, {APP: 'host', DB: 'container'})
        run = self.store.start_prepared_run('U1', self.cid, 4, 'build-v1')
        self.assertEqual(self.store.start_prepared_run('U1', self.cid, 4, 'build-v1'), run)
        with self.assertRaisesRegex(ValueError, 'use_prepared_run_entry'):
            self.store.start_run('U1', self.cid, 4, {'task': task1}, 'build-v1')
        with self.assertRaisesRegex(ValueError, 'use_atomic_task_preparation'):
            self.checkpoint('M4', 3, {'updates': []})
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(len(history['tasks']), 2)
        self.assertEqual([p['task_id'] for p in history['preparations']],
                         [task1['task_id'], task1['task_id'], second['task']['task_id'], task1['task_id']])
        self.assertEqual(history['runs'][0]['input_snapshot']['task'], task1)
        self.assertEqual(history['runs'][0]['input_snapshot']['context']['input_version'], 4)
        # An answer timeout does not undo either half of the committed preparation.
        self.store.finish_run('U1', self.cid, run, {'error': 'evaluation_timeout'}, status='failed')
        self.assertEqual(self.store.read_preparation('U1', self.cid)['prepared_task'], task1)
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])

    def test_failed_atomic_commit_keeps_raw_message_and_rolls_back_both_halves(self):
        self.store.append_message('U1', self.cid, 'M1', '排查数据库断连。')
        first = self.prepare('M1', 0, {'updates': []},
            {'operation': 'create', 'goal': '排查数据库断连', 'source_quote': '排查数据库断连'})
        self.store.append_message('U1', self.cid, 'M2', '换个问题。应用在宿主机。')
        proposal = {'updates': [add_fact('M2', APP, 'host', '应用在宿主机')]}
        selection = {'operation': 'create', 'goal': '解释应用部署', 'source_quote': '换个问题'}

        @contextmanager
        def rollback_connect():
            with self.connect() as conn:
                yield conn
                raise RuntimeError('injected_after_all_writes_before_commit')

        failing = PostgresConversationStore(rollback_connect, schema=self.schema)
        with self.assertRaisesRegex(RuntimeError, 'injected_after_all_writes_before_commit'):
            self.prepare('M2', 1, proposal, selection, failing)
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(len(history['messages']), 2)
        self.assertEqual(len(history['tasks']), 1)
        self.assertEqual(len(history['contexts']), 1)
        self.assertEqual(len(history['preparations']), 1)
        pending = self.store.read_preparation('U1', self.cid)
        self.assertEqual(pending['prepared_task'], first['task'])
        self.assertEqual(pending['next_message'].message_id, 'M2')
        self.assertEqual(pending['state'].current_facts, ())
        self.assertFalse(pending['ready_for_run'])
        with self.assertRaisesRegex(ValueError, 'turn_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 2, 'build-v1')
        recovered = self.prepare('M2', 1, proposal, selection)
        self.assertTrue(recovered['ready_for_run'])
        self.assertNotEqual(recovered['task']['task_id'], first['task']['task_id'])

    def test_invalid_context_does_not_leave_new_task_or_allow_skipping_message(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        selection = {'operation': 'create', 'goal': '排查连接', 'source_quote': '应用在宿主机'}
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
            self.prepare('M1', 0, {'updates': [add_fact('M1', APP, 'host', '不存在的引文')]}, selection)
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(history['tasks'], [])
        self.assertEqual(history['contexts'], [])
        self.assertEqual(history['preparations'], [])
        self.store.append_message('U1', self.cid, 'M2', '继续')
        with self.assertRaisesRegex(ValueError, 'previous_turn_not_prepared'):
            self.prepare('M2', 1, {'updates': []}, {'operation': 'create', 'goal': '排查连接', 'source_quote': '继续'})
        with self.assertRaisesRegex(ValueError, 'turn_not_prepared'):
            self.store.start_prepared_run('U1', self.cid, 2, 'build-v1')

    def test_atomic_replay_does_not_reselect_old_task_or_reallocate_ids(self):
        self.store.append_message('U1', self.cid, 'M1', '排查连接，应用在宿主机')
        proposal = {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]}
        selection = {'operation': 'create', 'goal': '排查连接', 'source_quote': '排查连接'}
        first = self.prepare('M1', 0, proposal, selection)
        self.store.append_message('U1', self.cid, 'M2', '解释FastAPI')
        second = self.prepare('M2', 1, {'updates': []},
            {'operation': 'create', 'goal': '解释FastAPI', 'source_quote': '解释FastAPI'})
        replay = self.prepare('M1', 0, proposal, selection)
        self.assertTrue(replay['replayed'])
        self.assertFalse(replay['ready_for_run'])
        self.assertEqual(replay['task'], first['task'])
        self.assertEqual(replay['state'], first['state'])
        self.assertEqual(self.store.read_preparation('U1', self.cid)['prepared_task'], second['task'])
        with self.assertRaisesRegex(ValueError, 'task_selection_conflict'):
            self.prepare('M1', 0, proposal, {**selection, 'goal': '另一目标'})
        with self.assertRaisesRegex(ValueError, 'context_update_conflict'):
            self.prepare('M1', 0, {'updates': []}, selection)
        self.assertEqual(len(self.store.read_history('U1', self.cid)['tasks']), 2)

    def test_concurrent_atomic_create_saves_one_task_and_one_preparation(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        proposal = {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]}
        selection = {'operation': 'create', 'goal': '排查连接', 'source_quote': '应用在宿主机'}
        def save(_):
            store = PostgresConversationStore(self.connect, schema=self.schema)
            return self.prepare('M1', 0, proposal, selection, store)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, range(2)))
        self.assertEqual(sorted(r['replayed'] for r in results), [False, True])
        self.assertEqual(results[0]['task'], results[1]['task'])
        self.assertEqual(results[0]['state'], results[1]['state'])
        history = self.store.read_history('U1', self.cid)
        self.assertEqual(len(history['tasks']), 1)
        self.assertEqual(len(history['preparations']), 1)

    def test_task_selection_is_scoped_to_authorized_conversation(self):
        self.store.append_message('U1', self.cid, 'M1', '排查连接')
        first = self.prepare('M1', 0, {'updates': []},
            {'operation': 'create', 'goal': '排查连接', 'source_quote': '排查连接'})
        other = self.store.create('U1')
        self.store.append_message('U1', other, 'X1', '继续原任务')
        with self.assertRaisesRegex(ValueError, 'task_not_found'):
            self.store.commit_preparation('U1', other, 'X1', 0, {'updates': []},
                {'operation': 'select', 'task_id': first['task']['task_id'], 'source_quote': '继续原任务'}, allowed_targets=TARGETS)
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.commit_preparation('U2', self.cid, 'M1', 0, {'updates': []},
                {'operation': 'create', 'goal': '排查连接', 'source_quote': '排查连接'}, allowed_targets=TARGETS)
        with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
            self.store.start_prepared_run('U2', self.cid, 1, 'build-v1')
        self.assertEqual(self.store.read_history('U1', other)['contexts'], [])

    def test_atomic_preparation_restores_in_new_process(self):
        self.store.append_message('U1', self.cid, 'M1', '排查连接，应用在宿主机')
        prepared = self.prepare('M1', 0, {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]},
            {'operation': 'create', 'goal': '排查连接', 'source_quote': '排查连接'})
        script = '''import json, os, sys, psycopg
from app.conversation_store import PostgresConversationStore
def connect():
    return psycopg.connect(host="127.0.0.1", port=int(os.environ["RAG_CONVERSATION_TEST_PORT"]),
        dbname="postgres", user="rag_conversation_probe", connect_timeout=5)
store = PostgresConversationStore(connect, schema=sys.argv[1])
prepared = store.read_preparation("U1", sys.argv[2])
run = store.start_prepared_run("U1", sys.argv[2], 1, "build-v1")
print(json.dumps({"pid": os.getpid(), "task": prepared["prepared_task"],
    "context": prepared["state"].to_dict(), "ready": prepared["ready_for_run"], "run": run,
    "turn_question": prepared["turn_question"]}))
'''
        process = subprocess.run([sys.executable, '-c', script, self.schema, self.cid],
            capture_output=True, text=True, encoding='utf-8', timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(process.returncode, 0, process.stderr)
        loaded = json.loads(process.stdout)
        self.assertNotEqual(loaded['pid'], os.getpid())
        self.assertTrue(loaded['ready'])
        self.assertEqual(loaded['task'], prepared['task'])
        self.assertEqual(loaded['turn_question'], prepared['turn_question'])
        self.assertEqual(loaded['context'], json.loads(json.dumps(prepared['state'].to_dict())))
        self.assertEqual(self.store.start_prepared_run('U1', self.cid, 1, 'build-v1'), loaded['run'])

    def test_context_tuple_snapshot_replays_after_jsonb_roundtrip(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        state = self.checkpoint('M1', 0, {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]})['state']
        run = self.store.start_run('U1', self.cid, 1, {'context': state.to_dict()}, 'build-v1')
        self.assertEqual(self.store.start_run('U1', self.cid, 1, {'context': state.to_dict()}, 'build-v1'), run)

    def test_atomic_entry_does_not_adopt_an_unpaired_context_checkpoint(self):
        self.store.append_message('U1', self.cid, 'M1', '排查连接')
        self.checkpoint('M1', 0, {'updates': []})
        with self.assertRaisesRegex(ValueError, 'context_already_committed_without_task'):
            self.prepare('M1', 0, {'updates': []},
                {'operation': 'create', 'goal': '排查连接', 'source_quote': '排查连接'})
        self.assertEqual(self.store.read_history('U1', self.cid)['tasks'], [])
        self.assertFalse(self.store.read_preparation('U1', self.cid)['ready_for_run'])

    def test_context_survives_failed_answer_and_followup_resolves_only_pending_field(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在容器。数据库位置还不清楚。')
        initial = self.checkpoint('M1', 0, {'updates': [
            add_fact('M1', APP, 'container', '应用在容器'),
            {'operation': 'mark_pending', 'target': asdict(DB), 'previous_fact_id': None,
             'reason': 'uncertain_value', 'source': {'message_id': 'M1', 'quote': '数据库位置还不清楚'}},
        ]})['state']
        self.store.append_message('U1', self.cid, 'M2', '更正：应用在宿主机。')
        proposal = {'updates': [{'operation': 'replace', 'target': asdict(APP), 'value': 'host',
            'previous_fact_id': initial.current_facts[0].fact_id,
            'source': {'message_id': 'M2', 'quote': '应用在宿主机'}}]}
        updated = self.checkpoint('M2', 1, proposal)['state']
        run = self.store.start_run('U1', self.cid, 2, {'context': updated.to_dict()}, 'build-v1')
        self.store.finish_run('U1', self.cid, run, {'error': 'evaluation_timeout'}, status='failed')
        other = PostgresConversationStore(self.connect, schema=self.schema)
        restored = other.read_preparation('U1', self.cid)
        self.assertEqual(restored['state'], updated)
        self.assertIsNone(restored['next_message'])
        self.assertIsNone(other.read_current('U1', self.cid)['result'])
        other.append_message('U1', self.cid, 'M3', '确认：数据库在Docker容器。')
        preparation = other.read_preparation('U1', self.cid)
        self.assertEqual(preparation['state'].input_version, 2)
        self.assertEqual(preparation['next_message'].message_id, 'M3')
        result = self.checkpoint('M3', 2, {'updates': [{'operation': 'resolve', 'target': asdict(DB),
            'pending_id': updated.pending_facts[0].pending_id, 'value': 'container',
            'source': {'message_id': 'M3', 'quote': '数据库在Docker容器'}}]}, other)['state']
        self.assertEqual({f.target: f.value for f in result.current_facts}, {APP: 'host', DB: 'container'})
        self.assertEqual(result.pending_facts, ())
        self.assertEqual(next(f for f in result.current_facts if f.target == APP), updated.current_facts[0])
        history = self.store.read_history('U1', self.cid)
        self.assertEqual([c['input_version'] for c in history['contexts']], [1, 2, 3])
        self.assertEqual(history['runs'][0]['status'], 'failed')
        self.assertEqual(history['contexts'][0]['snapshot']['current_facts'][0]['value'], 'container')

    def test_checkpoint_is_restored_in_fresh_process_even_without_started_run(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        state = self.checkpoint('M1', 0, {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]})['state']
        script = '''import json, os, sys, psycopg
from app.conversation_store import PostgresConversationStore
def connect():
    return psycopg.connect(host="127.0.0.1", port=int(os.environ["RAG_CONVERSATION_TEST_PORT"]),
        dbname="postgres", user="rag_conversation_probe", connect_timeout=5)
store = PostgresConversationStore(connect, schema=sys.argv[1])
prepared = store.read_preparation("U1", sys.argv[2])
print(json.dumps({"pid": os.getpid(), "state": prepared["state"].to_dict(),
    "has_pending_message": prepared["next_message"] is not None,
    "runs": len(store.read_history("U1", sys.argv[2])["runs"])}))
'''
        process = subprocess.run([sys.executable, '-c', script, self.schema, self.cid],
            capture_output=True, text=True, encoding='utf-8', timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(process.returncode, 0, process.stderr)
        loaded = json.loads(process.stdout)
        self.assertNotEqual(loaded['pid'], os.getpid())
        self.assertEqual(loaded['state'], json.loads(json.dumps(state.to_dict())))
        self.assertFalse(loaded['has_pending_message'])
        self.assertEqual(loaded['runs'], 0)

    def test_failed_context_cannot_be_skipped_when_later_messages_arrive(self):
        self.store.append_message('U1', self.cid, 'M1', '请排查连接问题')
        initial = self.checkpoint('M1', 0, {'updates': []})['state']
        self.store.append_message('U1', self.cid, 'M2', '应用在宿主机')
        self.store.append_message('U1', self.cid, 'M3', '数据库在容器')
        valid = add_fact('M2', APP, 'host', '应用在宿主机')
        invalid = add_fact('M2', DB, 'host', '消息中不存在的依据')
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
            self.checkpoint('M2', 1, {'updates': [valid, invalid]})
        preparation = self.store.read_preparation('U1', self.cid)
        self.assertEqual(preparation['state'], initial)
        self.assertEqual(preparation['next_message'].message_id, 'M2')
        with self.assertRaisesRegex(ValueError, 'context_version_conflict'):
            self.checkpoint('M3', 2, {'updates': []})
        second = self.checkpoint('M2', 1, {'updates': [valid]})
        self.assertFalse(second['is_latest_input'])
        with self.assertRaisesRegex(ValueError, 'conversation_changed'):
            self.store.start_run('U1', self.cid, 2, {'context': second['state'].to_dict()}, 'build-v1')
        third = self.checkpoint('M3', 2, {'updates': [add_fact('M3', DB, 'container', '数据库在容器')]})
        self.assertTrue(third['is_latest_input'])
        self.assertEqual(third['state'].input_version, 3)

    def test_context_commit_failure_preserves_raw_message_but_not_uncommitted_state(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        @contextmanager
        def rollback_connect():
            with self.connect() as conn:
                yield conn
                raise RuntimeError('injected_before_commit')
        failing = PostgresConversationStore(rollback_connect, schema=self.schema)
        with self.assertRaisesRegex(RuntimeError, 'injected_before_commit'):
            self.checkpoint('M1', 0, {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]}, failing)
        preparation = self.store.read_preparation('U1', self.cid)
        self.assertEqual(preparation['state'].input_version, 0)
        self.assertEqual(preparation['next_message'].message_id, 'M1')
        self.assertEqual(len(self.store.read_history('U1', self.cid)['messages']), 1)

    def test_context_replay_preserves_ids_and_cannot_replace_newer_checkpoint(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        proposal = {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]}
        first = self.checkpoint('M1', 0, proposal)['state']
        self.store.append_message('U1', self.cid, 'M2', '继续')
        second = self.checkpoint('M2', 1, {'updates': []})['state']
        replay = self.checkpoint('M1', 0, proposal)
        self.assertTrue(replay['replayed'])
        self.assertFalse(replay['is_latest_input'])
        self.assertEqual(replay['state'], first)
        self.assertEqual(self.store.read_preparation('U1', self.cid)['state'], second)
        with self.assertRaisesRegex(ValueError, 'context_update_conflict'):
            self.checkpoint('M1', 0, {'updates': []})
        for call in (
            lambda: self.store.read_preparation('U2', self.cid),
            lambda: self.store.commit_context_update('U2', self.cid, 'M1', 0, proposal, allowed_targets=TARGETS),
        ):
            with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
                call()

    def test_concurrent_duplicate_preparation_commits_one_canonical_state(self):
        self.store.append_message('U1', self.cid, 'M1', '应用在宿主机')
        proposal = {'updates': [add_fact('M1', APP, 'host', '应用在宿主机')]}
        def save(_):
            store = PostgresConversationStore(self.connect, schema=self.schema)
            return self.checkpoint('M1', 0, proposal, store)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, range(2)))
        self.assertEqual(sorted(r['replayed'] for r in results), [False, True])
        self.assertEqual(results[0]['state'], results[1]['state'])
        self.assertEqual(len(self.store.read_history('U1', self.cid)['contexts']), 1)

    def test_late_result_saved_without_overwriting_new_result(self):
        old = self.start()
        new = self.start('M2', 'correction: app on host')
        self.store.finish_run('U1', self.cid, new, {'answer': 'host advice'})
        late = self.store.finish_run('U1', self.cid, old, {'answer': 'container advice'})
        self.assertEqual(late['publication_at_finish'], 'superseded')
        self.assertFalse(late['is_current'])
        self.assertEqual(self.store.read_current('U1', self.cid)['result'], {'answer': 'host advice'})
        history = self.store.read_history('U1', self.cid)
        self.assertEqual([r['content'] for r in history['messages']], ['app in container', 'correction: app on host'])
        self.assertEqual(history['runs'][0]['input_snapshot']['user_statement'], 'app in container')
        self.assertEqual(history['runs'][0]['result'], {'answer': 'container advice'})

    def test_new_input_invalidates_previously_current_result_and_replay_does_not_restore_it(self):
        run = self.start()
        first = self.store.finish_run('U1', self.cid, run, {'answer': 'old'})
        self.assertTrue(first['is_current'])
        self.store.append_message('U1', self.cid, 'M2', 'correction')
        replay = self.store.finish_run('U1', self.cid, run, {'answer': 'old'})
        self.assertTrue(replay['replayed'])
        self.assertFalse(replay['is_current'])
        self.assertEqual(replay['publication_at_finish'], 'current_at_commit')
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])

    def test_failed_new_round_keeps_message_history_but_no_current_advice(self):
        run = self.start()
        report = self.store.finish_run('U1', self.cid, run, {'error': 'evaluation_timeout'}, status='failed')
        self.assertFalse(report['is_current'])
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])
        self.assertEqual(self.store.read_history('U1', self.cid)['runs'][0]['status'], 'failed')
        self.assertEqual(len(self.store.read_history('U1', self.cid)['messages']), 1)

    def test_message_replay_and_conflicts_do_not_advance_version_or_change_snapshot(self):
        snapshot = {'source_message_id': 'M1', 'user_statement': 'app in container'}
        run = self.start(snapshot=snapshot)
        replay = self.store.append_message('U1', self.cid, 'M1', 'app in container')
        self.assertEqual(replay, {'input_version': 1, 'current_version': 1, 'replayed': True})
        self.assertEqual(self.store.start_run('U1', self.cid, 1, snapshot, 'build-v1'), run)
        with self.assertRaisesRegex(ValueError, 'message_id_conflict'):
            self.store.append_message('U1', self.cid, 'M1', 'changed')
        snapshot['user_statement'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'run_input_conflict'):
            self.store.start_run('U1', self.cid, 1, snapshot, 'build-v1')
        self.assertEqual(self.store.read_history('U1', self.cid)['runs'][0]['input_snapshot']['user_statement'], 'app in container')
        self.store.finish_run('U1', self.cid, run, {'answer': 'first'})
        with self.assertRaisesRegex(ValueError, 'run_result_conflict'):
            self.store.finish_run('U1', self.cid, run, {'answer': 'replacement'})

    def test_wrong_owner_and_stale_preparation_are_rejected(self):
        run = self.start()
        for call in (
            lambda: self.store.append_message('U2', self.cid, 'Mx', 'other'),
            lambda: self.store.start_run('U2', self.cid, 1, {}, 'build-v1'),
            lambda: self.store.finish_run('U2', self.cid, run, {}),
            lambda: self.store.read_current('U2', self.cid),
            lambda: self.store.read_history('U2', self.cid),
        ):
            with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
                call()
        self.store.append_message('U1', self.cid, 'M2', 'new input')
        with self.assertRaisesRegex(ValueError, 'conversation_changed'):
            self.store.start_run('U1', self.cid, 1, {}, 'build-v1')

    def test_independent_instances_concurrently_accept_distinct_messages(self):
        def send(i):
            store = PostgresConversationStore(self.connect, schema=self.schema)
            return store.append_message('U1', self.cid, f'M{i}', f'message {i}')['input_version']
        with ThreadPoolExecutor(max_workers=4) as pool:
            versions = list(pool.map(send, range(8)))
        self.assertEqual(sorted(versions), list(range(1, 9)))
        self.assertEqual(self.store.read_current('U1', self.cid)['input_version'], 8)
        self.assertEqual(len(self.store.read_history('U1', self.cid)['messages']), 8)

    def assert_lock_wait(self, app_name):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            with self.connect() as conn:
                row = conn.execute('SELECT count(*) FROM pg_stat_activity WHERE application_name=%s AND wait_event_type=%s',
                                   (app_name, 'Lock')).fetchone()
            if row[0]:
                return
            time.sleep(0.02)
        self.fail('did not observe PostgreSQL row-lock wait')

    def test_correction_committing_during_finish_blocks_old_publication(self):
        run = self.start()
        ready, release = Event(), Event()
        @contextmanager
        def delayed_connect():
            with self.connect() as conn:
                yield conn
                ready.set()
                if not release.wait(8):
                    raise TimeoutError('test_release_timeout')
        writer = PostgresConversationStore(delayed_connect, schema=self.schema)
        name = 'finish-' + uuid4().hex
        finisher = PostgresConversationStore(lambda: self.connect(application_name=name), schema=self.schema)
        with ThreadPoolExecutor(max_workers=2) as pool:
            update = pool.submit(writer.append_message, 'U1', self.cid, 'M2', 'correction')
            try:
                self.assertTrue(ready.wait(4))
                finish = pool.submit(finisher.finish_run, 'U1', self.cid, run, {'answer': 'old'})
                self.assert_lock_wait(name)
            finally:
                release.set()
            update.result(timeout=5)
            self.assertFalse(finish.result(timeout=5)['is_current'])
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])

    def test_finish_commits_first_then_waiting_correction_invalidates_it(self):
        run = self.start()
        ready, release = Event(), Event()
        @contextmanager
        def delayed_connect():
            with self.connect() as conn:
                yield conn
                ready.set()
                if not release.wait(8):
                    raise TimeoutError('test_release_timeout')
        finisher = PostgresConversationStore(delayed_connect, schema=self.schema)
        name = 'append-' + uuid4().hex
        writer = PostgresConversationStore(lambda: self.connect(application_name=name), schema=self.schema)
        with ThreadPoolExecutor(max_workers=2) as pool:
            finish = pool.submit(finisher.finish_run, 'U1', self.cid, run, {'answer': 'old'})
            try:
                self.assertTrue(ready.wait(4))
                update = pool.submit(writer.append_message, 'U1', self.cid, 'M2', 'correction')
                self.assert_lock_wait(name)
            finally:
                release.set()
            self.assertTrue(finish.result(timeout=5)['is_current'])
            update.result(timeout=5)
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])

