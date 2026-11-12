"""Real PostgreSQL leases, late writers and fresh-process recovery; no providers."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql

from app.conversation_attempts import RunAttempts, AttemptHeartbeat
from app.conversation_rag import run_prepared_snapshot
from app.conversation_service import ConversationExecution, ConversationService
from app.conversation_store import PostgresConversationStore


QUESTION = '数据库断连后应用应如何处理？'
PASSAGE = {'chunk_id': 'REC:1', 'content': '应用应检测断连，重新解析服务名并重连。',
    'metadata': {'document_id': 'recovery-fixture', 'section_id': 'REC', 'source_url': 'https://example.invalid/recovery'}}


def recovery_service(store, requested):
    def forbidden(*args):
        raise AssertionError('recovery_must_not_reprepare_or_resolve_active_build')
    def resume(build_id):
        requested.append(build_id)
        def evaluate(messages):
            question = json.loads(messages[1]['content'])['question']
            return json.dumps({'requirements': [{'requirement_id': 'R1', 'requirement': '说明恢复方法', 'question_quote': question}],
                'assessments': [{'requirement_id': 'R1', 'coverage': 'covered', 'gap': None,
                                'evidence': [{'chunk_id': 'REC:1', 'quote': PASSAGE['content']}]}],
                'action': 'answer', 'selected_chunk_ids': ['REC:1'], 'missing_knowledge': [],
                'missing_user_information': [], 'search_query': None, 'reason': 'Scripted recovery check.',
                'known_findings': [], 'pending_impacts': []})
        def execute(snapshot):
            return run_prepared_snapshot(snapshot, retrieve=lambda _: [deepcopy(PASSAGE)], evaluate_raw=evaluate,
                generate_raw=lambda _: json.dumps({'status': 'answer', 'answer': '检测断连、重新解析并重连。[REC:1]',
                    'citation_ids': ['REC:1'], 'missing_information': []}))
        return ConversationExecution(build_id, execute)
    return ConversationService(store, generate_task=forbidden, generate_context=forbidden,
                               bind_execution=forbidden, resume_execution=resume)


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'), 'requires isolated PostgreSQL probe')
class ConversationAttemptTests(unittest.TestCase):
    def setUp(self):
        self.connections = []
        self.schema = 'rag_conversation_attempt_' + uuid4().hex
        self.store = PostgresConversationStore(self.connect, schema=self.schema)
        self.store.initialize()
        self.cid = self.store.create('U1')
        self.store.append_message('U1', self.cid, 'M1', QUESTION)
        self.store.commit_preparation('U1', self.cid, 'M1', 0, {'updates': []},
            {'operation': 'create', 'goal': '解释数据库连接恢复', 'source_quote': QUESTION},
            allowed_targets=frozenset(), registry=(), turn_request={'mode': 'question', 'question': QUESTION,
                'source': {'message_id': 'M1', 'quote': QUESTION}})
        self.run_id = self.store.start_prepared_run('U1', self.cid, 1, 'build-v1')
        self.attempts = RunAttempts(self.store)
        self.addCleanup(self.cleanup)

    def connect(self):
        conn = psycopg.connect(host='127.0.0.1', port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres', user='rag_conversation_probe', connect_timeout=5)
        self.connections.append(conn)
        return conn

    def cleanup(self):
        with self.connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(c.closed for c in self.connections))

    def claim(self):
        return self.attempts.claim('U1', self.cid, self.run_id)

    def expire(self, attempt_id):
        # Fault injection advances the lease to the past; no wall-clock wait or
        # tampering with the saved input/build. The database makes the decision.
        with self.connect() as conn:
            conn.execute(sql.SQL('UPDATE {} SET lease_expires_at=clock_timestamp()-interval \'1 second\' WHERE attempt_id=%s')
                .format(self.store._table('run_attempts')), (attempt_id,))

    def finish(self, attempt_id, text='current'):
        return self.attempts.finish('U1', self.cid, self.run_id, attempt_id,
                                   {'status': 'answer', 'message': text, 'required_information': []})

    def test_takeover_keeps_late_result_in_history_without_publication(self):
        first = self.claim()
        self.expire(first['attempt_id'])
        second = self.claim()
        self.assertEqual(second['attempt_number'], 2)
        self.assertNotEqual(first['attempt_id'], second['attempt_id'])
        stale = self.finish(first['attempt_id'], 'stale')
        self.assertEqual(stale['decision'], 'lease_lost')
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])
        self.assertEqual(self.finish(second['attempt_id'])['decision'], 'accepted')
        self.assertEqual(self.store.read_current('U1', self.cid)['result']['message'], 'current')
        self.assertTrue(self.finish(first['attempt_id'], 'stale')['replayed'])
        with self.assertRaisesRegex(ValueError, 'attempt_result_conflict'):
            self.finish(first['attempt_id'], 'different')
        history = self.store.read_history('U1', self.cid)
        self.assertEqual([a['result']['message'] for a in history['attempts']], ['stale', 'current'])

    def test_only_one_of_concurrent_claimants_can_take_over(self):
        first = self.claim()
        self.expire(first['attempt_id'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.claim(), range(4)))
        self.assertEqual(sum(r['acquired'] for r in results), 1)
        self.assertEqual(len(self.store.read_history('U1', self.cid)['attempts']), 2)
        self.assertFalse(self.attempts.renew('U1', self.cid, self.run_id, first['attempt_id']))

    def test_expired_attempt_cannot_revive_or_publish_without_takeover(self):
        first = self.claim()
        self.assertTrue(self.attempts.renew('U1', self.cid, self.run_id, first['attempt_id']))
        self.assertFalse(self.claim()['acquired'])
        self.expire(first['attempt_id'])
        self.assertFalse(self.attempts.renew('U1', self.cid, self.run_id, first['attempt_id']))
        self.assertEqual(self.finish(first['attempt_id'])['decision'], 'lease_lost')
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])
        self.assertTrue(self.claim()['acquired'])

    def test_heartbeat_retains_ownership_during_slow_work(self):
        first = self.attempts.claim('U1', self.cid, self.run_id, lease_seconds=1.2)
        with AttemptHeartbeat(self.attempts, 'U1', self.cid, self.run_id, first['attempt_id'], 1.2):
            Event().wait(1.6)
            self.assertEqual(self.claim()['reason'], 'lease_active')
            self.assertEqual(self.finish(first['attempt_id'])['decision'], 'accepted')

    def test_version_guard_and_attempt_guard_are_both_required(self):
        first = self.claim()
        with self.assertRaisesRegex(ValueError, 'attempt_token_required'):
            self.store.finish_run('U1', self.cid, self.run_id, {'status': 'answer'})
        self.store.append_message('U1', self.cid, 'M2', '新增环境信息')
        done = self.finish(first['attempt_id'])
        self.assertEqual(done['decision'], 'accepted')
        self.assertEqual(done['publication']['publication_at_finish'], 'superseded')
        self.assertIsNone(self.store.read_current('U1', self.cid)['result'])
        self.assertFalse(self.claim()['acquired'])

    def test_attempt_methods_check_owner_and_terminal_runs_do_not_restart(self):
        first = self.claim()
        for call in (
            lambda: self.attempts.claim('U2', self.cid, self.run_id),
            lambda: self.attempts.renew('U2', self.cid, self.run_id, first['attempt_id']),
            lambda: self.attempts.finish('U2', self.cid, self.run_id, first['attempt_id'], {}),
        ):
            with self.assertRaisesRegex(ValueError, 'conversation_not_found'):
                call()
        self.finish(first['attempt_id'])
        self.assertEqual(self.claim()['reason'], 'run_finished')

    def test_wrong_build_binder_fails_without_changing_snapshot(self):
        before = self.store.read_run_input('U1', self.cid, self.run_id)['input_snapshot']
        requested = []
        service = recovery_service(self.store, requested)
        service.resume_execution = lambda build: ConversationExecution('build-v2', lambda _: self.fail('wrong build executed'))
        record, replayed = service.submit('U1', self.cid, 'M1', QUESTION)
        self.assertTrue(replayed)
        self.assertEqual(record['run']['status'], 'failed')
        self.assertEqual(record['run']['result']['error']['stage'], 'bind')
        stored = self.store.read_run_input('U1', self.cid, self.run_id)
        self.assertEqual((stored['build_id'], stored['input_snapshot']), ('build-v1', before))

    def test_real_catalogue_recovery_keeps_old_build_after_new_publication(self):
        from app.build_catalog import PostgresBuildCatalog
        from app.build_runtime import build_request_binder
        from app.run_build_pinning_demo import fixture_embedding, write_fixture_build
        root = Path(os.environ['RAG_CONVERSATION_TEST_OUTPUT']) / 'recovery_build_pin'
        root.mkdir()
        v1, v2 = (write_fixture_build(root, name) for name in ('build-v1', 'build-v2'))
        schema = 'rag_build_attempt_' + uuid4().hex
        catalog = PostgresBuildCatalog(self.connect, schema=schema)
        catalog.initialize()
        def drop_catalog():
            with self.connect() as conn:
                conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        self.addCleanup(drop_catalog)
        for manifest in (v1, v2):
            catalog.register(manifest)
            catalog.validate_ready(manifest.build_id, root)
        catalog.publish(v1.build_id, expected_active=None)
        catalog.publish(v2.build_id, expected_active=v1.build_id)
        self.assertEqual(catalog.read_active().build_id, 'build-v2')
        binder = build_request_binder(catalog, root, bind_embedding=fixture_embedding,
            evaluate=lambda *a: None, generate=lambda *a: None)
        requested = []
        service = recovery_service(self.store, requested)
        def resume(saved_id):
            pinned = binder(saved_id)
            requested.append(pinned.build.build_id)
            rows = pinned.retrieve(pinned.retriever.chunks[0]['content'])
            self.assertTrue(rows)
            self.assertTrue(all(r['metadata']['build_id'] == 'build-v1' for r in rows))
            return ConversationExecution(pinned.build.build_id,
                lambda _: {'status': 'answer', 'message': 'old build', 'required_information': []})
        service.resume_execution = resume
        record, _ = service.submit('U1', self.cid, 'M1', QUESTION)
        self.assertEqual(record['run']['status'], 'completed')
        self.assertEqual(requested, ['build-v1'])
        self.assertEqual(catalog.read_active().build_id, 'build-v2')

    def test_real_new_process_recovers_same_http_message_and_frozen_run(self):
        project = Path(__file__).resolve().parents[1]
        output = Path(os.environ['RAG_CONVERSATION_TEST_OUTPUT']) / 'new_process_attempt_recovery'
        output.mkdir()
        prefix = '''import json,os,sys
import psycopg
from app.conversation_store import PostgresConversationStore
from app.conversation_attempts import RunAttempts
def connect():
    return psycopg.connect(host="127.0.0.1",port=int(os.environ["RAG_CONVERSATION_TEST_PORT"]),dbname="postgres",user="rag_conversation_probe",connect_timeout=5)
store=PostgresConversationStore(connect,schema=sys.argv[1])
cid,run_id=sys.argv[2:4]
'''
        crash = prefix + '''claim=RunAttempts(store).claim("U1",cid,run_id)
print(json.dumps(claim),flush=True)
os._exit(0)
'''
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        def child(code):
            return subprocess.run([sys.executable, '-c', code, self.schema, self.cid, self.run_id], cwd=project,
                env=env, capture_output=True, text=True, encoding='utf-8', timeout=20, creationflags=flags)
        exited = child(crash)
        self.assertEqual(exited.returncode, 0, exited.stderr)
        first = json.loads(exited.stdout)
        self.expire(first['attempt_id'])
        before = self.store.read_run_input('U1', self.cid, self.run_id)
        recover = prefix + '''sys.path.insert(0,str(__import__("pathlib").Path.cwd()/"tests"))
from test_conversation_attempts import recovery_service,QUESTION
from app.conversation_api import bearer_authenticator
from app.main import create_app
from fastapi.testclient import TestClient
requested=[]
service=recovery_service(store,requested)
with TestClient(create_app(conversation_service=service,authenticate=bearer_authenticator({"fixture-token":"U1"}))) as client:
    response=client.post("/conversations/"+cid+"/messages",headers={"Authorization":"Bearer fixture-token"},json={"message_id":"M1","content":QUESTION})
print(json.dumps({"http_status":response.status_code,"body":response.json(),"requested_build_ids":requested}),flush=True)
'''
        recovered = child(recover)
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        report = json.loads(recovered.stdout)
        self.assertEqual((report['http_status'], report['body']['status']), (200, 'answer'))
        self.assertTrue(report['body']['replayed'])
        self.assertEqual(report['requested_build_ids'], ['build-v1'])
        after = self.store.read_run_input('U1', self.cid, self.run_id)
        self.assertEqual((before['input_snapshot'], before['build_id']), (after['input_snapshot'], after['build_id']))
        history = self.store.read_history('U1', self.cid)
        self.assertEqual((len(history['messages']),len(history['runs']),len(history['attempts'])), (1,1,2))
        report.update(first_process_exit=exited.returncode, lease_expiry='injected in isolated database',
                      original_snapshot_unchanged=True, message_count=1, run_count=1, attempt_count=2,
                      attempts=history['attempts'], real_model_calls=0, real_embedding_calls=0)
        (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
