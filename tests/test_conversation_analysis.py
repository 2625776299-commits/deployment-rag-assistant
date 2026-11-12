"""Opt-in native PostgreSQL/HTTP scope preparation; scripted model outputs."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql

from app.context_proposal import RegisteredTarget
from app.conversation_api import bearer_authenticator
from app.conversation_rag import run_prepared_snapshot
from app.conversation_service import ConversationExecution, ConversationService, deployment_registry
from app.conversation_store import PostgresConversationStore
from app.conversation_context import FactTarget
from app.main import create_app


FIRST = '当前应用在宿主机，数据库实际位置不确定。服务名db，容器端口5432，连接地址怎么填？'
HYPOTHESIS = '先假设应用和数据库都在同一个Compose网络，应用已在容器里；这只是方案，还没实际迁移。'
FOLLOWUP = '那连接失败时先检查什么？'
CURRENT = '回到实际部署，现在连接地址怎么填？'
CONFIRM = '我已按刚才唯一的迁移方案完成部署，连接地址怎么填？'
AMBIGUOUS = '实际容器端口改为6432，但这个连接地址该怎么算？'
RULE = {'chunk_id': 'SCOPE:1', 'content': '同一Compose网络使用服务名和容器端口连接；失败时先核对网络与连接配置。',
        'metadata': {'document_id': 'scope-fixture', 'section_id': 'SCOPE',
                     'source_url': 'https://example.invalid/scope-fixture'}}


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'), 'requires isolated PostgreSQL probe')
class ConversationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.schema = 'rag_conversation_analysis_' + uuid4().hex
        self.connections, self.calls, self.payloads = [], [], []
        self.store = PostgresConversationStore(self.connect, schema=self.schema)
        self.store.initialize()
        self.service = self.make_service()
        self.client = TestClient(create_app(conversation_service=self.service,
            authenticate=bearer_authenticator({'analysis-u1': 'U1'})))
        self.headers = {'Authorization': 'Bearer analysis-u1'}
        self.cid = self.client.post('/conversations', headers=self.headers).json()['conversation_id']
        self.addCleanup(self.cleanup)

    def connect(self):
        c = psycopg.connect(host='127.0.0.1', port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres', user='rag_conversation_probe', connect_timeout=5)
        self.connections.append(c)
        return c

    def cleanup(self):
        self.client.close()
        with self.connect() as c:
            c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(c.closed for c in self.connections))

    def task_model(self, messages):
        self.assertTrue(all(c.closed for c in self.connections))
        p = json.loads(messages[1]['content'])
        user = p['current_user_message']
        self.calls.append('task')
        task = next((t for t in p['existing_tasks'] if t['task_id'] == p['current_task_id']), None)
        selection = ({'operation': 'create', 'goal': '独立环境' if task else '连接排查', 'source_quote': user['content']}
                     if task is None or '另一个独立任务' in user['content'] else
                     {'operation': 'select', 'task_id': task['task_id'], 'source_quote': user['content']})
        return json.dumps({'selection': selection, 'turn_request': {'mode': 'question',
            'question': user['content'], 'source': {'message_id': user['message_id'], 'quote': user['content']}}})

    def context_model(self, messages):
        self.assertTrue(all(c.closed for c in self.connections))
        p = json.loads(messages[1]['content'])
        self.calls.append('context')
        self.payloads.append(p)
        user = p['current_user_message']
        mid, text = user['message_id'], user['content']
        source = {'message_id': mid, 'quote': text}
        def target(entity, field):
            return next(r['target'] for r in p['registered_targets']
                        if r['target']['entity_id'] == entity and r['target']['field'] == field)
        def update(entity, field, value, quote):
            t = target(entity, field)
            previous = next((f for f in p['current_user_stated_facts'] if f['target'] == t), None)
            pending = next((f for f in p['pending_clarifications'] if f['target'] == t), None)
            if pending:
                return {'operation': 'resolve', 'target': t, 'pending_id': pending['pending_id'], 'value': value,
                        'source': {'message_id': mid, 'quote': quote}}
            return {'operation': 'replace' if previous else 'add', 'target': t,
                    'previous_fact_id': previous['fact_id'] if previous else None, 'value': value,
                    'source': {'message_id': mid, 'quote': quote}}
        request = {'mode': 'current', 'source': source, 'confirmation_from_input_version': None}
        updates = []
        if text == FIRST:
            updates = [update('application', 'runtime_location', 'host', '应用在宿主机'),
                       update('database', 'service_name', 'db', '服务名db'),
                       update('database', 'container_port', '5432', '容器端口5432'),
                       {'operation': 'mark_pending', 'target': target('database', 'runtime_location'),
                        'previous_fact_id': None, 'reason': 'uncertain_value',
                        'source': {'message_id': mid, 'quote': '数据库实际位置不确定'}}]
        elif text == HYPOTHESIS:
            request = {'mode': 'hypothetical', 'source': source, 'assumptions': [
                {'target': target(entity, field), 'value': value, 'source': source}
                for entity, field, value in (('application','runtime_location','container'),
                    ('database','runtime_location','compose'), ('application','network_relation','same_compose_network'))]}
        elif text in (FOLLOWUP, '就是刚才的假设。'):
            request = {'mode': 'inherit', 'source': source, 'from_input_version': 2, 'assumption_updates': []}
        elif text == AMBIGUOUS:
            updates = [update('database', 'container_port', '6432', '实际容器端口改为6432')]
            request = {'mode': 'clarify', 'source': source, 'question': '你问实际部署还是刚才的假设？'}
        elif text == CONFIRM:
            request['confirmation_from_input_version'] = 2
            updates = [update(e, f, v, text) for e, f, v in (
                ('application','runtime_location','container'), ('database','runtime_location','compose'),
                ('application','network_relation','same_compose_network'))]
        return json.dumps({'updates': updates, 'analysis_request': request})

    def evaluate(self, messages):
        self.calls.append('evaluate')
        p = json.loads(messages[1]['content'])
        question, context = p['question'], p['conversation_context']
        blockers = [r['pending_id'] for r in context['pending_clarifications'] if not r.get('assumed_for_this_turn')]
        return json.dumps({'requirements': [{'requirement_id': 'R1','requirement': question,'question_quote': question}],
            'assessments': [{'requirement_id':'R1','coverage':'partial' if blockers else 'covered',
                'evidence':[{'chunk_id':'SCOPE:1','quote':RULE['content']}], 'gap':'实际位置未确认' if blockers else None}],
            'action':'clarify' if blockers else 'answer','selected_chunk_ids':['SCOPE:1'],'missing_knowledge':[],
            'missing_user_information':['数据库实际在哪里？'] if blockers else [],'search_query':None,
            'reason':'Scripted contract output; not a quality claim.','known_findings':[],
            'pending_impacts':[{'pending_id':r['pending_id'],'blocking_requirement_ids':['R1'] if r['pending_id'] in blockers else [],
                                'reason':'依据本轮分析范围判断'} for r in context['pending_clarifications']]})

    def make_service(self):
        def bind():
            self.calls.append('bind')
            def execute(snapshot):
                return run_prepared_snapshot(snapshot, retrieve=lambda _: [RULE], evaluate_raw=self.evaluate,
                    generate_raw=lambda _: json.dumps({'status':'answer','answer':'在说明的条件下使用服务名及容器端口。[SCOPE:1]',
                                                       'citation_ids':['SCOPE:1'],'missing_information':[]}))
            return ConversationExecution('scope-fixture', execute)
        return ConversationService(self.store, generate_task=self.task_model,
            generate_context=self.context_model, bind_execution=bind)

    def send(self, mid, text):
        return self.client.post(f'/conversations/{self.cid}/messages', headers=self.headers,
                                json={'message_id':mid,'content':text})

    def start_hypothesis(self):
        self.assertEqual(self.send('M1', FIRST).json()['status'], 'needs_user')
        self.assertEqual(self.send('M2', HYPOTHESIS).json()['status'], 'answer')

    def test_full_inheritance_current_return_and_confirmation(self):
        self.start_hypothesis()
        self.assertEqual(self.send('M3', FOLLOWUP).json()['status'], 'answer')
        history = self.store.read_history('U1', self.cid)
        p1, p2, p3 = history['preparations']
        self.assertEqual(p2['analysis_scope']['assumptions'], p3['analysis_scope']['assumptions'])
        self.assertEqual(p3['analysis_scope']['inherited_from_input_version'], 2)
        self.assertNotEqual(p3['turn_question'], p2['turn_question'])
        self.assertEqual(len({p['task_id'] for p in history['preparations']}), 1)
        state = self.store.read_preparation('U1', self.cid)['state']
        self.assertEqual(next(f.value for f in state.current_facts if f.target.entity_id=='application'), 'host')
        self.assertEqual(len(state.pending_facts), 1)
        self.assertEqual(self.send('M4', CURRENT).json()['status'], 'needs_user')
        self.assertEqual(self.send('M5', CONFIRM).json()['status'], 'answer')
        final = self.store.read_preparation('U1', self.cid)
        self.assertEqual(final['analysis_scope']['mode'], 'current')
        self.assertEqual(final['state'].pending_facts, ())
        changed = [c for c in final['state'].changes if c.source.message_id == 'M5']
        self.assertEqual(len(changed), 3)
        self.assertTrue(all(c.source.quote == CONFIRM for c in changed))
        frozen = self.store.read_history('U1', self.cid)['runs'][-1]['input_snapshot']['turn']
        self.assertEqual(frozen['analysis_request']['confirmation_from_input_version'], 2)
        calls = list(self.calls)
        self.assertTrue(self.send('M5', CONFIRM).json()['replayed'])
        self.assertEqual(calls, self.calls)

    def test_sql_rejects_reanchored_hypothesis_and_preserves_original_after_explicit_inheritance(self):
        self.start_hypothesis()
        before = self.store.read_history('U1', self.cid)
        view = self.store.read_preparation('U1', self.cid)
        task = view['task_catalog'][0]
        registry = tuple(RegisteredTarget(FactTarget(**r['target']), r['description'], r.get('value_contract'))
                         for r in task['registered_targets'])
        text = '只分析刚才的假设场景。'
        self.store.append_message('U1', self.cid, 'M3', text)
        source = {'message_id': 'M3', 'quote': text}
        original = deepcopy(before['preparations'][1]['analysis_scope'])
        repeated = {'mode': 'hypothetical', 'source': source, 'assumptions': [
            {**a, 'source': source} for a in original['assumptions']]}
        def commit(request):
            return self.store.commit_preparation('U1', self.cid, 'M3', 2, {'updates': []},
                {'operation': 'select', 'task_id': task['task_id'], 'source_quote': text}, registry=registry,
                allowed_targets=frozenset(r.target for r in registry),
                turn_request={'mode': 'question', 'question': text, 'source': source}, analysis_request=request)
        with self.assertRaisesRegex(ValueError, 'requires_inheritance'):
            commit(repeated)
        self.assertEqual(self.store.read_history('U1', self.cid)['preparations'], before['preparations'])
        self.assertEqual(self.store.read_preparation('U1', self.cid)['state'], view['state'])
        commit({'mode': 'inherit', 'source': source, 'from_input_version': 2,
                'assumption_updates': [repeated['assumptions'][0]]})
        after = self.store.read_history('U1', self.cid)
        scope = after['preparations'][-1]['analysis_scope']
        self.assertEqual(scope['assumptions'], original['assumptions'])
        self.assertEqual(scope['inherited_from_input_version'], 2)
        self.assertTrue(all(a['source']['message_id'] == 'M2' for a in scope['assumptions']))
        self.assertEqual(after['runs'], before['runs'])
        self.assertEqual(after['preparations'][:2], before['preparations'])

    def test_scope_clarification_keeps_real_updates_and_resumes_the_same_task(self):
        self.start_hypothesis()
        before = self.calls.count('bind')
        response = self.send('M3', AMBIGUOUS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'needs_user')
        self.assertEqual(self.calls.count('bind'), before)
        view = self.store.read_preparation('U1', self.cid)
        self.assertEqual(view['prepared_status'], 'needs_analysis_scope')
        self.assertFalse(view['ready_for_run'])
        self.assertEqual(next(f.value for f in view['state'].current_facts if f.target.field=='container_port'), '6432')
        with self.assertRaisesRegex(ValueError, 'analysis_scope_not_ready'):
            self.store.start_prepared_run('U1', self.cid, 3, 'scope-fixture')
        replay_calls = list(self.calls)
        self.assertTrue(self.send('M3', AMBIGUOUS).json()['replayed'])
        self.assertEqual(self.calls, replay_calls)
        dialogue = self.store.read_dialogue('U1', self.cid, 3)
        self.assertTrue(any(d.role=='assistant' and d.content==response.json()['message'] for d in dialogue))
        self.assertEqual(self.send('M4', '就是刚才的假设。').json()['status'], 'answer')
        history = self.client.get(f'/conversations/{self.cid}/messages', headers=self.headers).json()
        self.assertEqual(history['turns'][2]['status'], 'needs_user')

    def test_foreign_task_reference_is_rechecked_in_the_database_transaction(self):
        self.start_hypothesis()
        self.assertEqual(self.send('M3', '另一个独立任务，只解释网络的一般机制。').status_code, 200)
        history = self.store.read_history('U1', self.cid)
        task_id = history['preparations'][-1]['task_id']
        task = next(t for t in self.store.read_preparation('U1', self.cid)['task_catalog'] if t['task_id']==task_id)
        registry = tuple(RegisteredTarget(FactTarget(**r['target']),
                                         r['description'], r.get('value_contract')) for r in task['registered_targets'])
        text='承接这个任务的假设。'
        self.store.append_message('U1', self.cid, 'M4', text)
        with self.assertRaisesRegex(ValueError, 'reference_not_in_selected_task'):
            self.store.commit_preparation('U1',self.cid,'M4',3,{'updates':[]},
                {'operation':'select','task_id':task_id,'source_quote':text},
                allowed_targets=frozenset(r.target for r in registry),registry=registry,
                turn_request={'mode':'question','question':text,'source':{'message_id':'M4','quote':text}},
                analysis_request={'mode':'inherit','from_input_version':2,'assumption_updates':[],
                                  'source':{'message_id':'M4','quote':text}})
        self.assertEqual(len(self.store.read_history('U1',self.cid)['preparations']),3)
        self.assertEqual(self.store.read_preparation('U1',self.cid)['state'].input_version,3)

    def test_fabricated_source_rolls_back_task_creation_and_context(self):
        registry = deployment_registry(self.cid,'M1')
        text='先假设应用在容器。'
        self.store.append_message('U1',self.cid,'M1',text)
        with self.assertRaisesRegex(ValueError,'analysis_source_not_authorized'):
            self.store.commit_preparation('U1',self.cid,'M1',0,{'updates':[]},
                {'operation':'create','goal':'条件问题','source_quote':text},registry=registry,
                allowed_targets=frozenset(r.target for r in registry),
                turn_request={'mode':'question','question':text,'source':{'message_id':'M1','quote':text}},
                analysis_request={'mode':'hypothetical','source':{'message_id':'M1','quote':text},
                    'assumptions':[{'target':asdict(registry[0].target),'value':'container',
                                    'source':{'message_id':'M1','quote':'不存在的引文'}}]})
        history=self.store.read_history('U1',self.cid)
        self.assertEqual(len(history['messages']),1)
        for key in ('tasks','contexts','preparations'):
            self.assertEqual(history[key],[])

    def test_replay_cannot_change_the_saved_analysis_request(self):
        self.start_hypothesis()
        history=self.store.read_history('U1',self.cid)
        prep=history['preparations'][1]
        registry=tuple(RegisteredTarget(FactTarget(**r['target']),
            r['description'],r.get('value_contract')) for r in self.store.read_preparation('U1',self.cid)['task_catalog'][0]['registered_targets'])
        kwargs={'registry':registry,'allowed_targets':frozenset(r.target for r in registry),
                'turn_request':prep['turn_request'],'analysis_request':deepcopy(prep['analysis_request'])}
        self.assertTrue(self.store.commit_preparation('U1',self.cid,'M2',1,history['contexts'][1]['proposal'],
                                                    prep['selection'],**kwargs)['replayed'])
        kwargs['analysis_request']={'mode':'current','source':{'message_id':'M2','quote':HYPOTHESIS},
                                    'confirmation_from_input_version':None}
        with self.assertRaisesRegex(ValueError,'analysis_scope_conflict'):
            self.store.commit_preparation('U1',self.cid,'M2',1,history['contexts'][1]['proposal'],prep['selection'],**kwargs)

    def test_new_process_reads_identical_saved_and_frozen_conditions(self):
        self.start_hypothesis()
        self.assertEqual(self.send('M3',FOLLOWUP).json()['status'],'answer')
        script='''import json,sys,psycopg
from app.conversation_store import PostgresConversationStore
def connect():
 return psycopg.connect(host="127.0.0.1",port=int(sys.argv[1]),dbname="postgres",user="rag_conversation_probe")
s=PostgresConversationStore(connect,schema=sys.argv[2])
h=s.read_history("U1",sys.argv[3])
print(json.dumps({"saved":h["preparations"][-1]["analysis_scope"],"frozen":h["runs"][-1]["input_snapshot"]["turn"]["analysis_scope"]}))
'''
        result=subprocess.run([sys.executable,'-c',script,os.environ['RAG_CONVERSATION_TEST_PORT'],self.schema,self.cid],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15,check=True)
        restored=json.loads(result.stdout)
        self.assertEqual(restored['saved'],restored['frozen'])
        self.assertEqual(restored['saved']['inherited_from_input_version'],2)
        self.assertTrue(all(a['source']['message_id']=='M2' for a in restored['saved']['assumptions']))

    def test_omitted_scope_fails_instead_of_silently_losing_conditions(self):
        self.service.generate_context=lambda _:json.dumps({'updates':[]})
        response=self.send('M1',FIRST)
        self.assertEqual(response.status_code,500)
        self.assertEqual(self.store.read_history('U1',self.cid)['preparations'],[])
        self.assertEqual(len(self.store.read_history('U1',self.cid)['messages']),1)

    def test_client_cannot_supply_analysis_scope(self):
        response=self.client.post(f'/conversations/{self.cid}/messages',headers=self.headers,
            json={'message_id':'M1','content':FIRST,'analysis_scope':{'mode':'current'}})
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.store.read_history('U1',self.cid)['messages'],[])

    def test_inherit_current_mode_keeps_new_actual_fields_instead_of_restoring_old_values(self):
        self.assertEqual(self.send('M1', FIRST).json()['status'], 'needs_user')
        view = self.store.read_preparation('U1', self.cid)
        row = view['task_catalog'][0]
        registry = tuple(RegisteredTarget(FactTarget(**r['target']), r['description'], r.get('value_contract')) for r in row['registered_targets'])
        port_fact = next(f for f in view['state'].current_facts if f.target.field=='container_port')
        text = '实际容器端口改为6432，继续解释当前部署的端口选择。'
        self.store.append_message('U1', self.cid, 'M2', text)
        result = self.store.commit_preparation('U1', self.cid, 'M2', 1,
            {'updates':[{'operation':'replace','target':asdict(port_fact.target),'previous_fact_id':port_fact.fact_id,
                         'value':'6432','source':{'message_id':'M2','quote':'实际容器端口改为6432'}}]},
            {'operation':'select','task_id':row['task_id'],'source_quote':text},registry=registry,
            allowed_targets=frozenset(r.target for r in registry),
            turn_request={'mode':'question','question':text,'source':{'message_id':'M2','quote':text}},
            analysis_request={'mode':'inherit','from_input_version':1,'assumption_updates':[],
                              'source':{'message_id':'M2','quote':text}})
        self.assertEqual(result['analysis_scope']['mode'], 'current')
        self.assertEqual(result['analysis_request']['from_input_version'], 1)
        run = self.store.start_prepared_run('U1', self.cid, 2, 'scope-fixture')
        frozen = self.store.read_run_input('U1', self.cid, run)['input_snapshot']
        current_port = next(f for f in frozen['context']['current_facts'] if f['target']['field']=='container_port')
        self.assertEqual(current_port['value'], '6432')
        self.assertEqual(current_port['source']['message_id'], 'M2')


if __name__ == '__main__':
    unittest.main()
