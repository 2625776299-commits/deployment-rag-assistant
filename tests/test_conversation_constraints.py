"""Isolated PostgreSQL/HTTP evidence-constraint integration; scripted providers."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
import subprocess
import sys
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql

from app.conversation_api import bearer_authenticator
from app.conversation_rag import run_prepared_snapshot
from app.conversation_service import ConversationExecution, ConversationService, deployment_registry
from app.conversation_store import PostgresConversationStore
from app.context_proposal import RegisteredTarget
from app.conversation_context import FactTarget
from app.main import create_app


FIRST = '第一次连接故障是不是密码错误？日志和报错暂时还没拿到。'
UNAVAILABLE = '刚才第一次故障的日志和报错均已清理，其他当时现场证据也拿不到，仍想确定原因。'
GENERAL = '不判断刚才那次根因，先解释同一Compose网络中的服务发现原理。'
SECOND = '同一任务的第二次独立连接故障，日志和报错都还在，先看这次。'
RETURN = '回到第一次故障，继续确定是不是密码错。'
AMBIGUOUS = '那次故障的日志现在也清理了。'
RESOLVE = '说的是第二次那份日志。'
RULE = {'chunk_id':'RULE:1','content':'同一Compose网络中的服务名注册到内部DNS。',
        'metadata':{'document_id':'fixture','section_id':'RULE','source_url':'https://example.invalid/rule'}}


def request(operation, source, *, iid=None, description=None, question=None, candidates=None,
            updates=None, resolves=None):
    return {'incident':{'operation':operation,'incident_id':iid,'description':description,'question':question,
            'candidate_incident_ids':candidates or [],'source':source},
            'updates':updates or [],'resolves_input_version':resolves}


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'), 'requires isolated PostgreSQL probe')
class ConversationConstraintTests(unittest.TestCase):
    def setUp(self):
        self.schema = 'rag_conversation_constraints_' + uuid4().hex
        self.connections, self.calls, self.payloads = [], [], []
        self.store = PostgresConversationStore(self.connect,schema=self.schema)
        self.store.initialize()
        self.service = ConversationService(self.store,generate_task=self.task_model,
            generate_context=self.context_model,bind_execution=self.bind,include_constraints=True)
        self.client = TestClient(create_app(conversation_service=self.service,
            authenticate=bearer_authenticator({'constraint-u1':'U1','constraint-u2':'U2'})))
        self.headers = {'Authorization':'Bearer constraint-u1'}
        self.cid = self.client.post('/conversations',headers=self.headers).json()['conversation_id']
        self.addCleanup(self.cleanup)

    def connect(self):
        conn = psycopg.connect(host='127.0.0.1',port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres',user='rag_conversation_probe',connect_timeout=5)
        self.connections.append(conn)
        return conn

    def cleanup(self):
        self.client.close()
        with self.connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(c.closed for c in self.connections))

    def send(self, mid, content):
        return self.client.post(f'/conversations/{self.cid}/messages',headers=self.headers,
                                json={'message_id':mid,'content':content})

    def task_model(self, messages):
        self.assertTrue(all(c.closed for c in self.connections))
        self.calls.append('task')
        p = json.loads(messages[1]['content']); u = p['current_user_message']
        task = next((t for t in p['existing_tasks'] if t['task_id']==p['current_task_id']),None)
        create = task is None or u['content']=='另一个独立任务的首次故障。'
        selection = ({'operation':'create','goal':'诊断连接故障','source_quote':u['content']} if create else
                     {'operation':'select','task_id':task['task_id'],'source_quote':u['content']})
        turn = ({'mode':'continue','from_input_version':1} if u['content']==UNAVAILABLE else
                {'mode':'question','question':u['content'],'source':{'message_id':u['message_id'],'quote':u['content']}})
        return json.dumps({'selection':selection,'turn_request':turn})

    def context_model(self, messages):
        self.assertTrue(all(c.closed for c in self.connections))
        self.calls.append('context')
        p = json.loads(messages[1]['content']); self.payloads.append(deepcopy(p))
        u = p['current_user_message']; source={'message_id':u['message_id'],'quote':u['content']}
        catalog = sorted(p['incident_catalog'],key=lambda r:r['introduced_in_input_version'])
        text = u['content']
        if text==GENERAL:
            evidence = request('general',source)
        elif text==AMBIGUOUS:
            evidence = request('clarify',source,question='你说第一次还是第二次故障？',
                               candidates=[c['incident_id'] for c in catalog])
        else:
            create = not catalog or text==SECOND
            iid = None if create else catalog[-1 if text==RESOLVE else 0]['incident_id']
            evidence = request('create' if create else 'select',source,iid=iid,
                description=text if create else None,resolves=6 if text==RESOLVE else None)
            status = 'not_obtained' if text==FIRST else 'available' if text==SECOND else 'unavailable'
            kinds = (['incident_logs','incident_error_details','other_incident_evidence'] if text==UNAVAILABLE
                     else ['incident_logs','incident_error_details'] if text in (FIRST,SECOND)
                     else ['incident_logs'] if text==RESOLVE else [])
            original = p['pending_incidents'][0] if text==RESOLVE else None
            for kind in kinds:
                old = next((c for c in p['constraint_state'] if c['incident']['incident_id']==iid
                            and c['evidence_kind']==kind),None)
                evidence['updates'].append({'evidence_kind':kind,'availability':status,
                    'previous_constraint_id':old['constraint_id'] if old else None,
                    'source':{'message_id':original['message_id'],'quote':original['content']} if original else source})
        return json.dumps({'updates':[],'analysis_request':{'mode':'current','source':source,
            'confirmation_from_input_version':None},'evidence_request':evidence})

    def evaluate(self, messages):
        self.calls.append('evaluate')
        p=json.loads(messages[1]['content']);self.payloads.append(deepcopy(p))
        constraints=p['task_constraints']; general=constraints['scope']['mode']=='general'
        unavailable={c['evidence_kind'] for c in constraints['evidence_constraints'] if c['availability']=='unavailable'}
        action='answer' if general else 'insufficient' if len(unavailable)==3 else 'clarify'
        evidence=[{'chunk_id':'RULE:1','quote':RULE['content']}] if general else []
        return json.dumps({'requirements':[{'requirement_id':'R1','requirement':p['question'],'question_quote':p['question']}],
            'assessments':[{'requirement_id':'R1','coverage':'covered' if general else 'missing',
                            'evidence':evidence,'gap':None if general else '缺少本次现场错误证据'}],
            'action':action,'selected_chunk_ids':['RULE:1'] if general else [],
            'missing_knowledge':['无法取得必要历史证据，不能确认根因'] if action=='insufficient' else [],
            'missing_user_information':['请提供本次实际取得的完整报错内容'] if action=='clarify' else [],
            'search_query':None,'reason':'scripted contract test, not semantic validation',
            'known_findings':[],'pending_impacts':[]})

    def generate(self, messages):
        self.calls.append('generate');self.payloads.append(deepcopy(json.loads(messages[1]['content'])))
        return json.dumps({'status':'answer','answer':'同网络使用内部DNS发现服务。[RULE:1]',
                           'citation_ids':['RULE:1'],'missing_information':[]})

    def bind(self):
        self.calls.append('bind')
        return ConversationExecution('constraint-fixture',lambda s:run_prepared_snapshot(s,
            retrieve=lambda _:[RULE],evaluate_raw=self.evaluate,generate_raw=self.generate))

    def seed_two(self):
        self.assertEqual(self.send('M1',FIRST).json()['status'],'needs_user')
        self.assertEqual(self.send('M2',UNAVAILABLE).json()['status'],'insufficient')
        self.assertEqual(self.send('M3',GENERAL).json()['status'],'answer')
        self.assertEqual(self.send('M4',SECOND).json()['status'],'needs_user')

    def test_continued_old_question_receives_new_constraints_and_old_run_is_immutable(self):
        self.send('M1',FIRST)
        old=deepcopy(self.store.read_history('U1',self.cid)['runs'][0])
        response=self.send('M2',UNAVAILABLE)
        self.assertEqual(response.json()['status'],'insufficient')
        history=self.store.read_history('U1',self.cid)
        self.assertEqual(history['runs'][0],old)
        self.assertEqual(history['preparations'][1]['turn_question'],FIRST)
        latest=history['runs'][1]['input_snapshot']['turn']['task_constraints']
        self.assertEqual({c['availability'] for c in latest['evidence_constraints']},{'unavailable'})
        self.assertTrue(all(c['source']['message_id']=='M2' for c in latest['evidence_constraints']))
        seen=next(p for p in self.payloads if p.get('question')==FIRST and
                  p.get('task_constraints',{}).get('input_version')==2)
        self.assertEqual(seen['task_constraints'],latest)
        calls=list(self.calls)
        self.assertTrue(self.send('M2',UNAVAILABLE).json()['replayed'])
        self.assertEqual(self.calls,calls)

    def test_general_scope_does_not_delete_constraints_and_second_incident_does_not_inherit_them(self):
        self.seed_two()
        self.assertEqual(self.send('M5',RETURN).json()['status'],'insufficient')
        history=self.store.read_history('U1',self.cid)
        first=history['preparations'][1]['incident_scope']['incident_id']
        second=history['preparations'][3]['incident_scope']['incident_id']
        self.assertNotEqual(first,second)
        general=history['runs'][2]['input_snapshot']['turn']['task_constraints']
        self.assertEqual(general['evidence_constraints'],[])
        state=self.store.read_preparation('U1',self.cid)['constraint_state']
        self.assertEqual(len(state.current),5)
        self.assertEqual(history['preparations'][4]['incident_scope']['incident_id'],first)
        new=history['runs'][3]['input_snapshot']['turn']['task_constraints']
        self.assertTrue(all(c['incident']['incident_id']==second and c['availability']=='available'
                            for c in new['evidence_constraints']))
        self.assertEqual(self.payloads[-1]['task_constraints']['scope']['incident_id'],first)

    def test_ambiguous_incident_saves_original_without_running_and_resolution_keeps_original_source(self):
        self.seed_two();self.send('M5',RETURN)
        state=self.store.read_preparation('U1',self.cid)['constraint_state']
        calls=self.calls.count('bind');response=self.send('M6',AMBIGUOUS)
        self.assertEqual(response.json()['status'],'needs_user');self.assertEqual(calls,self.calls.count('bind'))
        view=self.store.read_preparation('U1',self.cid)
        self.assertEqual(view['prepared_status'],'needs_incident')
        self.assertEqual(view['constraint_state'].current,state.current)
        with self.assertRaisesRegex(ValueError,'incident_scope_not_ready'):
            self.store.start_prepared_run('U1',self.cid,6,'constraint-fixture')
        self.assertTrue(any(d.content==response.json()['message'] for d in self.store.read_dialogue('U1',self.cid,6)))
        self.assertEqual(self.send('M7',RESOLVE).json()['status'],'needs_user')
        latest=self.store.read_preparation('U1',self.cid)
        self.assertEqual(latest['pending_incidents'],[])
        second=latest['incident_scope']['incident_id']
        log=next(c for c in latest['constraint_state'].current if c.incident.incident_id==second
                 and c.evidence_kind=='incident_logs')
        self.assertEqual(log.source.message_id,'M6');self.assertEqual(log.introduced_in_input_version,7)
        self.assertEqual(log.availability,'unavailable')
        self.assertTrue(log.supersedes_constraint_id)

    def test_invalid_constraint_rolls_back_new_task_incident_and_environment_together(self):
        self.store.append_message('U1',self.cid,'M1',FIRST)
        registry=deployment_registry(self.cid,'M1');source={'message_id':'M1','quote':FIRST}
        evidence=request('create',source,description='首次故障',updates=[{'evidence_kind':'incident_logs',
            'availability':'unavailable','previous_constraint_id':None,
            'source':{'message_id':'M1','quote':'不存在的原文'}}])
        with self.assertRaisesRegex(ValueError,'constraint_quote_not_in_message'):
            self.store.commit_preparation('U1',self.cid,'M1',0,{'updates':[]},
                {'operation':'create','goal':'诊断','source_quote':FIRST},registry=registry,
                allowed_targets=frozenset(r.target for r in registry),
                turn_request={'mode':'question','question':FIRST,'source':source},evidence_request=evidence)
        history=self.store.read_history('U1',self.cid)
        self.assertEqual(len(history['messages']),1)
        for name in ('tasks','contexts','preparations'):self.assertEqual(history[name],[])
        self.assertEqual(self.store.read_preparation('U1',self.cid)['incident_catalog'],[])

    def test_other_task_incident_and_replayed_change_are_rejected_in_store(self):
        self.send('M1',FIRST)
        original=self.store.read_history('U1',self.cid)
        self.send('M2','另一个独立任务的首次故障。')
        view=self.store.read_preparation('U1',self.cid);first,second=view['incident_catalog']
        self.assertNotEqual(first['task_id'],second['task_id'])
        text='查询这个任务的故障。';self.store.append_message('U1',self.cid,'M3',text)
        registry=tuple(RegisteredTarget(FactTarget(**r['target']),r['description'],r.get('value_contract'))
            for r in next(t for t in view['task_catalog'] if t['task_id']==second['task_id'])['registered_targets'])
        with self.assertRaisesRegex(ValueError,'incident_not_in_selected_task'):
            self.store.commit_preparation('U1',self.cid,'M3',2,{'updates':[]},
                {'operation':'select','task_id':second['task_id'],'source_quote':text},registry=registry,
                allowed_targets=frozenset(r.target for r in registry),
                turn_request={'mode':'question','question':text,'source':{'message_id':'M3','quote':text}},
                evidence_request=request('select',{'message_id':'M3','quote':text},iid=first['incident_id']))
        prep=original['preparations'][0]
        with self.assertRaisesRegex(ValueError,'evidence_request_conflict'):
            self.store.commit_preparation('U1',self.cid,'M1',0,original['contexts'][0]['proposal'],prep['selection'],
                allowed_targets=frozenset(),turn_request=prep['turn_request'],analysis_request=prep['analysis_request'],
                evidence_request=request('general',{'message_id':'M1','quote':FIRST}))

    def test_new_process_reads_persisted_sources_and_frozen_constraint_versions(self):
        self.send('M1',FIRST);self.send('M2',UNAVAILABLE)
        code='''import json,os,sys,psycopg
from app.conversation_store import PostgresConversationStore
def connect():
 return psycopg.connect(host='127.0.0.1',port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),dbname='postgres',user='rag_conversation_probe')
s=PostgresConversationStore(connect,schema=sys.argv[1]);v=s.read_preparation('U1',sys.argv[2]);h=s.read_history('U1',sys.argv[2])
print(json.dumps({'state':v['constraint_state'].to_dict(),'old':h['runs'][0]['input_snapshot']['turn']['task_constraints']}))
'''
        r=subprocess.run([sys.executable,'-c',code,self.schema,self.cid],capture_output=True,text=True,
                         check=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result=json.loads(r.stdout)
        self.assertEqual({c['availability'] for c in result['state']['current']},{'unavailable'})
        self.assertEqual({c['availability'] for c in result['old']['evidence_constraints']},{'not_obtained'})

    def test_deferred_constraint_cannot_override_newer_same_value_confirmation(self):
        self.seed_two();self.send('M5',RETURN);self.send('M6',AMBIGUOUS)
        view=self.store.read_preparation('U1',self.cid)
        task_id=view['incident_catalog'][0]['task_id'];second=view['incident_catalog'][1]['incident_id']
        registry=tuple(RegisteredTarget(FactTarget(**r['target']),r['description'],r.get('value_contract'))
            for r in next(t for t in view['task_catalog'] if t['task_id']==task_id)['registered_targets'])
        old=next(c for c in view['constraint_state'].current if c.incident.incident_id==second
                 and c.evidence_kind=='incident_logs')
        def commit(mid,version,text,evidence):
            self.store.append_message('U1',self.cid,mid,text)
            source={'message_id':mid,'quote':text}
            return self.store.commit_preparation('U1',self.cid,mid,version-1,{'updates':[]},
                {'operation':'select','task_id':task_id,'source_quote':text},registry=registry,
                allowed_targets=frozenset(r.target for r in registry),
                turn_request={'mode':'question','question':text,'source':source},
                analysis_request={'mode':'current','source':source,'confirmation_from_input_version':None},
                evidence_request=evidence)
        text='再明确一下：第二次的日志仍然保存着。';source={'message_id':'M7','quote':text}
        commit('M7',7,text,request('select',source,iid=second,updates=[{
            'evidence_kind':'incident_logs','availability':'available','previous_constraint_id':old.constraint_id,'source':source}]))
        self.assertIn(old,self.store.read_preparation('U1',self.cid)['constraint_state'].current)
        text='前面那句话指第二次故障。'
        with self.assertRaisesRegex(ValueError,'deferred_constraint_would_overwrite_newer_record'):
            commit('M8',8,text,request('select',{'message_id':'M8','quote':text},iid=second,resolves=6,
                updates=[{'evidence_kind':'incident_logs','availability':'unavailable',
                          'previous_constraint_id':old.constraint_id,'source':{'message_id':'M6','quote':AMBIGUOUS}}]))
        history=self.store.read_history('U1',self.cid)
        self.assertEqual(len(history['messages']),8);self.assertEqual(len(history['preparations']),7)
        self.assertIn(old,self.store.read_preparation('U1',self.cid)['constraint_state'].current)

    def test_external_identity_or_constraint_fields_are_not_accepted_and_other_owner_is_denied(self):
        self.send('M1',FIRST);calls=list(self.calls)
        bad=self.client.post(f'/conversations/{self.cid}/messages',headers=self.headers,
            json={'message_id':'M2','content':'继续','incident_id':'injected'})
        self.assertEqual(bad.status_code,422);self.assertEqual(calls,self.calls)
        other=self.client.get(f'/conversations/{self.cid}/messages',headers={'Authorization':'Bearer constraint-u2'})
        self.assertEqual(other.status_code,404)

    def test_direct_commit_cannot_store_availability_as_error_and_rolls_back_all_changes(self):
        self.send('M1', FIRST)
        view = self.store.read_preparation('U1', self.cid)
        before = self.store.read_history('U1', self.cid)
        task = view['task_catalog'][0]
        registry = tuple(RegisteredTarget(FactTarget(**r['target']), r['description'], r.get('value_contract'))
                         for r in task['registered_targets'])
        self.assertEqual(next(r.value_contract for r in registry if r.target.field == 'error_text'), 'error_literal')
        text = UNAVAILABLE + '服务名未变化。'
        self.store.append_message('U1', self.cid, 'M2', text)
        source = {'message_id': 'M2', 'quote': text}
        iid = view['incident_catalog'][0]['incident_id']
        old = next(c for c in view['constraint_state'].current if c.evidence_kind == 'incident_error_details')
        evidence = request('select', source, iid=iid, updates=[{
            'evidence_kind': 'incident_error_details', 'availability': 'unavailable',
            'previous_constraint_id': old.constraint_id, 'source': source}])
        for field, value, error in [('error_text', '报错均已清理', 'availability_is_not_error'),
                                    ('service_name', '未变化', 'not_a_service_name')]:
            target = next(r.target for r in registry if r.target.field == field)
            patch = {'updates': [{'operation': 'add', 'target': asdict(target), 'previous_fact_id': None,
                                  'value': value, 'source': source}]}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                self.store.commit_preparation('U1', self.cid, 'M2', 1, patch,
                    {'operation': 'select', 'task_id': task['task_id'], 'source_quote': text}, registry=registry,
                    allowed_targets=frozenset(r.target for r in registry),
                    turn_request={'mode': 'question', 'question': text, 'source': source},
                    analysis_request={'mode': 'current', 'source': source, 'confirmation_from_input_version': None},
                    evidence_request=evidence)
            after = self.store.read_history('U1', self.cid)
            self.assertEqual(after['preparations'], before['preparations'])
            self.assertEqual(after['runs'], before['runs'])
            current = self.store.read_preparation('U1', self.cid)
            self.assertEqual(current['state'], view['state'])
            self.assertEqual(current['constraint_state'], view['constraint_state'])
            self.assertEqual(current['next_message'].content, text)

    def test_http_correction_keeps_evidence_constraint_and_omits_fake_error(self):
        self.send('M1', FIRST)
        model = self.service.generate_context
        proposals = []
        def context(messages):
            result = json.loads(model(messages))
            proposals.append(deepcopy(result))
            if len(proposals) == 1:
                payload = json.loads(messages[1]['content'])
                target = next(r['target'] for r in payload['registered_targets'] if r['target']['field'] == 'error_text')
                result['updates'] = [{'operation': 'add', 'target': target, 'previous_fact_id': None,
                    'value': '报错均已清理', 'source': {'message_id': 'M2', 'quote': UNAVAILABLE}}]
            return json.dumps(result)
        self.service.generate_context = context
        response = self.send('M2', UNAVAILABLE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(proposals), 2)
        view = self.store.read_preparation('U1', self.cid)
        self.assertFalse(any(f.target.field == 'error_text' for f in view['state'].current_facts))
        self.assertEqual({c.availability for c in view['constraint_state'].current}, {'unavailable'})
