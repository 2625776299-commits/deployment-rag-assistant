"""Canonical question sources across SQL, continuation, task return and processes."""
from copy import deepcopy
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
from app.conversation_service import ConversationService,ConversationExecution
from app.conversation_store import PostgresConversationStore
from app.main import create_app


ORIGINAL='升级后数据库连接失败，现在能否排除其他原因，认定就是密码错误？'
BAD_HINT='在已经排除其他原因的前提下，确认连接失败是密码错误。'
SUPPLEMENT='报错已清理，其他现场信息也拿不到。'
GENERAL='先只解释同一Compose网络内的服务名发现机制。'
OTHER='另一个独立任务：解释事务提交。'
RETURN='回到第一次连接故障，继续原先的问题。'
RULE={'chunk_id':'QA:1','content':'同一网络使用内部DNS发现服务。',
      'metadata':{'document_id':'fixture','section_id':'QA','source_url':'https://example.invalid/qa'}}


@unittest.skipUnless(os.getenv('RAG_CONVERSATION_TEST_PORT'),'requires isolated PostgreSQL probe')
class ConversationQuestionAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.schema='rag_conversation_question_'+uuid4().hex
        self.connections=[];self.payloads=[];self.calls=[];self.queries=[]
        self.store=PostgresConversationStore(self.connect,schema=self.schema);self.store.initialize()
        self.service=ConversationService(self.store,generate_task=self.task_model,
            generate_context=self.context_model,bind_execution=self.bind)
        self.client=TestClient(create_app(conversation_service=self.service,
            authenticate=bearer_authenticator({'question-u1':'U1'})))
        self.headers={'Authorization':'Bearer question-u1'}
        self.cid=self.client.post('/conversations',headers=self.headers).json()['conversation_id']
        self.addCleanup(self.cleanup)

    def connect(self):
        c=psycopg.connect(host='127.0.0.1',port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),
            dbname='postgres',user='rag_conversation_probe',connect_timeout=5)
        self.connections.append(c);return c

    def cleanup(self):
        self.client.close()
        with self.connect() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        self.assertTrue(all(c.closed for c in self.connections))

    def send(self,mid,text):
        return self.client.post(f'/conversations/{self.cid}/messages',headers=self.headers,
                                json={'message_id':mid,'content':text})

    def task_model(self,messages):
        self.assertTrue(all(c.closed for c in self.connections));self.calls.append('task')
        p=json.loads(messages[1]['content']);u=p['current_user_message'];text=u['content']
        task=(next((t for t in p['existing_tasks'] if t['source_message_id']=='M1'),None) if text==RETURN
              else next((t for t in p['existing_tasks'] if t['task_id']==p['current_task_id']),None))
        selection=({'operation':'create','goal':'事务解释' if text==OTHER else '诊断连接失败','source_quote':text}
                   if task is None or text==OTHER else {'operation':'select','task_id':task['task_id'],'source_quote':text})
        request=({'mode':'continue','from_input_version':task['latest_input_version']} if text in (SUPPLEMENT,RETURN)
                 else {'mode':'question','question':BAD_HINT if text==ORIGINAL else '模型解释：'+text,
                       'source':{'message_id':u['message_id'],'quote':'排除其他原因' if text==ORIGINAL else text}})
        return json.dumps({'selection':selection,'turn_request':request})

    def context_model(self,messages):
        self.assertTrue(all(c.closed for c in self.connections));self.calls.append('context')
        p=json.loads(messages[1]['content']);u=p['current_user_message']
        return json.dumps({'updates':[],'analysis_request':{'mode':'current','confirmation_from_input_version':None,
            'source':{'message_id':u['message_id'],'quote':u['content']}}})

    def evaluate(self,messages):
        self.calls.append('evaluate');p=json.loads(messages[1]['content']);self.payloads.append(deepcopy(p))
        question=p['question'];answer=question in (GENERAL,OTHER)
        return json.dumps({'requirements':[{'requirement_id':'R1','requirement':question,'question_quote':question}],
            'assessments':[{'requirement_id':'R1','coverage':'covered' if answer else 'missing',
                'evidence':[{'chunk_id':'QA:1','quote':RULE['content']}] if answer else [],
                'gap':None if answer else '没有本次根因的证据'}],
            'action':'answer' if answer else 'insufficient','selected_chunk_ids':['QA:1'] if answer else [],
            'missing_knowledge':[] if answer else ['无法证明原问题的结论'],
            'missing_user_information':[],'search_query':None,'reason':'scripted contract output',
            'known_findings':[],'pending_impacts':[]})

    def generate(self,messages):
        self.calls.append('generate');self.payloads.append(deepcopy(json.loads(messages[1]['content'])))
        return json.dumps({'status':'answer','answer':'资料中的一般机制。[QA:1]',
                           'citation_ids':['QA:1'],'missing_information':[]})

    def bind(self):
        def retrieve(q):self.queries.append(q);return [RULE]
        return ConversationExecution('question-fixture',lambda s:run_prepared_snapshot(s,
            retrieve=retrieve,evaluate_raw=self.evaluate,generate_raw=self.generate))

    def test_bad_rewrite_never_replaces_sql_frozen_original_or_evaluation_question(self):
        self.assertEqual(self.send('M1',ORIGINAL).json()['status'],'insufficient')
        h=self.store.read_history('U1',self.cid);p=h['preparations'][0];turn=h['runs'][0]['input_snapshot']['turn']
        self.assertEqual(p['turn_question'],BAD_HINT)
        self.assertEqual(p['question_anchor']['text'],ORIGINAL)
        self.assertEqual(turn['question'],ORIGINAL);self.assertEqual(turn['retrieval_question'],BAD_HINT)
        self.assertEqual(self.payloads[-1]['question'],ORIGINAL)
        self.assertEqual(self.payloads[-1]['question_interpretation'],BAD_HINT)
        self.assertTrue(self.queries[-1].startswith(BAD_HINT))

    def test_supplement_inherits_same_source_and_replay_cannot_rewrite_it(self):
        self.send('M1',ORIGINAL);old=deepcopy(self.store.read_history('U1',self.cid)['runs'][0])
        self.send('M2',SUPPLEMENT);h=self.store.read_history('U1',self.cid)
        self.assertEqual(h['preparations'][0]['question_anchor'],h['preparations'][1]['question_anchor'])
        self.assertEqual(h['runs'][0],old)
        self.assertEqual(self.payloads[-1]['current_user_message']['content'],SUPPLEMENT)
        self.assertEqual(self.payloads[-1]['question'],ORIGINAL)
        calls=list(self.calls);self.assertTrue(self.send('M2',SUPPLEMENT).json()['replayed'])
        self.assertEqual(self.calls,calls)

    def test_new_question_changes_source_and_both_consumers_see_the_raw_text(self):
        self.send('M1',ORIGINAL);self.send('M2',GENERAL)
        h=self.store.read_history('U1',self.cid)
        self.assertEqual(h['preparations'][1]['question_anchor']['message_id'],'M2')
        for p in self.payloads[-2:]:
            self.assertEqual(p['question'],GENERAL);self.assertEqual(p['question_source']['message_id'],'M2')
        self.assertEqual(h['preparations'][0]['question_anchor']['text'],ORIGINAL)

    def test_task_return_preserves_original_task_question_source(self):
        self.send('M1',ORIGINAL);self.send('M2',SUPPLEMENT);self.send('M3',OTHER);self.send('M4',RETURN)
        h=self.store.read_history('U1',self.cid);anchors=[p['question_anchor'] for p in h['preparations']]
        self.assertEqual(anchors[0],anchors[3]);self.assertNotEqual(anchors[2]['task_id'],anchors[3]['task_id'])
        self.assertEqual(self.payloads[-1]['question'],ORIGINAL)

    def test_queryable_legacy_request_can_recover_source_without_backfilling_old_preparation(self):
        self.send('M1',ORIGINAL)
        with self.connect() as c:
            c.execute(sql.SQL('UPDATE {} SET question_anchor=NULL WHERE conversation_id=%s')
                .format(self.store._table('preparations')),(self.cid,))
        old=deepcopy(self.store.read_history('U1',self.cid)['runs'][0])
        self.send('M2',SUPPLEMENT);h=self.store.read_history('U1',self.cid)
        self.assertIsNone(h['preparations'][0]['question_anchor'])
        self.assertEqual(h['preparations'][1]['question_anchor']['text'],ORIGINAL)
        self.assertEqual(h['runs'][0],old)

    def test_new_process_reads_identical_original_source_and_retrieval_hint(self):
        self.send('M1',ORIGINAL);self.send('M2',SUPPLEMENT)
        code='''import json,os,sys,psycopg
from app.conversation_store import PostgresConversationStore
def connect():return psycopg.connect(host='127.0.0.1',port=int(os.environ['RAG_CONVERSATION_TEST_PORT']),dbname='postgres',user='rag_conversation_probe')
s=PostgresConversationStore(connect,schema=sys.argv[1]);h=s.read_history('U1',sys.argv[2]);print(json.dumps(h['runs'][-1]['input_snapshot']['turn']))
'''
        r=subprocess.run([sys.executable,'-c',code,self.schema,self.cid],capture_output=True,text=True,
            check=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        turn=json.loads(r.stdout)
        self.assertEqual(turn['question'],ORIGINAL);self.assertEqual(turn['retrieval_question'],BAD_HINT)
        self.assertEqual(turn['question_anchor']['message_id'],'M1')

