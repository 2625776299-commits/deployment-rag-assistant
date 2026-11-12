import unittest
from fastapi.testclient import TestClient
from app.bounded_retrieval import run_bounded_retrieval
from app.main import create_app
from app.rag_service import run_rag_service


def chunk(key):
    return {'chunk_id':key,'content':key,'metadata':{'document_id':'doc','section_id':key,'source_url':'https://example.org/'+key}}


def plan(q,gaps):
    return {'specific_query':'specific rule','mechanism_query':'general mechanism','technical_anchors':[]}


def decision(action, selected=None):
    return {'status':'proposed','proposal':{'action':action,'search_query':'legacy query',
        'missing_knowledge':['needed rule'] if action=='retrieve' else [],'missing_user_information':['logs']},
        'selected_evidence':selected or []}


class PlannedIntegrationTests(unittest.TestCase):
    def test_queries_share_one_round_and_original_question_is_reevaluated(self):
        observed=[]; calls=[]
        def evaluate(q,pool):
            observed.append((q,[c['chunk_id'] for c in pool]))
            return decision('retrieve' if len(observed)==1 else 'answer',pool[:1])
        def search(request):
            calls.append(request)
            return [chunk(request.query)]
        result=run_bounded_retrieval('original question',[chunk('a')],evaluate,self.fail,
            request_retrieval=search,plan_queries=plan,build_id='v1')
        self.assertEqual(result['status'],'answer_proposed')
        self.assertEqual(result['retrieval_count'],1)
        self.assertEqual(len(calls),2)
        self.assertTrue(all(c.build_id=='v1' and c.exclude_chunk_ids==('a',) and c.max_results==9 for c in calls))
        self.assertEqual([q for q,_ in observed],['original question']*2)

    def test_capacity_one_is_shared_not_one_slot_per_query(self):
        n=0
        def evaluate(q,p):
            nonlocal n
            n+=1
            self.assertLessEqual(len(p),10)
            return decision('retrieve' if n==1 else 'answer',p[:1])
        r=run_bounded_retrieval('q',[chunk(str(i)) for i in range(9)],evaluate,
            lambda query:[chunk(query)],plan_queries=plan)
        self.assertEqual(len(r['candidate_pool']),10)

    def test_one_query_failure_keeps_successful_evidence(self):
        n=0
        def evaluate(q,p):
            nonlocal n
            n+=1
            return decision('retrieve' if n==1 else 'answer',p[:1])
        def retrieve(q):
            if q=='specific rule':
                raise TimeoutError()
            return [chunk('b')]
        r=run_bounded_retrieval('q',[chunk('a')],evaluate,retrieve,plan_queries=plan)
        self.assertEqual(r['status'],'answer_proposed')
        self.assertIn('b',[c['chunk_id'] for c in r['candidate_pool']])
        self.assertTrue(any(t.get('status')=='failed' for t in r['trace']))

    def test_all_query_failures_are_system_failure(self):
        def broken(q):
            raise TimeoutError()
        r=run_rag_service('q',retrieve=lambda q:[chunk('a')],evaluate=lambda q,p:decision('retrieve'),
            generate=self.fail,request_retrieval=lambda r:[chunk('a')] if r.query=='q' else broken(r.query),plan_queries=plan)
        self.assertEqual(r['status'],'failed')
        self.assertEqual(r['error']['stage'],'additional_retrieve')

    def test_invalid_plan_fails_before_any_supplementary_query(self):
        def invalid(q,g):
            return {'specific_query':None,'mechanism_query':None,'technical_anchors':[]}
        r=run_rag_service('q',retrieve=lambda q:[chunk('a')],evaluate=lambda q,p:decision('retrieve'),
            generate=self.fail,plan_queries=invalid)
        self.assertEqual(r['status'],'failed')
        self.assertEqual(r['error']['stage'],'plan_queries')
        self.assertEqual(r['counts']['additional_retrievals'],0)

    def test_terminal_decisions_do_not_plan_queries(self):
        for action in ('answer','clarify','insufficient'):
            with self.subTest(action=action):
                r=run_bounded_retrieval('q',[chunk('a')],lambda q,p:decision(action,p),
                    self.fail,plan_queries=self.fail)
                self.assertEqual(r['retrieval_count'],0)

    def test_no_new_evidence_stops(self):
        r=run_bounded_retrieval('q',[chunk('a')],lambda q,p:decision('retrieve'),
            lambda query:[chunk('a')],plan_queries=plan)
        self.assertEqual(r['stop_reason'],'no_new_evidence')

    def test_repeated_queries_do_not_execute(self):
        r=run_bounded_retrieval('q',[chunk('a')],lambda q,p:decision('retrieve'),self.fail,
            plan_queries=plan,known_queries=('specific rule','general mechanism'))
        self.assertEqual(r['stop_reason'],'repeated_query')

    def test_http_entry_invokes_planner_and_counts_real_query_callbacks(self):
        qs=[]; es=[]
        def retrieve(q):
            qs.append(q)
            return [chunk('a:000' if q=='q' else q)]
        def evaluate(q,p):
            es.append(q)
            return decision('retrieve' if len(es)==1 else 'answer',p[:1])
        def generate(q,e):
            return {'status':'answer','answer':'Supported [a:000]','citation_ids':['a:000'],'missing_information':[]}
        app=create_app(retrieve=retrieve,evaluate=evaluate,generate=generate,plan_queries=plan)
        with TestClient(app) as client:
            response=client.post('/ask',json={'question':'q'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['status'],'answer')
        self.assertEqual(qs,['q','specific rule','general mechanism'])
        self.assertEqual(es,['q','q'])


if __name__=='__main__':
    unittest.main()
