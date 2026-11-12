"""Conditional runtime contracts with scripted models; no recognition accuracy."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import Mock

from app.analysis_scope import (
    AnalysisScope, Assumption, build_analysis_view, restore_analysis_scope,
)
from app.conversation_context import ContextState, FactTarget, PendingFact, SourceQuote, UserFact
from app.conversation_rag import build_initial_query, run_contextual_rag, run_prepared_snapshot
from app.requirement_evaluation import evaluate_requirements
from app.provider_errors import ModelProviderError


APP = FactTarget('application', 'T1_environment', 'runtime_location')
DB = FactTarget('database', 'T1_environment', 'runtime_location')
NETWORK = FactTarget('application', 'T1_environment', 'network_relation')
SERVICE = FactTarget('database', 'T1_environment', 'service_name')
PORT = FactTarget('database', 'T1_environment', 'container_port')
ERROR = FactTarget('application', 'T1_environment', 'error_text')
TARGETS = frozenset((APP, DB, NETWORK, SERVICE, PORT, ERROR))
SOURCE = SourceQuote('M2', '先假设应用和数据库都在同一个Compose网络，数据库服务名db，容器端口5432')
QUESTION = '在刚才假设的同一Compose网络中，应用的连接host和port如何配置？'
PASSAGE = {'chunk_id': 'RULE:1', 'content': '同一Compose网络的服务间连接可使用数据库服务名和容器端口。',
           'metadata': {'document_id': 'fixture', 'section_id': 'RULE',
                        'source_url': 'https://example.invalid/conditional-fixture'}}
CONTEXT = ContextState(3, (
    UserFact('APP_HOST', APP, 'host', SourceQuote('M1', '应用在宿主机')),
    UserFact('SERVICE_DB', SERVICE, 'db', SourceQuote('M1', '服务名db')),
), pending_facts=(
    PendingFact('P_DB', DB, UserFact('OLD_DB', DB, 'old-location', SourceQuote('M1', '旧位置')),
                SourceQuote('M2', '数据库实际位置不确定'), 'uncertain_value'),
    PendingFact('P_ERROR', ERROR, None, SourceQuote('M2', '真实报错还没拿到'), 'uncertain_value'),
))


def hypothesis():
    return AnalysisScope('hypothetical', tuple(Assumption(t, value, SOURCE) for t, value in (
        (APP, 'container'), (DB, 'compose'), (NETWORK, 'same_compose_network'), (PORT, '5432'))), 2)


def evaluation(question=QUESTION, *, blocked=None):
    return {
        'requirements': [{'requirement_id': 'R1', 'requirement': question, 'question_quote': question}],
        'assessments': [{'requirement_id': 'R1', 'coverage': 'partial' if blocked else 'covered',
                         'evidence': [{'chunk_id': 'RULE:1', 'quote': PASSAGE['content']}],
                         'gap': '必须核实实际环境' if blocked else None}],
        'action': 'clarify' if blocked else 'answer', 'selected_chunk_ids': ['RULE:1'],
        'missing_knowledge': [], 'missing_user_information': ['请提供实际环境'] if blocked else [],
        'search_query': None, 'reason': 'Scripted conditional contract test, not a semantic benchmark.',
        'known_findings': [],
        'pending_impacts': [{'pending_id': p.pending_id,
            'blocking_requirement_ids': ['R1'] if p.pending_id == blocked else [],
            'reason': '条件假设已提供所需位置' if p.pending_id == 'P_DB' else '逐需求判断该未知项'}
            for p in CONTEXT.pending_facts],
    }


def answer():
    return json.dumps({'status': 'answer', 'answer': '在上述同网假设下，host=db，port=5432。[RULE:1]',
                       'citation_ids': ['RULE:1'], 'missing_information': []}, ensure_ascii=False)


class AnalysisScopeTests(unittest.TestCase):
    def test_view_separates_assumptions_from_facts_and_preserves_actual_unknowns(self):
        before = CONTEXT.to_dict()
        view = build_analysis_view(CONTEXT, hypothesis(), allowed_targets=TARGETS)
        payload = view.to_payload()
        self.assertEqual(CONTEXT.to_dict(), before)
        self.assertEqual(view.actual_context.to_dict(), before)
        self.assertEqual([f['fact_id'] for f in payload['user_stated_facts']], ['SERVICE_DB'])
        self.assertEqual(payload['analysis_scope']['assumptions'][0]['value'], 'container')
        pending = {p['pending_id']: p for p in payload['pending_clarifications']}
        self.assertTrue(pending['P_DB']['assumed_for_this_turn'])
        self.assertFalse(pending['P_ERROR']['assumed_for_this_turn'])
        self.assertNotIn('old-location', json.dumps(payload))
        self.assertEqual(CONTEXT.current_facts[0].value, 'host')
        payload['analysis_scope']['assumptions'][0]['value'] = 'tampered'
        self.assertEqual(view.to_payload()['analysis_scope']['assumptions'][0]['value'], 'container')

    def test_query_uses_hypothesis_and_keeps_non_overridden_facts(self):
        view = build_analysis_view(CONTEXT, hypothesis(), allowed_targets=TARGETS)
        query = build_initial_query(QUESTION, view)
        self.assertIn('[假设]', query)
        self.assertIn('runtime_location: container', query)
        self.assertIn('service_name: db', query)
        self.assertNotIn('runtime_location: host', query)
        self.assertNotIn('old-location', query)
        self.assertNotIn('P_DB', query)

    def test_serialized_scope_roundtrip_keeps_original_assumption_sources(self):
        raw = hypothesis().to_dict()
        restored = restore_analysis_scope(json.loads(json.dumps(raw)))
        self.assertEqual(restored, hypothesis())
        self.assertEqual(restored.assumptions[0].source.message_id, 'M2')
        self.assertEqual(restored.inherited_from_input_version, 2)

    def test_invalid_modes_shapes_and_inheritance_are_not_silently_current(self):
        for raw in (None, {}, {'mode': 'hypothetical'},
                    {**hypothesis().to_dict(), 'mode': 'needs_clarification'},
                    {**hypothesis().to_dict(), 'assumptions': []},
                    {**hypothesis().to_dict(), 'inherited_from_input_version': True},
                    {**hypothesis().to_dict(), 'inherited_from_input_version': 0}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                restore_analysis_scope(raw)
        for version in (3, 4):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'must_precede'):
                build_analysis_view(CONTEXT, AnalysisScope('hypothetical', hypothesis().assumptions, version),
                                    allowed_targets=TARGETS)

    def test_duplicate_foreign_and_unregistered_targets_fail_before_retrieval(self):
        a = hypothesis().assumptions[0]
        with self.assertRaisesRegex(ValueError, 'duplicate_assumption_target'):
            AnalysisScope('hypothetical', (a, a))
        foreign = Assumption(FactTarget('application', 'T2_environment', 'runtime_location'), 'container', SOURCE)
        retrieve = Mock()
        for allowed, scope in ((None, hypothesis()), (TARGETS, AnalysisScope('hypothetical', (foreign,))),
                               (frozenset((APP,)), hypothesis())):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                run_contextual_rag(QUESTION, CONTEXT, analysis_scope=scope, allowed_targets=allowed,
                    retrieve=retrieve, evaluate_raw=Mock(), generate_raw=Mock())
        retrieve.assert_not_called()

    def test_runtime_passes_one_view_to_evaluation_and_generation_without_changing_question(self):
        captures = {'evaluate': [], 'generate': []}
        def evaluate(messages):
            captures['evaluate'].append(deepcopy(messages))
            return json.dumps(evaluation())
        def generate(messages):
            captures['generate'].append(deepcopy(messages))
            return answer()
        retrieve = Mock(return_value=[PASSAGE])
        before = CONTEXT.to_dict()
        result = run_contextual_rag(QUESTION, CONTEXT, analysis_scope=hypothesis(), allowed_targets=TARGETS,
            retrieve=retrieve, evaluate_raw=evaluate, generate_raw=generate)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(result['evaluation']['blocking_pending_ids'], [])
        self.assertEqual(result['analysis_scope']['mode'], 'hypothetical')
        payloads = [json.loads(captures[stage][0][1]['content']) for stage in ('evaluate', 'generate')]
        self.assertEqual(payloads[0]['conversation_context'], payloads[1]['conversation_context'])
        self.assertEqual([p['question'] for p in payloads], [QUESTION, QUESTION])
        for stage in ('evaluate', 'generate'):
            self.assertIn('不是已经发生的变更', captures[stage][0][0]['content'])
        self.assertEqual(CONTEXT.to_dict(), before)

    def test_assumed_unknown_cannot_be_reintroduced_as_a_blocker_by_evaluator(self):
        view = build_analysis_view(CONTEXT, hypothesis(), allowed_targets=TARGETS)
        result = evaluate_requirements(QUESTION, [PASSAGE], lambda _: json.dumps(evaluation(blocked='P_DB')),
            context=view, include_user_reply=True)
        self.assertEqual(result['status'], 'evaluation_failed')
        self.assertEqual(result['failure_reason'], 'assumed_pending_cannot_block_hypothetical_requirement')
        self.assertEqual(result['model_attempts'], 1)
        self.assertIsNone(result['proposal'])

    def test_an_unassumed_unknown_can_still_block_its_requested_conclusion(self):
        question = '在假设的网络下，真实应用报错的具体原因是什么？'
        view = build_analysis_view(CONTEXT, hypothesis(), allowed_targets=TARGETS)
        result = evaluate_requirements(question, [PASSAGE],
            lambda _: json.dumps(evaluation(question, blocked='P_ERROR')), context=view, include_user_reply=True)
        self.assertEqual(result['status'], 'proposed')
        self.assertEqual(result['proposal']['blocking_pending_ids'], ['P_ERROR'])
        self.assertEqual(len(CONTEXT.pending_facts), 2)

    def test_returning_to_current_scope_restores_unknown_impact_not_a_hypothetical_fact(self):
        # Executing a hypothesis does not resolve P_DB for a later actual request.
        run_contextual_rag(QUESTION, CONTEXT, analysis_scope=hypothesis(), allowed_targets=TARGETS,
            retrieve=lambda _: [PASSAGE], evaluate_raw=lambda _: json.dumps(evaluation()), generate_raw=lambda _: answer())
        question = '现在真实部署的数据库连接地址是什么？'
        generated = Mock()
        result = run_contextual_rag(question, CONTEXT, analysis_scope=AnalysisScope('current'), allowed_targets=TARGETS,
            retrieve=lambda _: [PASSAGE], evaluate_raw=lambda _: json.dumps(evaluation(question, blocked='P_DB')),
            generate_raw=generated)
        self.assertEqual(result['status'], 'needs_user')
        self.assertEqual(result['evaluation']['blocking_pending_ids'], ['P_DB'])
        self.assertIn('runtime_location: host', result['initial_query'])
        self.assertNotIn('[假设]', result['initial_query'])
        generated.assert_not_called()

    def snapshot(self):
        return {'task': {'task_id': 'T1', 'goal': '当前部署连接排查',
                         'registered_targets': [{'target': asdict(t), 'description': 'Server registered field'}
                                                for t in sorted(TARGETS, key=lambda t: (t.entity_id, t.field))]},
                'context': CONTEXT.to_dict(),
                'turn': {'input_version': 3, 'message_id': 'M3', 'question': QUESTION,
                         'analysis_scope': hypothesis().to_dict()}}

    def test_trusted_frozen_scope_reaches_runtime_and_cannot_be_overridden(self):
        snapshot = self.snapshot()
        before = deepcopy(snapshot)
        result = run_prepared_snapshot(snapshot, retrieve=lambda _: [PASSAGE],
            evaluate_raw=lambda _: json.dumps(evaluation()), generate_raw=lambda _: answer())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(result['task_id'], 'T1')
        self.assertEqual(result['analysis_scope']['inherited_from_input_version'], 2)
        self.assertEqual(snapshot, before)
        with self.assertRaisesRegex(ValueError, 'cannot_be_overridden'):
            run_prepared_snapshot(snapshot, retrieve=Mock(), evaluate_raw=Mock(), generate_raw=Mock(),
                                  analysis_scope=AnalysisScope('current'))

    def test_missing_registry_or_explicitly_invalid_saved_scope_stops_before_model(self):
        for variant in ('missing_registry', 'missing_target', 'null_scope', 'invalid_assumption'):
            snapshot = self.snapshot()
            if variant == 'missing_registry':
                del snapshot['task']['registered_targets']
            elif variant == 'missing_target':
                snapshot['task']['registered_targets'] = []
            elif variant == 'null_scope':
                snapshot['turn']['analysis_scope'] = None
            else:
                snapshot['turn']['analysis_scope']['assumptions'][0]['source'] = {'message_id': 'M2'}
            retrieve, evaluate = Mock(), Mock()
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                run_prepared_snapshot(snapshot, retrieve=retrieve, evaluate_raw=evaluate, generate_raw=Mock())
            retrieve.assert_not_called()
            evaluate.assert_not_called()

    def run_with_generation(self, callback, **extra):
        return run_contextual_rag(QUESTION, CONTEXT, analysis_scope=hypothesis(), allowed_targets=TARGETS,
            retrieve=lambda _: [PASSAGE], evaluate_raw=lambda _: json.dumps(evaluation()), generate_raw=callback, **extra)

    def test_answer_contract_repair_keeps_original_scope_evidence_and_failed_output(self):
        bad = json.loads(answer())
        del bad['citation_ids']
        raw = json.dumps(bad)
        callback = Mock(side_effect=[raw, answer()])
        result = self.run_with_generation(callback)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(callback.call_count, 2)
        first, second = (call.args[0] for call in callback.call_args_list)
        self.assertEqual(first, second[:2])
        self.assertEqual(second[2], {'role': 'assistant', 'content': raw})
        self.assertIn('RULE:1', second[-1]['content'])
        self.assertEqual(result['generation_checks'][0]['raw_text'], raw)
        self.assertIsNotNone(result['generation_checks'][0]['validation_error'])
        self.assertIsNone(result['generation_checks'][1]['validation_error'])

    def test_invalid_json_placeholder_and_fabricated_reference_can_be_repaired_once(self):
        for bad in (answer().replace('[RULE:1]', '[chunk_id]'), answer().replace('RULE:1', 'FAKE:999'),
                    '{"status":"answer"，"answer":"invalid JSON"}'):
            callback = Mock(side_effect=[bad, answer()])
            with self.subTest(bad=bad):
                self.assertEqual(self.run_with_generation(callback)['status'], 'answer')
                self.assertEqual(callback.call_count, 2)

    def test_repair_failure_stops_after_two_obtained_invalid_answers(self):
        callback = Mock(return_value='{}')
        result = self.run_with_generation(callback)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'generation_failed')
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(len(result['generation_checks']), 2)

    def test_provider_failure_or_timeout_does_not_trigger_contract_repair(self):
        for error in (TimeoutError(), ModelProviderError('provider_connection_failed')):
            callback = Mock(side_effect=error)
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.run_with_generation(callback)['status'], 'failed')
                self.assertEqual(callback.call_count, 1)
        callback = Mock(side_effect=['{}', TimeoutError()])
        result = self.run_with_generation(callback)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(callback.call_count, 2)


if __name__ == '__main__':
    unittest.main()
