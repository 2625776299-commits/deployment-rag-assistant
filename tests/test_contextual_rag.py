"""Context/evaluation/controller integration with explicit scripted model replies."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from app.conversation_context import ContextState, FactTarget, PendingFact, SourceQuote, UserFact
from app.conversation_rag import build_conversation_answer_messages, run_contextual_rag, run_prepared_snapshot
from app.requirement_evaluation import build_requirement_messages, evaluate_requirements


APP = FactTarget('app', 'incident', 'runtime_location')
DB = FactTarget('db', 'incident', 'runtime_location')
ERROR = FactTarget('app', 'incident', 'error_text')
CONTEXT = ContextState(3, (
    UserFact('F_APP', APP, 'host', SourceQuote('M3', '应用在宿主机')),
    UserFact('F_ERROR', ERROR, 'Connection refused', SourceQuote('M1', 'Connection refused')),
), pending_facts=(PendingFact('P_DB', DB,
    UserFact('F_OLD_DB', DB, 'old-container-value', SourceQuote('M1', '数据库在容器')),
    SourceQuote('M2', '数据库位置我不确定'), 'uncertain_value'),))
PASSAGE = {'chunk_id': 'TEST:1', 'content': '测试资料：连接失败可先核对客户端连接的目标。它不提供当前用户的具体地址。',
           'metadata': {'document_id': 'fixture', 'section_id': 'TEST',
                        'source_url': 'https://example.invalid/fixture'}}
A = '先解释连接错误的一般含义。'
B = '告诉我当前数据库应该配置成哪个具体连接地址。'


def reply(question=A, *, blocked=False):
    return {
        'requirements': [{'requirement_id': 'R1', 'requirement': question, 'question_quote': question}],
        'assessments': [{'requirement_id': 'R1', 'coverage': 'partial' if blocked else 'covered',
                         'evidence': [{'chunk_id': 'TEST:1', 'quote': PASSAGE['content']}],
                         'gap': '当前数据库的位置和可访问地址未确认' if blocked else None}],
        'action': 'clarify' if blocked else 'answer', 'selected_chunk_ids': ['TEST:1'],
        'missing_knowledge': [], 'missing_user_information': ['数据库运行在哪里？'] if blocked else [],
        'search_query': None, 'reason': 'Scripted test; not evidence of semantic correctness.',
        'known_findings': [],
        'pending_impacts': [{'pending_id': 'P_DB', 'blocking_requirement_ids': ['R1'] if blocked else [],
                             'reason': '具体地址需要位置' if blocked else '一般解释不依赖当前数据库位置'}],
    }


def answer():
    return json.dumps({'status': 'answer', 'answer': '可先核对客户端连接目标。[TEST:1]',
                       'citation_ids': ['TEST:1'], 'missing_information': []}, ensure_ascii=False)


class ContextualRagTests(unittest.TestCase):
    def test_frozen_incident_constraints_reach_both_evaluator_and_generator(self):
        from app.task_constraints import ConstraintSnapshot,IncidentRef,EvidenceConstraint
        record=EvidenceConstraint('C1',IncidentRef('T1','I1'),'incident_logs','unavailable',
                                  SourceQuote('M2','日志不可取得'),2)
        payload={'input_version':3,'scope':{'mode':'incident','task_id':'T1','incident_id':'I1'},
                 'evidence_constraints':json.loads(json.dumps(ConstraintSnapshot(3,(record,)).to_dict()))['current']}
        snapshot={'task':{'task_id':'T1','goal':'诊断故障'},'context':CONTEXT.to_dict(),
                  'turn':{'message_id':'M3','input_version':3,'question':A,'task_constraints':payload}}
        received=[]
        def evaluate(messages):
            received.append(json.loads(messages[1]['content'])['task_constraints'])
            return json.dumps(reply())
        def generate(messages):
            received.append(json.loads(messages[1]['content'])['task_constraints'])
            return answer()
        result=run_prepared_snapshot(snapshot,retrieve=lambda _:[PASSAGE],evaluate_raw=evaluate,generate_raw=generate)
        self.assertEqual(result['status'],'answer')
        self.assertEqual(received,[payload,payload])
        self.assertEqual(snapshot['turn']['task_constraints'],payload)
        snapshot['task']['task_id']='T2'
        retrieve=Mock()
        with self.assertRaisesRegex(ValueError,'frozen_incident_task_mismatch'):
            run_prepared_snapshot(snapshot,retrieve=retrieve,evaluate_raw=Mock(),generate_raw=Mock())
        retrieve.assert_not_called()

    def test_prepared_turn_controls_retrieval_evaluation_and_answer_scope(self):
        broad = '排查数据库重建后的全部原因并提供完整修复方案。'
        snapshot = {'task': {'task_id': 'T1', 'goal': broad},
                    'turn': {'input_version': 3, 'message_id': 'M3', 'question': A,
                             'raw_message': '应用在宿主机'}, 'context': CONTEXT.to_dict()}
        captured_eval, captured_gen = [], []
        def evaluate(messages):
            captured_eval.append(deepcopy(messages))
            return json.dumps(reply())
        def generate(messages):
            captured_gen.append(deepcopy(messages))
            return answer()
        retrieve = Mock(return_value=[PASSAGE])
        result = run_prepared_snapshot(snapshot, retrieve=retrieve, evaluate_raw=evaluate, generate_raw=generate)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(result['turn_question'], A)
        self.assertTrue(retrieve.call_args.args[0].startswith(A))
        self.assertNotIn(broad, retrieve.call_args.args[0])
        for messages in (captured_eval[0], captured_gen[0]):
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['question'], A)
            self.assertEqual(payload['task_background_goal'], broad)
            self.assertIn('不是额外的回答要求', messages[0]['content'])
        self.assertEqual(snapshot['task']['goal'], broad)

    def test_background_goal_is_not_an_allowed_requirement_question_quote(self):
        broad = '另一个不属于本轮问题的背景目标。'
        wrong = reply()
        wrong['requirements'][0]['question_quote'] = broad
        result = run_contextual_rag(A, CONTEXT, task_goal=broad, retrieve=lambda _: [PASSAGE],
            evaluate_raw=lambda _: json.dumps(wrong), generate_raw=Mock())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'requirement_quote_not_in_question')
        self.assertIsNone(result['evaluation'])

    def test_prepared_turn_requires_question_and_matching_context_before_retrieval(self):
        snapshot = {'task': {'task_id': 'T1', 'goal': B},
                    'turn': {'input_version': 3, 'message_id': 'M3', 'question': A}, 'context': CONTEXT.to_dict()}
        for turn in ({'input_version': 3, 'message_id': 'M3'},
                     {'input_version': 2, 'message_id': 'M3', 'question': A}):
            with self.subTest(turn=turn):
                retrieve = Mock()
                with self.assertRaises(ValueError):
                    run_prepared_snapshot({**snapshot, 'turn': turn}, retrieve=retrieve,
                                          evaluate_raw=Mock(), generate_raw=Mock())
                retrieve.assert_not_called()

    def evaluate(self, proposal, question=A, context=CONTEXT):
        return evaluate_requirements(question, [PASSAGE], generate=lambda _: json.dumps(proposal),
                                     include_user_reply=True, context=context)

    def test_general_answer_can_proceed_while_unknown_context_is_preserved(self):
        captured_eval, captured_gen = [], []
        def evaluate(messages):
            captured_eval.append(json.loads(messages[1]['content']))
            return json.dumps(reply())
        def generate(messages):
            captured_gen.append(json.loads(messages[1]['content']))
            return answer()
        retrieve = Mock(return_value=[PASSAGE])
        result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=evaluate, generate_raw=generate)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(result['context_input_version'], 3)
        self.assertEqual(captured_eval[0]['question'], A)
        self.assertEqual(captured_gen[0]['question'], A)
        self.assertEqual(captured_eval[0]['conversation_context'], captured_gen[0]['conversation_context'])
        query = retrieve.call_args.args[0]
        self.assertIn('host', query)
        self.assertIn('Connection refused', query)
        self.assertNotIn('P_DB', query)
        self.assertNotIn('old-container-value', query)
        self.assertNotIn('old-container-value', json.dumps(captured_eval))
        self.assertEqual(len(CONTEXT.pending_facts), 1)
        self.assertEqual(result['evaluation']['blocking_pending_ids'], [])

    def test_concrete_address_clarifies_without_answer_generation(self):
        generate = Mock()
        result = run_contextual_rag(B, CONTEXT, retrieve=lambda _: [PASSAGE],
            evaluate_raw=lambda _: json.dumps(reply(B, blocked=True)), generate_raw=generate)
        self.assertEqual(result['status'], 'needs_user')
        self.assertEqual(result['required_information'], ['数据库运行在哪里？'])
        generate.assert_not_called()
        self.assertEqual(result['evaluation']['blocking_pending_ids'], ['P_DB'])
        step = next(s for s in result['trace'] if s['step'] == 'evaluate')
        self.assertEqual(step['blocking_pending_ids'], ['P_DB'])
        self.assertEqual(step['context_input_version'], 3)

    def test_mixed_request_preserves_supported_part_before_clarification(self):
        question = A + B
        proposal = reply(question, blocked=True)
        proposal['requirements'] = [
            {'requirement_id': 'R1', 'requirement': A, 'question_quote': A},
            {'requirement_id': 'R2', 'requirement': B, 'question_quote': B}]
        supported = {'requirement_id': 'R1', 'coverage': 'covered',
                     'evidence': [{'chunk_id': 'TEST:1', 'quote': PASSAGE['content']}], 'gap': None}
        proposal['assessments'] = [supported, {**proposal['assessments'][0], 'requirement_id': 'R2'}]
        proposal['pending_impacts'][0]['blocking_requirement_ids'] = ['R2']
        proposal['known_findings'] = [{'requirement_id': 'R1', 'text': '资料建议核对客户端连接目标。',
                                      'evidence': deepcopy(supported['evidence'])}]
        generate = Mock()
        result = run_contextual_rag(question, CONTEXT, retrieve=lambda _: [PASSAGE],
            evaluate_raw=lambda _: json.dumps(proposal), generate_raw=generate)
        self.assertEqual(result['status'], 'needs_user')
        self.assertIn('资料建议核对客户端连接目标', result['message'])
        self.assertIn('数据库运行在哪里', result['message'])
        generate.assert_not_called()

    def test_blocker_and_covered_claim_conflict_is_rejected(self):
        proposal = reply()
        proposal['pending_impacts'][0]['blocking_requirement_ids'] = ['R1']
        result = self.evaluate(proposal)
        self.assertEqual(result['failure_reason'], 'blocked_requirement_marked_covered')
        self.assertIsNone(result['proposal'])

    def test_declared_runtime_blocker_cannot_trigger_document_retrieval(self):
        proposal = reply(B, blocked=True)
        proposal.update(action='retrieve', missing_user_information=[], missing_knowledge=['连接地址规则'],
                        search_query='database connection address')
        self.assertEqual(self.evaluate(proposal, B)['failure_reason'], 'pending_dependency_conflicts_with_action')

    def test_pending_mapping_must_account_for_each_known_pending_and_valid_requirement(self):
        for impacts, error in (
            ([], 'pending_impact_ids_mismatch'),
            ([{'pending_id': 'UNKNOWN', 'blocking_requirement_ids': [], 'reason': 'x'}], 'pending_impact_ids_mismatch'),
            ([{'pending_id': 'P_DB', 'blocking_requirement_ids': ['R9'], 'reason': 'x'}], 'invalid_blocking_requirement_ids'),
            ([{'pending_id': 'P_DB', 'blocking_requirement_ids': ['R1', 'R1'], 'reason': 'x'}], 'invalid_blocking_requirement_ids'),
        ):
            with self.subTest(error=error):
                proposal = reply()
                proposal['pending_impacts'] = impacts
                self.assertEqual(self.evaluate(proposal)['failure_reason'], error)

    def test_no_runtime_blocker_still_allows_a_knowledge_gap_and_supplementary_retrieval(self):
        first = reply()
        first['assessments'][0].update(coverage='partial', gap='一般机制的解释资料不全')
        first.update(action='retrieve', missing_knowledge=['一般机制'], search_query='general connection failure meaning')
        second = reply()
        calls = iter([json.dumps(first), json.dumps(second)])
        extra = {**deepcopy(PASSAGE), 'chunk_id': 'TEST:2', 'content': '不同的测试资料：补充一般机制。'}
        retrieve = Mock(side_effect=[[PASSAGE], [extra]])
        result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=lambda _: next(calls),
                                    generate_raw=lambda _: answer())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(result['counts']['additional_retrievals'], 1)
        self.assertEqual(retrieve.call_args_list[1].args[0], 'general connection failure meaning')

    def test_needed_user_fact_can_be_missing_without_existing_pending_record(self):
        proposal = reply(B, blocked=True)
        proposal['pending_impacts'] = []
        context = ContextState(3, CONTEXT.current_facts)
        self.assertEqual(self.evaluate(proposal, B, context)['status'], 'proposed')

    def test_explicitly_unavailable_runtime_fact_allows_insufficient_instead_of_repeated_request(self):
        question = B + '我无法取得数据库的位置或地址，也无法进一步提供。'
        proposal = reply(question, blocked=True)
        proposal.update(action='insufficient', missing_user_information=[],
                        missing_knowledge=['必要现场信息无法取得，无法确定具体地址'])
        result = self.evaluate(proposal, question)
        self.assertEqual(result['status'], 'proposed')
        self.assertEqual(result['proposal']['action'], 'insufficient')

    def test_timeout_is_failure_not_a_new_clarification(self):
        generate = Mock()
        result = run_contextual_rag(A, CONTEXT, retrieve=lambda _: [PASSAGE],
            evaluate_raw=Mock(side_effect=TimeoutError()), generate_raw=generate)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['required_information'], [])
        self.assertEqual(result['error']['code'], 'evaluation_timeout')
        generate.assert_not_called()

    def test_context_prompt_has_one_shape_and_does_not_require_old_question_to_supply_updated_facts(self):
        messages = build_requirement_messages(A, [PASSAGE], context=CONTEXT, include_user_reply=True)
        prompt = messages[0]['content']
        self.assertIn('"pending_impacts"', prompt)
        self.assertNotIn("Only facts the user explicitly states in the original question", prompt)
        answer_prompt = build_conversation_answer_messages(A, [PASSAGE], CONTEXT)
        self.assertNotIn('old-container-value', answer_prompt[1]['content'])
        self.assertNotIn('changes', answer_prompt[1]['content'])

    def test_invalid_context_or_dependencies_fail_before_retrieval(self):
        retrieve = Mock()
        with self.assertRaises(ValueError):
            run_contextual_rag(A, ContextState(), retrieve=retrieve, evaluate_raw=Mock(), generate_raw=Mock())
        with self.assertRaises(ValueError):
            run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=None, generate_raw=Mock())
        retrieve.assert_not_called()

    def test_structural_checks_do_not_prove_declared_irrelevance(self):
        # Intentionally wrong semantic proposal: code cannot prove that a generic
        # document suffices for a real deployment address when the model lies
        # about the dependency. Live/human review must check that decision.
        proposal = reply(B, blocked=False)
        self.assertEqual(self.evaluate(proposal, B)['status'], 'proposed')

    def test_public_finding_support_has_one_explicit_repair_without_changing_raw_failure(self):
        bad = reply(B, blocked=True)
        bad['known_findings'] = [{'requirement_id': 'R1', 'text': '资料只能帮助核对目标，不能确定当前地址。',
            'evidence': [{'chunk_id': 'TEST:1', 'quote': '它不提供当前用户的具体地址。'}]}]
        # The quote exists in the selected passage, but the assessment used a
        # different quote. Code must not silently rewrite either citation.
        fixed = deepcopy(bad)
        fixed['known_findings'][0]['evidence'] = deepcopy(fixed['assessments'][0]['evidence'])
        replies = iter([json.dumps(bad), json.dumps(fixed)])
        captured = []
        def generate(messages):
            captured.append(deepcopy(messages))
            return next(replies)
        result = evaluate_requirements(B, [PASSAGE], generate, context=CONTEXT,
            include_user_reply=True, repair_finding_support_once=True)
        self.assertEqual(result['status'], 'proposed')
        self.assertEqual(result['repair_reason'], 'finding_support_not_in_assessment')
        self.assertEqual(result['model_attempts'], 2)
        self.assertEqual(json.loads(result['raw_attempts'][0]), bad)
        self.assertIn('permitted_finding_support', captured[1][-1]['content'])
        self.assertEqual(result['proposal']['blocking_pending_ids'], ['P_DB'])

    def test_public_finding_repair_stops_if_same_error_or_timeout_remains(self):
        bad = reply(B, blocked=True)
        bad['known_findings'] = [{'requirement_id': 'R1', 'text': '资料不能确定当前地址。',
            'evidence': [{'chunk_id': 'TEST:1', 'quote': '它不提供当前用户的具体地址。'}]}]
        for second, failure in ((json.dumps(bad), 'finding_support_not_in_assessment'),
                                (TimeoutError(), 'evaluation_timeout')):
            callback = Mock(side_effect=[json.dumps(bad), second])
            result = evaluate_requirements(B, [PASSAGE], callback, context=CONTEXT,
                include_user_reply=True, repair_finding_support_once=True)
            self.assertEqual(result['status'], 'evaluation_failed')
            self.assertEqual(result['failure_reason'], failure)
            self.assertEqual(callback.call_count, 2)


    def test_assessment_quote_repair_uses_same_evidence_without_additional_retrieval(self):
        bad = reply()
        bad['assessments'][0]['evidence'][0]['quote'] = '这不是原片段中的连续原文。'
        callback = Mock(side_effect=[json.dumps(bad), json.dumps(reply())])
        retrieve = Mock(return_value=[PASSAGE])
        result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=callback,
                                    generate_raw=lambda _: answer())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(result['counts']['additional_retrievals'], 0)
        first, second = (c.args[0] for c in callback.call_args_list)
        self.assertEqual(first, second[:2])
        self.assertEqual(json.loads(second[2]['content']), bad)
        trace = next(s for s in result['trace'] if s['step']=='evaluate')
        self.assertEqual(trace['model_attempts'], 2)
        self.assertEqual(trace['repair_reason'], 'support_quote_not_in_passage')

    def test_assessment_quote_repair_does_not_relax_validation_or_retry_service_failure(self):
        bad = reply()
        bad['assessments'][0]['evidence'][0]['quote'] = '这不是原片段中的连续原文。'
        for second, expected in ((json.dumps(bad), 'support_quote_not_in_passage'),
                                 (TimeoutError(), 'evaluation_timeout')):
            callback = Mock(side_effect=[json.dumps(bad), second])
            generated = Mock()
            result = run_contextual_rag(A, CONTEXT, retrieve=lambda _: [PASSAGE],
                evaluate_raw=callback, generate_raw=generated)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error']['code'], expected)
            self.assertEqual(callback.call_count, 2)
            self.assertEqual(result['counts']['additional_retrievals'], 0)
            generated.assert_not_called()


class EvaluationShapeRepairTests(unittest.TestCase):
    def test_exact_id_typography_is_normalized_without_model_retry_and_raw_is_kept(self):
        raw = answer().replace('[TEST:1]', '[“TEST:1”]')
        generate = Mock(return_value=raw)
        result = run_contextual_rag(A, CONTEXT, retrieve=lambda _: [PASSAGE],
            evaluate_raw=lambda _: json.dumps(reply()), generate_raw=generate)
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result['generation_checks'][0]['raw_text'], raw)
        self.assertIn('[TEST:1]', result['answer']['answer'])

    def test_conversation_entry_repairs_unknown_selection_without_adding_candidates(self):
        bad = reply()
        bad['selected_chunk_ids'] = ['NOT-A-CANDIDATE']
        callback = Mock(side_effect=[json.dumps(bad), json.dumps(reply())])
        retrieve = Mock(return_value=[PASSAGE])
        result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=callback,
                                    generate_raw=lambda _: answer())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(result['counts']['additional_retrievals'], 0)
        first, second = (call.args[0] for call in callback.call_args_list)
        self.assertEqual(first, second[:2])
        self.assertIn('NOT-A-CANDIDATE', second[-1]['content'])
        trace = next(s for s in result['trace'] if s['step'] == 'evaluate')
        self.assertEqual(trace['repair_reason'], 'unknown_selected_chunk_id')

    def test_missing_field_is_repaired_on_same_inputs_without_another_retrieval(self):
        bad = reply()
        del bad['selected_chunk_ids']
        callback = Mock(side_effect=[json.dumps(bad), json.dumps(reply())])
        retrieve = Mock(return_value=[PASSAGE])
        result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=callback,
                                    generate_raw=lambda _: answer())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(result['counts']['additional_retrievals'], 0)
        first, second = (call.args[0] for call in callback.call_args_list)
        self.assertEqual(first, second[:2])
        self.assertEqual(json.loads(second[2]['content']), bad)
        self.assertIn('"missing_fields": ["selected_chunk_ids"]', second[3]['content'])
        trace = next(s for s in result['trace'] if s['step'] == 'evaluate')
        self.assertEqual(trace['repair_reason'], 'unexpected_requirement_evaluation_fields')
        self.assertEqual(trace['model_attempts'], 2)

    def test_malformed_json_can_be_repaired_and_original_is_retained(self):
        callback = Mock(side_effect=['{"requirements":', json.dumps(reply())])
        result = evaluate_requirements(A, [PASSAGE], callback, context=CONTEXT,
            include_user_reply=True, repair_response_shape_once=True)
        self.assertEqual(result['status'], 'proposed')
        self.assertEqual(result['raw_attempts'][0], '{"requirements":')
        self.assertEqual(result['repair_reason'], 'response_is_not_json')
        self.assertFalse(result['coverage_verified'])

    def test_nested_record_fields_use_the_same_single_correction(self):
        extra_requirement = reply()
        extra_requirement['requirements'][0]['assessments'] = None
        missing_requirement = reply()
        del missing_requirement['requirements'][0]['question_quote']
        extra_assessment = reply()
        extra_assessment['assessments'][0]['requirements'] = None
        for bad, code in ((extra_requirement, 'invalid_requirement'),
                          (missing_requirement, 'invalid_requirement'),
                          (extra_assessment, 'invalid_assessment')):
            with self.subTest(code=code):
                callback = Mock(side_effect=[json.dumps(bad), json.dumps(reply())])
                retrieve = Mock(return_value=[PASSAGE])
                result = run_contextual_rag(A, CONTEXT, retrieve=retrieve, evaluate_raw=callback,
                                            generate_raw=lambda _: answer())
                self.assertEqual(result['status'], 'answer')
                self.assertEqual(callback.call_count, 2)
                self.assertEqual(retrieve.call_count, 1)
                first, second = (c.args[0] for c in callback.call_args_list)
                self.assertEqual(first, second[:2])
                self.assertEqual(json.loads(second[2]['content']), bad)
                self.assertIn('required_record_fields', second[-1]['content'])
                self.assertEqual(next(s for s in result['trace'] if s['step'] == 'evaluate')['repair_reason'], code)
        callback = Mock(side_effect=[json.dumps(extra_requirement), json.dumps(extra_requirement), json.dumps(reply())])
        generated = Mock()
        result = run_contextual_rag(A, CONTEXT, retrieve=lambda _: [PASSAGE],
            evaluate_raw=callback, generate_raw=generated)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(callback.call_count, 2)
        generated.assert_not_called()

    def test_shape_and_quote_repairs_share_one_opportunity(self):
        missing = reply()
        del missing['selected_chunk_ids']
        wrong_quote = reply()
        wrong_quote['assessments'][0]['evidence'][0]['quote'] = 'not in passage'
        for first, second, expected in (
            (missing, wrong_quote, 'support_quote_not_in_passage'),
            (wrong_quote, missing, 'unexpected_requirement_evaluation_fields'),
            (missing, missing, 'unexpected_requirement_evaluation_fields'),
        ):
            with self.subTest(expected=expected):
                callback = Mock(side_effect=[json.dumps(first), json.dumps(second), json.dumps(reply())])
                generated = Mock()
                result = run_contextual_rag(A, CONTEXT, retrieve=lambda _: [PASSAGE],
                    evaluate_raw=callback, generate_raw=generated)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['error']['code'], expected)
                self.assertEqual(callback.call_count, 2)
                generated.assert_not_called()

    def test_provider_timeout_is_not_a_shape_correction(self):
        bad = reply()
        del bad['selected_chunk_ids']
        for outputs, count in (([TimeoutError()], 1), ([json.dumps(bad), TimeoutError()], 2)):
            callback = Mock(side_effect=outputs)
            result = evaluate_requirements(A, [PASSAGE], callback, context=CONTEXT,
                include_user_reply=True, repair_response_shape_once=True)
            self.assertEqual(result['status'], 'evaluation_failed')
            self.assertEqual(result['failure_reason'], 'evaluation_timeout')
            self.assertEqual(callback.call_count, count)

    def test_shape_correction_is_opt_in_and_never_fills_fields_in_code(self):
        bad = reply()
        del bad['selected_chunk_ids']
        callback = Mock(return_value=json.dumps(bad))
        result = evaluate_requirements(A, [PASSAGE], callback, context=CONTEXT, include_user_reply=True)
        self.assertEqual(callback.call_count, 1)
        self.assertEqual(result['failure_reason'], 'unexpected_requirement_evaluation_fields')
        self.assertIsNone(result['proposal'])
        self.assertNotIn('selected_chunk_ids', json.loads(result['raw_attempts'][0]))


if __name__ == '__main__':
    unittest.main()
