"""One bounded proposal correction, same authorized inputs, no service retries."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock
from dataclasses import asdict

from app.conversation_context import ContextState,UserMessage
from app.conversation_service import deployment_registry
from app.conversation_preparation import prepare_task_context
from app.task_proposal import PreparedTaskTurn
from app.task_constraints import ConstraintSnapshot
from app.analysis_preparation import resolve_analysis_request


class ConstraintPreparationTests(unittest.TestCase):
    def setUp(self):
        self.text='只解释通用机制，不再诊断刚才那次故障。'
        self.message=UserMessage('M1',1,self.text)
        self.task=PreparedTaskTurn({'operation':'create','goal':'理解机制','source_quote':self.text},None,'理解机制',self.text)
        source={'message_id':'M1','quote':self.text}
        self.valid={'updates':[],'analysis_request':{'mode':'current','source':source,'confirmation_from_input_version':None},
            'evidence_request':{'incident':{'operation':'general','incident_id':None,'description':None,'question':None,
                'candidate_incident_ids':[],'source':source},'resolves_input_version':None,'updates':[]}}

    def run_prepare(self,callback):
        return prepare_task_context(ContextState(),self.message,self.task,registry=deployment_registry('C1','M1'),
            generate=callback,include_analysis=True,constraint_view={'constraint_state':ConstraintSnapshot(),
                'incident_catalog':[],'pending_incidents':[]})

    def test_paraphrased_source_gets_one_correction_and_raw_is_retained(self):
        bad=deepcopy(self.valid);bad['analysis_request']['source']={'message_id':'M1','quote':'补问原理'}
        callback=Mock(side_effect=[json.dumps(bad),json.dumps(self.valid)])
        result=self.run_prepare(callback)
        self.assertEqual(result.validation_checks[0]['validation_error'],'analysis_source_not_authorized')
        self.assertEqual(json.loads(result.validation_checks[0]['raw_text']),bad)
        first,second=(c.args[0] for c in callback.call_args_list)
        self.assertEqual(first,second[:2]);self.assertEqual(callback.call_count,2)
        self.assertEqual(result.evidence_request,self.valid['evidence_request'])

    def test_replace_on_a_new_target_is_corrected_by_model_not_silently_converted(self):
        self.message = UserMessage('M1', 1, '测试应用在容器。' + self.text)
        registry = deployment_registry('C1', 'M1')
        target = next(r.target for r in registry if r.target.field == 'runtime_location'
                      and r.target.entity_id == 'application')
        valid = deepcopy(self.valid)
        valid['updates'] = [{'operation': 'add', 'target': asdict(target), 'previous_fact_id': None,
                            'value': 'container', 'source': {'message_id': 'M1', 'quote': '测试应用在容器'}}]
        bad = deepcopy(valid);bad['updates'][0]['operation'] = 'replace'
        callback = Mock(side_effect=[json.dumps(bad), json.dumps(valid)])
        result = self.run_prepare(callback)
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(result.validation_checks[0]['validation_error'], 'previous_fact_does_not_match_target')
        self.assertEqual(json.loads(result.validation_checks[0]['raw_text']), bad)
        self.assertEqual(result.proposal, {'updates': valid['updates']})
        callback = Mock(side_effect=[json.dumps(bad), json.dumps(bad), json.dumps(valid)])
        with self.assertRaisesRegex(ValueError, 'previous_fact_does_not_match_target'):
            self.run_prepare(callback)
        self.assertEqual(callback.call_count, 2)

    def test_repeated_invalid_output_and_provider_timeout_stop(self):
        bad=deepcopy(self.valid);del bad['evidence_request']
        for outputs,error,count in (([json.dumps(bad),json.dumps(bad)],ValueError,2),
                                    ([TimeoutError()],TimeoutError,1),
                                    ([json.dumps(bad),TimeoutError()],TimeoutError,2)):
            with self.subTest(count=count):
                callback=Mock(side_effect=outputs)
                with self.assertRaises(error):self.run_prepare(callback)
                self.assertEqual(callback.call_count,count)

    def test_repeated_hypothesis_and_field_errors_share_one_correction(self):
        registry = deployment_registry('C1', 'M1')
        target = next(r.target for r in registry if r.target.entity_id == 'application'
                      and r.target.field == 'runtime_location')
        m1 = UserMessage('M1', 1, '如果应用在容器里。')
        previous = resolve_analysis_request({'mode': 'hypothetical',
            'source': {'message_id': 'M1', 'quote': m1.content}, 'assumptions': [
                {'target': asdict(target), 'value': 'container',
                 'source': {'message_id': 'M1', 'quote': m1.content}}]}, m1,
            allowed_targets=frozenset(r.target for r in registry))
        m2 = UserMessage('M2', 2, '只分析刚才的假设场景。服务名未变化。')
        source = {'message_id': 'M2', 'quote': m2.content}
        valid = deepcopy(self.valid)
        valid['analysis_request'] = {'mode': 'inherit', 'source': source,
            'from_input_version': 1, 'assumption_updates': []}
        valid['evidence_request']['incident']['source'] = source
        repeated = deepcopy(valid)
        repeated['analysis_request'] = {'mode': 'hypothetical', 'source': source, 'assumptions': [
            {'target': asdict(target), 'value': 'container', 'source': source}]}
        bad_field = deepcopy(valid)
        db_target = next(r.target for r in registry if r.target.field == 'service_name')
        bad_field['updates'] = [{'operation': 'add', 'target': asdict(db_target),
            'previous_fact_id': None, 'value': '未变化', 'source': source}]
        task = PreparedTaskTurn({'operation': 'select', 'task_id': 'T1', 'source_quote': m2.content},
            None, '理解机制', m2.content)
        def prepare(callback):
            return prepare_task_context(ContextState(1), m2, task, registry=registry, generate=callback,
                include_analysis=True, analysis_history=({'input_version': 1, 'analysis_scope': previous},),
                constraint_view={'constraint_state': ConstraintSnapshot(1), 'incident_catalog': [], 'pending_incidents': []})
        callback = Mock(side_effect=[json.dumps(repeated), json.dumps(valid)])
        result = prepare(callback)
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(result.validation_checks[0]['validation_error'], 'repeated_hypothesis_requires_inheritance')
        self.assertEqual(result.analysis_request, valid['analysis_request'])
        callback = Mock(side_effect=[json.dumps(bad_field), json.dumps(repeated), json.dumps(valid)])
        with self.assertRaisesRegex(ValueError, 'requires_inheritance'):
            prepare(callback)
        self.assertEqual(callback.call_count, 2)  # no fresh repair allowance per error type

