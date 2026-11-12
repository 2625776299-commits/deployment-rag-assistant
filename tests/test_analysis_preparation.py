"""Provenance and condition inheritance, with canonical scripted inputs."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from app.analysis_preparation import resolve_analysis_request, guard_hypothetical_context_sources
from app.conversation_context import FactTarget, UserMessage


APP = FactTarget('application', 'T1', 'runtime_location')
NETWORK = FactTarget('application', 'T1', 'network_relation')
TARGETS = frozenset((APP, NETWORK))
M1 = UserMessage('M1', 1, '先假设应用在容器内，且在同一Compose网络。')
M2 = UserMessage('M2', 2, '那连接失败怎么办？')


def source(m):
    return {'message_id': m.message_id, 'quote': m.content}


def assumption(target, value, message=M1):
    return {'target': asdict(target), 'value': value, 'source': source(message)}


def initial():
    return {'mode': 'hypothetical', 'source': source(M1),
            'assumptions': [assumption(APP, 'container'), assumption(NETWORK, 'same_compose_network')]}


class AnalysisPreparationTests(unittest.TestCase):
    def history(self):
        scope = resolve_analysis_request(initial(), M1, allowed_targets=TARGETS)
        return [{'input_version': 1, 'analysis_scope': scope}]

    def test_inheritance_reads_original_conditions_and_does_not_rewrite_their_source(self):
        history = self.history()
        before = deepcopy(history)
        scope = resolve_analysis_request({'mode': 'inherit', 'source': source(M2),
            'from_input_version': 1, 'assumption_updates': []}, M2, allowed_targets=TARGETS, history=history)
        self.assertEqual(scope['assumptions'], history[0]['analysis_scope']['assumptions'])
        self.assertEqual(scope['inherited_from_input_version'], 1)
        self.assertEqual(scope['assumptions'][0]['source']['message_id'], 'M1')
        self.assertEqual(history, before)

    def test_changed_and_removed_conditions_keep_separate_current_sources(self):
        message = UserMessage('M2', 2, '改为假设应用在宿主机，不再假设同一Compose网络。')
        scope = resolve_analysis_request({'mode': 'inherit', 'source': source(message), 'from_input_version': 1,
            'assumption_updates': [assumption(APP, 'host', message), assumption(NETWORK, None, message)]},
            message, allowed_targets=TARGETS, history=self.history())
        self.assertEqual(scope['assumptions'], [assumption(APP, 'host', message)])
        self.assertEqual(self.history()[0]['analysis_scope']['assumptions'][0]['value'], 'container')

    def test_repeated_e05_conditions_require_an_explicit_authorized_version(self):
        # Reproduce E05/M3: a vague anaphoric quote is real text, but not a new
        # origin for the same complete set of conditions.
        m = UserMessage('M2', 2, '只分析刚才的假设场景。')
        request = {'mode': 'hypothetical', 'source': source(m),
                   'assumptions': [assumption(APP, 'container', m),
                                   assumption(NETWORK, 'same_compose_network', m)]}
        before = deepcopy(self.history())
        with self.assertRaisesRegex(ValueError, 'requires_inheritance'):
            resolve_analysis_request(request, m, allowed_targets=TARGETS, history=before)
        scope = resolve_analysis_request({'mode': 'inherit', 'source': source(m),
            'from_input_version': 1, 'assumption_updates': [assumption(APP, 'container', m)]},
            m, allowed_targets=TARGETS, history=before)
        self.assertEqual(scope['assumptions'], before[0]['analysis_scope']['assumptions'])
        self.assertEqual(scope['assumptions'][0]['source']['message_id'], 'M1')
        self.assertEqual(scope['inherited_from_input_version'], 1)
        self.assertEqual(before, self.history())

    def test_different_new_scenario_is_not_silently_merged_with_older_conditions(self):
        m = UserMessage('M2', 2, '另一个假设：应用在宿主机，不假设任何网络关系。')
        scope = resolve_analysis_request({'mode': 'hypothetical', 'source': source(m),
            'assumptions': [assumption(APP, 'host', m)]}, m, allowed_targets=TARGETS, history=self.history())
        self.assertEqual(scope['assumptions'], [assumption(APP, 'host', m)])
        self.assertIsNone(scope['inherited_from_input_version'])

    def test_unknown_future_current_or_unregistered_references_fail(self):
        for version in (0, True, 2, 99):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolve_analysis_request({'mode': 'inherit', 'source': source(M2),
                    'from_input_version': version, 'assumption_updates': []}, M2,
                    allowed_targets=TARGETS, history=self.history())
        request = {'mode': 'inherit', 'source': source(M2), 'from_input_version': 1, 'assumption_updates': []}
        with self.assertRaisesRegex(ValueError, 'selected_task'):
            resolve_analysis_request(request, M2, allowed_targets=TARGETS, history=[])
        current_history = [{'input_version': 1, 'analysis_scope': {'mode': 'current', 'assumptions': [],
                                                                  'inherited_from_input_version': None}}]
        self.assertEqual(resolve_analysis_request(request, M2, allowed_targets=TARGETS,
                         history=current_history)['mode'], 'current')
        with self.assertRaisesRegex(ValueError, 'not_hypothetical'):
            resolve_analysis_request({'mode':'current','source':source(M2),'confirmation_from_input_version':1},
                                     M2, allowed_targets=TARGETS, history=current_history)
        request['assumption_updates'] = [assumption(APP, 'container', M2)]
        with self.assertRaisesRegex(ValueError, 'cannot_modify_assumptions'):
            resolve_analysis_request(request, M2, allowed_targets=TARGETS, history=current_history)
        with self.assertRaisesRegex(ValueError, 'not_registered'):
            resolve_analysis_request(initial(), M1, allowed_targets=frozenset((APP,)))

    def test_old_or_fabricated_quotes_cannot_be_used_as_a_new_hypothesis(self):
        for changed in ('quote', 'message_id'):
            request = initial()
            request['assumptions'][0]['source'][changed] = 'fabricated'
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'source_not_authorized'):
                resolve_analysis_request(request, M1, allowed_targets=TARGETS)
        with self.assertRaisesRegex(ValueError, 'source_not_authorized'):
            resolve_analysis_request(initial(), M2, allowed_targets=TARGETS, history=self.history())

    def test_confirmation_records_a_current_source_and_validates_the_referenced_plan(self):
        m = UserMessage('M2', 2, '已按刚才唯一方案迁移完成。')
        request = {'mode': 'current', 'source': source(m), 'confirmation_from_input_version': 1}
        result = resolve_analysis_request(request, m, allowed_targets=TARGETS, history=self.history())
        self.assertEqual(result['mode'], 'current')
        self.assertEqual(result['assumptions'], [])
        with self.assertRaisesRegex(ValueError, 'selected_task'):
            resolve_analysis_request(request, m, allowed_targets=TARGETS, history=[])

    def test_same_hypothetical_clause_cannot_also_assert_a_real_update(self):
        scope = self.history()[0]['analysis_scope']
        update = {'target': asdict(APP), 'source': source(M1)}
        with self.assertRaisesRegex(ValueError, 'cannot_update_actual'):
            guard_hypothetical_context_sources({'updates': [update]}, scope)
        # A hypothetical clause cannot establish real state for another field either.
        update['target'] = {'entity_id': 'database', 'scope_id': 'T1', 'field': 'runtime_location'}
        with self.assertRaisesRegex(ValueError, 'cannot_update_actual'):
            guard_hypothetical_context_sources({'updates': [update]}, scope)
        update['source'] = source(M2)
        guard_hypothetical_context_sources({'updates': [update]}, scope)

    def test_ambiguity_produces_no_executable_hypothesis(self):
        result = resolve_analysis_request({'mode': 'clarify', 'source': source(M2),
            'question': '你问实际部署还是刚才的假设？'}, M2, allowed_targets=TARGETS)
        self.assertEqual(result['mode'], 'needs_clarification')
        self.assertNotIn('assumptions', result)


if __name__ == '__main__':
    unittest.main()
