"""Offline adapter checks; the injected responses are not model quality evidence."""
from dataclasses import asdict
import json
import unittest

from app.context_proposal import (
    DialogueMessage, RegisteredTarget, build_context_update_messages, prepare_context,
)
from app.conversation_context import ContextState, FactTarget, SourceQuote, UserFact, UserMessage


TARGET = FactTarget('app', 'incident', 'runtime_location')
REGISTRY = (RegisteredTarget(TARGET, '当前故障应用的运行位置'),)


class ContextProposalTests(unittest.TestCase):
    def setUp(self):
        self.fact = UserFact('F1', TARGET, 'container', SourceQuote('M1', '应用在容器'))
        self.state = ContextState(1, (self.fact,))
        self.message = UserMessage('M2', 2, '更正一下，应用在宿主机。')
        self.patch = {'updates': [{'operation': 'replace', 'target': asdict(TARGET),
                     'previous_fact_id': 'F1', 'value': 'host',
                     'source': {'message_id': 'M2', 'quote': '应用在宿主机'}}]}

    def test_input_has_exact_roles_sources_registry_and_no_changes_or_test_labels(self):
        dialogue = (DialogueMessage('M1', 'user', '应用在容器'),
                    DialogueMessage('A1', 'assistant', '应用是在宿主机吗？'))
        messages = build_context_update_messages(self.state, self.message, registry=REGISTRY,
                                                  recent_dialogue=dialogue)
        payload = json.loads(messages[1]['content'])
        self.assertEqual(set(payload), {'input_version', 'current_user_message', 'registered_targets',
                         'current_user_stated_facts', 'pending_clarifications', 'recent_dialogue'})
        self.assertEqual(payload['current_user_stated_facts'][0]['fact_id'], 'F1')
        self.assertEqual(payload['recent_dialogue'][1]['role'], 'assistant')
        self.assertEqual(payload['current_user_message']['message_id'], 'M2')

    def test_valid_model_patch_applies_and_keeps_original_state_unchanged(self):
        before = self.state.to_dict()
        captured = []
        def generate(messages):
            captured.append(messages)
            return json.dumps(self.patch)
        result = prepare_context(self.state, self.message, registry=REGISTRY, generate=generate)
        self.assertEqual(len(captured), 1)
        self.assertEqual(result.proposal, self.patch)
        self.assertEqual(result.state.current_facts[0].value, 'host')
        self.assertEqual(self.state.to_dict(), before)

    def test_failure_never_becomes_successful_empty_update(self):
        def timed_out(messages):
            raise TimeoutError('model_timeout')
        with self.assertRaises(TimeoutError):
            prepare_context(self.state, self.message, registry=REGISTRY, generate=timed_out)
        for raw in ('not-json', '{"updates":null}', '{"updates":[{"operation":"replace"}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare_context(self.state, self.message, registry=REGISTRY, generate=lambda _: raw)
        self.assertEqual(self.state.input_version, 1)

    def test_empty_update_is_distinct_from_model_failure(self):
        result = prepare_context(self.state, self.message, registry=REGISTRY,
                                 generate=lambda _: '{"updates":[]}')
        self.assertEqual(result.state.input_version, 2)
        self.assertEqual(result.state.current_facts, self.state.current_facts)

    def test_invalid_inputs_rejected_before_model_call(self):
        called = []
        def generate(messages):
            called.append(messages)
            return '{"updates":[]}'
        with self.assertRaisesRegex(ValueError, 'message_version_not_next'):
            prepare_context(self.state, UserMessage('M4', 4, 'next'), registry=REGISTRY, generate=generate)
        with self.assertRaisesRegex(ValueError, 'duplicate_registered_target'):
            prepare_context(self.state, self.message, registry=REGISTRY * 2, generate=generate)
        with self.assertRaisesRegex(ValueError, 'duplicate_dialogue_message'):
            prepare_context(self.state, self.message, registry=REGISTRY, generate=generate,
                            recent_dialogue=(DialogueMessage('M2', 'user', '重复当前消息'),))
        self.assertEqual(called, [])

    def test_source_must_still_be_current_user_message_despite_history(self):
        self.patch['updates'][0]['source'] = {'message_id': 'A1', 'quote': '应用在宿主机'}
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
            prepare_context(self.state, self.message, registry=REGISTRY,
                            generate=lambda _: json.dumps(self.patch),
                            recent_dialogue=(DialogueMessage('A1', 'assistant', '应用在宿主机'),))
