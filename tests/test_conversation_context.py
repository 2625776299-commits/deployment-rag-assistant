"""Deterministic patch semantics; fixtures are not claims of model comprehension."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest

from app.conversation_context import (
    ContextState, FactTarget, UserMessage, apply_context_updates,
    build_reasoning_context, parse_context_updates, restore_context,
)


APP = FactTarget('app_1', 'incident_environment', 'runtime_location')
ERROR = FactTarget('app_1', 'incident_environment', 'error_text')
DB = FactTarget('db_1', 'incident_environment', 'runtime_location')
LOCAL_APP = FactTarget('app_1', 'local_test', 'runtime_location')
TARGETS = frozenset((APP, ERROR, DB, LOCAL_APP))


def update(operation, target, value, message_id, quote, previous=None):
    return {'operation': operation, 'target': asdict(target), 'value': value,
            'previous_fact_id': previous, 'source': {'message_id': message_id, 'quote': quote}}


def apply(state, message, *updates):
    return apply_context_updates(state, message, {'updates': list(updates)}, allowed_targets=TARGETS)


def mark_pending(target, message, previous=None, reason='ambiguous_target'):
    return {'operation': 'mark_pending', 'target': asdict(target), 'previous_fact_id': previous,
            'reason': reason, 'source': {'message_id': message.message_id, 'quote': message.content}}


def resolve(item, value, message):
    return {'operation': 'resolve', 'target': asdict(item.target), 'pending_id': item.pending_id,
            'value': value, 'source': {'message_id': message.message_id, 'quote': message.content}}


class ConversationContextTests(unittest.TestCase):
    def setUp(self):
        self.initial = apply(ContextState(), UserMessage('M1', 1, '应用在容器，报错 Connection refused。数据库在容器。'),
            update('add', APP, 'container', 'M1', '应用在容器'),
            update('add', ERROR, 'Connection refused', 'M1', 'Connection refused'),
            update('add', DB, 'container', 'M1', '数据库在容器'))
        self.old = next(f for f in self.initial.current_facts if f.target == APP)
        self.message = UserMessage('M2', 2, '刚才说错了，应用在宿主机。')
        self.correction = update('replace', APP, 'host', 'M2', '应用在宿主机', self.old.fact_id)

    def ambiguous_state(self):
        msg = UserMessage('M2', 2, '前面说错了，有一个其实在宿主机，但我没记清是应用还是数据库。')
        return apply(self.initial, msg, *(
            mark_pending(f.target, msg, f.fact_id)
            for f in self.initial.current_facts if f.target in (APP, DB)
        ))

    def test_stored_snapshot_restores_pending_history_ids_and_partial_resolution(self):
        ambiguous = self.ambiguous_state()
        message = UserMessage('M3', 3, '应用在宿主机，数据库还不确定。')
        item = next(p for p in ambiguous.pending_facts if p.target == APP)
        state = apply(ambiguous, message, resolve(item, 'host', message))
        restored = restore_context(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored, state)
        self.assertEqual(restored.pending_facts[0].target, DB)
        self.assertEqual(restored.pending_facts[0].previous_fact.value, 'container')
        self.assertEqual(restore_context(ContextState().to_dict()), ContextState())

    def test_corrupt_snapshots_are_rejected_instead_of_resetting_context(self):
        snapshot = json.loads(json.dumps(self.initial.to_dict()))
        null_fact = deepcopy(snapshot)
        null_fact['current_facts'][0] = None
        duplicate = deepcopy(snapshot)
        duplicate['current_facts'].append(duplicate['current_facts'][0])
        wrong_source = deepcopy(snapshot)
        wrong_source['current_facts'][0]['source'] = {'message_id': 'M1'}
        for broken in (None, {}, {**snapshot, 'input_version': True},
                       {**snapshot, 'input_version': 0}, null_fact, duplicate, wrong_source):
            with self.subTest(snapshot=broken), self.assertRaises(ValueError):
                restore_context(broken)

    def test_correction_preserves_other_facts_and_keeps_before_after_sources(self):
        before = self.initial.to_dict()
        result = apply(self.initial, self.message, self.correction)
        values = {f.target: f for f in result.current_facts}
        self.assertEqual(values[APP].value, 'host')
        self.assertNotEqual(values[APP].fact_id, self.old.fact_id)
        for fact in self.initial.current_facts:
            if fact.target != APP:
                self.assertEqual(values[fact.target], fact)
        change = result.changes[-1]
        self.assertEqual(change.before, self.old)
        self.assertEqual(change.after, values[APP])
        self.assertEqual(change.source.message_id, 'M2')
        self.assertEqual(self.initial.to_dict(), before)
        self.assertEqual(result.input_version, 2)
        json.dumps(result.to_dict(), ensure_ascii=False)  # Snapshot is persistable.

    def test_different_environment_is_added_without_replacing_incident(self):
        msg = UserMessage('M2', 2, '本地测试的应用在宿主机。')
        result = apply(self.initial, msg, update('add', LOCAL_APP, 'host', 'M2', msg.content))
        values = {f.target: f.value for f in result.current_facts}
        self.assertEqual(values[APP], 'container')
        self.assertEqual(values[LOCAL_APP], 'host')

    def test_old_fact_id_cannot_modify_another_entity_or_environment(self):
        for target in (DB, LOCAL_APP):
            with self.subTest(target=target):
                wrong = {**self.correction, 'target': asdict(target)}
                with self.assertRaisesRegex(ValueError, 'previous_fact_does_not_match_target'):
                    apply(self.initial, self.message, wrong)

    def test_stale_fact_and_unregistered_field_are_rejected(self):
        for wrong, reason in (
            ({**self.correction, 'previous_fact_id': 'not-current'}, 'previous_fact_does_not_match_target'),
            ({**self.correction, 'target': {**asdict(APP), 'field': 'is_admin'}}, 'unregistered_target'),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                apply(self.initial, self.message, wrong)

    def test_missing_fake_or_historical_quote_is_rejected(self):
        for source in ({'message_id': 'M1', 'quote': '应用在容器'},
                       {'message_id': 'M2', 'quote': '数据库在宿主机'},
                       {'message_id': 'M2', 'quote': ''}):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
                apply(self.initial, self.message, {**self.correction, 'source': source})

    def test_invalid_second_change_does_not_partially_modify_input(self):
        before = deepcopy(self.initial.to_dict())
        bad = update('replace', DB, 'host', 'M2', '应用在宿主机', 'wrong-id')
        with self.assertRaises(ValueError):
            apply(self.initial, self.message, self.correction, bad)
        self.assertEqual(self.initial.to_dict(), before)

    def test_retract_removes_current_fact_without_inventing_opposite(self):
        msg = UserMessage('M2', 2, '之前说应用在容器不准确，实际运行位置我不清楚。')
        change = update('retract', APP, None, 'M2', msg.content, self.old.fact_id)
        result = apply(self.initial, msg, change)
        self.assertNotIn(APP, {f.target for f in result.current_facts})
        self.assertEqual(result.changes[-1].before, self.old)
        self.assertIsNone(result.changes[-1].after)
        self.assertEqual(len(result.current_facts), 2)

    def test_empty_patch_keeps_facts_and_advances_version_but_skipped_message_is_rejected(self):
        result = apply(self.initial, self.message)
        self.assertEqual(result.current_facts, self.initial.current_facts)
        self.assertEqual(result.input_version, 2)
        for version in (1, 3):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'message_version_not_next'):
                apply(self.initial, UserMessage('Mx', version, 'some text'))

    def test_duplicate_updates_and_implicit_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple_updates_for_same_target'):
            apply(self.initial, self.message, self.correction, self.correction)
        with self.assertRaisesRegex(ValueError, 'add_cannot_overwrite_current_fact'):
            apply(self.initial, self.message, update('add', APP, 'host', 'M2', '应用在宿主机'))

    def test_structural_acceptance_does_not_prove_interpretation(self):
        # Intentionally wrong interpretation of a real quote. This check records
        # the boundary honestly; target + quote matching cannot verify meaning.
        wrong = {**self.correction, 'value': 'container'}
        result = apply(self.initial, self.message, wrong)
        self.assertEqual(next(f for f in result.current_facts if f.target == APP).value, 'container')

    def test_raw_proposal_shape_is_strict(self):
        proposal = {'updates': [self.correction]}
        self.assertEqual(parse_context_updates(json.dumps(proposal)), proposal)
        for raw in ('not-json', '[]', '{"updates":[],"diagnosis":"network"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_context_updates(raw)

    def test_ambiguity_suspends_only_locations_and_does_not_project_old_values(self):
        before = self.initial.to_dict()
        result = self.ambiguous_state()
        self.assertEqual([f.target for f in result.current_facts], [ERROR])
        self.assertEqual(result.current_facts[0].value, 'Connection refused')
        self.assertEqual({p.target for p in result.pending_facts}, {APP, DB})
        for item in result.pending_facts:
            self.assertEqual(item.previous_fact.value, 'container')
            self.assertEqual(item.previous_fact.source.message_id, 'M1')
            self.assertEqual(item.source.message_id, 'M2')
        self.assertEqual([e.operation for e in result.changes[-2:]], ['mark_pending'] * 2)
        projected = build_reasoning_context(result)
        self.assertEqual(len(projected['user_stated_facts']), 1)
        self.assertEqual(len(projected['pending_clarifications']), 2)
        self.assertNotIn('container', json.dumps(projected))
        self.assertNotIn('previous_fact', json.dumps(projected))
        self.assertNotIn('quote', json.dumps(projected))
        self.assertEqual(self.initial.to_dict(), before)
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_empty_update_does_not_clear_pending_or_restore_old_values(self):
        pending = self.ambiguous_state()
        result = apply(pending, UserMessage('M3', 3, '先等一下，我去确认。'))
        self.assertEqual(result.pending_facts, pending.pending_facts)
        self.assertEqual(result.current_facts, pending.current_facts)
        self.assertEqual(result.input_version, 3)

    def test_resolving_app_keeps_database_pending(self):
        pending = self.ambiguous_state()
        items = {p.target: p for p in pending.pending_facts}
        msg = UserMessage('M3', 3, '我确认了，应用在宿主机；数据库的位置还不确定。')
        result = apply(pending, msg, resolve(items[APP], 'host', msg))
        facts = {f.target: f for f in result.current_facts}
        self.assertEqual(facts[APP].value, 'host')
        self.assertEqual(facts[APP].source.message_id, 'M3')
        self.assertNotIn(DB, facts)
        self.assertEqual(result.pending_facts, (items[DB],))
        self.assertEqual(result.changes[-1].pending_before, items[APP])
        self.assertEqual(result.changes[-1].after, facts[APP])
        self.assertEqual(len(pending.pending_facts), 2)

    def test_confirming_old_value_creates_new_fact_with_current_message_source(self):
        pending = self.ambiguous_state()
        item = next(p for p in pending.pending_facts if p.target == APP)
        msg = UserMessage('M3', 3, '确认应用还是在容器里。')
        change = resolve(item, 'container', msg)
        result = apply(pending, msg, change)
        fact = next(f for f in result.current_facts if f.target == APP)
        self.assertNotEqual(fact.fact_id, self.old.fact_id)
        self.assertEqual(fact.source.message_id, 'M3')
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_current_user_message'):
            apply(pending, msg, {**change, 'source': asdict(self.old.source)})

    def test_pending_targets_require_explicit_resolution(self):
        pending = self.ambiguous_state()
        msg = UserMessage('M3', 3, '应用在宿主机。')
        changes = [
            update('add', APP, 'host', 'M3', msg.content),
            update('replace', APP, 'host', 'M3', msg.content, self.old.fact_id),
            update('retract', APP, None, 'M3', msg.content, self.old.fact_id),
            mark_pending(APP, msg, self.old.fact_id),
        ]
        for change in changes:
            with self.subTest(operation=change['operation']):
                with self.assertRaisesRegex(ValueError, 'pending_target_requires_resolution'):
                    apply(pending, msg, change)

    def test_resolve_checks_target_and_pending_id_and_rejects_stale_resolution(self):
        pending = self.ambiguous_state()
        items = {p.target: p for p in pending.pending_facts}
        msg = UserMessage('M3', 3, '应用在宿主机。')
        change = resolve(items[APP], 'host', msg)
        for wrong in ({**change, 'pending_id': items[DB].pending_id},
                      {**change, 'target': asdict(LOCAL_APP)},
                      {**change, 'pending_id': 'outdated'}):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, 'pending_fact_does_not_match_target'):
                apply(pending, msg, wrong)
        resolved = apply(pending, msg, change)
        fact = next(f for f in resolved.current_facts if f.target == APP)
        msg4 = UserMessage('M4', 4, '应用位置又需要确认。')
        uncertain_again = apply(resolved, msg4, mark_pending(APP, msg4, fact.fact_id, 'uncertain_value'))
        msg5 = UserMessage('M5', 5, '应用在宿主机。')
        with self.assertRaisesRegex(ValueError, 'pending_fact_does_not_match_target'):
            apply(uncertain_again, msg5, resolve(items[APP], 'host', msg5))

    def test_pending_location_without_old_value_is_supported_but_not_invented(self):
        msg = UserMessage('M2', 2, '本地测试的应用在哪里运行，我还不清楚。')
        change = mark_pending(LOCAL_APP, msg, reason='uncertain_value')
        result = apply(self.initial, msg, change)
        self.assertEqual(result.current_facts, self.initial.current_facts)
        self.assertIsNone(result.pending_facts[0].previous_fact)
        self.assertEqual(result.pending_facts[0].target, LOCAL_APP)
        for wrong, reason in (
            ({**change, 'previous_fact_id': self.old.fact_id}, 'previous_fact_does_not_match_target'),
            ({**change, 'reason': 'network_error'}, 'invalid_pending_reason'),
            ({**change, 'value': 'host'}, 'invalid_update_fields'),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                apply(self.initial, msg, wrong)

    def test_invalid_second_resolution_does_not_partially_modify_pending_state(self):
        pending = self.ambiguous_state()
        before = pending.to_dict()
        items = {p.target: p for p in pending.pending_facts}
        msg = UserMessage('M3', 3, '应用在宿主机，数据库在容器。')
        valid = resolve(items[APP], 'host', msg)
        invalid = {**resolve(items[DB], 'container', msg), 'pending_id': 'wrong-id'}
        with self.assertRaises(ValueError):
            apply(pending, msg, valid, invalid)
        self.assertEqual(pending.to_dict(), before)

    def test_pending_needs_correct_old_id_and_resolution_needs_value(self):
        msg = UserMessage('M2', 2, '应用在哪里我现在不确定。')
        with self.assertRaisesRegex(ValueError, 'previous_fact_does_not_match_target'):
            apply(self.initial, msg, mark_pending(APP, msg))
        pending = self.ambiguous_state()
        item = next(p for p in pending.pending_facts if p.target == APP)
        msg3 = UserMessage('M3', 3, '还是不清楚。')
        with self.assertRaisesRegex(ValueError, 'require_nonempty_fact_value'):
            apply(pending, msg3, resolve(item, None, msg3))

    def test_pending_and_resolution_proposals_parse_but_duplicate_target_is_rejected(self):
        msg = UserMessage('M2', 2, '应用在哪里我不确定。')
        change = mark_pending(APP, msg, self.old.fact_id, 'uncertain_value')
        raw = json.dumps({'updates': [change]}, ensure_ascii=False)
        self.assertEqual(parse_context_updates(raw), {'updates': [change]})
        with self.assertRaisesRegex(ValueError, 'multiple_updates_for_same_target'):
            apply(self.initial, msg, change, change)

    def test_identical_replace_keeps_fact_id_and_history_while_real_change_applies(self):
        db = next(f for f in self.initial.current_facts if f.target == DB)
        msg = UserMessage('M2', 2, '应用在宿主机，数据库仍在容器里。')
        noop = update('replace', DB, 'container', 'M2', '数据库仍在容器里', db.fact_id)
        result = apply(self.initial, msg, self.correction, noop)
        self.assertEqual(next(f for f in result.current_facts if f.target == DB), db)
        self.assertEqual(len(result.changes), len(self.initial.changes) + 1)
        self.assertEqual(result.changes[-1].after.target, APP)
        self.assertEqual(result.input_version, 2)

    def test_identical_replace_still_requires_valid_old_id_and_current_source(self):
        msg = UserMessage('M2', 2, '应用仍在容器里。')
        valid = update('replace', APP, 'container', 'M2', msg.content, self.old.fact_id)
        for wrong in ({**valid, 'previous_fact_id': 'wrong-id'},
                      {**valid, 'source': asdict(self.old.source)}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                apply(self.initial, msg, wrong)
