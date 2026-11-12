"""Incident identity, provenance and immutable updates, without SQL or models."""
import unittest

from app.conversation_context import UserMessage
from app.task_constraints import (
    ConstraintSnapshot, IncidentRef, project_incident_constraints, revise_evidence_availability,
    apply_evidence_updates, restore_constraint_snapshot,
)


I1 = IncidentRef('T1', 'I1')
I2 = IncidentRef('T1', 'I2')
OTHER_TASK = IncidentRef('T2', 'I1')
REGISTRY = frozenset({I1, I2, OTHER_TASK})


def change(state, *, version, incident=I1, kind='incident_logs', status='unavailable',
           quote='这次日志不可取得', previous=None, content=None):
    return revise_evidence_availability(state,
        UserMessage(f'M{version}', version, content or quote), incident=incident,
        evidence_kind=kind, availability=status, quote=quote,
        previous_constraint_id=previous, allowed_incidents=REGISTRY)


class TaskConstraintTests(unittest.TestCase):
    def test_atomic_multi_category_updates_restore_json_and_keep_deferred_source(self):
        import json
        original=UserMessage('M2',2,'那次日志和报错已经清理')
        message=UserMessage('M3',3,'说的是第一次故障')
        updates=[{'evidence_kind':kind,'availability':'unavailable','previous_constraint_id':None,
                  'source':{'message_id':'M2','quote':original.content}}
                 for kind in ('incident_logs','incident_error_details')]
        old=ConstraintSnapshot(2)
        new=apply_evidence_updates(old,message,incident=I1,updates=updates,
                                  allowed_incidents=REGISTRY,deferred_messages=(original,))
        self.assertEqual(len(new.current),2)
        self.assertTrue(all(c.source.message_id=='M2' and c.introduced_in_input_version==3 for c in new.current))
        self.assertEqual(restore_constraint_snapshot(json.loads(json.dumps(new.to_dict()))),new)
        updates[1]['source']['quote']='伪造来源'
        with self.assertRaisesRegex(ValueError,'constraint_quote_not_in_message'):
            apply_evidence_updates(old,message,incident=I1,updates=updates,
                                   allowed_incidents=REGISTRY,deferred_messages=(original,))
        self.assertEqual(old.current,())

    def test_deferred_message_cannot_replace_a_more_recent_change(self):
        old=change(ConstraintSnapshot(),version=3,status='available',quote='日志仍然在')
        previous=old.current[0]
        with self.assertRaisesRegex(ValueError,'deferred_constraint_would_overwrite_newer_record'):
            apply_evidence_updates(old,UserMessage('M4',4,'说的是第一次'),incident=I1,
                updates=[{'evidence_kind':'incident_logs','availability':'unavailable',
                    'previous_constraint_id':previous.constraint_id,'source':{'message_id':'M2','quote':'日志没有了'}}],
                allowed_incidents=REGISTRY,deferred_messages=(UserMessage('M2',2,'日志没有了'),))

    def test_two_incidents_in_same_task_and_other_task_are_isolated(self):
        state = change(ConstraintSnapshot(), version=1)
        original = state.current[0]
        state = change(state, version=2, incident=I2, status='available', quote='这次日志还在')
        state = change(state, version=3, incident=OTHER_TASK, status='not_obtained', quote='暂时还没拿到日志')
        result = project_incident_constraints(state, I1, allowed_incidents=REGISTRY)
        self.assertEqual(len(result['evidence_constraints']), 1)
        self.assertEqual(result['evidence_constraints'][0]['constraint_id'], original.constraint_id)
        self.assertEqual(result['evidence_constraints'][0]['availability'], 'unavailable')
        self.assertEqual([c.availability for c in state.current], ['unavailable', 'available', 'not_obtained'])

    def test_new_evidence_updates_only_one_category_without_rewriting_old_snapshot(self):
        old = change(ConstraintSnapshot(), version=1)
        old = change(old, version=2, kind='incident_error_details', quote='报错也清理了')
        frozen = old.to_dict()
        updated = change(old, version=4, status='available', quote='找回了这次日志',
                         previous=old.current[0].constraint_id)
        self.assertEqual(old.to_dict(), frozen)
        logs = next(c for c in updated.current if c.evidence_kind == 'incident_logs')
        self.assertEqual(logs.supersedes_constraint_id, old.current[0].constraint_id)
        self.assertEqual(logs.source.message_id, 'M4')
        self.assertEqual(logs.introduced_in_input_version, 4)
        self.assertIn(old.current[1], updated.current)

    def test_repeated_value_preserves_original_id_source_and_introduction_version(self):
        first = change(ConstraintSnapshot(), version=1)
        later = change(first, version=2, quote='仍然拿不到这次日志', previous=first.current[0].constraint_id)
        self.assertEqual(later.current, first.current)
        self.assertEqual(later.input_version, 2)
        self.assertEqual(later.current[0].source.message_id, 'M1')

    def test_ambiguous_unauthorized_or_stale_changes_do_not_mutate_state(self):
        state = change(ConstraintSnapshot(), version=1)
        before = state.to_dict()
        for incident, previous, error in ((IncidentRef('T1', 'unknown'), None, 'incident_not_authorized'),
                                          (I2, state.current[0].constraint_id, 'constraint_previous_id_conflict'),
                                          (I1, 'stale-id', 'constraint_previous_id_conflict')):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                change(state, version=2, incident=incident, previous=previous)
            self.assertEqual(state.to_dict(), before)
        with self.assertRaisesRegex(ValueError, 'require_explicit_incident_reference'):
            IncidentRef('T1', '')

    def test_bad_source_old_version_and_unknown_availability_are_rejected(self):
        state = change(ConstraintSnapshot(), version=1)
        for options, error in (({'version': 2, 'content': '无关原话'}, 'constraint_quote_not_in_message'),
                               ({'version': 1}, 'constraint_version_not_newer'),
                               ({'version': 2, 'status': 'probably_available',
                                 'previous': state.current[0].constraint_id}, 'invalid_evidence_constraint')):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                change(state, **options)

    def test_withdrawn_certainty_is_unknown_and_projection_does_not_choose_an_action(self):
        state = change(ConstraintSnapshot(), version=1)
        state = change(state, version=2, status='uncertain', quote='我不确定那份日志是否还在',
                       previous=state.current[0].constraint_id)
        result = project_incident_constraints(state, I1, allowed_incidents=REGISTRY)
        self.assertEqual(result['evidence_constraints'][0]['availability'], 'uncertain')
        self.assertNotIn('action', result)
        result['evidence_constraints'][0]['availability'] = 'available'
        self.assertEqual(state.current[0].availability, 'uncertain')


if __name__ == '__main__':
    unittest.main()
