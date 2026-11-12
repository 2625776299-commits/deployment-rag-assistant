"""A finished HTTP probe is not an automatic semantic pass."""
from copy import deepcopy
import unittest

from app.run_multiturn_suite import classify_report, require_same_fingerprint


CASE = {'turns': [{'message_id': 'M1'}, {'message_id': 'M2'}],
        'expected': [{'message_id': 'M1', 'action': 'needs_user'}, {'message_id': 'M2', 'action': 'answer'}]}


def report():
    return {'status': 'finished', 'source_unchanged': True, 'suite_unchanged': True,
            'http_server_stopped': True, 'database_stopped': True,
            'turns': [{'message_id': mid, 'http_status': 200, 'response': {'status': action}}
                      for mid, action in (('M1', 'needs_user'), ('M2', 'answer'))]}


class MultiturnSuiteTests(unittest.TestCase):
    def test_matching_actions_still_require_semantic_review(self):
        result = classify_report(CASE, report())
        self.assertEqual(result['status'], 'responses_observed_pending_semantic_review')
        self.assertEqual(result['semantic_review'], 'pending')

    def test_finished_probe_with_failed_request_is_not_a_pass(self):
        data = report()
        data['turns'][1].update(http_status=503, response={'status': 'failed', 'error_code': 'provider_connection_failed'})
        result = classify_report(CASE, data)
        self.assertEqual(result['status'], 'request_failures')
        self.assertEqual(result['request_failures'][0]['error_code'], 'provider_connection_failed')

    def test_missing_dependent_turns_are_preserved_as_interruption(self):
        data = report()
        data['turns'] = data['turns'][:1]
        result = classify_report(CASE, data)
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['missing_messages'], ['M2'])

    def test_drift_invalidates_comparison(self):
        data = report()
        data['source_unchanged'] = False
        self.assertEqual(classify_report(CASE, data)['status'], 'incomparable')
        with self.assertRaisesRegex(ValueError, 'changed'):
            require_same_fingerprint({'corpus': 'v1'}, {'corpus': 'v2'})

    def test_unexpected_action_is_not_hidden_by_http_success(self):
        data = report()
        data['turns'][1]['response']['status'] = 'needs_user'
        result = classify_report(CASE, data)
        self.assertEqual(result['status'], 'action_mismatch')
        self.assertFalse(result['action_checks'][1]['matches'])

    def test_identical_fingerprint_is_accepted_without_mutation(self):
        frozen = {'source': 'abc', 'configuration': {'model': 'model'}, 'corpus': 'v1'}
        before = deepcopy(frozen)
        require_same_fingerprint(frozen, deepcopy(frozen))
        self.assertEqual(frozen, before)


if __name__ == '__main__':
    unittest.main()
