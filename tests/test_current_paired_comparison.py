import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.run_current_paired_comparison import direct_answer, execute_arm
from app.retrieval_contract import RetrievalRequest


def chunk(key):
    return {'chunk_id': key, 'content': 'Original evidence ' + key,
        'metadata': {'document_id': 'doc', 'section_id': key,
            'source_url': 'https://example.org/docs', 'build_id': 'v4'}}


class CurrentPairedComparisonTests(unittest.TestCase):
    def test_shared_initial_is_copied_and_supplement_preserves_structured_request(self):
        initial = [chunk('A')]
        before = deepcopy(initial)
        pinned = SimpleNamespace(build=SimpleNamespace(build_id='v4'),
            retrieve=Mock(), search=Mock(return_value=[chunk('D')]))
        supplement = RetrievalRequest('missing rule', build_id='v4',
            exclude_chunk_ids=('A',), max_results=1)
        def service(question, context, **kwargs):
            first = kwargs['request_retrieval'](RetrievalRequest(question, build_id='v4'))
            first[0]['content'] = 'mutated consumer copy'
            second = kwargs['request_retrieval'](supplement)
            self.assertEqual(second[0]['chunk_id'], 'D')
            return {'status': 'answer'}
        with patch('app.run_current_paired_comparison.run_contextual_rag', side_effect=service):
            result = execute_arm('question', initial, arm='B_coverage_flow', pinned=pinned, raw=Mock())
        self.assertEqual(result['status'], 'answer')
        self.assertEqual(initial, before)
        pinned.search.assert_called_once_with(supplement)
        pinned.retrieve.assert_not_called()

    def test_direct_arm_repairs_obtained_invalid_citation_once(self):
        def answer(key):
            return json.dumps({'status': 'answer', 'answer': 'Grounded [' + key + ']',
                'citation_ids': [key], 'missing_information': []})
        raw = Mock(side_effect=[answer('NOT_SELECTED:000'), answer('D:000')])
        result = direct_answer('question', [chunk('D:000')], raw)
        self.assertEqual(result['citation_ids'], ['D:000'])
        self.assertEqual(raw.call_count, 2)
        self.assertIn('allowed_citation_ids', raw.call_args.args[1][-1]['content'])

    def test_direct_arm_does_not_retry_provider_timeout(self):
        raw = Mock(side_effect=TimeoutError('fixture'))
        with self.assertRaises(TimeoutError):
            direct_answer('question', [chunk('D')], raw)
        raw.assert_called_once()


if __name__ == '__main__':
    unittest.main()
