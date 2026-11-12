"""Typographic normalization preserves strict citation membership."""
import json
import unittest

from app.answering import validate_answer


PASSAGES = [{'chunk_id': name, 'content': '测试资料',
             'metadata': {'document_id': 'test', 'section_id': 'TEST',
                          'source_url': 'https://example.invalid/test'}}
            for name in ('TEST:1', 'TEST:2')]


def response(text):
    return json.dumps({'status': 'answer', 'answer': text,
        'citation_ids': ['TEST:1'], 'missing_information': []}, ensure_ascii=False)


class CitationFormatTests(unittest.TestCase):
    def test_spaces_and_balanced_quotes_normalize_only_the_reference_marker(self):
        for marker in ('[ TEST:1 ]', '[“TEST:1”]', '[ "TEST:1" ]', "['TEST:1']", '[‘ TEST:1 ’]', '[TEST:1]'):
            with self.subTest(marker=marker):
                raw = response('保留正文“原话”和全部结论。' + marker)
                result = validate_answer(raw, PASSAGES, normalize_citation_format=True)
                self.assertEqual(result['answer'], '保留正文“原话”和全部结论。[TEST:1]')
                self.assertEqual(result['citation_ids'], ['TEST:1'])
                self.assertEqual(result['sources'][0]['chunk_id'], 'TEST:1')
                self.assertIn(marker, json.loads(raw)['answer'])

    def test_unknown_unlisted_and_malformed_markers_are_not_hidden_or_fuzzy_matched(self):
        for marker in ('[“UNKNOWN:1”]', '[ TEST:2 ]', '[test:1]', '[“TEST:1"]', '[ TEST :1 ]'):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, 'Inline references must match'):
                    validate_answer(response('[TEST:1]'+marker), PASSAGES, normalize_citation_format=True)

    def test_does_not_fix_json_or_rewrite_citation_ids(self):
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            validate_answer('not JSON', PASSAGES, normalize_citation_format=True)
        raw = json.loads(response('[ TEST:1 ]'))
        raw['citation_ids'] = [' TEST:1 ']
        with self.assertRaisesRegex(ValueError, 'outside the retrieved set'):
            validate_answer(json.dumps(raw), PASSAGES, normalize_citation_format=True)

    def test_legacy_entry_remains_strict_by_default(self):
        with self.assertRaisesRegex(ValueError, 'Inline references must match'):
            validate_answer(response('[ TEST:1 ]'), PASSAGES)


if __name__ == '__main__':
    unittest.main()
