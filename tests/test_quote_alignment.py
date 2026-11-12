import unittest
from app.quote_alignment import align_quote


class QuoteAlignmentTests(unittest.TestCase):
    def test_lost_inline_markup_recovers_original_span(self):
        source='Heading\n`COMMIT` commits the transaction.'
        match=align_quote(source,'COMMIT commits the transaction.')
        self.assertEqual(match['mode'],'inline_code_markup')
        self.assertEqual(source[match['source_start']:match['source_end']],match['source_quote'])
        self.assertEqual(match['source_quote'],'`COMMIT` commits the transaction.')

    def test_multiple_code_names_preserve_words(self):
        self.assertIsNotNone(align_quote('A `SELECT` may use `EXPLAIN` or `SHOW`.','A SELECT may use EXPLAIN or SHOW.'))

    def test_exact_match_preserves_original_behavior(self):
        self.assertEqual(align_quote('`x` stays.','`x` stays.')['mode'],'exact')

    def test_changed_negation_number_case_or_punctuation_rejected(self):
        source='`COMMIT` does not retry 2 writes.'
        for q in ['COMMIT does retry 2 writes.','COMMIT does not retry 3 writes.',
                  'commit does not retry 2 writes.','COMMIT does not retry 2 writes!']:
            with self.subTest(q=q):
                self.assertIsNone(align_quote(source,q))

    def test_code_fence_literal_backticks_are_not_removed(self):
        self.assertIsNone(align_quote('```python\nx = `literal`\n```','x = literal'))
        self.assertIsNone(align_quote('~~~python\nx = `literal`\n~~~','x = literal'))

    def test_changed_newline_or_spaces_not_normalized(self):
        self.assertIsNone(align_quote('`SELECT`\nreturns rows.','SELECT returns rows.'))

    def test_noncontiguous_or_fabricated_quote_rejected(self):
        self.assertIsNone(align_quote('`COMMIT` ends one transaction. Another starts later.','COMMIT ends one transaction. starts later.'))

    def test_ambiguous_normalized_locations_rejected(self):
        self.assertIsNone(align_quote('`A` does B.\nA does `B`.','A does B.'))


if __name__=='__main__':
    unittest.main()
