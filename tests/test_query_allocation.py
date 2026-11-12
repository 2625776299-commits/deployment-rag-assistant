import unittest

from app.query_allocation import allocate_query_candidates


def chunk(key, body=None):
    return {'chunk_id': key, 'content': body or key, 'metadata': {'source_url': key}, 'score': 1.0}


class QueryAllocationTests(unittest.TestCase):
    def test_both_respect_capacity_and_preserve_complementary_routes(self):
        ranks = {'specific': [chunk(str(i)) for i in range(5)],
                 'mechanism': [chunk('m'+str(i)) for i in range(5)]}
        selected = allocate_query_candidates(ranks, 5, policy='reserved_then_rrf')
        self.assertEqual(len(selected), 5)
        self.assertGreaterEqual(sum(c['chunk_id'].startswith('m') for c in selected), 2)
        self.assertGreaterEqual(sum(not c['chunk_id'].startswith('m') for c in selected), 2)

    def test_overlap_does_not_consume_two_places(self):
        shared = chunk('shared')
        ranks = {'a': [shared, chunk('a')], 'b': [shared, chunk('b')]}
        for policy in ('unified_rrf', 'reserved_then_rrf'):
            result = allocate_query_candidates(ranks, 5, policy=policy)
            self.assertEqual({c['chunk_id'] for c in result}, {'shared', 'a', 'b'})
            self.assertEqual(len(result), 3)

    def test_empty_route_releases_places(self):
        ranks = {'a': [], 'b': [chunk(str(i)) for i in range(5)]}
        self.assertEqual(len(allocate_query_candidates(ranks, 5, policy='reserved_then_rrf')), 5)

    def test_exact_body_aliases_are_deduplicated(self):
        ranks = {'a': [chunk('a', 'same')], 'b': [chunk('alias', 'same')]}
        self.assertEqual(len(allocate_query_candidates(ranks, 5, policy='unified_rrf')), 1)

    def test_conflicting_identity_is_rejected(self):
        ranks = {'a': [chunk('same', 'one')], 'b': [chunk('same', 'two')]}
        with self.assertRaises(ValueError):
            allocate_query_candidates(ranks, 5, policy='reserved_then_rrf')

    def test_small_capacity_is_bounded(self):
        ranks = {'a': [chunk('a')], 'b': [chunk('b')]}
        self.assertEqual(len(allocate_query_candidates(ranks, 1, policy='reserved_then_rrf')), 1)


if __name__ == '__main__':
    unittest.main()
