"""Request contracts and source boundaries; no providers or real index changes."""
from copy import deepcopy
import unittest

from app.bounded_retrieval import run_bounded_retrieval
from app.retrieval_contract import RetrievalRequest, SourceScope
from app.scoped_retrieval import ScopedHybridRetriever
from app.run_scoped_retrieval_demo import QUERY, fixture, run_demo


class ScopedRetrievalTests(unittest.TestCase):
    def test_sources_are_filtered_before_top_k_and_existing_chunks_are_excluded(self):
        chunks, vectors = fixture()
        before = deepcopy((chunks, vectors))
        retriever = ScopedHybridRetriever(chunks, vectors, self.fail, result_k=5)
        result = retriever.search(RetrievalRequest(QUERY, ('D0',),
            SourceScope(('sqlalchemy-pooling',), ('psycopg-pooling',))))
        self.assertEqual({c['chunk_id'] for c in result}, {'S1', 'D1'})
        self.assertEqual(retriever.trace[0]['eligible_ids'], ['S1', 'D1'])
        self.assertEqual((chunks, vectors), before)

    def test_exclusion_can_reach_next_candidate_not_just_remove_top_result(self):
        chunks, vectors = fixture()
        retriever = ScopedHybridRetriever(chunks, vectors, self.fail, route_k=1, result_k=1)
        result = retriever.search(RetrievalRequest(QUERY, ('D0', 'S1'),
            SourceScope(excluded_document_ids=('psycopg-pooling',))))
        self.assertEqual([c['chunk_id'] for c in result], ['D1'])

    def test_preference_breaks_ties_without_overriding_stronger_relevance(self):
        chunks = [{'chunk_id': 'a', 'content': 'topic topic topic', 'metadata': {'document_id': 'docker'}},
                  {'chunk_id': 'z', 'content': 'topic', 'metadata': {'document_id': 'sqlalchemy'}}]
        # Opposite rank-1/rank-2 routes give equal fused RRF contributions.
        retriever = ScopedHybridRetriever(chunks, {'topic topic topic': [0., 1.], 'topic': [1., 0.], 'unused': [0., 1.]}, self.fail)
        equal = retriever.search(RetrievalRequest('topic', scope=SourceScope(('sqlalchemy',))))
        self.assertEqual([c['chunk_id'] for c in equal], ['z', 'a'])
        stronger = retriever.search(RetrievalRequest('unused', scope=SourceScope(('sqlalchemy',))))
        self.assertEqual([c['chunk_id'] for c in stronger], ['a', 'z'])

    def test_empty_scope_result_avoids_embedding_and_unknown_document_fails(self):
        chunks, vectors = fixture()
        retriever = ScopedHybridRetriever(chunks, vectors, self.fail)
        result = retriever.search(RetrievalRequest('uncached', tuple(c['chunk_id'] for c in chunks)))
        self.assertEqual(result, [])
        self.assertFalse(retriever.trace[0]['embedding_requested'])
        with self.assertRaisesRegex(ValueError, 'unknown_documents'):
            retriever.search(RetrievalRequest('uncached', scope=SourceScope(('not-in-this-corpus',))))

    def test_contract_rejects_conflicting_and_mutable_identifiers(self):
        for make in (lambda: SourceScope(('sqlalchemy',), ('sqlalchemy',)),
                     lambda: SourceScope(['sqlalchemy']),
                     lambda: RetrievalRequest('query', ('same', 'same')),
                     lambda: RetrievalRequest('query', ({},))):
            with self.subTest(make=make), self.assertRaises(ValueError):
                make()

    def test_feedback_changes_scope_and_old_evidence_survives(self):
        report = run_demo()
        self.assertEqual(report['evaluated_pools'], [['D0'], ['D0', 'S1'], ['D0', 'S1', 'D1']])
        requests = [t['request'] for t in report['retrieval_trace']]
        self.assertEqual(requests[0]['exclude_chunk_ids'], ('D0',))
        self.assertEqual(requests[1]['exclude_chunk_ids'], ('D0', 'S1'))
        self.assertEqual(requests[1]['scope']['preferred_document_ids'], ('docker-networking',))
        self.assertFalse(report['semantic_quality_verified'])

    def test_same_query_unchanged_scope_stops_and_policy_failure_does_not_lookup(self):
        chunks, _ = fixture()
        def evaluation(*_):
            return {'status': 'proposed', 'proposal': {'action': 'retrieve', 'search_query': QUERY,
                                                       'missing_knowledge': ['checkout handling']}}
        result = run_bounded_retrieval('question', chunks[:1], evaluation, self.fail,
            request_retrieval=self.fail, known_queries=(QUERY,), initial_scope=SourceScope())
        self.assertEqual(result['stop_reason'], 'repeated_query')
        self.assertEqual(result['retrieval_count'], 0)
        result = run_bounded_retrieval('question', chunks[:1], evaluation, self.fail,
            request_retrieval=self.fail, scope_policy=lambda _: 'not-a-SourceScope')
        self.assertEqual(result['stop_reason'], 'retrieval_or_merge_error:ValueError')
        self.assertEqual(result['retrieval_count'], 0)

    def test_scope_options_cannot_be_silently_ignored_by_legacy_path(self):
        chunks, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'scope_requires_request_retrieval'):
            run_bounded_retrieval('question', chunks[:1], self.fail, self.fail, initial_scope=SourceScope())

    def test_request_limit_is_local_and_cannot_raise_configured_result_size(self):
        chunks, vectors = fixture()
        search = ScopedHybridRetriever(chunks, vectors, self.fail, result_k=2)
        limited = search.search(RetrievalRequest(QUERY, max_results=1))
        self.assertEqual(len(limited), 1)
        self.assertEqual(search.trace[-1]['result_limit'], 1)
        self.assertEqual(search.result_k, 2)
        self.assertEqual(len(search.search(RetrievalRequest(QUERY))), 2)
        self.assertEqual(len(search.search(RetrievalRequest(QUERY, max_results=99))), 2)
        for value in (0, -1, True, 1.5, '1', []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RetrievalRequest(QUERY, max_results=value)

    def test_controller_caps_supplement_to_remaining_space_and_reevaluates(self):
        chunks = [{'chunk_id': f'C{i}', 'content': f'connection topic {i}',
                   'metadata': {'document_id': 'fixture'}} for i in range(12)]
        vectors = {c['content']: [1., 0.] for c in chunks}
        vectors[QUERY] = [1., 0.]
        for initial_count in (7, 9):
            with self.subTest(initial_count=initial_count):
                search = ScopedHybridRetriever(chunks, vectors, self.fail, route_k=10, result_k=10)
                pools = []

                def evaluate(question, pool):
                    pools.append([c['chunk_id'] for c in pool])
                    return {'status': 'proposed', 'proposal': {
                        'action': 'retrieve' if len(pools) == 1 else 'answer',
                        'search_query': QUERY, 'missing_knowledge': ['controlled gap']},
                        'selected_evidence': pool[-1:]}

                result = run_bounded_retrieval('controlled question', chunks[:initial_count],
                    evaluate, self.fail, request_retrieval=search.search)
                request = search.trace[0]['request']
                self.assertEqual(request['max_results'], 10 - initial_count)
                self.assertEqual(request['exclude_chunk_ids'], tuple(c['chunk_id'] for c in chunks[:initial_count]))
                self.assertEqual(len(pools[1]), 10)
                self.assertEqual(result['status'], 'answer_proposed')
                self.assertEqual(result['retrieval_count'], 1)
                self.assertEqual(search.result_k, 10)

    def test_full_pool_avoids_search_but_can_still_answer(self):
        chunks = [{'chunk_id': str(i), 'content': f'body {i}'} for i in range(10)]

        def evaluation(action):
            return lambda *_: {'status': 'proposed', 'proposal': {
                'action': action, 'search_query': QUERY, 'missing_knowledge': ['gap']}}

        stopped = run_bounded_retrieval('question', chunks, evaluation('retrieve'),
                                       self.fail, request_retrieval=self.fail)
        self.assertEqual(stopped['stop_reason'], 'candidate_budget_exhausted')
        self.assertEqual(stopped['retrieval_count'], 0)
        answered = run_bounded_retrieval('question', chunks, evaluation('answer'),
                                        self.fail, request_retrieval=self.fail)
        self.assertEqual(answered['status'], 'answer_proposed')

    def test_remaining_gap_and_oversized_adapter_output_do_not_force_answer(self):
        chunks = [{'chunk_id': str(i), 'content': f'body {i}'} for i in range(11)]
        requests = []

        def evaluate(*_):
            return {'status': 'proposed', 'proposal': {
                'action': 'retrieve', 'search_query': QUERY, 'missing_knowledge': ['still missing']}}

        def retrieve(request):
            requests.append(request)
            return chunks[9:10]

        stopped = run_bounded_retrieval('question', chunks[:9], evaluate, self.fail,
                                       request_retrieval=retrieve)
        self.assertEqual(requests[0].max_results, 1)
        self.assertEqual(stopped['status'], 'stopped')
        self.assertEqual(stopped['stop_reason'], 'retrieval_budget_exhausted')
        self.assertEqual(stopped['evaluation_count'], 2)
        overflow = run_bounded_retrieval('question', chunks[:9], evaluate, self.fail,
                                       request_retrieval=lambda _: chunks[9:])
        self.assertEqual(overflow['stop_reason'], 'candidate_budget_exhausted')
        self.assertEqual(overflow['evaluation_count'], 1)


if __name__ == '__main__':
    unittest.main()
