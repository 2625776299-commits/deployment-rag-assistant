"""Offline checks of evidence preservation, finite loops and failure exits."""

import copy
import unittest

from app.bounded_retrieval import merge_evidence, run_bounded_retrieval


def decision(action, evidence=None, query="focused missing evidence"):
    return {"status": "proposed", "proposal": {"action": action, "search_query": query},
            "selected_evidence": evidence or [], "coverage_verified": False}


class BoundedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.a = {"chunk_id": "a", "content": "Old connections close.", "metadata": {"url": "one"}}
        self.b = {"chunk_id": "b", "content": "Old connections close. Detect and reconnect."}

    def test_merge_deduplicates_exact_content_keeps_complementary_text_and_origins(self):
        alias = {"chunk_id": "alias", "content": self.a["content"], "metadata": {"url": "two"}}
        old, incoming = [self.a], [self.a, alias, self.b]
        before = copy.deepcopy((old, incoming))
        result = merge_evidence(old, incoming)
        self.assertEqual(result["candidates"], [self.a, self.b])
        self.assertEqual(result["added_chunk_ids"], ["b"])
        self.assertEqual([d["reason"] for d in result["duplicates"]], ["same_id", "same_body"])
        self.assertEqual(result["duplicates"][1]["duplicate_metadata"], {"url": "two"})
        self.assertEqual((old, incoming), before)

    def test_conflicting_identity_fails_without_silently_choosing_a_version(self):
        for other in ({**self.a, "content": "Different"}, {**self.a, "metadata": {"url": "other"}}):
            with self.assertRaisesRegex(ValueError, "conflicting_chunk_identity"):
                merge_evidence([self.a], [other])

    def test_answer_clarify_and_insufficient_do_not_retrieve(self):
        for action, status in (("answer", "answer_proposed"), ("clarify", "needs_user"),
                               ("insufficient", "insufficient")):
            with self.subTest(action=action):
                def forbidden(_):
                    self.fail("Do not retrieve on this branch")

                result = run_bounded_retrieval("question", [self.a],
                                               lambda q, p: decision(action, p), forbidden)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["evaluation_count"], 1)
                self.assertFalse(result["coverage_verified"])

    def test_one_lookup_merges_then_reevaluates(self):
        evaluated, queries = [], []

        def evaluate(question, pool):
            evaluated.append([c["chunk_id"] for c in pool])
            return decision("retrieve" if len(evaluated) == 1 else "answer", pool)

        def retrieve(query):
            queries.append(query)
            return [self.a, self.b]

        result = run_bounded_retrieval("question", [self.a], evaluate, retrieve)
        self.assertEqual(result["status"], "answer_proposed")
        self.assertEqual(evaluated, [["a"], ["a", "b"]])
        self.assertEqual(len(queries), 1)
        self.assertEqual(result["retrieval_count"], 1)

    def test_duplicate_results_stop_without_another_evaluation(self):
        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: decision("retrieve"), lambda q: [self.a])
        self.assertEqual(result["stop_reason"], "no_new_evidence")
        self.assertEqual(result["evaluation_count"], 1)

    def test_repeated_query_is_not_executed(self):
        def forbidden(_):
            self.fail("Repeated query must not execute")

        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: decision("retrieve", query=" FOCUSED   QUERY "),
                                      forbidden, known_queries=("focused query",))
        self.assertEqual(result["stop_reason"], "repeated_query")
        self.assertEqual(result["retrieval_count"], 0)

    def test_persistent_gap_stops_at_retrieval_budget(self):
        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: decision("retrieve"), lambda q: [self.b])
        self.assertEqual(result["stop_reason"], "retrieval_budget_exhausted")
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["retrieval_count"], 1)

    def test_failed_evaluation_and_timeout_do_not_trigger_blind_lookup(self):
        def forbidden(_):
            self.fail("No lookup after failed evaluation")

        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: {"status": "evaluation_failed"}, forbidden)
        self.assertEqual(result["stop_reason"], "evaluation_failed")

        def timeout(q, p):
            raise TimeoutError("private provider message")

        result = run_bounded_retrieval("question", [self.a], timeout, forbidden)
        self.assertEqual(result["stop_reason"], "evaluation_error:TimeoutError")
        self.assertNotIn("private provider message", str(result))

    def test_evaluation_and_candidate_budgets_stop_before_unneeded_work(self):
        def forbidden(_):
            self.fail("No lookup with exhausted evaluation budget")

        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: decision("retrieve"), forbidden, max_evaluations=1)
        self.assertEqual(result["stop_reason"], "evaluation_budget_exhausted")
        result = run_bounded_retrieval("question", [self.a],
                                      lambda q, p: decision("retrieve"), lambda q: [self.b], max_candidates=1)
        self.assertEqual(result["stop_reason"], "candidate_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
