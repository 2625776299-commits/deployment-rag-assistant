"""Offline checks for model-output failures and candidate-pool boundaries."""

import copy
import json
import unittest

from app.reranking import build_rerank_messages, rerank_candidates


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"chunk_id": f"c{i}", "content": f"Evidence {i}",
                      "routes": {"vector": {"rank": i}}, "rrf_score": 1 / (60 + i)}
                     for i in range(1, 5)]

    def test_lower_rrf_candidate_can_enter_final_evidence_without_rewriting_it(self):
        before = copy.deepcopy(self.pool)
        calls = []

        def generate(messages):
            calls.append(messages)
            return json.dumps({"ranked_chunk_ids": ["c4", "c1", "c2", "c3"]})

        result = rerank_candidates("Which evidence?", self.pool, generate, final_k=2)
        self.assertEqual(result["status"], "reranked")
        self.assertEqual(result["final_evidence"], [self.pool[3], self.pool[0]])
        self.assertEqual(self.pool, before)
        self.assertEqual(len(calls), 1)
        payload = json.loads(calls[0][1]["content"])
        self.assertEqual(set(payload), {"question", "passages"})
        self.assertEqual(len(payload["passages"]), 4)
        self.assertTrue(all(set(c) == {"chunk_id", "content"} for c in payload["passages"]))

    def test_unknown_duplicate_omitted_and_invalid_outputs_restore_original_order(self):
        invalid = [
            '{"ranked_chunk_ids": ["invented", "c1", "c2", "c3"]}',
            '{"ranked_chunk_ids": ["c1", "c1", "c2", "c3"]}',
            '{"ranked_chunk_ids": ["c1", "c2", "c3"]}',
            '{"ranked_chunk_ids": [1, "c2", "c3", "c4"]}',
            '{"ranked_chunk_ids": ["c1", "c2", "c3", "c4"], "answer": "extra"}',
            'not json',
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                calls = []

                def generate(messages):
                    calls.append(messages)
                    return raw

                result = rerank_candidates("Question", self.pool, generate)
                self.assertEqual(result["status"], "fallback_rrf")
                self.assertIsNotNone(result["fallback_reason"])
                self.assertEqual(result["ranked_chunk_ids"], ["c1", "c2", "c3", "c4"])
                self.assertEqual(result["final_evidence"], self.pool[:3])
                self.assertEqual(len(calls), 1)

    def test_timeout_is_one_attempt_and_does_not_leak_exception_text(self):
        calls = []

        def generate(messages):
            calls.append(messages)
            raise TimeoutError("private-provider-details")

        result = rerank_candidates("Question", self.pool, generate)
        self.assertEqual(result["fallback_reason"], "rerank_timeout")
        self.assertEqual(result["final_evidence"], self.pool[:3])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("private-provider-details", json.dumps(result))

    def test_empty_pool_skips_model(self):
        def forbidden(messages):
            self.fail("No model request should be made")

        result = rerank_candidates("Question", [], forbidden)
        self.assertEqual(result["status"], "skipped_empty")
        self.assertEqual(result["model_attempts"], 0)
        self.assertEqual(result["final_evidence"], [])

    def test_duplicate_input_ids_fail_before_call(self):
        with self.assertRaisesRegex(ValueError, "duplicate_input_chunk_id"):
            build_rerank_messages("Question", [self.pool[0], self.pool[0]])


if __name__ == "__main__":
    unittest.main()
