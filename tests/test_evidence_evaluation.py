"""Offline contract checks; generated replies here are fixtures, not LLM results."""

import copy
import json
import unittest

from app.evidence_evaluation import build_evaluation_messages, evaluate_evidence


class EvidenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"chunk_id": "cause", "content": "The IP can change.", "metadata": {"source_url": "example"}},
            {"chunk_id": "noise", "content": "Unrelated startup information."},
            {"chunk_id": "steps", "content": "Detect the condition and reconnect."},
        ]
        self.reply = {"action": "answer", "selected_chunk_ids": ["cause", "steps"],
                      "covered_points": [
                          {"point": "Reason", "evidence": [{"chunk_id": "cause", "quote": "IP can change"}]},
                          {"point": "Action", "evidence": [{"chunk_id": "steps", "quote": "reconnect"}]},
                      ], "missing_knowledge": [], "missing_user_information": [],
                      "search_query": None, "reason": "Selected passages support both parts."}

    def call(self, reply, **kwargs):
        return evaluate_evidence("Why and what next?", self.pool,
                                 lambda _: json.dumps(reply), **kwargs)

    def test_selects_complementary_passage_without_mutating_or_claiming_semantic_validity(self):
        before = copy.deepcopy(self.pool)
        result = self.call(self.reply)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["selected_evidence"], [self.pool[0], self.pool[2]])
        self.assertEqual(self.pool, before)
        self.assertFalse(result["coverage_verified"])
        self.assertEqual(result["semantic_review"], "pending")

    def test_model_payload_excludes_labels_answers_ranks_and_metadata(self):
        pool = [{**self.pool[0], "expected_action": "answer", "reference_answer": "secret",
                 "score": 999, "human_required_points": ["secret"]}]
        messages = build_evaluation_messages("question", pool)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"question", "passages", "selection_budget"})
        self.assertEqual(set(payload["passages"][0]), {"chunk_id", "content"})
        self.assertNotIn("secret", messages[1]["content"])

    def test_source_applicability_is_projected_without_unrelated_fields(self):
        candidate = {**self.pool[0], "metadata": {
            "technology": "EF Core", "applicable_version": None,
            "evidence_scope": "Commit failure", "generalizable_as": "Response loss",
            "source_url": "https://example.org/?token=secret", "expected_answer": "secret"}}
        before = copy.deepcopy(candidate)
        passage = json.loads(build_evaluation_messages("question", [candidate])[1]["content"])["passages"][0]
        self.assertEqual(passage["source_metadata"], {"technology": "EF Core", "applicable_version": None,
            "evidence_scope": "Commit failure", "generalizable_as": "Response loss"})
        self.assertNotIn("secret", json.dumps(passage))
        self.assertEqual(candidate, before)

    def test_invalid_or_oversized_metadata_is_not_forwarded(self):
        candidate = {**self.pool[0], "metadata": {"technology": "x" * 161,
                                                   "applicable_version": ["2.0"]}}
        passage = json.loads(build_evaluation_messages("question", [candidate])[1]["content"])["passages"][0]
        self.assertNotIn("source_metadata", passage)

    def test_invalid_ids_duplicates_unselected_support_and_invented_quotes_fail(self):
        invalid = []
        item = copy.deepcopy(self.reply)
        item["selected_chunk_ids"] = ["invented"]
        invalid.append(item)
        item = copy.deepcopy(self.reply)
        item["selected_chunk_ids"] = ["cause", "cause"]
        invalid.append(item)
        item = copy.deepcopy(self.reply)
        item["selected_chunk_ids"] = ["cause"]
        invalid.append(item)
        item = copy.deepcopy(self.reply)
        item["covered_points"][0]["evidence"][0]["quote"] = "fabricated quotation"
        invalid.append(item)
        for reply in invalid:
            with self.subTest(reply=reply):
                result = self.call(reply)
                self.assertEqual(result["status"], "evaluation_failed")
                self.assertIsNone(result["proposal"])
                self.assertEqual(result["selected_evidence"], [])

    def test_action_gap_contracts(self):
        retrieve = {**self.reply, "action": "retrieve", "selected_chunk_ids": [], "covered_points": [],
                    "missing_knowledge": ["How the application handles a replaced container"],
                    "search_query": "application handling connections after container replacement"}
        clarify = {**retrieve, "action": "clarify", "missing_knowledge": [],
                   "missing_user_information": ["Exact error", "Runtime locations"], "search_query": None}
        insufficient = {**retrieve, "action": "insufficient", "search_query": None,
                        "missing_knowledge": ["Private company policy is unavailable"]}
        for reply in (retrieve, clarify, insufficient):
            self.assertEqual(self.call(reply)["status"], "proposed")
        for action in ("answer", "retrieve", "clarify", "insufficient"):
            invalid = {**self.reply, "action": action, "missing_user_information": ["Missing fact"]}
            if action == "clarify":
                invalid["search_query"] = "must not search instead of asking"
            self.assertEqual(self.call(invalid)["status"], "evaluation_failed")

    def test_count_and_text_budget_are_enforced(self):
        self.assertEqual(self.call(self.reply, final_k=1)["status"], "evaluation_failed")
        self.assertEqual(self.call(self.reply, max_evidence_chars=1)["status"], "evaluation_failed")

    def test_timeout_and_malformed_output_do_not_retry_or_silently_answer(self):
        calls = []

        def timeout(messages):
            calls.append(messages)
            raise TimeoutError("private provider secret")

        result = evaluate_evidence("question", self.pool, timeout)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["failure_reason"], "evaluation_timeout")
        self.assertIsNone(result["proposal"])
        self.assertNotIn("private provider secret", json.dumps(result))
        malformed = ["not json", "[]", json.dumps({**self.reply, "action": []}),
                     json.dumps({**self.reply, "extra": "field"})]
        for raw in malformed:
            result = evaluate_evidence("question", self.pool, lambda _: raw)
            self.assertEqual(result["status"], "evaluation_failed")
            self.assertEqual(result["model_attempts"], 1)

    def test_empty_or_duplicate_inputs_do_not_call_generator(self):
        def forbidden(_):
            self.fail("Generator must not run")

        result = evaluate_evidence("question", [], forbidden)
        self.assertEqual(result["model_attempts"], 0)
        self.assertIsNone(result["proposal"])
        with self.assertRaisesRegex(ValueError, "duplicate_input_chunk_id"):
            evaluate_evidence("question", [self.pool[0], self.pool[0]], forbidden)

    def test_exact_quote_does_not_prove_the_claim_is_true(self):
        wrong = copy.deepcopy(self.reply)
        wrong["covered_points"][0]["point"] = "The IP is guaranteed never to change."
        result = self.call(wrong)
        # Deliberately demonstrates the mechanical validator's limitation.
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["coverage_verified"])


if __name__ == "__main__":
    unittest.main()
