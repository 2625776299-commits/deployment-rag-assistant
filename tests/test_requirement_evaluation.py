"""Offline adversarial checks; all generation callbacks return hand-built fixtures."""

import copy
import json
import unittest

from app.bounded_retrieval import run_bounded_retrieval
from app.requirement_evaluation import build_requirement_messages, evaluate_requirements


class RequirementEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.question = "Why does the old connection fail? What must the application do?"
        self.pool = [{"chunk_id": "P1", "content": "The service name stays stable; the IP may change."},
                     {"chunk_id": "P2", "content": "Old connections close. Detect this and reconnect."}]
        self.reply = {
            "requirements": [
                {"requirement_id": "R1", "requirement": "Explain connection failure",
                 "question_quote": "Why does the old connection fail?"},
                {"requirement_id": "R2", "requirement": "Explain application handling",
                 "question_quote": "What must the application do?"}],
            "assessments": [
                {"requirement_id": "R1", "coverage": "covered", "evidence": [
                    {"chunk_id": "P1", "quote": "the IP may change"},
                    {"chunk_id": "P2", "quote": "Old connections close"}], "gap": None},
                {"requirement_id": "R2", "coverage": "covered", "evidence": [
                    {"chunk_id": "P2", "quote": "Detect this and reconnect"}], "gap": None}],
            "action": "answer", "selected_chunk_ids": ["P1", "P2"],
            "missing_knowledge": [], "missing_user_information": [],
            "search_query": None, "reason": "Both requirements have selected support."}

    def call(self, reply, **kwargs):
        return evaluate_requirements(self.question, self.pool,
            lambda _: json.dumps(reply), **kwargs)

    def gap_reply(self):
        reply = copy.deepcopy(self.reply)
        reply["assessments"][1].update(coverage="missing", evidence=[], gap="Application recovery")
        reply.update(action="retrieve", missing_knowledge=["Application recovery"],
                     search_query="application recovery after old connection closes")
        return reply

    def test_valid_proposal_is_derived_but_not_semantically_verified(self):
        before = copy.deepcopy(self.reply)
        result = self.call(self.reply)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["proposal"]["overall_coverage"], "complete")
        self.assertFalse(result["coverage_verified"])
        self.assertEqual(result["semantic_review"], "pending")
        self.assertEqual(self.reply, before)

    def test_original_question_quote_gets_only_one_shared_quote_correction(self):
        bad = copy.deepcopy(self.reply)
        bad['requirements'][0]['question_quote'] = 'A paraphrased question from the retrieval hint'
        for second, expected in ((self.reply, 'proposed'), (bad, 'evaluation_failed')):
            calls = []
            def generate(messages):
                calls.append(copy.deepcopy(messages))
                return json.dumps(bad if len(calls) == 1 else second)
            result = evaluate_requirements(self.question, self.pool, generate, repair_evidence_quote_once=True)
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['repair_reason'], 'requirement_quote_not_in_question')
            self.assertEqual(result['model_attempts'], 2)
            self.assertEqual(calls[0], calls[1][:2])
            self.assertIn('ORIGINAL question', calls[1][-1]['content'])
            self.assertEqual(json.loads(result['raw_attempts'][0]), bad)
        # A second, different quote error does not create another allowance.
        wrong_evidence = copy.deepcopy(self.reply)
        wrong_evidence['assessments'][0]['evidence'][0]['quote'] = 'invented'
        replies = iter((bad, wrong_evidence, self.reply))
        result = evaluate_requirements(self.question, self.pool, lambda _: json.dumps(next(replies)),
                                       repair_evidence_quote_once=True)
        self.assertEqual(result['model_attempts'], 2)
        self.assertEqual(result['failure_reason'], 'support_quote_not_in_passage')

    def test_duplicate_missing_and_unknown_assessment_ids_fail(self):
        variants = []
        duplicate = copy.deepcopy(self.reply)
        duplicate["assessments"][1]["requirement_id"] = "R1"
        variants.append((duplicate, "duplicate_assessment_id"))
        missing = copy.deepcopy(self.reply)
        missing["assessments"].pop()
        variants.append((missing, "requirement_assessment_ids_mismatch"))
        unknown = copy.deepcopy(self.reply)
        unknown["assessments"][1]["requirement_id"] = "R3"
        variants.append((unknown, "requirement_assessment_ids_mismatch"))
        for reply, reason in variants:
            with self.subTest(reason=reason):
                result = self.call(reply)
                self.assertEqual(result["failure_reason"], reason)
                self.assertIsNone(result["proposal"])

    def test_requirement_duplicates_and_invented_question_quotes_fail(self):
        reply = copy.deepcopy(self.reply)
        reply["requirements"][1]["requirement_id"] = "R1"
        self.assertEqual(self.call(reply)["failure_reason"], "duplicate_requirement_id")
        reply = copy.deepcopy(self.reply)
        reply["requirements"][0]["question_quote"] = "Human answer key not in question"
        self.assertEqual(self.call(reply)["failure_reason"], "requirement_quote_not_in_question")

    def test_incomplete_answer_fails_and_gap_actions_remain_available(self):
        reply = self.gap_reply()
        result = self.call(reply)
        self.assertEqual(result["proposal"]["overall_coverage"], "partial")
        wrong = {**reply, "action": "answer", "missing_knowledge": [], "search_query": None}
        self.assertEqual(self.call(wrong)["failure_reason"], "answer_requires_complete_coverage")
        clarify = {**reply, "action": "clarify", "search_query": None,
                   "missing_knowledge": [], "missing_user_information": ["Exact runtime error"]}
        insufficient = {**reply, "action": "insufficient", "search_query": None,
                        "missing_knowledge": ["Private records unavailable"]}
        self.assertEqual(self.call(clarify)["status"], "proposed")
        self.assertEqual(self.call(insufficient)["status"], "proposed")

    def test_state_evidence_and_gap_inconsistencies_fail(self):
        mutations = [
            {"coverage": "covered", "evidence": []},
            {"coverage": "covered", "gap": "Still missing something"},
            {"coverage": "partial", "gap": None},
            {"coverage": "missing", "gap": "Absent", "evidence": self.reply["assessments"][1]["evidence"]},
            {"coverage": []}]
        for change in mutations:
            with self.subTest(change=change):
                reply = copy.deepcopy(self.reply)
                reply["assessments"][1].update(change)
                self.assertEqual(self.call(reply)["status"], "evaluation_failed")
        reply = {**self.reply, "missing_knowledge": ["Contradicts complete"]}
        self.assertEqual(self.call(reply)["failure_reason"], "complete_coverage_conflicts_with_reported_gaps")

    def test_existing_quote_selection_and_budget_protections_are_reused(self):
        wrong_quote = copy.deepcopy(self.reply)
        wrong_quote["assessments"][1]["evidence"][0]["quote"] = "invented quotation"
        self.assertEqual(self.call(wrong_quote)["failure_reason"], "support_quote_not_in_passage")
        unselected = {**self.reply, "selected_chunk_ids": ["P1"]}
        self.assertEqual(self.call(unselected)["failure_reason"], "support_from_unselected_passage")
        self.assertEqual(self.call(self.reply, final_k=1)["status"], "evaluation_failed")
        self.assertEqual(self.call(self.reply, max_evidence_chars=1)["status"], "evaluation_failed")

    def test_bounded_repair_for_unselected_support(self):
        invalid = {**self.reply, "selected_chunk_ids": ["P1"]}
        calls = []
        replies = iter([json.dumps(invalid), json.dumps(self.reply)])
        def scripted(messages):
            calls.append(copy.deepcopy(messages))
            return next(replies)
        result = evaluate_requirements(self.question, self.pool, scripted, repair_unselected_once=True)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["model_attempts"], 2)
        self.assertEqual(len(result["raw_attempts"]), 2)
        self.assertFalse(result["coverage_verified"])
        self.assertIn('P2', calls[1][-1]["content"])
        self.assertIn('at most 3 passages', calls[1][-1]["content"])

        calls.clear()
        def still_invalid(messages):
            calls.append(messages)
            return json.dumps(invalid)
        result = evaluate_requirements(self.question, self.pool, still_invalid, repair_unselected_once=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertEqual(result["failure_reason"], "support_from_unselected_passage")
        self.assertEqual(result["selected_evidence"], [])

    def test_repair_is_not_used_for_other_errors_or_timeouts(self):
        calls = []
        def malformed(messages):
            calls.append(messages)
            return "not JSON"
        result = evaluate_requirements(self.question, self.pool, malformed, repair_unselected_once=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["failure_reason"], "response_is_not_json")

    def test_repair_selection_count_unknown_and_duplicates(self):
        for selected, expected_reason in (
            (["P1", "P2", "P3", "P4"], "invalid_selected_chunk_ids"),
            (["P1", "UNKNOWN"], "unknown_selected_chunk_id"),
            (["P1", "P1"], "invalid_selected_chunk_ids"),
        ):
            with self.subTest(selected=selected):
                bad = copy.deepcopy(self.reply)
                bad["selected_chunk_ids"] = selected
                calls = []
                replies = iter([json.dumps(bad), json.dumps(self.reply)])
                def scripted(messages):
                    calls.append(copy.deepcopy(messages))
                    return next(replies)
                result = evaluate_requirements(self.question, self.pool, scripted, repair_unselected_once=True)
                self.assertEqual(result["status"], "proposed")
                self.assertEqual(result["repair_reason"], expected_reason)
                feedback = calls[1][-1]["content"]
                self.assertIn("distinct IDs from the provided candidates", feedback)
                if "P3" in selected:
                    self.assertIn('"selected_count": 4', feedback)
                    self.assertIn('P3', feedback)
                if "UNKNOWN" in selected:
                    self.assertIn('UNKNOWN', feedback)

        # A different error in the repair result does not unlock another retry.
        bad = {**self.reply, "selected_chunk_ids": ["P1", "P2", "P3", "P4"]}
        unknown = {**self.reply, "selected_chunk_ids": ["P1", "UNKNOWN"]}
        calls = []
        replies = iter([json.dumps(bad), json.dumps(unknown)])
        def scripted_twice(messages):
            calls.append(messages)
            return next(replies)
        result = evaluate_requirements(self.question, self.pool, scripted_twice, repair_unselected_once=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertEqual(result["failure_reason"], "unknown_selected_chunk_id")

    def test_split_obligations_share_anchor_and_keep_missing_action_through_retrieval(self):
        # A model fixture checks the existing contract and control flow, not
        # whether a live model obeys the decomposition instructions.
        pool = copy.deepcopy(self.pool)
        pool[1]["content"] = "Old connections close. Detect this condition."
        gap = copy.deepcopy(self.reply)
        anchor = "What must the application do?"
        gap["requirements"][1:] = [
            {"requirement_id": "R2", "requirement": "Detect the closed connection",
             "question_quote": anchor},
            {"requirement_id": "R3", "requirement": "Restore connectivity after detection",
             "question_quote": anchor}]
        gap["assessments"][1:] = [
            {"requirement_id": "R2", "coverage": "covered", "evidence": [
                {"chunk_id": "P2", "quote": "Detect this condition"}], "gap": None},
            {"requirement_id": "R3", "coverage": "missing", "evidence": [],
             "gap": "How to restore connectivity after detecting a closed connection"}]
        gap.update(action="retrieve", missing_knowledge=[gap["assessments"][2]["gap"]],
                   search_query="application reconnect after detecting closed connection")
        proposal = evaluate_requirements(self.question, pool, lambda _: json.dumps(gap))
        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["proposal"]["overall_coverage"], "partial")
        self.assertEqual(proposal["proposal"]["requirements"][1]["question_quote"],
                         proposal["proposal"]["requirements"][2]["question_quote"])

        wrong_answer = {**gap, "action": "answer", "missing_knowledge": [], "search_query": None}
        rejected = evaluate_requirements(self.question, pool, lambda _: json.dumps(wrong_answer))
        self.assertEqual(rejected["failure_reason"], "answer_requires_complete_coverage")

        recovery = {"chunk_id": "P3", "content": "Establish a new connection using the service name."}
        complete = copy.deepcopy(gap)
        complete["assessments"][2].update(coverage="covered", gap=None, evidence=[
            {"chunk_id": "P3", "quote": recovery["content"]}])
        complete.update(action="answer", selected_chunk_ids=["P1", "P2", "P3"],
                        missing_knowledge=[], search_query=None)
        queries, evaluated_pools = [], []

        def evaluate(question, candidates):
            ids = [item["chunk_id"] for item in candidates]
            evaluated_pools.append(ids)
            reply = complete if "P3" in ids else gap
            return evaluate_requirements(question, candidates, lambda _: json.dumps(reply))

        def retrieve(query):
            queries.append(query)
            return [recovery]

        controlled = run_bounded_retrieval(self.question, pool, evaluate, retrieve)
        self.assertEqual(queries, [gap["search_query"]])
        self.assertEqual(evaluated_pools, [["P1", "P2"], ["P1", "P2", "P3"]])
        self.assertEqual(controlled["status"], "answer_proposed")
        self.assertEqual([item["chunk_id"] for item in controlled["selected_evidence"]],
                         ["P1", "P2", "P3"])
        self.assertFalse(controlled["coverage_verified"])

    def test_real_quote_with_wrong_meaning_still_passes_mechanical_checks(self):
        wrong = copy.deepcopy(self.reply)
        wrong["assessments"][1]["evidence"] = [
            {"chunk_id": "P1", "quote": "The service name stays stable"}]
        result = self.call(wrong)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["proposal"]["overall_coverage"], "complete")
        self.assertFalse(result["coverage_verified"])

    def test_omitted_requirement_is_not_detectable_from_matching_ids_alone(self):
        wrong = copy.deepcopy(self.reply)
        wrong["requirements"] = wrong["requirements"][:1]
        wrong["assessments"] = wrong["assessments"][:1]
        result = self.call(wrong)
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["coverage_verified"])

    def test_input_projection_failures_and_no_hidden_retries(self):
        pool = [{**self.pool[0], "reference_answer": "DO_NOT_SEND", "expected_action": "answer"}]
        payload = build_requirement_messages(self.question, pool)[1]["content"]
        self.assertNotIn("DO_NOT_SEND", payload)
        self.assertEqual(set(json.loads(payload)), {"question", "passages", "selection_budget"})
        calls = []
        def timeout(messages):
            calls.append(messages)
            raise TimeoutError("private provider message")
        result = evaluate_requirements(self.question, self.pool, timeout)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["failure_reason"], "evaluation_timeout")
        self.assertNotIn("private provider message", json.dumps(result))
        result = evaluate_requirements(self.question, [], lambda _: self.fail("Unexpected call"))
        self.assertEqual(result["model_attempts"], 0)
        for raw in ("invalid", "[]", json.dumps({**self.reply, "unexpected": True})):
            result = evaluate_requirements(self.question, self.pool, lambda _: raw)
            self.assertEqual(result["status"], "evaluation_failed")

    def test_new_proposal_can_drive_existing_controller_without_changing_it(self):
        reply = self.gap_reply()
        reply["selected_chunk_ids"] = ["P1"]
        reply["assessments"][0].update(coverage="partial", gap="Connection lifetime")
        reply["assessments"][0]["evidence"] = reply["assessments"][0]["evidence"][:1]
        used = []
        def evaluate(question, pool):
            fixture = self.reply if len(pool) == 2 else reply
            return evaluate_requirements(question, pool, lambda _: json.dumps(fixture))
        def retrieve(query):
            used.append(query)
            return [self.pool[1]]
        result = run_bounded_retrieval(self.question, self.pool[:1], evaluate, retrieve)
        self.assertEqual(len(used), 1)
        self.assertEqual(result["retrieval_count"], 1)
        self.assertEqual(result["status"], "answer_proposed")
        self.assertFalse(result["coverage_verified"])


if __name__ == "__main__":
    unittest.main()
