"""Fixed-plan contracts only; no claims about model semantic correctness."""

from copy import deepcopy
import json
import unittest

from app.requirement_planning import (
    build_fixed_evaluation_messages, build_planning_messages,
    parse_fixed_evaluation, parse_requirement_plan,
)


class RequirementPlanningTests(unittest.TestCase):
    def setUp(self):
        self.question = "Why did the connection close? What must the application do?"
        self.rows = [
            {"requirement_id": "R1", "requirement": "Explain closure",
             "question_quote": "Why did the connection close?"},
            {"requirement_id": "R2", "requirement": "Explain recovery action",
             "question_quote": "What must the application do?"}]
        self.plan = parse_requirement_plan(self.question, json.dumps({"requirements": self.rows}))
        self.pool = [{"chunk_id": "C1", "content": "The server restarted, closing old connections."},
                     {"chunk_id": "C2", "content": "Reconnect using the service name."}]
        self.complete = {"assessments": [
            {"requirement_id": "R1", "coverage": "covered", "evidence": [
                {"chunk_id": "C1", "quote": self.pool[0]["content"]}], "gap": None},
            {"requirement_id": "R2", "coverage": "covered", "evidence": [
                {"chunk_id": "C2", "quote": self.pool[1]["content"]}], "gap": None}],
            "known_findings": [], "action": "answer", "selected_chunk_ids": ["C1", "C2"],
            "missing_knowledge": [], "missing_user_information": [], "search_query": None,
            "reason": "Both fixed obligations have support."}

    def test_planner_receives_only_original_question(self):
        messages = build_planning_messages(self.question)
        self.assertEqual(json.loads(messages[1]["content"]), {"question": self.question})
        with self.assertRaisesRegex(ValueError, "unexpected_plan_fields"):
            parse_requirement_plan(self.question, json.dumps({"requirements": self.rows, "answer": "leak"}))

    def test_plan_rejects_fabricated_anchor_and_duplicate_ids_but_allows_shared_anchor(self):
        bad = deepcopy(self.rows)
        bad[0]["question_quote"] = "invented span"
        with self.assertRaisesRegex(ValueError, "plan_quote_not_in_question"):
            parse_requirement_plan(self.question, json.dumps({"requirements": bad}))
        bad = deepcopy(self.rows)
        bad[1]["requirement_id"] = "R1"
        with self.assertRaisesRegex(ValueError, "duplicate_plan_requirement_id"):
            parse_requirement_plan(self.question, json.dumps({"requirements": bad}))
        same_anchor = deepcopy(self.rows)
        same_anchor[0]["question_quote"] = same_anchor[1]["question_quote"]
        plan = parse_requirement_plan(self.question, json.dumps({"requirements": same_anchor}))
        self.assertEqual(len(plan.requirements), 2)

    def test_evaluator_cannot_replace_or_drop_fixed_requirements(self):
        with self.assertRaisesRegex(ValueError, "evaluation_must_not_replace_plan"):
            parse_fixed_evaluation(self.plan, self.pool, json.dumps({**self.complete, "requirements": []}))
        missing = deepcopy(self.complete)
        missing["assessments"].pop()
        with self.assertRaisesRegex(ValueError, "requirement_assessment_ids_mismatch"):
            parse_fixed_evaluation(self.plan, self.pool, json.dumps(missing))
        renamed = deepcopy(self.complete)
        renamed["assessments"][1]["requirement_id"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "requirement_assessment_ids_mismatch"):
            parse_fixed_evaluation(self.plan, self.pool, json.dumps(renamed))

    def test_evidence_changes_coverage_without_changing_plan(self):
        partial = deepcopy(self.complete)
        partial["assessments"][1].update(coverage="missing", evidence=[], gap="Application recovery action")
        partial.update(action="retrieve", selected_chunk_ids=["C1"],
                       missing_knowledge=["Application recovery action"], search_query="application connection recovery")
        first = parse_fixed_evaluation(self.plan, self.pool[:1], json.dumps(partial))
        second = parse_fixed_evaluation(self.plan, self.pool, json.dumps(self.complete))
        self.assertEqual(first["overall_coverage"], "partial")
        self.assertEqual(second["overall_coverage"], "complete")
        self.assertEqual(first["requirements"], second["requirements"])
        first["requirements"][0]["requirement"] = "Mutate external projection"
        self.assertEqual(self.plan.records(), self.rows)
        payload = json.loads(build_fixed_evaluation_messages(self.plan, self.pool)[1]["content"])
        self.assertEqual(payload["requirements"], self.rows)


if __name__ == "__main__":
    unittest.main()
