"""Check source context at model boundaries, without asserting semantic quality."""
from copy import deepcopy
from functools import partial
import json
import unittest

from app.answering import build_messages, validate_answer
from app.evidence_evaluation import build_evaluation_messages
from app.rag_service import run_rag_service
from app.requirement_evaluation import evaluate_requirements


def passage():
    return {"chunk_id": "T1:001", "content": "A response can be lost after processing.",
            "metadata": {"document_id": "fixture", "section_id": "T1",
                         "source_url": "https://example.invalid/source"}}


class SourceContextTests(unittest.TestCase):
    def test_both_model_boundaries_keep_same_source_fields_without_labels_or_mutation(self):
        item = passage()
        expected = {"technology": "EF Core", "applicable_version": None,
                    "evidence_scope": "Commit failure", "generalizable_as": "Response loss"}
        item["metadata"].update(expected, expected_answer="DO_NOT_SEND", score=100,
                                arbitrary_field="DO_NOT_SEND")
        item["reference_answer"] = "DO_NOT_SEND"
        before = deepcopy(item)
        evaluation = build_evaluation_messages("question", [item])
        generation = build_messages("question", [item])
        self.assertEqual(json.loads(evaluation[1]["content"])["passages"][0]["source_metadata"], expected)
        self.assertEqual(json.loads(generation[1]["content"])["evidence"][0]["source_metadata"], expected)
        self.assertNotIn("DO_NOT_SEND", str(evaluation) + str(generation))
        self.assertEqual(item, before)

    def test_missing_and_invalid_applicability_does_not_invent_defaults(self):
        for metadata in ({}, {"technology": "x" * 161, "applicable_version": ["2.0"],
                              "evidence_scope": " ", "generalizable_as": False}):
            with self.subTest(metadata=metadata):
                item = passage()
                item["metadata"].update(metadata)
                payload = json.loads(build_messages("question", [item])[1]["content"])
                self.assertNotIn("source_metadata", payload["evidence"][0])

    def test_source_context_is_not_added_to_public_citation_contract(self):
        item = passage()
        item["metadata"].update(technology="EF Core", applicable_version=None)
        answer = {"status": "answer", "answer": "Example. [T1:001]",
                  "citation_ids": ["T1:001"], "missing_information": []}
        parsed = validate_answer(json.dumps(answer), [item])
        self.assertEqual(set(parsed["sources"][0]),
                         {"chunk_id", "document_id", "section_id", "source_url"})

    def test_service_preserves_context_through_selection_to_generation(self):
        question = "Could the response be lost?"
        item = passage()
        expected = {"technology": "Generic API", "applicable_version": None}
        item["metadata"].update(expected)
        observed = {}

        def raw_evaluation(messages):
            observed["evaluation"] = json.loads(messages[1]["content"])["passages"][0]["source_metadata"]
            return json.dumps({
                "requirements": [{"requirement_id": "R1", "requirement": "Explain response loss",
                                  "question_quote": question}],
                "assessments": [{"requirement_id": "R1", "coverage": "covered",
                    "evidence": [{"chunk_id": item["chunk_id"], "quote": item["content"]}], "gap": None}],
                "action": "answer", "selected_chunk_ids": [item["chunk_id"]],
                "missing_knowledge": [], "missing_user_information": [],
                "search_query": None, "reason": "Scripted data-flow fixture"})

        def generate(q, evidence):
            payload = json.loads(build_messages(q, evidence)[1]["content"])
            observed["generation"] = payload["evidence"][0]["source_metadata"]
            return {"status": "answer", "answer": "It can be lost. [T1:001]",
                    "citation_ids": ["T1:001"], "missing_information": []}

        result = run_rag_service(question, retrieve=lambda _: [item],
            evaluate=partial(evaluate_requirements, generate=raw_evaluation), generate=generate)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(observed, {"evaluation": expected, "generation": expected})
        self.assertFalse(result["coverage_verified"])
