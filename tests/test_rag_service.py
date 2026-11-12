"""Offline integration: real coverage parser/controller, hand-written model replies."""

from copy import deepcopy
from functools import partial
import json
import unittest
from unittest.mock import Mock

from app.rag_service import run_rag_service
from app.requirement_evaluation import evaluate_requirements


QUESTION = "连接关闭后应用应该怎么办？"


def passage(key, content):
    return {"chunk_id": key, "content": content, "metadata": {
        "document_id": "fixture-network", "section_id": key.split(":")[0],
        "source_url": "https://example.invalid/fixture",
    }}


P1 = passage("P1:1", "旧连接可能关闭。")
P2 = passage("P2:1", "应用应检测连接中断，重新解析服务名并重连。")


def proposal(action):
    complete = action == "answer"
    return {
        "requirements": [{"requirement_id": "R1", "requirement": "说明应用恢复方式",
                          "question_quote": QUESTION}],
        "assessments": [{"requirement_id": "R1", "coverage": "covered" if complete else "missing",
            "evidence": [{"chunk_id": "P2:1", "quote": P2["content"]}] if complete else [],
            "gap": None if complete else "缺少恢复方法或用户现场信息"}],
        "action": action,
        "selected_chunk_ids": ["P2:1"] if complete else ["P1:1"],
        "missing_knowledge": ["应用恢复连接的方法"] if action in {"retrieve", "insufficient"} else [],
        "missing_user_information": ["完整报错", "应用和数据库启动方式"] if action == "clarify" else [],
        "search_query": "application reconnect after connection closes" if action == "retrieve" else None,
        "reason": "Hand-written control-flow fixture, not a measured model judgment.",
    }


def evaluator(action):
    return partial(evaluate_requirements,
                   generate=lambda _: json.dumps(proposal(action), ensure_ascii=False))


def generated_answer():
    return {"status": "answer", "answer": "应用应检测中断并重连。[P2:1]",
            "citation_ids": ["P2:1"], "missing_information": []}


class RagServiceTests(unittest.TestCase):
    def test_answer_runs_generation_after_coverage_proposal(self):
        retrieve = Mock(return_value=[P2])
        generate = Mock(return_value=generated_answer())
        result = run_rag_service(QUESTION, retrieve=retrieve, evaluate=evaluator("answer"), generate=generate)
        self.assertEqual(result["status"], "answer")
        self.assertFalse(result["coverage_verified"])
        self.assertEqual(result["counts"], {"initial_retrievals": 1, "additional_retrievals": 0,
                                           "evaluations": 1, "generations": 1})
        self.assertEqual(result["answer"]["sources"][0]["source_url"], P2["metadata"]["source_url"])
        generate.assert_called_once_with(QUESTION, [P2])

    def test_retrieval_merge_reassessment_then_answer_preserves_inputs(self):
        before = deepcopy([P1, P2])
        retrieve = Mock(side_effect=[[P1], [deepcopy(P1), P2]])
        def raw_reply(messages):
            payload = json.loads(messages[1]["content"])
            complete = any(p["chunk_id"] == "P2:1" for p in payload["passages"])
            return json.dumps(proposal("answer" if complete else "retrieve"), ensure_ascii=False)
        result = run_rag_service(QUESTION, retrieve=retrieve,
            evaluate=partial(evaluate_requirements, generate=raw_reply),
            generate=lambda q, evidence: generated_answer())
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["counts"], {"initial_retrievals": 1, "additional_retrievals": 1,
                                           "evaluations": 2, "generations": 1})
        self.assertEqual([step["step"] for step in result["trace"]],
                         ["initial_retrieve", "evaluate", "retrieve_and_merge", "evaluate", "generate"])
        evaluations = [step for step in result["trace"] if step["step"] == "evaluate"]
        self.assertEqual(evaluations[0]["overall_coverage"], "missing")
        self.assertEqual(evaluations[0]["missing_knowledge"], ["应用恢复连接的方法"])
        self.assertEqual(evaluations[0]["selected_chunk_ids"], ["P1:1"])
        self.assertEqual(evaluations[1]["overall_coverage"], "complete")
        self.assertEqual(evaluations[1]["missing_knowledge"], [])
        self.assertEqual(evaluations[1]["selected_chunk_ids"], ["P2:1"])
        self.assertEqual(result["evidence"], [P2])
        self.assertEqual([P1, P2], before)

    def test_clarification_and_insufficient_do_not_generate_or_retrieve_again(self):
        for action, status in (("clarify", "needs_user"), ("insufficient", "insufficient")):
            with self.subTest(action=action):
                retrieve, generate = Mock(return_value=[P1]), Mock()
                result = run_rag_service(QUESTION, retrieve=retrieve,
                                        evaluate=evaluator(action), generate=generate)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["error"])
                retrieve.assert_called_once_with(QUESTION)
                generate.assert_not_called()
                self.assertEqual(result["counts"]["generations"], 0)
                if action == "clarify":
                    self.assertEqual(result["required_information"], ["完整报错", "应用和数据库启动方式"])

    def test_evaluation_timeout_and_invalid_json_are_failures_not_gaps(self):
        for output in (TimeoutError("private provider detail"), "not JSON"):
            with self.subTest(output=type(output).__name__):
                raw = Mock(side_effect=output) if isinstance(output, Exception) else Mock(return_value=output)
                retrieve, generate = Mock(return_value=[P1]), Mock()
                result = run_rag_service(QUESTION, retrieve=retrieve,
                    evaluate=partial(evaluate_requirements, generate=raw), generate=generate)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["stage"], "evaluate")
                evaluation_step = next(step for step in result["trace"] if step["step"] == "evaluate")
                self.assertEqual(evaluation_step["failure_reason"], result["error"]["code"])
                self.assertEqual(evaluation_step["missing_knowledge"], [])
                self.assertEqual(result["required_information"], [])
                self.assertNotIn("private provider detail", json.dumps(result))
                self.assertEqual(raw.call_count, 1)
                self.assertEqual(retrieve.call_count, 1)
                generate.assert_not_called()

    def test_retrieval_exceptions_identify_initial_or_additional_stage(self):
        for responses, stage in (([TimeoutError()], "initial_retrieve"),
                                 ([[P1], TimeoutError()], "additional_retrieve")):
            with self.subTest(stage=stage):
                result = run_rag_service(QUESTION, retrieve=Mock(side_effect=responses),
                                        evaluate=evaluator("retrieve"), generate=Mock())
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["stage"], stage)

    def test_empty_retrieval_does_not_claim_a_system_failure(self):
        evaluate, generate = Mock(), Mock()
        result = run_rag_service(QUESTION, retrieve=lambda _: [], evaluate=evaluate, generate=generate)
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["error"])
        evaluate.assert_not_called()
        generate.assert_not_called()

    def test_budget_stops_do_not_authorize_generation(self):
        for limits, reason in (({"max_retrievals": 0}, "retrieval_budget_exhausted"),
                               ({"max_evaluations": 1}, "evaluation_budget_exhausted")):
            with self.subTest(limits=limits):
                retrieve, generate = Mock(return_value=[P1]), Mock()
                result = run_rag_service(QUESTION, retrieve=retrieve,
                                        evaluate=evaluator("retrieve"), generate=generate, **limits)
                self.assertEqual(result["status"], "insufficient")
                self.assertEqual(result["stop_reason"], reason)
                self.assertIn('尚缺少以下依据', result['message'])
                self.assertNotIn('流程已停止', result['message'])
                self.assertEqual(result['answer']['status'], 'insufficient')
                self.assertEqual(retrieve.call_count, 1)
                generate.assert_not_called()
        generate = Mock()
        result = run_rag_service(QUESTION, retrieve=lambda _: [P2], evaluate=evaluator("answer"),
                                generate=generate, max_generations=0)
        self.assertEqual(result["stop_reason"], "generation_budget_exhausted")
        generate.assert_not_called()

    def test_generation_failure_or_fabricated_citation_is_not_an_answer(self):
        for generate in (Mock(side_effect=TimeoutError("private detail")),
                         Mock(return_value={**generated_answer(), "citation_ids": ["fake:1"]})):
            with self.subTest(generate=generate):
                result = run_rag_service(QUESTION, retrieve=lambda _: [P2],
                                        evaluate=evaluator("answer"), generate=generate)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["stage"], "generate")
                self.assertEqual(result["counts"]["generations"], 1)
                self.assertIsNone(result["answer"])

    def test_generator_can_downgrade_an_evaluator_answer_proposal(self):
        reply = {"status": "clarify", "answer": "请补充完整报错。", "citation_ids": [],
                 "missing_information": ["完整报错"]}
        result = run_rag_service(QUESTION, retrieve=lambda _: [P2], evaluate=evaluator("answer"),
                                generate=lambda q, passages: reply)
        self.assertEqual(result["status"], "needs_user")
        self.assertEqual(result["required_information"], ["完整报错"])

    def test_limits_are_checked_before_external_work_and_pool_is_not_silently_trimmed(self):
        retrieve = Mock()
        for limits in ({"max_candidates": 11}, {"max_retrievals": True}, {"max_evaluations": 0}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                run_rag_service(QUESTION, retrieve=retrieve, evaluate=Mock(), generate=Mock(), **limits)
        retrieve.assert_not_called()
        evaluate, generate = Mock(), Mock()
        result = run_rag_service(QUESTION, retrieve=lambda _: [P1, P2], evaluate=evaluate,
                                generate=generate, max_candidates=1)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stop_reason"], "candidate_budget_exhausted")
        evaluate.assert_not_called()
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
