"""ASGI -> real RAG service/parser/controller; no external model or embedding calls."""

from functools import partial
import json
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.requirement_evaluation import evaluate_requirements
from app.provider_errors import ModelProviderError


QUESTION = "数据库连接关闭后应用应如何恢复？"
PASSAGE = {
    "chunk_id": "H1:1", "content": "应用应检测中断、重新解析服务名并重连。",
    "metadata": {"document_id": "http-fixture", "section_id": "H1",
                 "source_url": "https://example.invalid/http-fixture"},
}


def evaluation(action):
    complete = action == "answer"
    proposal = {
        "requirements": [{"requirement_id": "R1", "requirement": "说明恢复方式",
                          "question_quote": QUESTION}],
        "assessments": [{"requirement_id": "R1", "coverage": "covered" if complete else "missing",
            "evidence": [{"chunk_id": "H1:1", "quote": PASSAGE["content"]}] if complete else [],
            "gap": None if complete else "仍需恢复方法或现场信息"}],
        "action": action, "selected_chunk_ids": ["H1:1"],
        "missing_knowledge": ["恢复方法"] if action in {"retrieve", "insufficient"} else [],
        "missing_user_information": ["具体错误文本", "应用和数据库的运行方式"] if action == "clarify" else [],
        "search_query": "connection recovery details" if action == "retrieve" else None,
        "reason": "Constructed fixture, not a real model judgment.",
    }
    return partial(evaluate_requirements,
                   generate=lambda _: json.dumps(proposal, ensure_ascii=False))


def answer():
    return {"status": "answer", "answer": "检测中断并重连。[H1:1]",
            "citation_ids": ["H1:1"], "missing_information": []}


class HttpApiTests(unittest.TestCase):
    def client(self, *, evaluate=None, retrieve=None, generate=None):
        return TestClient(create_app(
            retrieve=retrieve if retrieve is not None else Mock(return_value=[PASSAGE]),
            evaluate=evaluate if evaluate is not None else evaluation("answer"),
            generate=generate if generate is not None else Mock(return_value=answer()),
        ))

    def test_default_app_is_live_but_not_silently_configured(self):
        client = TestClient(create_app())
        self.assertEqual(client.get("/health").json(), {"status": "ok"})
        response = client.post("/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(response.headers["X-Request-ID"], response.json()["request_id"])
        with self.assertRaises(ValueError):
            create_app(retrieve=Mock())

    def test_invalid_requests_stop_before_retrieval(self):
        retrieve = Mock(return_value=[PASSAGE])
        client = self.client(retrieve=retrieve)
        for payload in ({}, {"question": "   "}, {"question": 123},
                        {"question": "x" * 4001}, {"question": QUESTION, "max_evaluations": 100}):
            with self.subTest(payload_type=list(payload)):
                self.assertEqual(client.post("/ask", json=payload).status_code, 422)
        retrieve.assert_not_called()

    def test_answer_passes_through_real_service_with_only_public_fields(self):
        retrieve, generate = Mock(return_value=[PASSAGE]), Mock(return_value=answer())
        response = self.client(retrieve=retrieve, generate=generate).post(
            "/ask", json={"question": "  " + QUESTION + "  "})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"request_id", "status", "message", "required_information", "sources"})
        self.assertEqual(body["status"], "answer")
        self.assertEqual(body["sources"], [{"chunk_id": "H1:1", **PASSAGE["metadata"]}])
        self.assertEqual(body["required_information"], [])
        retrieve.assert_called_once_with(QUESTION)
        generate.assert_called_once_with(QUESTION, [PASSAGE])

    def test_needs_user_is_successful_processing_and_does_not_generate(self):
        generate = Mock()
        response = self.client(evaluate=evaluation("clarify"), generate=generate).post(
            "/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "needs_user")
        self.assertEqual(body["required_information"], ["具体错误文本", "应用和数据库的运行方式"])
        self.assertEqual(body["sources"], [])  # Retrieved does not mean cited.
        generate.assert_not_called()

    def test_evaluation_timeout_is_503_without_clarification_or_generation(self):
        raw = Mock(side_effect=TimeoutError("private provider detail"))
        generate, retrieve = Mock(), Mock(return_value=[PASSAGE])
        response = self.client(evaluate=partial(evaluate_requirements, generate=raw),
                               retrieve=retrieve, generate=generate).post("/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["required_information"], [])
        self.assertEqual(body["sources"], [])
        self.assertIn("评价服务", body["message"])
        self.assertNotIn("private provider detail", response.text)
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(retrieve.call_count, 1)
        generate.assert_not_called()

    def test_typed_provider_failure_survives_evaluation_and_generation(self):
        for stage in ('evaluate', 'generate'):
            error = ModelProviderError('provider_balance_insufficient', upstream_status=402)
            callback = Mock(side_effect=error)
            with self.subTest(stage=stage):
                client = (self.client(evaluate=partial(evaluate_requirements, generate=callback)) if stage == 'evaluate'
                          else self.client(generate=callback))
                response = client.post('/ask', json={'question': QUESTION})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['status'], 'failed')
                self.assertIn('额度不足', response.json()['message'])
                self.assertEqual(response.json()['required_information'], [])
                self.assertEqual(callback.call_count, 1)

    def test_insufficient_and_bounded_stop_are_not_server_errors(self):
        for action, expected in (("insufficient", "insufficient"), ("retrieve", "insufficient")):
            with self.subTest(action=action):
                generate = Mock()
                response = self.client(evaluate=evaluation(action), generate=generate).post(
                    "/ask", json={"question": QUESTION})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], expected)
                generate.assert_not_called()

    def test_unclassified_failure_is_500_not_a_claim_of_temporary_unavailability(self):
        generate = Mock()
        response = self.client(retrieve=Mock(side_effect=RuntimeError("private backend detail")),
                               generate=generate).post("/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(response.json()["required_information"], [])
        self.assertNotIn("private backend detail", response.text)
        generate.assert_not_called()

    def test_assessment_cannot_rely_on_a_candidate_omitted_from_selection(self):
        # New real-run failure class: authentic candidate, but not selected.
        raw = json.loads(evaluation("answer").keywords["generate"]([]))
        other = {**PASSAGE, "chunk_id": "H2:1", "content": "另一条恢复证据。"}
        raw["assessments"][0]["evidence"] = [
            {"chunk_id": "H2:1", "quote": other["content"]},
        ]
        generate = Mock()
        response = self.client(
            retrieve=Mock(return_value=[PASSAGE, other]), generate=generate,
            evaluate=partial(evaluate_requirements, generate=lambda _: json.dumps(raw)),
        ).post("/ask", json={"question": QUESTION})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(response.json()["sources"], [])
        self.assertEqual(response.json()["required_information"], [])
        generate.assert_not_called()

    def test_repair_succeeds_or_stops_without_generating(self):
        good = json.loads(evaluation("answer").keywords["generate"]([]))
        other = {**PASSAGE, "chunk_id": "H2:1", "content": "另一条恢复证据。"}
        bad = json.loads(json.dumps(good))
        bad["assessments"][0]["evidence"] = [{"chunk_id": "H2:1", "quote": other["content"]}]
        for fixed in (True, False):
            with self.subTest(fixed=fixed):
                model = Mock(side_effect=[json.dumps(bad), json.dumps(good if fixed else bad)])
                generator = Mock(return_value=answer())
                response = self.client(retrieve=Mock(return_value=[PASSAGE, other]), generate=generator,
                    evaluate=partial(evaluate_requirements, generate=model, repair_unselected_once=True)
                ).post("/ask", json={"question": QUESTION})
                self.assertEqual(model.call_count, 2)
                self.assertEqual(response.status_code, 200 if fixed else 500)
                self.assertEqual(response.json()["status"], "answer" if fixed else "failed")
                self.assertEqual(generator.call_count, 1 if fixed else 0)
                if not fixed:
                    self.assertEqual(response.json()["sources"], [])


if __name__ == "__main__":
    unittest.main()
