"""Public findings survive rendering; traceability is not semantic verification."""

from copy import deepcopy
from functools import partial
import json
import unittest
from unittest.mock import Mock

from app.main import to_public_response
from app.rag_service import run_rag_service
from app.requirement_evaluation import evaluate_requirements
from app.clarification_reply import render_insufficiency


QUESTION = "启动检查通过后还是断连，是密码错误吗？"
CONTENT = "Compose waits for healthchecks on startup."
CHUNK = {"chunk_id": "S2:1", "content": CONTENT, "metadata": {
    "document_id": "startup", "section_id": "S2", "source_url": "https://example.invalid/startup"}}


def proposal():
    return {"requirements": [
        {"requirement_id": "R1", "requirement": "解释启动检查的作用范围", "question_quote": "启动检查通过后还是断连"},
        {"requirement_id": "R2", "requirement": "判断此次断连根因", "question_quote": "是密码错误吗"}],
        "assessments": [
            {"requirement_id": "R1", "coverage": "covered", "evidence": [{"chunk_id": "S2:1", "quote": CONTENT}], "gap": None},
            {"requirement_id": "R2", "coverage": "missing", "evidence": [], "gap": "缺少故障时完整报错"}],
        "action": "clarify", "selected_chunk_ids": ["S2:1"], "missing_knowledge": [],
        "missing_user_information": ["连接失败时的完整报错"], "search_query": None,
        "reason": "INTERNAL_REASON_DO_NOT_RENDER",
        "known_findings": [{"requirement_id": "R1", "text": "健康检查说明启动时就绪，不能保证此后连接一直正常。",
                            "evidence": [{"chunk_id": "S2:1", "quote": CONTENT}]}]}


def evaluate(reply, **kwargs):
    return evaluate_requirements(QUESTION, [CHUNK], lambda _: json.dumps(reply, ensure_ascii=False),
                                 include_user_reply=True, **kwargs)


class ClarificationReplyTests(unittest.TestCase):
    def test_finding_reaches_http_with_sources_without_an_answer_model_call(self):
        reply = proposal()
        raw = Mock(return_value=json.dumps(reply, ensure_ascii=False))
        generate = Mock(side_effect=AssertionError("No additional LLM call is required"))
        result = run_rag_service(QUESTION, retrieve=lambda _: [CHUNK],
            evaluate=partial(evaluate_requirements, generate=raw, include_user_reply=True),
            generate=generate)
        self.assertEqual(result["status"], "needs_user")
        self.assertIn(reply["known_findings"][0]["text"], result["message"])
        self.assertIn("[S2:1]", result["message"])
        self.assertNotIn("INTERNAL_REASON_DO_NOT_RENDER", result["message"])
        self.assertEqual(result["counts"]["generations"], 0)
        self.assertEqual(raw.call_count, 1)
        self.assertFalse(result["coverage_verified"])
        generate.assert_not_called()
        code, public = to_public_response(result, "fixture-request")
        self.assertEqual(code, 200)
        self.assertEqual(public.sources[0].chunk_id, "S2:1")
        self.assertEqual(public.required_information, ["连接失败时的完整报错"])

    def test_covered_requirement_cannot_silently_drop_its_public_finding(self):
        reply = proposal()
        reply["known_findings"] = []
        result = evaluate(reply)
        self.assertEqual(result["failure_reason"], "covered_requirement_missing_public_finding")

    def test_missing_and_duplicate_requirements_cannot_supply_findings(self):
        for rid in ("R2", "UNKNOWN"):
            with self.subTest(rid=rid):
                reply = proposal()
                reply["known_findings"][0]["requirement_id"] = rid
                self.assertEqual(evaluate(reply)["status"], "evaluation_failed")
        reply = proposal()
        reply["known_findings"] *= 2
        self.assertEqual(evaluate(reply)["status"], "evaluation_failed")

    def test_finding_quote_must_be_in_same_assessment_and_selected_source(self):
        for ref in ({"chunk_id": "UNKNOWN", "quote": CONTENT},
                    {"chunk_id": "S2:1", "quote": "invented quotation"}):
            with self.subTest(ref=ref):
                reply = proposal()
                reply["known_findings"][0]["evidence"] = [ref]
                self.assertEqual(evaluate(reply)["status"], "evaluation_failed")

    def test_no_supported_findings_is_allowed_when_all_requirements_are_missing(self):
        reply = proposal()
        reply["assessments"][0].update(coverage="missing", evidence=[], gap="缺少支持资料")
        reply["known_findings"] = []
        result = run_rag_service(QUESTION, retrieve=lambda _: [CHUNK],
            evaluate=partial(evaluate_requirements, generate=lambda _: json.dumps(reply), include_user_reply=True),
            generate=Mock())
        self.assertEqual(result["status"], "needs_user")
        self.assertEqual(result["answer"]["citation_ids"], [])
        self.assertNotIn("健康检查", result["message"])

    def test_existing_format_remains_usable_only_in_original_mode(self):
        reply = proposal()
        reply.pop("known_findings")
        raw = lambda _: json.dumps(reply)
        old = evaluate_requirements(QUESTION, [CHUNK], raw)
        new = evaluate_requirements(QUESTION, [CHUNK], raw, include_user_reply=True)
        self.assertEqual(old["status"], "proposed")
        self.assertEqual(new["status"], "evaluation_failed")

    def test_unsupported_meaning_can_pass_structural_checks_and_stays_unverified(self):
        reply = proposal()
        reply["known_findings"][0]["text"] = "已经确定是密码错误。"
        result = evaluate(reply)
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["coverage_verified"])
        self.assertEqual(result["semantic_review"], "pending")

    def test_two_strings_do_not_prove_two_semantic_information_categories(self):
        reply = proposal()
        reply["missing_user_information"] = ["报错、部署配置、监控、重启记录都提供", "数据库连接池参数也提供"]
        result = evaluate(reply)
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["coverage_verified"])

    def test_service_rechecks_findings_even_when_an_injected_evaluator_skips_parser(self):
        reply = proposal()
        reply["known_findings"][0]["evidence"][0]["quote"] = "invented quotation"
        result = run_rag_service(QUESTION, retrieve=lambda _: [CHUNK],
            evaluate=lambda q, c: {"status": "proposed", "proposal": deepcopy(reply), "selected_evidence": [CHUNK]},
            generate=Mock())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "invalid_public_clarification")


class InsufficiencyReplyTests(unittest.TestCase):
    def test_missing_evidence_reply_preserves_verbatim_support_without_claiming_completion(self):
        reply=proposal()
        rendered=render_insufficiency(reply,[CHUNK],['故障时段的证据。','故障时段的证据。'])
        self.assertEqual(rendered['status'],'insufficient')
        self.assertIn(CONTENT,rendered['answer'])
        self.assertEqual(rendered['citation_ids'],['S2:1'])
        self.assertEqual(rendered['missing_information'],['故障时段的证据'])
        self.assertNotIn('。。',rendered['answer'])
        self.assertNotIn('请提供',rendered['answer'])

    def test_unselected_or_fabricated_support_cannot_escape_through_terminal_rendering(self):
        for ref in ({'chunk_id':'unknown:1','quote':CONTENT},
                    {'chunk_id':'S2:1','quote':'invented source text'}):
            with self.subTest(ref=ref):
                reply=proposal()
                reply['assessments'][0]['evidence']=[ref]
                with self.assertRaises(ValueError):
                    render_insufficiency(reply,[CHUNK],['缺少现场依据'])


if __name__ == "__main__":
    unittest.main()
