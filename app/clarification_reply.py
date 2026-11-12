"""Carry evidence-linked findings into a clarification without another LLM call.

Checks establish traceability/shape, not truth, entailment or full question coverage.
"""

import json
import re

from app.answering import validate_answer
from app.quote_alignment import align_quote


def validate_known_findings(proposal: dict, candidates: list[dict]) -> None:
    findings = proposal.get("known_findings")
    if not isinstance(findings, list):
        raise ValueError("invalid_known_findings")
    assessments = proposal["assessments"]
    if len(findings) > len(assessments):
        raise ValueError("invalid_known_findings")
    if proposal["action"] != "clarify":
        if findings:
            raise ValueError("known_findings_only_for_clarify")
        return
    by_requirement = {item["requirement_id"]: item for item in assessments}
    by_chunk = {item["chunk_id"]: item for item in candidates}
    selected = set(proposal["selected_chunk_ids"])
    seen = set()
    for finding in findings:
        if (not isinstance(finding, dict)
                or set(finding) != {"requirement_id", "text", "evidence"}
                or not isinstance(finding["requirement_id"], str)):
            raise ValueError("invalid_known_finding")
        requirement = finding["requirement_id"]
        if requirement in seen or requirement not in by_requirement:
            raise ValueError("invalid_known_finding_requirement")
        seen.add(requirement)
        assessment = by_requirement[requirement]
        if assessment["coverage"] == "missing":
            raise ValueError("finding_for_missing_requirement")
        text, evidence = finding["text"], finding["evidence"]
        if (not isinstance(text, str) or not text.strip()
                or re.search(r"\[[^\[\]\r\n]+:\d+\]", text)):
            raise ValueError("invalid_known_finding_text")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
            raise ValueError("invalid_known_finding_evidence")
        permitted = {(e["chunk_id"], e["quote"]) for e in assessment["evidence"]}
        used = set()
        for ref in evidence:
            if (not isinstance(ref, dict) or set(ref) != {"chunk_id", "quote"}
                    or not all(isinstance(ref[k], str) and ref[k].strip() for k in ref)):
                raise ValueError("invalid_known_finding_reference")
            key, quote = ref["chunk_id"], ref["quote"]
            if key not in selected or key not in by_chunk:
                raise ValueError("finding_support_not_selected")
            if (key, quote) not in permitted or align_quote(by_chunk[key]["content"], quote) is None:
                raise ValueError("finding_support_not_in_assessment")
            if (key, quote) in used:
                raise ValueError("duplicate_known_finding_reference")
            used.add((key, quote))
    covered = {a["requirement_id"] for a in assessments if a["coverage"] == "covered"}
    if not covered <= seen:
        raise ValueError("covered_requirement_missing_public_finding")


def render_clarification(proposal: dict, candidates: list[dict]) -> dict:
    """Use validated public text, not internal reason/analysis or hardcoded causes."""
    validate_known_findings(proposal, candidates)
    parts, citation_ids = [], []
    for finding in proposal["known_findings"]:
        ids = list(dict.fromkeys(ref["chunk_id"] for ref in finding["evidence"]))
        citation_ids.extend(key for key in ids if key not in citation_ids)
        parts.append(finding["text"].strip() + " " + "".join("[" + key + "]" for key in ids))
    parts.append("为回答本轮问题，还需要确认以下信息：")
    required = proposal["missing_user_information"]
    parts.extend(f"{index}. {text.strip()}" for index, text in enumerate(required, 1))
    return validate_answer(json.dumps({"status": "clarify", "answer": "\n\n".join(parts),
        "citation_ids": citation_ids, "missing_information": required}, ensure_ascii=False), candidates)


def render_insufficiency(proposal: dict, candidates: list[dict], gaps: list[str]) -> dict:
    """Explain a terminal evidence gap without inventing a conclusion or retrying.

    Show only source-aligned quotations from selected evidence, labelled as raw
    evidence rather than as a completed answer. No user information is requested.
    """
    missing = list(dict.fromkeys(text.strip().rstrip('。；; ') for text in gaps
        if isinstance(text, str) and text.strip().rstrip('。；; ')))
    if not missing:
        missing = ['足以完整支持本轮结论的资料']
    by_id = {c['chunk_id']: c for c in candidates}
    quotations, citation_ids, quote_chars = [], [], 0
    for assessment in proposal.get('assessments', []):
        if assessment.get('coverage') not in ('covered', 'partial'):
            continue
        for ref in assessment.get('evidence', []):
            key = ref['chunk_id']
            if key not in by_id:
                raise ValueError('insufficiency_support_not_selected')
            aligned = align_quote(by_id[key]['content'], ref['quote'])
            if aligned is None:
                raise ValueError('insufficiency_quote_not_in_passage')
            quote = aligned['source_quote']
            if key in citation_ids or len(citation_ids) >= 3 or quote_chars + len(quote) > 600:
                continue
            quote_chars += len(quote)
            citation_ids.append(key)
            quotations.append('> ' + quote.replace('\n', '\n> ') + ' [' + key + ']')
    parts = ['本次检索所得资料不足以完整回答，暂时无法给出有充分依据的结论。']
    if quotations:
        parts.append('已有证据原文（供核对，不代表已覆盖全部问题）：\n\n' + '\n\n'.join(quotations))
    parts.append('尚缺少以下依据：\n' + '\n'.join(f'{i}. {text}。' for i,text in enumerate(missing, 1)))
    return validate_answer(json.dumps({'status':'insufficient','answer':'\n\n'.join(parts),
        'citation_ids':citation_ids,'missing_information':missing},ensure_ascii=False),candidates)
