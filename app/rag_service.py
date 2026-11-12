"""Unified RAG service with injected, already configured dependencies.

No clients, retries, HTTP routes, embedding calls, or hidden provider defaults.
Evaluation callbacks return the evaluate_requirements result contract; generation
callbacks return the generate_answer result contract. Counts are callback attempts,
not proof of underlying network requests or model tokens. Traces are backend data.
"""

from collections.abc import Callable
from copy import deepcopy
import json

from app.answering import build_messages, validate_answer
from app.bounded_retrieval import merge_evidence, run_bounded_retrieval
from app.evidence_evaluation import build_evaluation_messages
from app.clarification_reply import render_clarification, render_insufficiency
from app.retrieval_contract import RetrievalRequest, SourceScope
from app.provider_errors import ModelProviderError


def run_rag_service(
    question: str, *,
    retrieve: Callable[[str], list[dict]],
    evaluate: Callable[[str, list[dict]], dict],
    generate: Callable[[str, list[dict]], dict],
    max_retrievals: int = 1, max_evaluations: int = 2,
    max_candidates: int = 10, final_k: int = 3,
    max_evidence_chars: int = 3600, max_generations: int = 1,
    request_retrieval: Callable[[RetrievalRequest], list[dict]] | None = None,
    initial_scope: SourceScope | None = None, scope_policy: Callable | None = None,
    build_id: str | None = None,
    plan_queries: Callable | None = None, allocation_policy: str = 'unified_rrf',
    initial_query: str | None = None,
) -> dict:
    """max_retrievals limits supplementary rounds; initial retrieval is separate.

    An optional planned round can execute up to two query callbacks. Service
    additional_retrievals counts those actual callbacks, not rounds.

    Use functools.partial to bind evaluate_requirements(generate=...) and
    generate_answer(client, model, ...). Retrieval strategy stays replaceable.
    The existing evaluator accepts at most ten candidates; never silently trim.
    """
    build_evaluation_messages(question, [], final_k=final_k,
                              max_evidence_chars=max_evidence_chars)
    if initial_query is not None and (not isinstance(initial_query, str) or not initial_query.strip()):
        raise ValueError('initial_query_must_be_nonempty')
    search_query = question if initial_query is None else initial_query
    for value, minimum, name in (
        (max_retrievals, 0, "max_retrievals"),
        (max_evaluations, 1, "max_evaluations"),
        (max_candidates, 1, "max_candidates"),
        (max_generations, 0, "max_generations"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("invalid_" + name)
    if max_candidates > 10:
        raise ValueError("evaluator_accepts_at_most_10_candidates")
    if not all(callable(fn) for fn in (retrieve, evaluate, generate)):
        raise ValueError("dependencies_must_be_callable")
    if request_retrieval is not None and not callable(request_retrieval):
        raise ValueError('request_retrieval_must_be_callable')
    if request_retrieval is None and (initial_scope is not None or scope_policy is not None):
        raise ValueError('scope_requires_request_retrieval')
    if initial_scope is not None and not isinstance(initial_scope, SourceScope):
        raise ValueError('invalid_initial_scope')
    if scope_policy is not None and not callable(scope_policy):
        raise ValueError('scope_policy_must_be_callable')
    if plan_queries is not None and not callable(plan_queries):
        raise ValueError('query_planner_must_be_callable')
    if build_id is not None and request_retrieval is None:
        raise ValueError('pinned_build_requires_request_retrieval')

    result = {
        "status": "failed", "message": "本轮处理未完成。",
        "required_information": [], "missing_knowledge": [],
        "evidence": [], "answer": None, "evaluation": None, "error": None,
        "stop_reason": None, "trace": [], "coverage_verified": False,
        "counts": {"initial_retrievals": 0, "additional_retrievals": 0,
                   "evaluations": 0, "generations": 0},
    }
    if build_id is not None:
        result['build_id'] = build_id
        result['trace'].append({'step': 'pin_build', 'build_id': build_id})

    def fail(stage: str, code: str, detail=None) -> dict:
        result.update(status="failed", message="本轮处理暂时失败，请稍后重试。",
                      stop_reason=code, error={"stage": stage, "code": code})
        if detail is not None:
            result['error'].update(detail)
        result["trace"].append({"step": stage, "status": "failed", "code": code})
        return result

    result["counts"]["initial_retrievals"] = 1
    try:
        initial_request = (RetrievalRequest(search_query, scope=initial_scope or SourceScope(), build_id=build_id)
                           if request_retrieval is not None else None)
        if initial_request is not None:
            result['trace'].append({'step': 'prepare_initial_retrieval', 'request': initial_request.to_dict()})
        fetched = request_retrieval(initial_request) if initial_request is not None else retrieve(search_query)
        initial = merge_evidence([], deepcopy(fetched))["candidates"]
    except Exception:
        return fail("initial_retrieve", "initial_retrieval_failed")
    result["trace"].append({"step": "initial_retrieve",
                            "candidate_ids": [item["chunk_id"] for item in initial]})
    if not initial:
        result.update(status="insufficient", stop_reason="empty_candidates",
                      message="本次检索未找到可用资料，暂时无法据此回答。",
                      missing_knowledge=["本次检索未返回可用证据"])
        return result

    def checked_evaluate(q, candidates):
        # Enforce evaluator input limits before making an evaluation callback.
        build_evaluation_messages(q, candidates, final_k=final_k,
                                  max_evidence_chars=max_evidence_chars)
        result["counts"]["evaluations"] += 1
        return evaluate(q, deepcopy(candidates))

    def additional_retrieve(query):
        result["counts"]["additional_retrievals"] += 1
        return deepcopy(retrieve(query))

    def additional_request_retrieval(request):
        result['counts']['additional_retrievals'] += 1
        return deepcopy(request_retrieval(request))

    def checked_plan(q,gaps):
        result['counts']['query_plans']=result['counts'].get('query_plans',0)+1
        return plan_queries(q,list(gaps))

    scoped_options = ({'request_retrieval': additional_request_retrieval,
                       'initial_scope': initial_scope, 'scope_policy': scope_policy, 'build_id': build_id}
                      if request_retrieval is not None else {})

    control = run_bounded_retrieval(
        question, initial, checked_evaluate, additional_retrieve,
        max_retrievals=max_retrievals, max_evaluations=max_evaluations,
        max_candidates=max_candidates, known_queries=(search_query,),
        plan_queries=checked_plan if plan_queries is not None else None,
        allocation_policy=allocation_policy,
        **scoped_options,
    )
    for step in control["trace"]:
        if step["step"] == "evaluate":
            report = step["result"]
            proposal = (report.get("proposal") if isinstance(report, dict) else None)
            proposal = proposal if isinstance(proposal, dict) else {}
            result["trace"].append({"step": "evaluate", "candidate_ids": step["candidate_ids"],
                "status": report.get("status") if isinstance(report, dict) else "invalid",
                "action": proposal.get("action"),
                "overall_coverage": proposal.get("overall_coverage"),
                "selected_chunk_ids": deepcopy(proposal.get("selected_chunk_ids", [])),
                "missing_knowledge": deepcopy(proposal.get("missing_knowledge", [])),
                "missing_user_information": deepcopy(proposal.get("missing_user_information", [])),
                "failure_reason": report.get("failure_reason") if isinstance(report, dict) else None})
            if isinstance(report, dict):
                result['trace'][-1].update(model_attempts=report.get('model_attempts'),
                                          repair_reason=report.get('repair_reason'))
            if 'pending_impacts' in proposal:
                result['trace'][-1].update(
                    pending_impacts=deepcopy(proposal['pending_impacts']),
                    blocking_pending_ids=deepcopy(proposal.get('blocking_pending_ids', [])),
                    context_input_version=proposal.get('context_input_version'))
        else:
            result["trace"].append(deepcopy(step))
    result["stop_reason"] = control["stop_reason"]
    last = control["last_evaluation"]
    proposal = last.get("proposal") if isinstance(last, dict) else None
    result["evaluation"] = deepcopy(proposal)
    if isinstance(proposal, dict):
        result["missing_knowledge"] = list(proposal.get("missing_knowledge", []))

    reason = control["stop_reason"] or "controller_failed"
    if reason.startswith('query_planning_error:'):
        return fail('plan_queries','query_planning_failed')
    if reason.startswith("evaluation_error:") or reason in {"evaluation_failed", "invalid_action"}:
        code = last.get("failure_reason") if isinstance(last, dict) else None
        return fail("evaluate", code or "evaluation_failed",
                    last.get('provider_error') if isinstance(last, dict) else None)
    if reason.startswith("retrieval_or_merge_error:"):
        return fail("additional_retrieve", "additional_retrieval_failed")

    selected = control["selected_evidence"] or (
        last.get("selected_evidence", []) if isinstance(last, dict) else [])
    try:
        by_id = {item["chunk_id"]: item for item in control["candidate_pool"]}
        ids = [item["chunk_id"] for item in selected]
        if len(ids) != len(set(ids)) or len(ids) > final_k:
            raise ValueError("invalid_selection")
        canonical = [by_id[key] for key in ids]
        if any(a["content"] != b["content"] or a.get("metadata") != b.get("metadata")
               for a, b in zip(selected, canonical)):
            raise ValueError("changed_evidence")
        if sum(len(item["content"]) for item in canonical) > max_evidence_chars:
            raise ValueError("evidence_budget_exceeded")
        build_messages(question, canonical)  # Source metadata must also be valid.
        result["evidence"] = deepcopy(canonical)
    except (KeyError, TypeError, ValueError):
        return fail("evaluate", "invalid_selected_evidence")

    if control["status"] == "needs_user":
        required = proposal["missing_user_information"]
        if "known_findings" in proposal:
            try:
                public = render_clarification(proposal, canonical)
            except (KeyError, TypeError, ValueError):
                return fail("evaluate", "invalid_public_clarification")
            result.update(status="needs_user", required_information=list(required),
                          message=public["answer"], answer=public)
            result["trace"].append({"step": "render_clarification", "origin": "evaluation_findings",
                                    "citation_ids": public["citation_ids"]})
            return result
        result.update(status="needs_user", required_information=list(required),
                      message="目前无法确定具体情况，请补充：" + "；".join(required) + "。")
        return result
    normal_evidence_stop = (control['status'] == 'stopped' and isinstance(proposal,dict)
        and proposal.get('action') == 'retrieve' and reason in {
            'retrieval_budget_exhausted','evaluation_budget_exhausted',
            'candidate_budget_exhausted','no_new_evidence','repeated_query'})
    if control["status"] == "insufficient" or normal_evidence_stop:
        try:
            public = render_insufficiency(proposal, canonical, result['missing_knowledge'])
        except (KeyError,TypeError,ValueError):
            return fail('evaluate','invalid_public_insufficiency')
        result.update(status="insufficient", message=public['answer'], answer=public)
        result['trace'].append({'step':'render_insufficiency','origin':'validated_evaluation',
            'terminal_reason':reason,'citation_ids':public['citation_ids']})
        return result
    if control["status"] != "answer_proposed" or max_generations == 0:
        result.update(status="stopped", message="本轮流程已停止，尚未生成完整答案。")
        if control["status"] == "answer_proposed":
            result["stop_reason"] = "generation_budget_exhausted"
        return result
    if not canonical:
        return fail("evaluate", "answer_requires_evidence")

    result["counts"]["generations"] = 1
    try:
        generated = generate(question, deepcopy(canonical))
        fields = ("status", "answer", "citation_ids", "missing_information")
        answer = validate_answer(json.dumps({key: generated[key] for key in fields},
                                            ensure_ascii=False), canonical)
    except ModelProviderError as error:
        return fail('generate', error.code, error.detail('generate'))
    except Exception:
        return fail("generate", "generation_failed")
    status = {"answer": "answer", "clarify": "needs_user", "insufficient": "insufficient"}[answer["status"]]
    result.update(status=status, message=answer["answer"], answer=answer,
                  stop_reason="generation_" + answer["status"])
    if status == "needs_user":
        result["required_information"] = answer["missing_information"]
    elif status == "insufficient":
        result["missing_knowledge"] = answer["missing_information"]
    result["trace"].append({"step": "generate", "status": answer["status"],
                            "citation_ids": answer["citation_ids"]})
    return result
