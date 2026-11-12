"""Bounded evidence feedback control with injected evaluation/retrieval functions."""

from collections.abc import Callable


def merge_evidence(existing: list[dict], incoming: list[dict]) -> dict:
    """Deduplicate identity/exact body only; preserve originals and alias origins."""
    if not isinstance(existing, list) or not isinstance(incoming, list):
        raise ValueError("evidence_must_be_lists")
    merged, duplicates = [], []
    by_id, by_body = {}, {}
    for item in existing + incoming:
        if (not isinstance(item, dict) or not isinstance(item.get("chunk_id"), str)
                or not item["chunk_id"].strip() or not isinstance(item.get("content"), str)
                or not item["content"].strip()):
            raise ValueError("invalid_evidence")
        key, body = item["chunk_id"], item["content"].strip()
        if key in by_id:
            previous = by_id[key]
            if (previous["content"], previous.get("metadata")) != (item["content"], item.get("metadata")):
                raise ValueError("conflicting_chunk_identity")
            kept = by_body[body]
            reason = "same_id"
        elif body in by_body:
            kept = by_body[body]
            reason = "same_body"
            by_id[key] = item
        else:
            merged.append(item)
            by_id[key] = by_body[body] = item
            continue
        duplicates.append({"duplicate_chunk_id": key, "kept_chunk_id": kept["chunk_id"],
                           "reason": reason, "duplicate_metadata": item.get("metadata")})
    old_bodies = {item["content"].strip() for item in existing}
    return {"candidates": merged, "duplicates": duplicates,
            "added_chunk_ids": [c["chunk_id"] for c in merged if c["content"].strip() not in old_bodies]}


def _query_key(text: str) -> str:
    return " ".join(text.casefold().split())


def run_bounded_retrieval(question: str, initial_candidates: list[dict],
                          evaluate: Callable[[str, list[dict]], dict],
                          retrieve: Callable[[str], list[dict]], *,
                          max_retrievals: int = 1, max_evaluations: int = 2,
                          max_candidates: int = 10, known_queries: tuple[str, ...] = (),
                          request_retrieval: Callable | None = None,
                          initial_scope=None, scope_policy: Callable | None = None,
                          build_id: str | None = None,
                          plan_queries: Callable | None = None,
                          allocation_policy: str = 'unified_rrf') -> dict:
    """Control proposals, not semantic truth. No hidden retries or LLM clients."""
    for value, minimum, name in ((max_retrievals, 0, "max_retrievals"),
                                 (max_evaluations, 1, "max_evaluations"),
                                 (max_candidates, 1, "max_candidates")):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("invalid_" + name)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("invalid_question")
    from app.retrieval_contract import SourceScope, query_scope_key
    if request_retrieval is not None and not callable(request_retrieval):
        raise ValueError('request_retrieval_must_be_callable')
    if request_retrieval is None and (initial_scope is not None or scope_policy is not None):
        raise ValueError('scope_requires_request_retrieval')
    if scope_policy is not None and not callable(scope_policy):
        raise ValueError('scope_policy_must_be_callable')
    if plan_queries is not None and not callable(plan_queries):
        raise ValueError('query_planner_must_be_callable')
    if allocation_policy not in ('unified_rrf','reserved_then_rrf'):
        raise ValueError('invalid_allocation_policy')
    scope = initial_scope if initial_scope is not None else SourceScope()
    if not isinstance(scope, SourceScope):
        raise ValueError('invalid_initial_scope')
    pool = merge_evidence([], initial_candidates)["candidates"]
    result = {"status": "stopped", "stop_reason": None, "evaluation_count": 0,
              "retrieval_count": 0, "trace": [], "candidate_pool": pool,
              "selected_evidence": [], "last_evaluation": None, "coverage_verified": False}
    seen = {_query_key(q) for q in known_queries}
    scoped_seen = {query_scope_key(q, scope) for q in known_queries}

    def stop(reason: str) -> dict:
        result["stop_reason"] = reason
        return result

    if not pool:
        return stop("empty_candidates")
    if len(pool) > max_candidates:
        return stop("candidate_budget_exhausted")
    while result["evaluation_count"] < max_evaluations:
        result["evaluation_count"] += 1
        try:
            evaluation = evaluate(question, list(pool))
        except Exception as error:
            return stop("evaluation_error:" + type(error).__name__)
        result["last_evaluation"] = evaluation
        result["trace"].append({"step": "evaluate", "candidate_ids": [c["chunk_id"] for c in pool],
                                "result": evaluation})
        if (not isinstance(evaluation, dict) or evaluation.get("status") != "proposed"
                or not isinstance(evaluation.get("proposal"), dict)):
            return stop("evaluation_failed")
        proposal = evaluation["proposal"]
        action = proposal.get("action")
        if action in {"answer", "clarify", "insufficient"}:
            result.update(status={"answer": "answer_proposed", "clarify": "needs_user",
                                  "insufficient": "insufficient"}[action], stop_reason=action,
                          selected_evidence=evaluation.get("selected_evidence", []))
            return result
        if action != "retrieve":
            return stop("invalid_action")
        if result["retrieval_count"] >= max_retrievals:
            return stop("retrieval_budget_exhausted")
        if result["evaluation_count"] >= max_evaluations:
            return stop("evaluation_budget_exhausted")
        remaining = max_candidates - len(pool)
        if remaining <= 0:
            return stop("candidate_budget_exhausted")
        if plan_queries is not None:
            from app.planned_retrieval import execute_planned_round
            gaps=proposal.get('missing_knowledge',[])
            if not isinstance(gaps,list) or not gaps or any(not isinstance(g,str) or not g.strip() for g in gaps):
                return stop('query_planning_error:invalid_missing_knowledge')
            planned=execute_planned_round(question,gaps,list(pool),planner=plan_queries,
                retrieve=retrieve,request_retrieval=request_retrieval,remaining=remaining,
                scope=scope,scope_policy=scope_policy,build_id=build_id,
                round_number=result['retrieval_count']+1,seen=seen,scoped_seen=scoped_seen,
                allocation_policy=allocation_policy)
            result['trace'].extend(planned['trace'])
            if planned['attempted_queries']:
                # Count supplementary rounds here; per-query callbacks are counted by the service.
                result['retrieval_count']+=1
            if planned['status']!='ok':
                return stop(planned['stop_reason'])
            merged=merge_evidence(pool,planned['incoming'])
            result['trace'].append({'step':'merge_planned_round','added_chunk_ids':merged['added_chunk_ids'],
                'duplicates':merged['duplicates']})
            pool=merged['candidates']
            result['candidate_pool']=pool
            if not merged['added_chunk_ids']:
                return stop('no_new_evidence')
            continue
        query = proposal.get("search_query")
        if not isinstance(query, str) or not query.strip():
            return stop("missing_search_query")
        request = None
        if request_retrieval is not None:
            from app.retrieval_contract import RetrievalFeedback, RetrievalRequest
            try:
                gaps = proposal.get('missing_knowledge', [])
                if not isinstance(gaps, list) or any(not isinstance(g, str) for g in gaps):
                    raise ValueError('invalid_missing_knowledge')
                feedback = RetrievalFeedback(question, query, tuple(gaps),
                                             tuple(c['chunk_id'] for c in pool), scope,
                                             result['retrieval_count'] + 1, build_id)
                next_scope = scope_policy(feedback) if scope_policy is not None else scope
                request = RetrievalRequest(query=query, exclude_chunk_ids=feedback.candidate_ids,
                                           scope=next_scope, build_id=build_id, max_results=remaining)
            except Exception as error:
                return stop('retrieval_or_merge_error:' + type(error).__name__)
            identity = query_scope_key(query, request.scope)
            if identity in scoped_seen:
                return stop('repeated_query')
            scoped_seen.add(identity)
            scope = request.scope
            result['trace'].append({'step': 'prepare_retrieval', 'request': request.to_dict(),
                                    'scope_origin': 'feedback_policy' if scope_policy is not None else 'configured'})
        else:
            if _query_key(query) in seen:
                return stop("repeated_query")
            seen.add(_query_key(query))
        result["retrieval_count"] += 1
        try:
            incoming = request_retrieval(request) if request is not None else retrieve(query)
            merged = merge_evidence(pool, incoming)
        except Exception as error:
            return stop("retrieval_or_merge_error:" + type(error).__name__)
        result["trace"].append({"step": "retrieve_and_merge", "query": query,
                                "retrieved_ids": [c["chunk_id"] for c in incoming],
                                "added_chunk_ids": merged["added_chunk_ids"],
                                "duplicates": merged["duplicates"]})
        pool = merged["candidates"]
        result["candidate_pool"] = pool
        if not merged["added_chunk_ids"]:
            return stop("no_new_evidence")
        if len(pool) > max_candidates:
            return stop("candidate_budget_exhausted")
    return stop("evaluation_budget_exhausted")
