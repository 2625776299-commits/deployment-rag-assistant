"""One controller-owned planned lookup round; callbacks perform provider I/O."""
import json

from app.query_planning import parse_query_plan
from app.query_allocation import allocate_query_candidates
from app.retrieval_contract import RetrievalFeedback, RetrievalRequest, query_scope_key


def execute_planned_round(question, gaps, pool, *, planner, retrieve,
                          request_retrieval, remaining, scope, scope_policy,
                          build_id, round_number, seen, scoped_seen, allocation_policy):
    result={'status':'failed','stop_reason':None,'incoming':[], 'trace':[], 'attempted_queries':0}
    try:
        plan=parse_query_plan(json.dumps(planner(question,list(gaps)),ensure_ascii=False),question)
    except Exception as error:
        result['stop_reason']='query_planning_error:'+type(error).__name__
        return result
    result['trace'].append({'step':'plan_queries','plan':plan,'build_id':build_id})
    queries={k:plan[k] for k in ('specific_query','mechanism_query') if plan[k] is not None}
    ranks={}
    successful=0
    from app.bounded_retrieval import merge_evidence
    for kind,query in queries.items():
        request=None
        try:
            if request_retrieval is not None:
                feedback=RetrievalFeedback(question,query,tuple(gaps),tuple(c['chunk_id'] for c in pool),
                    scope,round_number,build_id)
                next_scope=scope_policy(feedback) if scope_policy is not None else scope
                request=RetrievalRequest(query,exclude_chunk_ids=feedback.candidate_ids,
                    scope=next_scope,build_id=build_id,max_results=remaining)
                key=query_scope_key(query,request.scope)
                if key in scoped_seen:
                    result['trace'].append({'step':'skip_planned_query','kind':kind,'reason':'repeated_query'})
                    continue
                scoped_seen.add(key)
            else:
                key=' '.join(query.casefold().split())
                if key in seen:
                    result['trace'].append({'step':'skip_planned_query','kind':kind,'reason':'repeated_query'})
                    continue
                seen.add(key)
        except Exception as error:
            result['stop_reason']='retrieval_or_merge_error:'+type(error).__name__
            return result
        result['attempted_queries']+=1
        entry={'step':'planned_query','kind':kind,'query':query,
               'request':request.to_dict() if request else None}
        try:
            incoming=request_retrieval(request) if request else retrieve(query)
            # Check conflicting identities before removing duplicates of known evidence.
            merged=merge_evidence(pool,incoming)
            fresh=set(merged['added_chunk_ids'])
            ranks[kind]=[c for c in incoming if c['chunk_id'] in fresh]
            entry.update(status='ok',retrieved_ids=[c['chunk_id'] for c in incoming],
                         duplicates=merged['duplicates'])
            successful+=1
        except Exception as error:
            entry.update(status='failed',error_type=type(error).__name__)
        result['trace'].append(entry)
    if not result['attempted_queries']:
        result.update(status='stopped',stop_reason='repeated_query')
        return result
    if not successful:
        result['stop_reason']='retrieval_or_merge_error:all_planned_queries_failed'
        return result
    try:
        incoming=allocate_query_candidates(ranks,remaining,policy=allocation_policy)
    except Exception as error:
        result['stop_reason']='retrieval_or_merge_error:'+type(error).__name__
        return result
    result['trace'].append({'step':'allocate_planned_candidates','policy':allocation_policy,
        'capacity':remaining,'selected_ids':[c['chunk_id'] for c in incoming],
        'attempted_queries':result['attempted_queries'],'successful_queries':successful})
    result.update(status='ok',incoming=incoming)
    return result
