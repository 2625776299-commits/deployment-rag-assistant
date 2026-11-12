"""Experimental query-list allocation; not wired into the application controller."""
from copy import deepcopy

from app.bounded_retrieval import merge_evidence
from app.hybrid_retrieval import rrf_fuse


def allocate_query_candidates(rankings, capacity, *, policy):
    if type(capacity) is not int or capacity < 1:
        raise ValueError('positive_capacity_required')
    if policy not in ('unified_rrf', 'reserved_then_rrf') or not rankings:
        raise ValueError('invalid_allocation_policy_or_rankings')
    # Validate identities before allowing the fusion to choose between records.
    merged = merge_evidence([], [c for items in rankings.values() for c in items])
    canonical = {c['content'].strip(): c for c in merged['candidates']}
    normalized = {}
    for route, items in rankings.items():
        seen, normalized[route] = set(), []
        for item in items:
            base = canonical[item['content'].strip()]
            if base['chunk_id'] in seen:
                continue
            seen.add(base['chunk_id'])
            normalized[route].append({**base, 'score': item.get('rrf_score', item.get('score', 0.0))})
    if not merged['candidates']:
        return []
    fused = rrf_fuse(normalized, top_k=len(merged['candidates']))
    if policy == 'unified_rrf':
        return deepcopy(fused[:capacity])
    by_id = {c['chunk_id']: c for c in fused}
    selected, used = [], set()
    quota = capacity // len(normalized)
    for items in normalized.values():
        added = 0
        for item in items:
            if added >= quota:
                break
            key = item['chunk_id']
            if key not in used:
                selected.append(by_id[key])
                used.add(key)
                added += 1
    for item in fused:
        if len(selected) >= capacity:
            break
        if item['chunk_id'] not in used:
            selected.append(item)
            used.add(item['chunk_id'])
    return deepcopy(selected)
