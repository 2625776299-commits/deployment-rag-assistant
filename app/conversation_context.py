"""Apply explicit changes to user-stated facts without rewriting all context.

Pure functions: no SQL, model calls, or automatic fact extraction. The caller
supplies canonical user messages and registered targets from the same authorized
conversation. Structural acceptance does not establish semantic correctness.
"""
from dataclasses import asdict, dataclass
import json
from uuid import uuid4


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class FactTarget:
    entity_id: str
    scope_id: str
    field: str

    def __post_init__(self):
        if not all(_text(v) for v in (self.entity_id, self.scope_id, self.field)):
            raise ValueError('invalid_fact_target')


@dataclass(frozen=True)
class UserMessage:
    message_id: str
    input_version: int
    content: str

    def __post_init__(self):
        if (not _text(self.message_id) or not _text(self.content)
                or type(self.input_version) is not int or self.input_version < 1):
            raise ValueError('invalid_user_message')


@dataclass(frozen=True)
class SourceQuote:
    message_id: str
    quote: str


@dataclass(frozen=True)
class UserFact:
    fact_id: str
    target: FactTarget
    value: str
    source: SourceQuote


@dataclass(frozen=True)
class PendingFact:
    pending_id: str
    target: FactTarget
    previous_fact: UserFact | None
    source: SourceQuote
    reason: str


@dataclass(frozen=True)
class FactChange:
    operation: str
    before: UserFact | None
    after: UserFact | None
    source: SourceQuote
    pending_before: PendingFact | None = None
    pending_after: PendingFact | None = None


@dataclass(frozen=True)
class ContextState:
    input_version: int = 0
    current_facts: tuple[UserFact, ...] = ()
    changes: tuple[FactChange, ...] = ()
    pending_facts: tuple[PendingFact, ...] = ()

    def to_dict(self):
        return asdict(self)


def restore_context(snapshot: dict) -> ContextState:
    """Restore a stored snapshot, preserving IDs and provenance, without a model.

    This checks the storage shape, not the truth of user statements. Only the
    authorized repository supplies snapshots; clients cannot import context.
    """
    def record(value, keys):
        if not isinstance(value, dict) or set(value) != set(keys.split()):
            raise ValueError('invalid_context_snapshot')
        return value

    def source(value):
        record(value, 'message_id quote')
        if not all(_text(v) for v in value.values()):
            raise ValueError('invalid_context_snapshot')
        return SourceQuote(**value)

    def target(value):
        record(value, 'entity_id scope_id field')
        return FactTarget(**value)

    def fact(value):
        if value is None:
            return None
        record(value, 'fact_id target value source')
        if not _text(value['fact_id']) or not _text(value['value']):
            raise ValueError('invalid_context_snapshot')
        return UserFact(value['fact_id'], target(value['target']), value['value'], source(value['source']))

    def pending(value):
        if value is None:
            return None
        record(value, 'pending_id target previous_fact source reason')
        if not _text(value['pending_id']) or value['reason'] not in ('ambiguous_target', 'uncertain_value'):
            raise ValueError('invalid_context_snapshot')
        item = PendingFact(value['pending_id'], target(value['target']), fact(value['previous_fact']),
                           source(value['source']), value['reason'])
        if item.previous_fact is not None and item.previous_fact.target != item.target:
            raise ValueError('invalid_context_snapshot')
        return item

    record(snapshot, 'input_version current_facts changes pending_facts')
    if type(snapshot['input_version']) is not int or snapshot['input_version'] < 0:
        raise ValueError('invalid_context_snapshot')
    if any(not isinstance(snapshot[key], (list, tuple)) for key in ('current_facts', 'changes', 'pending_facts')):
        raise ValueError('invalid_context_snapshot')
    changes = []
    for change in snapshot['changes']:
        record(change, 'operation before after source pending_before pending_after')
        if change['operation'] not in ('add', 'replace', 'retract', 'mark_pending', 'resolve'):
            raise ValueError('invalid_context_snapshot')
        changes.append(FactChange(change['operation'], fact(change['before']), fact(change['after']),
            source(change['source']), pending(change['pending_before']), pending(change['pending_after'])))
    state = ContextState(snapshot['input_version'], tuple(fact(f) for f in snapshot['current_facts']),
                         tuple(changes), tuple(pending(p) for p in snapshot['pending_facts']))
    current = (*state.current_facts, *state.pending_facts)
    if (any(item is None for item in current)
            or len({item.target for item in current}) != len(current)
            or len({f.fact_id for f in state.current_facts}) != len(state.current_facts)
            or len({p.pending_id for p in state.pending_facts}) != len(state.pending_facts)
            or (state.input_version == 0 and (current or changes))):
        raise ValueError('invalid_context_snapshot')
    return state


def apply_context_updates(state: ContextState, message: UserMessage, proposal: dict,
                          *, allowed_targets: frozenset[FactTarget],
                          deferred_message: UserMessage | None = None) -> ContextState:
    """Apply one message's patch atomically to a new in-memory state.

    Each accepted message advances the input version, including an empty patch.
    Context preparation processes messages in order; it must not silently skip an
    unprocessed message. Database publication still needs its own version guard.

    A trusted repository may supply the ONE earlier message whose task ambiguity
    this turn resolves. It must validate that resolution and guard against stale
    updates. This is not permission to cite arbitrary conversation history.
    The original message ID/quote is preserved; task attribution is stored in
    the repository's resolution record, never fabricated as a current quote.
    """
    if message.input_version != state.input_version + 1:
        raise ValueError('message_version_not_next')
    if not isinstance(proposal, dict) or set(proposal) != {'updates'} or not isinstance(proposal['updates'], list):
        raise ValueError('invalid_update_proposal')
    if not isinstance(allowed_targets, frozenset) or any(not isinstance(t, FactTarget) for t in allowed_targets):
        raise ValueError('invalid_registered_targets')
    authorized_messages = {message.message_id: message}
    if deferred_message is not None:
        if (not isinstance(deferred_message, UserMessage)
                or deferred_message.input_version >= message.input_version
                or deferred_message.message_id == message.message_id):
            raise ValueError('invalid_deferred_message')
        authorized_messages[deferred_message.message_id] = deferred_message
    facts = {fact.target: fact for fact in state.current_facts}
    if len(facts) != len(state.current_facts):
        raise ValueError('duplicate_current_target')
    pending = {item.target: item for item in state.pending_facts}
    if len(pending) != len(state.pending_facts) or facts.keys() & pending.keys():
        raise ValueError('conflicting_pending_targets')
    events, touched = [], set()
    for update in proposal['updates']:
        if not isinstance(update, dict):
            raise ValueError('invalid_update_fields')
        operation = update.get('operation')
        if not isinstance(operation, str) or operation not in ('add', 'replace', 'retract', 'mark_pending', 'resolve'):
            raise ValueError('invalid_update_operation')
        fields = {'operation', 'target', 'source'}
        if operation == 'mark_pending':
            fields |= {'previous_fact_id', 'reason'}
        elif operation == 'resolve':
            fields |= {'pending_id', 'value'}
        else:
            fields |= {'previous_fact_id', 'value'}
        if set(update) != fields:
            raise ValueError('invalid_update_fields')
        target_record = update['target']
        if not isinstance(target_record, dict) or set(target_record) != {'entity_id', 'scope_id', 'field'}:
            raise ValueError('invalid_fact_target')
        target = FactTarget(**target_record)
        if target not in allowed_targets:
            raise ValueError('unregistered_target')
        if target in touched:
            raise ValueError('multiple_updates_for_same_target')
        touched.add(target)
        source = update['source']
        if (not isinstance(source, dict) or set(source) != {'message_id', 'quote'}
                or not _text(source['message_id']) or source['message_id'] not in authorized_messages
                or not _text(source['quote'])
                or source['quote'] not in authorized_messages[source['message_id']].content):
            raise ValueError('source_quote_not_in_authorized_user_message' if deferred_message is not None
                             else 'source_quote_not_in_current_user_message')
        quote = SourceQuote(**source)
        before = facts.get(target)
        pending_before = pending.get(target)
        if operation == 'resolve':
            if pending_before is None or update['pending_id'] != pending_before.pending_id:
                raise ValueError('pending_fact_does_not_match_target')
            if not _text(update['value']):
                raise ValueError('require_nonempty_fact_value')
            after = UserFact(uuid4().hex, target, update['value'], quote)
            facts[target] = after
            del pending[target]
            events.append(FactChange(operation, None, after, quote, pending_before=pending_before))
            continue
        # A disputed value cannot silently reappear via an ordinary add/replace.
        if pending_before is not None:
            raise ValueError('pending_target_requires_resolution')
        if operation == 'mark_pending':
            expected_id = before.fact_id if before is not None else None
            if update['previous_fact_id'] != expected_id:
                raise ValueError('previous_fact_does_not_match_target')
            if update['reason'] not in ('ambiguous_target', 'uncertain_value'):
                raise ValueError('invalid_pending_reason')
            item = PendingFact(uuid4().hex, target, before, quote, update['reason'])
            pending[target] = item
            facts.pop(target, None)
            events.append(FactChange(operation, before, None, quote, pending_after=item))
            continue
        if operation == 'add':
            if before is not None or update['previous_fact_id'] is not None:
                raise ValueError('add_cannot_overwrite_current_fact')
        elif before is None or update['previous_fact_id'] != before.fact_id:
            raise ValueError('previous_fact_does_not_match_target')
        if operation == 'retract':
            if update['value'] is not None:
                raise ValueError('retraction_has_no_replacement_value')
            after = None
            del facts[target]
        else:
            if not _text(update['value']):
                raise ValueError('require_nonempty_fact_value')
            # A fresh message/proposal is still retained by the caller. Repeating
            # an unchanged, non-pending value must not churn the current fact ID.
            # Run this only after validating the target, old ID and new source.
            if operation == 'replace' and update['value'] == before.value:
                continue
            # Fact identifiers are assigned by code, not supplied by the model.
            after = UserFact(uuid4().hex, target, update['value'], quote)
            facts[target] = after
        events.append(FactChange(operation, before, after, quote))
    return ContextState(message.input_version, tuple(facts.values()), state.changes + tuple(events), tuple(pending.values()))


def build_reasoning_context(state: ContextState) -> dict:
    """Project current user statements and unknowns, excluding historical values.

    This is a proposed input boundary for retrieval/evaluation/generation, not an
    integration into /ask. Full snapshots and quotes remain available for audit
    and the update proposer; they must not be blindly appended to RAG context.
    User statements are not independently verified observations.
    """
    return {
        'input_version': state.input_version,
        'user_stated_facts': [
            {'fact_id': fact.fact_id, 'target': asdict(fact.target), 'value': fact.value,
             'source_message_id': fact.source.message_id}
            for fact in state.current_facts
        ],
        'pending_clarifications': [
            {'pending_id': item.pending_id, 'target': asdict(item.target), 'reason': item.reason,
             'source_message_id': item.source.message_id}
            for item in state.pending_facts
        ],
    }


def parse_context_updates(raw: str) -> dict:
    """Parse only; all target/source/version checks happen when applying the patch."""
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError('update_proposal_is_not_json') from None
    if not isinstance(result, dict) or set(result) != {'updates'} or not isinstance(result['updates'], list):
        raise ValueError('invalid_update_proposal')
    return result
