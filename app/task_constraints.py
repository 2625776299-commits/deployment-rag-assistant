"""Versioned user statements about evidence availability, scoped to an incident.

Pure data operations, not an incident recognizer, repository or controller. The
trusted caller supplies an authorized incident registry and selected incident.
Quote membership checks establish provenance, not semantic attribution or truth.
"""
from dataclasses import asdict, dataclass
from uuid import uuid4

from app.conversation_context import SourceQuote, UserMessage


AVAILABILITIES = frozenset({'not_obtained', 'unavailable', 'available', 'uncertain'})
EVIDENCE_KINDS = frozenset({'incident_logs', 'incident_error_details', 'other_incident_evidence'})


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class IncidentRef:
    task_id: str
    incident_id: str

    def __post_init__(self):
        if not all(_text(v) for v in (self.task_id, self.incident_id)):
            raise ValueError('require_explicit_incident_reference')


@dataclass(frozen=True)
class EvidenceConstraint:
    constraint_id: str
    incident: IncidentRef
    evidence_kind: str
    availability: str
    source: SourceQuote
    introduced_in_input_version: int
    supersedes_constraint_id: str | None = None

    def __post_init__(self):
        if (not _text(self.constraint_id) or not isinstance(self.incident, IncidentRef)
                or not isinstance(self.evidence_kind, str) or self.evidence_kind not in EVIDENCE_KINDS
                or not isinstance(self.availability, str) or self.availability not in AVAILABILITIES
                or not isinstance(self.source, SourceQuote)
                or not _text(self.source.message_id) or not _text(self.source.quote)
                or type(self.introduced_in_input_version) is not int
                or self.introduced_in_input_version < 1
                or (self.supersedes_constraint_id is not None
                    and (not _text(self.supersedes_constraint_id)
                         or self.supersedes_constraint_id == self.constraint_id))):
            raise ValueError('invalid_evidence_constraint')

    @property
    def key(self):
        return self.incident, self.evidence_kind


@dataclass(frozen=True)
class ConstraintSnapshot:
    input_version: int = 0
    current: tuple[EvidenceConstraint, ...] = ()

    def __post_init__(self):
        if (type(self.input_version) is not int or self.input_version < 0
                or not isinstance(self.current, tuple)
                or any(not isinstance(c, EvidenceConstraint) for c in self.current)):
            raise ValueError('invalid_constraint_snapshot')
        if (len({c.key for c in self.current}) != len(self.current)
                or len({c.constraint_id for c in self.current}) != len(self.current)
                or any(c.introduced_in_input_version > self.input_version for c in self.current)):
            raise ValueError('invalid_constraint_snapshot')

    def to_dict(self):
        return asdict(self)


def _authorize(incident, allowed_incidents):
    if (not isinstance(incident, IncidentRef) or not isinstance(allowed_incidents, frozenset)
            or any(not isinstance(i, IncidentRef) for i in allowed_incidents)):
        raise ValueError('require_authorized_incident_registry')
    if incident not in allowed_incidents:
        raise ValueError('incident_not_authorized')


def revise_evidence_availability(state: ConstraintSnapshot, message: UserMessage, *,
        incident: IncidentRef, evidence_kind: str, availability: str, quote: str,
        previous_constraint_id: str | None, allowed_incidents: frozenset[IncidentRef]) -> ConstraintSnapshot:
    """Make a new snapshot; preserve all other incidents/categories and old input.

    Input versions are conversation versions: other tasks may occupy intervening
    versions. SQL must later compare the base snapshot under its transaction lock.
    A repeated value preserves the original record/source. Unknown is explicit;
    withdrawing certainty never silently becomes 'available'.
    """
    return apply_evidence_updates(state, message, incident=incident,
        updates=[{'evidence_kind': evidence_kind, 'availability': availability,
                  'previous_constraint_id': previous_constraint_id,
                  'source': {'message_id': message.message_id, 'quote': quote}}],
        allowed_incidents=allowed_incidents)


def apply_evidence_updates(state, message, *, incident, updates, allowed_incidents,
                           deferred_messages=()):
    """Apply one message's category deltas atomically; retain deferred sources."""
    if not isinstance(state, ConstraintSnapshot) or not isinstance(message, UserMessage):
        raise ValueError('require_constraint_state_and_message')
    if message.input_version <= state.input_version:
        raise ValueError('constraint_version_not_newer')
    _authorize(incident, allowed_incidents)
    if not isinstance(updates, list) or not isinstance(deferred_messages, tuple):
        raise ValueError('invalid_evidence_updates')
    sources = {message.message_id: message}
    for deferred in deferred_messages:
        if (not isinstance(deferred, UserMessage) or deferred.input_version >= message.input_version
                or deferred.message_id in sources):
            raise ValueError('invalid_constraint_deferred_source')
        sources[deferred.message_id] = deferred
    current = {c.key: c for c in state.current}
    seen = set()
    for update in updates:
        if (not isinstance(update, dict) or set(update) != {
                'evidence_kind', 'availability', 'previous_constraint_id', 'source'}
                or not isinstance(update['source'], dict)
                or set(update['source']) != {'message_id', 'quote'}
                or not _text(update['evidence_kind'])):
            raise ValueError('invalid_evidence_update')
        source = update['source']
        canonical = sources.get(source['message_id']) if _text(source['message_id']) else None
        if canonical is None or not _text(source['quote']) or source['quote'] not in canonical.content:
            raise ValueError('constraint_quote_not_in_message')
        key = incident, update['evidence_kind']
        if key in seen:
            raise ValueError('duplicate_constraint_target')
        seen.add(key)
        previous = current.get(key)
        if update['previous_constraint_id'] != (previous.constraint_id if previous else None):
            raise ValueError('constraint_previous_id_conflict')
        if (previous is not None and canonical.input_version < message.input_version
                and previous.introduced_in_input_version > canonical.input_version
                and previous.availability != update['availability']):
            raise ValueError('deferred_constraint_would_overwrite_newer_record')
        candidate = EvidenceConstraint(uuid4().hex, incident, update['evidence_kind'], update['availability'],
            SourceQuote(**source), message.input_version, previous.constraint_id if previous else None)
        if previous is None or previous.availability != candidate.availability:
            current[key] = candidate
    return ConstraintSnapshot(message.input_version, tuple(current.values()))


def restore_constraint_snapshot(raw):
    """Restore trusted stored JSON, without re-extracting or changing sources."""
    if (not isinstance(raw, dict) or set(raw) != {'input_version', 'current'}
            or not isinstance(raw['current'], (list, tuple))):
        raise ValueError('invalid_constraint_snapshot')
    constraints = []
    for item in raw['current']:
        if (not isinstance(item, dict) or set(item) != {'constraint_id', 'incident', 'evidence_kind',
                'availability', 'source', 'introduced_in_input_version', 'supersedes_constraint_id'}
                or not isinstance(item['incident'], dict) or set(item['incident']) != {'task_id','incident_id'}
                or not isinstance(item['source'], dict) or set(item['source']) != {'message_id','quote'}):
            raise ValueError('invalid_constraint_snapshot')
        constraints.append(EvidenceConstraint(**{**item, 'incident': IncidentRef(**item['incident']),
                                                'source': SourceQuote(**item['source'])}))
    return ConstraintSnapshot(raw['input_version'], tuple(constraints))


def project_incident_constraints(state: ConstraintSnapshot, incident: IncidentRef, *,
        allowed_incidents: frozenset[IncidentRef]) -> dict:
    """Return only the selected incident's statements, without choosing an action.

    General-mechanism questions can retain incident context without treating it
    as a blocker; requirement assessment, not this projection, decides relevance.
    """
    if not isinstance(state, ConstraintSnapshot):
        raise ValueError('invalid_constraint_snapshot')
    _authorize(incident, allowed_incidents)
    return {'input_version': state.input_version, 'incident': asdict(incident),
            'evidence_constraints': [asdict(c) for c in state.current if c.incident == incident]}
