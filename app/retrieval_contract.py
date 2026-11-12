"""Explicit controller-to-retriever requests; no model/provider dependencies."""
from dataclasses import asdict, dataclass, field


def _identifiers(values):
    if (not isinstance(values, tuple)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))):
        raise ValueError('identifiers_must_be_unique_nonempty_strings_in_tuple')


@dataclass(frozen=True)
class SourceScope:
    preferred_document_ids: tuple[str, ...] = ()
    excluded_document_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _identifiers(self.preferred_document_ids)
        _identifiers(self.excluded_document_ids)
        if set(self.preferred_document_ids) & set(self.excluded_document_ids):
            raise ValueError('a_document_cannot_be_preferred_and_excluded')


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    exclude_chunk_ids: tuple[str, ...] = ()
    scope: SourceScope = field(default_factory=SourceScope)
    build_id: str | None = None
    max_results: int | None = None

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError('query_must_be_nonempty')
        _identifiers(self.exclude_chunk_ids)
        if not isinstance(self.scope, SourceScope):
            raise ValueError('invalid_source_scope')
        if self.build_id is not None and (not isinstance(self.build_id, str) or not self.build_id.strip()):
            raise ValueError('invalid_build_id')
        if self.max_results is not None and (type(self.max_results) is not int or self.max_results < 1):
            raise ValueError('max_results_must_be_a_positive_integer_or_none')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RetrievalFeedback:
    question: str
    search_query: str
    missing_knowledge: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    previous_scope: SourceScope
    retrieval_number: int
    build_id: str | None = None


def query_scope_key(query: str, scope: SourceScope):
    """Changing source scope is a new lookup; changing tuple order is not."""
    return (' '.join(query.casefold().split()),
            tuple(sorted(scope.preferred_document_ids)), tuple(sorted(scope.excluded_document_ids)))
