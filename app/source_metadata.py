"""Shared, bounded source-applicability projection; these fields are not user facts."""

APPLICABILITY_FIELDS = (
    "technology", "applicable_version", "evidence_scope", "generalizable_as",
)


def project_source_metadata(metadata: object) -> dict[str, str | None]:
    """Preserve explicit unknowns; omit missing/invalid fields and unrelated labels.

    This is a data projection, not a semantic or prompt-injection safety check.
    The 160-character bound preserves the existing evaluator contract.
    """
    if not isinstance(metadata, dict):
        return {}
    source = {}
    for key in APPLICABILITY_FIELDS:
        value = metadata.get(key)
        if value is None and key in metadata:
            source[key] = None
        elif isinstance(value, str) and value.strip() and len(value) <= 160:
            source[key] = value
    return source
