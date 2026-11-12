import json
from pathlib import Path


# loader.py -> app -> deploy-rag; independent of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "source1.json"


def load_sources(path: Path) -> list[dict]:
    """Read source records and validate the minimum ingestion requirements."""
    # File and JSON parsing errors propagate so the actual cause stays visible.
    sources = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(sources, list):
        raise ValueError("Source JSON must contain a list of records")

    seen_section_ids: set[str] = set()
    for position, record in enumerate(sources, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {position} must be an object")

        for field in ("document_id", "section_id", "content","source_url"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Record {position}: {field!r} must be a non-empty string"
                )

        section_id = record["section_id"]
        if section_id in seen_section_ids:
            raise ValueError(
                f"Record {position}: duplicate section_id {section_id!r}"
            )
        seen_section_ids.add(section_id)

    return sources
def get_sections_by_document(
    sources: list[dict],
    document_id: str,
) -> list[dict]:
    matches = []

    for record in sources:
        if record["document_id"] == document_id:
            matches.append(record)

    return matches


def main() -> None:
    sources = load_sources(SOURCE_PATH)
    selected = get_sections_by_document(
        sources, "docker-compose-networking"
    )
    print(f"Matched sections: {len(selected)}")

    for record in selected:
        print(record["section_id"])
    document_ids = {record["document_id"] for record in sources}
    first_section_id = sources[0]["section_id"] if sources else "(none)"
    print(f"Source sections: {len(sources)}")
    print(f"Source documents: {len(document_ids)}")
    print(f"First section_id: {first_section_id}")


if __name__ == "__main__":
    main()
