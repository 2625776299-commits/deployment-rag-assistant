"""Structure-aware experimental chunks with exact source-span provenance.

This is a deterministic approximation, not a semantic necessity classifier.
Unmarked code is treated as prose; code we cannot safely divide is preserved
and may be marked requires_review instead of silently losing source content.
"""

import hashlib
import json
import re

from app.yaml_chunker import split_yaml_block


STRATEGY = "structure-v1"
_SENTENCE_END = re.compile(r'[。！？][”’」』]*|[.!?]["\')\]]*(?=\s|$)')
_OPEN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """A punctuation heuristic; abbreviations and unmarked code may be ambiguous."""
    spans = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        a, b = _trim(text, cursor, match.end())
        if a < b:
            spans.append((a, b))
        cursor = match.end()
    a, b = _trim(text, cursor, end)
    if a < b:
        spans.append((a, b))
    return spans


def _blocks(text: str):
    """Yield prose spans and fenced-code spans, retaining all source offsets."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    cursor = 0
    i = 0
    while i < len(lines):
        match = _OPEN_FENCE.match(lines[i])
        if match is None:
            i += 1
            continue
        if cursor < offsets[i]:
            yield {"kind": "text", "start": cursor, "end": offsets[i]}
        marker, language = match.groups()
        closer = re.compile(r"^[ \t]{0,3}" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}[ \t]*[\r\n]*$")
        j = i + 1
        while j < len(lines) and not closer.match(lines[j]):
            j += 1
        closed = j < len(lines)
        yield {
            "kind": "code", "start": offsets[i],
            "end": offsets[j + 1] if closed else len(text),
            "body_start": offsets[i + 1], "body_end": offsets[j],
            "language": language.strip(), "closed": closed,
        }
        i = j + 1
        cursor = offsets[i] if closed else len(text)
    if cursor < len(text):
        yield {"kind": "text", "start": cursor, "end": len(text)}


def _text_pieces(text: str, start: int, end: int, target: int, maximum: int) -> list[dict]:
    atoms = []
    cursor = start
    boundaries = list(re.finditer(r"\r?\n[ \t]*\r?\n", text[start:end]))
    paragraph_spans = []
    for boundary in boundaries:
        paragraph_spans.append(_trim(text, cursor, start + boundary.start()))
        cursor = start + boundary.end()
    paragraph_spans.append(_trim(text, cursor, end))
    for a, b in paragraph_spans:
        if a == b:
            continue
        heading = bool(re.match(r"#{1,6}\s", text[a:b]))
        units = [(a, b)] if b - a <= target else _sentences(text, a, b)
        for unit_start, unit_end in units:
            forced = unit_end - unit_start > maximum
            while unit_end - unit_start > maximum:
                window = text[unit_start:unit_start + maximum]
                positions = [m.end() for m in re.finditer(r"[，,；;]\s*|\s+", window)]
                # Avoid pathological one-character progress at the start of a unit.
                split = unit_start + (max(positions) if positions and max(positions) >= maximum // 2 else maximum)
                atoms.append({"start": unit_start, "end": split, "forced": True, "heading": heading})
                unit_start = split
            if unit_start < unit_end:
                atoms.append({"start": unit_start, "end": unit_end, "forced": forced, "heading": heading})

    pieces = []
    current = None
    for atom in atoms:
        can_merge = (
            current is not None and not atom["heading"]
            and not current["forced"] and not atom["forced"]
            and atom["end"] - current["start"] <= target
        )
        if can_merge:
            current["end"] = atom["end"]
        else:
            if current is not None:
                pieces.append(current)
            current = dict(atom)
    if current is not None:
        pieces.append(current)
    return pieces


def build_structured_chunks(
    sources: list[dict], target_chars: int = 800, max_chars: int = 1200,
    context_budget: int = 100,
) -> list[dict]:
    """Build a separate corpus; all metadata stays compatible with flat storage.

    max_chars bounds prose; an indivisible code unit exceeding it is preserved
    with requires_review=True. Such records must be reviewed before indexing.
    Provenance lists are JSON-encoded in metadata to keep values primitive.
    """
    if target_chars <= 0 or max_chars < target_chars or context_budget < 0:
        raise ValueError("Require 0 < target_chars <= max_chars and context_budget >= 0")
    profile = {"strategy": STRATEGY, "target_chars": target_chars,
               "max_chars": max_chars, "context_budget": context_budget}
    profile_id = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:8]
    chunks = []
    seen_sections = set()
    for source in sources:
        if source["section_id"] in seen_sections:
            raise ValueError("section_id must be unique within the source snapshot")
        seen_sections.add(source["section_id"])
        text = source["content"]
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parent_id = f"{source['document_id']}:{source['section_id']}:{source_hash[:12]}"
        local = []

        def emit(parts: list[tuple[str, int, int]], *, kind: str, prefix: str = "", suffix: str = "", warning: str = "", yaml_path: list | None = None) -> None:
            rendered = prefix
            spans = []
            for role, a, b in parts:
                if rendered and not rendered.endswith("\n") and role == "body" and spans:
                    rendered += "\n\n"
                content_start = len(rendered)
                rendered += text[a:b]
                spans.append({"role": role, "source_start": a, "source_end": b,
                              "content_start": content_start, "content_end": len(rendered)})
            rendered += suffix
            index = len(local)
            source_spans = json.dumps(spans, ensure_ascii=False, separators=(",", ":"))
            local.append({
                "chunk_id": f"{source['section_id']}-s{profile_id}-{source_hash[:8]}:{index:03d}",
                "content": rendered,
                "metadata": {
                    "document_id": source["document_id"], "section_id": source["section_id"],
                    "source_url": source["source_url"], "title": source.get("title", ""),
                    "section": source.get("section", ""), "fetched_at": source.get("fetched_at", ""),
                    "parent_id": parent_id, "source_sha256": source_hash,
                    "chunk_index": index, "chunking_strategy": STRATEGY, "profile_id": profile_id,
                    "content_kind": kind, "target_chars": target_chars, "max_chars": max_chars,
                    "context_budget": context_budget, "source_spans": source_spans,
                    "actual_overlap_chars": sum(b - a for role, a, b in parts if role == "context"),
                    "requires_review": len(rendered) > max_chars or bool(warning),
                    "warning": warning, "yaml_path": json.dumps(yaml_path or [], ensure_ascii=False),
                },
            })

        for block in _blocks(text):
            if block["kind"] == "text":
                previous = None
                for piece in _text_pieces(text, block["start"], block["end"], target_chars, max_chars):
                    parts = []
                    if (
                        previous is not None and not previous["forced"]
                        and not piece["heading"] and not piece["forced"]
                    ):
                        candidates = _sentences(text, previous["start"], previous["end"])
                        if candidates:
                            a, b = candidates[-1]
                            ending = list(_SENTENCE_END.finditer(text, a, b))
                            complete = bool(ending) and ending[-1].end() == b
                            if complete and b - a <= context_budget and b - a + 2 + piece["end"] - piece["start"] <= max_chars:
                                parts.append(("context", a, b))
                    parts.append(("body", piece["start"], piece["end"]))
                    emit(parts, kind="text", warning="forced_text_boundary" if piece["forced"] else "")
                    previous = piece
                continue

            a, b = block["body_start"], block["body_end"]
            language = block["language"].split()[0].lower() if block["language"] else ""
            body = text[a:b]
            is_yaml = language in {"yaml", "yml"} or (not language and bool(re.match(r"\s*(services|env_file):", body)))
            warning = "unclosed_code_fence" if not block["closed"] else ""
            if is_yaml and not warning:
                try:
                    fragments = split_yaml_block(body, max_chars=max(1, target_chars - 16))
                except ValueError:
                    fragments = None
                    warning = "yaml_requires_shared_context_or_parse_review"
                if fragments is not None:
                    # Empty/comment-only YAML remains visible rather than disappearing.
                    if not fragments:
                        emit([("body", a, b)], kind="yaml", prefix="```yaml\n", suffix="\n```")
                    for fragment in fragments:
                        parts = [("context", a + s["start"], a + s["end"]) for s in fragment["ancestor_spans"]]
                        parts.append(("body", a + fragment["source_start"], a + fragment["source_end"]))
                        emit(parts, kind="yaml", prefix="```yaml\n", suffix="\n```", yaml_path=fragment["yaml_path"])
                    continue
            # Preserve opaque code as an original continuous span, including fences.
            emit([("body", block["start"], block["end"])], kind="code", warning=warning)
        chunks.extend(local)
    return chunks
