"""Prototype: split YAML mappings while retaining original text and parent keys."""

import argparse
import json
import sys

import yaml
from yaml.nodes import MappingNode, ScalarNode
from yaml.tokens import AliasToken, AnchorToken


def split_yaml_block(
    text: str, max_chars: int = 800, *, preserve_healthcheck: bool = True,
) -> list[dict]:
    """Return contextual excerpts; max_chars is a target, not a hard limit.

    Split only block mappings. Lists, scalars and flow mappings stay atomic.
    Compose services.<name>.healthcheck stays atomic by default, even above
    the target. Set preserve_healthcheck=False for a comparison experiment.
    Anchors/aliases are rejected because their definitions may cross excerpts.
    source_start/end refer to the input YAML body, excluding copied parent keys.
    These excerpts provide evidence; they are not complete deployment files.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text.strip():
        return []

    try:
        if any(isinstance(token, (AnchorToken, AliasToken)) for token in yaml.scan(text)):
            raise ValueError("YAML anchors/aliases need shared context; keep this block whole")
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError("Expected one valid YAML document") from exc

    if root is None:
        return []

    lines = text.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    chunks = []

    def visit(
        node, start: int, end: int, headers: str, path: tuple[str, ...],
        header_spans: list[tuple[int, int]],
    ) -> None:
        content = headers + text[start:end]
        can_split = (
            isinstance(node, MappingNode)
            and not node.flow_style
            and bool(node.value)
        )
        if len(content) <= max_chars or not can_split:
            chunks.append({
                "content": content,
                "yaml_path": list(path),
                "ancestor_headers": headers,
                "ancestor_spans": [{"start": a, "end": b} for a, b in header_spans],
                "source_start": start,
                "source_end": end,
                "oversized": len(content) > max_chars,
            })
            return

        # Each child retains the key line introducing its enclosing mapping.
        for index, (key, value) in enumerate(node.value):
            if not isinstance(key, ScalarNode):
                raise ValueError("Complex YAML keys are not supported by this prototype")
            child_start = line_starts[key.start_mark.line]
            child_end = (
                line_starts[node.value[index + 1][0].start_mark.line]
                if index + 1 < len(node.value)
                else end
            )
            # Keep comments preceding the first key with that first excerpt.
            if index == 0:
                child_start = start
            header_end = line_starts[key.start_mark.line + 1]
            is_block_mapping = (
                isinstance(value, MappingNode)
                and not value.flow_style
                and value.start_mark.line > key.start_mark.line
            )
            child_path = path + (key.value,)
            child_content = headers + text[child_start:child_end]
            keep_group = (
                preserve_healthcheck
                and len(child_path) == 3
                and child_path[0] == "services"
                and child_path[2] == "healthcheck"
            )
            if len(child_content) > max_chars and is_block_mapping and not keep_group:
                visit(
                    value,
                    header_end,
                    child_end,
                    headers + text[child_start:header_end],
                    child_path,
                    header_spans + [(child_start, header_end)],
                )
            else:
                chunks.append({
                    "content": child_content,
                    "yaml_path": list(child_path),
                    "ancestor_headers": headers,
                    "ancestor_spans": [{"start": a, "end": b} for a, b in header_spans],
                    "source_start": child_start,
                    "source_end": child_end,
                    "oversized": len(child_content) > max_chars,
                })

    visit(root, 0, len(text), "", (), [])
    return chunks


def main() -> None:
    """Preview one real source block without changing the indexed corpus."""
    import re

    from app.compose_relations import extract_dependencies
    from app.loader import SOURCE_PATH, load_sources

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-id", default="S2")
    parser.add_argument("--max-chars", type=int, default=240)
    args = parser.parse_args()
    sources = load_sources(SOURCE_PATH)
    selected = next((s for s in sources if s["section_id"] == args.section_id), None)
    if selected is None:
        parser.error("section_id was not found")
    # This preview targets the triple-backtick blocks used in the current corpus.
    blocks = re.findall(r"```[^\n]*\n(.*?)```", selected["content"], re.S)
    if not blocks:
        parser.error("the selected section has no fenced code blocks")
    for block_index, body in enumerate(blocks, start=1):
        chunks = split_yaml_block(body, max_chars=args.max_chars)
        config_id = f"{selected['document_id']}:{args.section_id}:block-{block_index}"
        dependencies = extract_dependencies(body, config_id=config_id)
        print(json.dumps({
            "config_id": config_id,
            "section_id": args.section_id,
            "block_index": block_index,
            "source_url": selected["source_url"],
            "original_characters": len(body),
            "target_characters": args.max_chars,
            "dependencies": dependencies,
            "chunks": chunks,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
