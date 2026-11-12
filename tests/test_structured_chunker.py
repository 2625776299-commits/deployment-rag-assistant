"""Offline checks for evidence preservation and chunk boundary regressions."""

import json
import re
import unittest

import yaml

from app.loader import SOURCE_PATH, load_sources
from app.structured_chunker import build_structured_chunks


def source(content: str) -> dict:
    return {"document_id": "example", "section_id": "T1", "content": content,
            "source_url": "https://example.com/source"}


def merge(target: dict, incoming: dict) -> None:
    for key, value in incoming.items():
        if key in target and isinstance(value, dict):
            merge(target[key], value)
        else:
            if key in target and target[key] != value:
                raise AssertionError(f"Conflicting YAML value: {key}")
            target[key] = value


class StructuredChunkTests(unittest.TestCase):
    def test_paragraph_packing_with_windows_and_unix_newlines(self):
        for newline in ("\n\n", "\r\n\r\n"):
            text = newline.join(("a" * 450, "b" * 500, "c" * 150))
            chunks = build_structured_chunks([source(text)], context_budget=0)
            self.assertEqual([c["content"] for c in chunks],
                             ["a" * 450, "b" * 500 + newline + "c" * 150])

    def test_soft_target_and_forced_boundary_are_distinct(self):
        whole = build_structured_chunks([source("x" * 900 + ".")])
        self.assertEqual(len(whole), 1)
        self.assertFalse(whole[0]["metadata"]["requires_review"])
        forced = build_structured_chunks([source("x" * 1300 + ".")])
        self.assertEqual("".join(c["content"] for c in forced), "x" * 1300 + ".")
        self.assertTrue(all(c["metadata"]["requires_review"] for c in forced))
        self.assertTrue(all(len(c["content"]) <= 1200 for c in forced))

    def test_overlap_uses_a_complete_sentence_within_budget(self):
        for ending, expected in ((".", 51), ("", 0)):
            text = "a" * 50 + ending + "\n\n" + "b" * 80 + "."
            chunks = build_structured_chunks([source(text)], 100, 200, 60)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[1]["metadata"]["actual_overlap_chars"], expected)
        chinese = build_structured_chunks([source("甲" * 50 + "。\n\n" + "乙" * 80 + "。")], 100, 200, 60)
        self.assertEqual(chinese[1]["metadata"]["actual_overlap_chars"], 51)

    def test_code_fence_separates_prose_context(self):
        text = "Before.\n\n```python\nprint('hello')\n```\n\nAfter."
        chunks = build_structured_chunks([source(text)])
        self.assertEqual([c["metadata"]["content_kind"] for c in chunks], ["text", "code", "text"])
        self.assertEqual(chunks[-1]["content"], "After.")
        self.assertEqual(chunks[-1]["metadata"]["actual_overlap_chars"], 0)

    def test_yaml_values_and_protected_healthcheck_survive_integration(self):
        item = next(s for s in load_sources(SOURCE_PATH) if s["section_id"] == "S2")
        body = re.findall(r"```[^\n]*\n(.*?)```", item["content"], re.S)[0]
        chunks = build_structured_chunks([item], 100, 300, 0)
        combined = {}
        yaml_chunks = [c for c in chunks if c["metadata"]["content_kind"] == "yaml"]
        for chunk in yaml_chunks:
            merge(combined, yaml.safe_load(chunk["content"].split("\n", 1)[1].rsplit("\n```", 1)[0]))
        self.assertEqual(combined, yaml.safe_load(body))
        groups = [c for c in yaml_chunks if json.loads(c["metadata"]["yaml_path"]) == ["services", "db", "healthcheck"]]
        self.assertEqual(len(groups), 1)
        self.assertIn("start_period:", groups[0]["content"])
        self.assertIn("timeout:", groups[0]["content"])

    def test_unsafe_or_oversized_code_is_preserved_and_flagged(self):
        cases = [
            "```yaml\ndefault: &base\n  value: 1\ncopy: *base\n```",
            "```yaml\nservices: [\n```",
            "```yaml\nservices:\n  db:\n    image: postgres\n",
            "```python\n" + "x" * 1300 + "\n```",
        ]
        for text in cases:
            with self.subTest(text=text[:35]):
                chunks = build_structured_chunks([source(text)])
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]["content"], text)
                self.assertTrue(chunks[0]["metadata"]["requires_review"])

    def test_real_corpus_spans_cover_evidence_and_reconstruct_exact_text(self):
        sources = load_sources(SOURCE_PATH)
        for item in sources:
            text = item["content"]
            covered = set()
            for chunk in build_structured_chunks([item]):
                for span in json.loads(chunk["metadata"]["source_spans"]):
                    a, b = span["source_start"], span["source_end"]
                    self.assertEqual(text[a:b], chunk["content"][span["content_start"]:span["content_end"]])
                    covered.update(range(a, b))
            # Fence markers can be regenerated; all other non-whitespace must survive.
            fence_positions = set()
            for match in re.finditer(r"(?m)^```[^\n]*(?:\n|$)", text):
                fence_positions.update(range(match.start(), match.end()))
            missing = [i for i, char in enumerate(text) if not char.isspace() and i not in covered and i not in fence_positions]
            self.assertFalse(missing, (item["section_id"], missing[:10]))

    def test_ids_track_input_and_profile_and_empty_input_is_supported(self):
        original = build_structured_chunks([source("Example.")])
        self.assertEqual(original, build_structured_chunks([source("Example.")]))
        self.assertNotEqual(original[0]["chunk_id"], build_structured_chunks([source("Changed.")])[0]["chunk_id"])
        self.assertNotEqual(original[0]["chunk_id"], build_structured_chunks([source("Example.")], context_budget=0)[0]["chunk_id"])
        self.assertEqual(build_structured_chunks([]), [])
        self.assertEqual(build_structured_chunks([source(" \n ")]), [])


if __name__ == "__main__":
    unittest.main()
