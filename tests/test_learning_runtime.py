"""Profile adapter and attempt accounting; no SDK/client/network requests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.learning_runtime import HybridRetriever, RunAccounting, create_learning_app


def chunk(key, content):
    return {"chunk_id": key, "content": content,
            "metadata": {"document_id": "fixture", "section_id": key,
                         "source_url": "https://example.invalid/runtime"}}


class LearningRuntimeTests(unittest.TestCase):
    def test_same_corpus_fusion_keeps_both_routes_and_returns_isolated_data(self):
        a, b = chunk("A:1", "network overview"), chunk("B:1", "PoolTimeout limit")
        embed = Mock()
        retrieve = HybridRetriever([a, b], {
            a["content"]: [1.0, 0.0], b["content"]: [0.0, 1.0], "PoolTimeout": [1.0, 0.0],
        }, embed)
        result = retrieve("PoolTimeout")
        self.assertEqual({x["chunk_id"] for x in result}, {"A:1", "B:1"})
        b_hit = next(x for x in result if x["chunk_id"] == "B:1")
        self.assertEqual(set(b_hit["routes"]), {"vector", "bm25"})
        result[0]["content"] = "changed by caller"
        self.assertNotEqual(retrieve.chunks[0]["content"], "changed by caller")
        self.assertTrue(retrieve.trace[0]["query_vector_cached"])
        embed.assert_not_called()

    def test_new_query_is_embedded_once_and_empty_bm25_is_a_valid_route(self):
        a = chunk("A:1", "English only")
        embed = Mock(return_value=[1.0, 0.0])
        retrieve = HybridRetriever([a], {a["content"]: [1.0, 0.0]}, embed)
        self.assertEqual(retrieve("完全中文")[0]["chunk_id"], "A:1")
        retrieve("完全中文")
        embed.assert_called_once_with("完全中文")
        self.assertEqual(retrieve.trace[0]["bm25_ids"], [])
        self.assertFalse(retrieve.trace[0]["query_vector_cached"])
        self.assertTrue(retrieve.trace[1]["query_vector_cached"])

    def test_missing_corpus_vectors_never_trigger_implicit_embedding(self):
        embed = Mock()
        with self.assertRaises(ValueError):
            HybridRetriever([chunk("A:1", "missing")], {}, embed)
        embed.assert_not_called()

    def test_attempts_are_recorded_without_daily_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            accounting = RunAccounting(path, "2026-10-24")
            for attempt in range(1, 4):
                self.assertEqual(accounting.reserve("embedding", "query_embedding"), attempt)
            for attempt in range(1, 7):
                self.assertEqual(accounting.reserve("generation", "evaluate"), attempt)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                RunAccounting(path, "2026-10-24")
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(json.loads(before)["attempts"], {"generation": 6, "embedding": 3})

    def test_multiple_apps_same_day_create_separate_records(self):
        a = chunk("A:1", "fixture")
        snapshot = ([a], {"fixture": [1.0]},
                    {"embedding_model": "fixture", "embedding_base_url": "https://fixture"})
        with tempfile.TemporaryDirectory() as directory, \
             patch("app.learning_runtime.PROJECT", Path(directory)), \
             patch("app.learning_runtime.load_snapshot", return_value=snapshot), \
             patch("app.ask.create_generation_client", side_effect=lambda: (Mock(), "fixture-model")), \
             patch("app.embedding_runtime.MODEL", "fixture"), \
             patch("app.embedding_runtime.BASE_URL", "https://fixture"), \
             patch("app.embedding_runtime.DIMENSIONS", None):
            first = create_learning_app(learning_date="2026-10-27").state.learning_runtime
            second = create_learning_app(learning_date="2026-10-27").state.learning_runtime
            self.assertNotEqual(first["output"], second["output"])
            first["accounting"].reserve("generation", "evaluate")
            self.assertEqual(first["accounting"].data["attempts"]["generation"], 1)
            self.assertEqual(second["accounting"].data["attempts"]["generation"], 0)
            self.assertTrue((first["output"] / "attempts.json").exists())
            self.assertTrue((second["output"] / "attempts.json").exists())


if __name__ == "__main__":
    unittest.main()
