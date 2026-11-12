"""Check that fusion preserves evidence rather than filtering to agreement."""

import unittest

from app.hybrid_retrieval import bm25_search, cosine_search, rrf_fuse, tokenize


def hit(key, score=1.0):
    return {"chunk_id": key, "content": f"Evidence {key}", "metadata": {"section_id": key}, "score": score}


class HybridRetrievalTests(unittest.TestCase):
    def test_union_preserves_single_route_evidence_and_accumulates_shared_votes(self):
        result = rrf_fuse({"bm25": [hit("A"), hit("B")], "vector": [hit("B"), hit("C")]}, 5)
        self.assertEqual({c["chunk_id"] for c in result}, {"A", "B", "C"})
        self.assertEqual(result[0]["chunk_id"], "B")
        self.assertAlmostEqual(result[0]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertEqual(result[0]["content"], "Evidence B")
        self.assertEqual(set(result[0]["routes"]), {"bm25", "vector"})

    def test_fewer_hits_and_an_empty_route_do_not_trigger_padding(self):
        result = rrf_fuse({"bm25": [], "vector": [hit("A")]}, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chunk_id"], "A")

    def test_duplicate_votes_and_conflicting_bodies_are_rejected(self):
        with self.assertRaises(ValueError):
            rrf_fuse({"bm25": [hit("A"), hit("A")]})
        changed = hit("A")
        changed["content"] = "A different source under the same ID"
        with self.assertRaises(ValueError):
            rrf_fuse({"bm25": [hit("A")], "vector": [changed]})

    def test_technical_tokens_and_chinese_bigrams_are_explicit(self):
        self.assertEqual(tokenize("service_healthy override.env PoolTimeout 数据库"),
                         ["service_healthy", "override.env", "pooltimeout", "数据", "据库"])

    def test_no_keyword_overlap_yields_no_bm25_hit(self):
        self.assertEqual(bm25_search([hit("A")], "数据库重建"), [])
        result = bm25_search([hit("A"), hit("B")], "Evidence A")
        self.assertEqual(result[0]["chunk_id"], "A")

    def test_vector_ranking_and_dimension_mismatch(self):
        chunks = [hit("A"), hit("B")]
        vectors = {"Evidence A": [1.0, 0.0], "Evidence B": [0.0, 1.0]}
        self.assertEqual(cosine_search(chunks, vectors, [1.0, 0.0])[0]["chunk_id"], "A")
        with self.assertRaises(ValueError):
            cosine_search(chunks, vectors, [1.0])


if __name__ == "__main__":
    unittest.main()
