"""Compare a saved query conversion/fallback with the original hybrid baseline.

Offline only: reuse frozen Chinese query vectors, with no model client or calls.
Successful conversions remain subject to their saved human semantic review.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.hybrid_retrieval import bm25_search, cosine_search, rrf_fuse
from app.loader import PROJECT_ROOT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_lexical_query(saved: dict, original: str) -> str | None:
    """Fail closed on inconsistent artifacts instead of silently changing routes."""
    if saved["original_query"] != original or saved["vector_query"] != original:
        raise ValueError("Saved original/vector query differs from the frozen question")
    if saved["status"] == "fallback_vector_only":
        if saved["bm25_query"] is not None or not saved["fallback_reason"]:
            raise ValueError("Inconsistent fallback record")
        return None
    if saved["status"] == "converted":
        query = saved["bm25_query"]
        if not isinstance(query, str) or not query.strip() or saved["fallback_reason"]:
            raise ValueError("Inconsistent converted record")
        return query
    raise ValueError("Unsupported saved translation status")


def rank_arm(chunks: list[dict], vector: list[dict], lexical_query: str | None) -> dict:
    started = perf_counter()
    lexical = [] if lexical_query is None else bm25_search(chunks, lexical_query, 5)
    full = rrf_fuse({"vector": vector, "bm25": lexical}, max(1, len(vector) + len(lexical)))
    return {
        "bm25_query": lexical_query,
        "bm25_executed": lexical_query is not None,
        "bm25_candidates": lexical,
        "vector_candidates": vector,
        "full_rrf_ranking": full,
        "final_evidence": full[:3],
        "lexical_and_fusion_ms": round((perf_counter() - started) * 1000, 3),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-run", type=Path, required=True)
    args = parser.parse_args()
    saved_dir = args.translation_run.resolve()
    translation_manifest = read_json(saved_dir / "manifest.json")
    if translation_manifest["status"] != "complete" or not translation_manifest["inputs_and_code_unchanged"]:
        raise ValueError("Require a completed translation attempt with unchanged inputs")
    snapshot = Path(translation_manifest["source_snapshot"])
    frozen_manifest = read_json(snapshot / "manifest.json")
    hashes = dict(translation_manifest["input_and_code_sha256"])
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Protected translation/baseline inputs changed")
    qids = translation_manifest["question_ids"]
    extra = [saved_dir / "manifest.json", snapshot / "structured_context0_chunks.json",
             snapshot / "embedding_vectors.json", Path(__file__), PROJECT_ROOT / "app/hybrid_retrieval.py"]
    extra += [saved_dir / f"{qid}.json" for qid in qids]
    hashes.update({str(p): digest(p) for p in extra})
    questions = {q["question_id"]: q["question"] for q in read_json(snapshot / "questions_snapshot.json")}
    chunks = read_json(snapshot / "structured_context0_chunks.json")
    cache = read_json(snapshot / "embedding_vectors.json")
    if (len(cache["texts"]) != len(cache["vectors"])
            or len(set(cache["texts"])) != len(cache["texts"])
            or cache["usage"]["model"] != frozen_manifest["embedding_model"]
            or any(len(v) != frozen_manifest["embedding_dimension"] for v in cache["vectors"])):
        raise ValueError("Invalid cached vector alignment, model or dimension")
    vectors = dict(zip(cache["texts"], cache["vectors"]))
    output = PROJECT_ROOT / "data/query_retrieval_experiments" / (
        "comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "started", "translation_run": str(saved_dir), "source_snapshot": str(snapshot),
        "question_ids": qids, "candidate_k_per_route": 5, "final_k": 3, "rrf_constant": 60,
        "vector_search": "Exact cosine over frozen vectors, same candidates in both arms",
        "input_and_code_sha256": hashes, "model_requests_this_run": 0,
        "semantic_and_evidence_review": "pending", "development_only": True,
    }
    write_json(output / "manifest.json", manifest)
    summaries = []
    lines = ["# 已保存查询转换/回退的检索对照", "",
             "复用中文查询向量；两组相同语料、各路最多5候选、RRF常数60、最终最多3。",
             "本次0模型请求。相似度/排序不是证据充分性结论；未生成回答。", ""]
    try:
        for qid in qids:
            saved = read_json(saved_dir / f"{qid}.json")
            query = questions[qid]
            lexical_query = select_lexical_query(saved, query)
            started = perf_counter()
            vector = cosine_search(chunks, vectors, vectors[query], 5)
            vector_ms = round((perf_counter() - started) * 1000, 3)
            baseline = rank_arm(chunks, vector, query)
            variant = rank_arm(chunks, vector, lexical_query)
            ids = lambda arm: [c["chunk_id"] for c in arm["final_evidence"]]
            summary = {
                "question_id": qid, "translation_status": saved["status"],
                "fallback_reason": saved["fallback_reason"],
                "baseline_bm25_count": len(baseline["bm25_candidates"]),
                "variant_bm25_executed": variant["bm25_executed"],
                "variant_bm25_count": len(variant["bm25_candidates"]),
                "vector_count": len(vector), "baseline_final_ids": ids(baseline),
                "variant_final_ids": ids(variant), "same_final_order": ids(baseline) == ids(variant),
                "human_evidence_review": "pending",
            }
            write_json(output / f"{qid}.json", {
                "question": query, "question_id": qid, "saved_translation": saved,
                "vector_retrieval_ms": vector_ms, "baseline": baseline, "variant": variant,
                "summary": summary,
            })
            summaries.append(summary)
            lines += [f"## {qid}", "", query, "", f"转换状态：{saved['status']}；原因：{saved['fallback_reason']}。",
                      f"A原始混合基线BM25命中{len(baseline['bm25_candidates'])}；B实际分支BM25执行：{variant['bm25_executed']}。",
                      f"最终顺序一致：{summary['same_final_order']}。", ""]
            for rank, chunk in enumerate(variant["final_evidence"], 1):
                lines += [f"### B证据{rank}：{chunk['chunk_id']}", "", "````text", chunk["content"], "````", ""]
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise ValueError("Inputs or code changed during run")
        write_json(output / "summary.json", {"questions": summaries, "model_requests_this_run": 0,
                                               "inputs_and_code_unchanged": True})
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.update(status="complete", inputs_and_code_unchanged=True)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(f"Saved: {output}")
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
