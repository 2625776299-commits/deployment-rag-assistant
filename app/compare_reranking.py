"""One-attempt LLM reranking of saved, unchanged query-retrieval candidate pools."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import APIStatusError, APITimeoutError

from app.ask import create_generation_client, generation_options
from app.compare_query_retrieval import digest, read_json, write_json
from app.loader import PROJECT_ROOT
from app.reranking import build_rerank_messages, rerank_candidates


def request_order(client, model: str, messages: list[dict], path: Path) -> str:
    try:
        response = client.chat.completions.create(model=model, messages=messages,
                                                 **generation_options(client, evaluate=True))
    except APITimeoutError:
        write_json(path, {"requested_model": model, "error_type": "APITimeoutError"})
        raise TimeoutError("rerank_timeout") from None
    except APIStatusError as error:
        write_json(path, {"requested_model": model, "error_type": "APIStatusError", "http_status": error.status_code})
        raise RuntimeError("rerank_api_error") from None
    except Exception as error:
        write_json(path, {"requested_model": model, "error_type": type(error).__name__})
        raise RuntimeError("rerank_transport_error") from None
    choice = response.choices[0] if response.choices else None
    write_json(path, {"requested_model": model, "returned_model": response.model,
                     "finish_reason": choice.finish_reason if choice else None,
                     "raw_text": choice.message.content if choice else None,
                     "usage": response.usage.model_dump() if response.usage else None})
    if choice is None or choice.finish_reason != "stop":
        raise ValueError("rerank_response_incomplete")
    return choice.message.content


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--questions", nargs="+", choices=["Q03", "Q06", "P02"], default=["Q03"])
    parser.add_argument("--final-k", type=int, default=3)
    args = parser.parse_args()
    if len(set(args.questions)) != len(args.questions) or args.final_k <= 0:
        parser.error("Question IDs must be unique and final-k must be positive")
    hashes = {}
    sources = {}
    for source_dir in args.comparison_runs:
        source_dir = source_dir.resolve()
        source_manifest = read_json(source_dir / "manifest.json")
        if source_manifest["status"] != "complete" or not source_manifest["inputs_and_code_unchanged"]:
            raise ValueError("Require completed retrieval runs with unchanged inputs")
        for path, expected in source_manifest["input_and_code_sha256"].items():
            if digest(Path(path)) != expected or (path in hashes and hashes[path] != expected):
                raise ValueError("Protected retrieval input changed")
            hashes[path] = expected
        hashes[str(source_dir / "manifest.json")] = digest(source_dir / "manifest.json")
        for qid in set(args.questions) & set(source_manifest["question_ids"]):
            if qid in sources:
                raise ValueError("Ambiguous question source")
            path = source_dir / f"{qid}.json"
            sources[qid] = (path, read_json(path))
            hashes[str(path)] = digest(path)
    if set(sources) != set(args.questions):
        raise ValueError("A requested question is missing from the saved comparisons")
    extra = [Path(__file__), PROJECT_ROOT / "app/reranking.py", PROJECT_ROOT / "app/compare_query_retrieval.py",
             PROJECT_ROOT / "app/ask.py", PROJECT_ROOT / "app/embeddings.py"]
    hashes.update({str(path): digest(path) for path in extra})
    for _, saved in sources.values():
        arm = saved["variant"]
        pool = arm["full_rrf_ranking"]
        union = {c["chunk_id"] for c in arm["vector_candidates"] + arm["bm25_candidates"]}
        if {c["chunk_id"] for c in pool} != union or len(pool) != len(union):
            raise ValueError("Candidate pool is not the complete deduplicated retrieval union")
        build_rerank_messages(saved["question"], pool)
    client, model = create_generation_client()
    output = PROJECT_ROOT / "data/reranking_experiments" / (
        "rerank_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "question_ids": args.questions, "requested_model": model,
                "input_and_code_sha256": hashes, "candidate_pool": "Saved variant full retrieval union in RRF order",
                "sampling": "Provider defaults; no temperature or seed supplied", "timeout_seconds": 60,
                "retries": 0, "planned_rerank_requests": len(args.questions), "attempted_rerank_requests": 0,
                "embedding_requests": 0, "answer_generation_requests": 0, "final_k": args.final_k,
                "development_only": True, "human_evidence_review": "pending",
                "position_bias_control": "None; candidate presentation order is RRF order"}
    write_json(output / "manifest.json", manifest)
    # Freeze every question and complete pool before any paid model request.
    for qid, (path, saved) in sources.items():
        pool = saved["variant"]["full_rrf_ranking"]
        write_json(output / f"{qid}_input.json", {"source_record": str(path), "question": saved["question"], "candidate_pool": pool})
        write_json(output / f"{qid}_messages.json", build_rerank_messages(saved["question"], pool))
    print(f"Output: {output}", flush=True)
    summaries = []
    usages = []
    lines = ["# 同候选池：RRF与LLM重排", "", "每题一次请求；沿用原中文问题和完整候选池，无新Embedding、无回答生成。",
             "完整排列校验通过仅表示ID合法；证据效果仍需人工核对。候选按RRF顺序输入，未控制位置偏差。", ""]
    try:
        with client:
            for qid in args.questions:
                if any(digest(Path(path)) != expected for path, expected in hashes.items()):
                    raise ValueError("Input or implementation changed before request")
                _, saved = sources[qid]
                pool = saved["variant"]["full_rrf_ranking"]
                raw_path = output / f"{qid}_raw.json"

                def generate(messages):
                    manifest["attempted_rerank_requests"] += 1
                    write_json(output / "manifest.json", manifest)
                    return request_order(client, model, messages, raw_path)

                result = rerank_candidates(saved["question"], pool, generate, final_k=args.final_k)
                raw = read_json(raw_path) if raw_path.exists() else None
                usages.append((raw or {}).get("usage"))
                before = [c["chunk_id"] for c in pool[:args.final_k]]
                after = [c["chunk_id"] for c in result["final_evidence"]]
                summary = {"question_id": qid, "status": result["status"], "fallback_reason": result["fallback_reason"],
                           "candidate_count": len(pool), "rrf_final_ids": before, "rerank_final_ids": after,
                           "same_final_order": before == after, "same_final_set": set(before) == set(after),
                           "elapsed_ms": result["elapsed_ms"], "human_evidence_review": "pending"}
                write_json(output / f"{qid}.json", {"question_id": qid, "question": saved["question"],
                                                   "result": result, "raw_generation": raw, "summary": summary})
                summaries.append(summary)
                lines += [f"## {qid}", "", saved["question"], "", f"状态：{result['status']}；原因：{result['fallback_reason']}。",
                          "", "| 候选ID | RRF排名 | 重排/回退排名 | 进入最终证据 |", "|---|---:|---:|---|"]
                by_id = {c["chunk_id"]: c for c in pool}
                old_rank = {c["chunk_id"]: i for i, c in enumerate(pool, 1)}
                for rank, key in enumerate(result["ranked_chunk_ids"], 1):
                    lines.append(f"| {key} | {old_rank[key]} | {rank} | {'是' if key in after else '否'} |")
                for key in after:
                    lines += ["", f"### 最终证据：{key}", "", "````text", by_id[key]["content"], "````"]
                print(json.dumps(summary, ensure_ascii=False), flush=True)
        if any(digest(Path(path)) != expected for path, expected in hashes.items()):
            raise ValueError("Inputs or implementation changed during run")
        manifest.update(status="complete", inputs_and_code_unchanged=True)
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", {"questions": summaries, "reported_total_tokens": sum((u or {}).get("total_tokens", 0) or 0 for u in usages),
                                             "missing_usage_question_ids": [s["question_id"] for s, u in zip(summaries, usages) if u is None],
                                             "attempted_rerank_requests": manifest["attempted_rerank_requests"],
                                             "embedding_requests": 0, "answer_generation_requests": 0,
                                             "inputs_and_code_unchanged": True})
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        write_json(output / "manifest.json", manifest)
        print(f"Stopped: {type(error).__name__}; inspect saved artifacts", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
