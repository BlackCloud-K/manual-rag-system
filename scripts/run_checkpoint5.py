"""Checkpoint 5: end-to-end retrieval evaluation on testset_f16c.json."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import retriever as retriever_mod  # noqa: E402
from src.retriever.retriever import (  # noqa: E402
    expand_neighbors,
    hybrid_search,
    load_chunks_lookup,
    rerank,
)

TESTSET_PATH = ROOT / "data" / "testset_f16c.json"
F16C_PDF_NAME = "F-16C\u201c蝰蛇\u201d.pdf"
TOP_N = 5
OUT_OF_SCOPE_SCORE_THRESHOLD = 0.05


def _warmup_models() -> None:
    retriever_mod._get_embed_model()
    retriever_mod._get_rerank_model()


def _find_hit_rank(ranked: list[dict[str, Any]], source_id: str) -> tuple[int | None, float | None]:
    for i, row in enumerate(ranked, start=1):
        if str(row.get("chunk_id", "")) == source_id:
            score = row.get("rerank_score")
            rs = float(score) if score is not None else None
            return i, rs
    return None, None


def _top1_info(ranked: list[dict[str, Any]]) -> tuple[str, float]:
    if not ranked:
        return "", 0.0
    row = ranked[0]
    cid = str(row.get("chunk_id", ""))
    score = float(row.get("rerank_score", 0.0))
    return cid, score


def run() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    raw = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("ERROR: testset must be a JSON array", file=sys.stderr)
        return

    items = [x for x in raw if isinstance(x, dict)]
    chunks_lookup = load_chunks_lookup(pdf_name=F16C_PDF_NAME)

    _warmup_models()

    per_results: list[dict[str, Any]] = []
    times_by_id: dict[str, float] = {}

    for item in items:
        qid = str(item.get("id", ""))
        question = str(item.get("question", ""))
        expected = bool(item.get("expected_in_topn", True))
        source_id = item.get("source_chunk_id")
        source_str = str(source_id) if source_id is not None else ""

        t0 = time.perf_counter()
        hits = hybrid_search(
            query_text=question,
            top_k=50,
            pdf_name_filter=F16C_PDF_NAME,
        )
        candidates = expand_neighbors(hits, chunks_lookup)
        ranked = rerank(question, candidates, top_n=TOP_N)
        elapsed = time.perf_counter() - t0
        times_by_id[qid] = elapsed

        top1_id, top1_score = _top1_info(ranked)

        detail: dict[str, Any] = {
            "id": qid,
            "question": question,
            "expected_in_topn": expected,
            "source_chunk_id": source_id,
            "elapsed_seconds": round(elapsed, 4),
            "candidate_count_before_rerank": len(candidates),
            "top_n": TOP_N,
            "ranked_top": [
                {
                    "chunk_id": str(r.get("chunk_id", "")),
                    "rerank_score": float(r.get("rerank_score", 0.0)),
                    "source": r.get("source"),
                }
                for r in ranked
            ],
        }

        if expected:
            hit_rank, hit_score = _find_hit_rank(ranked, source_str)
            passed = hit_rank is not None
            detail["judgment"] = "PASS" if passed else "FAIL"
            detail["hit_rank"] = hit_rank
            detail["hit_rerank_score"] = hit_score
            detail["top1_chunk_id"] = top1_id
            detail["top1_rerank_score"] = top1_score

            if passed:
                rs = hit_score if hit_score is not None else 0.0
                print(
                    f"[PASS] {qid} | {question}\n"
                    f"       source: {source_str} | 命中位置: Top {hit_rank} | "
                    f"rerank_score: {rs:.4f} | 耗时: {elapsed:.2f}s\n"
                )
            else:
                print(
                    f"[FAIL] {qid} | {question}\n"
                    f"       source: {source_str} | 未出现在Top5中 | "
                    f"Top1: {top1_id} (score={top1_score:.2f}) | 耗时: {elapsed:.2f}s\n"
                )
        else:
            if not ranked:
                passed = True
                detail["judgment"] = "PASS"
                detail["reason"] = "empty_candidates"
                detail["top1_rerank_score"] = None
                print(
                    f"[PASS] {qid} | {question}（手册外）\n"
                    f"       候选为空，视为正确拒绝 | 耗时: {elapsed:.2f}s\n"
                )
            else:
                passed = top1_score < OUT_OF_SCOPE_SCORE_THRESHOLD
                detail["judgment"] = "PASS" if passed else "FAIL"
                detail["top1_chunk_id"] = top1_id
                detail["top1_rerank_score"] = top1_score
                detail["threshold"] = OUT_OF_SCOPE_SCORE_THRESHOLD
                if passed:
                    print(
                        f"[PASS] {qid} | {question}（手册外）\n"
                        f"       Top1 score: {top1_score:.3f} < 阈值{OUT_OF_SCOPE_SCORE_THRESHOLD} | "
                        f"耗时: {elapsed:.2f}s\n"
                    )
                else:
                    print(
                        f"[FAIL] {qid} | {question}（手册外）\n"
                        f"       Top1 score: {top1_score:.3f} >= 阈值{OUT_OF_SCOPE_SCORE_THRESHOLD} | "
                        f"耗时: {elapsed:.2f}s\n"
                    )

        per_results.append(detail)

    n_total = len(items)
    in_scope = [d for d in per_results if d.get("expected_in_topn")]
    out_scope = [d for d in per_results if not d.get("expected_in_topn")]

    pass_in = sum(1 for d in in_scope if d.get("judgment") == "PASS")
    fail_in = len(in_scope) - pass_in
    pass_out = sum(1 for d in out_scope if d.get("judgment") == "PASS")
    fail_out = len(out_scope) - pass_out

    total_pass = pass_in + pass_out
    total_fail = fail_in + fail_out

    hit_rate = (pass_in / len(in_scope) * 100.0) if in_scope else 0.0
    reject_rate = (pass_out / len(out_scope) * 100.0) if out_scope else 0.0

    total_time = sum(times_by_id.values())
    avg_time = total_time / n_total if n_total else 0.0
    slowest_id = ""
    slowest_t = 0.0
    for qid, t in times_by_id.items():
        if t >= slowest_t:
            slowest_t = t
            slowest_id = qid

    print("=" * 60)
    print("Checkpoint 5 验收结果汇总")
    print("=" * 60)
    print(f"题目总数:     {n_total}")
    print(f"PASS:         {total_pass}")
    print(f"FAIL:         {total_fail}")
    print()
    print("分类统计:")
    print(
        f"  expected_in_topn=true  ({len(in_scope)}题): "
        f"{pass_in} PASS / {fail_in} FAIL  命中率: {hit_rate:.1f}%"
    )
    print(
        f"  expected_in_topn=false ({len(out_scope)}题):  "
        f"{pass_out} PASS / {fail_out} FAIL  拒绝率: {reject_rate:.0f}%"
    )
    print()
    print("耗时统计（不含模型加载）:")
    print(f"  总耗时:   {total_time:.1f}s")
    print(f"  平均每题:  {avg_time:.2f}s")
    print(f"  最慢一题:  {slowest_t:.1f}s ({slowest_id})")
    print()

    crit_hit = hit_rate > 80.0
    crit_time = avg_time < 3.0
    crit_oos = reject_rate == 100.0 if out_scope else True

    print("验收标准对照:")
    print(
        f"  [{'✅' if crit_hit else '❌'}] 精确召回命中率 > 80%:  {hit_rate:.1f}%"
    )
    print(
        f"  [{'✅' if crit_time else '❌'}] 端到端平均耗时 < 3s:   {avg_time:.2f}s"
    )
    print(
        f"  [{'✅' if crit_oos else '❌'}] 手册外问题拒绝率=100%: "
        f"{reject_rate:.0f}%"
    )
    print("=" * 60)

    summary = {
        "checkpoint": "checkpoint5",
        "testset_path": str(TESTSET_PATH.relative_to(ROOT)),
        "f16c_pdf_name": F16C_PDF_NAME,
        "top_n": TOP_N,
        "out_of_scope_score_threshold": OUT_OF_SCOPE_SCORE_THRESHOLD,
        "totals": {
            "questions": n_total,
            "pass": total_pass,
            "fail": total_fail,
        },
        "by_category": {
            "expected_in_topn_true": {
                "count": len(in_scope),
                "pass": pass_in,
                "fail": fail_in,
                "hit_rate_percent": round(hit_rate, 2),
            },
            "expected_in_topn_false": {
                "count": len(out_scope),
                "pass": pass_out,
                "fail": fail_out,
                "reject_rate_percent": round(reject_rate, 2),
            },
        },
        "timing_seconds": {
            "total": round(total_time, 3),
            "average_per_question": round(avg_time, 4),
            "slowest": {"question_id": slowest_id, "seconds": round(slowest_t, 3)},
        },
        "acceptance": {
            "hit_rate_above_80": crit_hit,
            "avg_latency_under_3s": crit_time,
            "out_of_scope_reject_100": crit_oos,
        },
        "per_question": per_results,
    }

    out_dir = ROOT / "scripts" / "temp_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"checkpoint5_result_{ts}.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n完整结果已写入: {out_path}")


if __name__ == "__main__":
    run()
