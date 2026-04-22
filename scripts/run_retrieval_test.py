from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever.retriever import (  # noqa: E402
    expand_neighbors,
    hybrid_search,
    load_chunks_lookup,
    rerank,
)

F16C_PDF_NAME = "F-16C\u201c蝰蛇\u201d.pdf"
TEST_QUERIES = [
    "TWS模式有什么限制",
    "发动机滑油压力正常范围是多少",
    "飞机启动后仪表盘上有个黄色警告灯一直亮着是什么情况？",
]
TOP_N = 5
TOKEN_BUDGET = 6000

TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def _full_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _apply_token_budget(text: str, token_budget: int) -> tuple[str, int, int]:
    token_ids = TOKENIZER.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    if total_tokens <= token_budget:
        return text, total_tokens, 0
    kept_ids = token_ids[:token_budget]
    cropped_text = TOKENIZER.decode(kept_ids, skip_special_tokens=True)
    removed_tokens = total_tokens - token_budget
    return cropped_text, total_tokens, removed_tokens


def run_one_query(query: str, chunks_lookup: dict[str, dict]) -> tuple[list[str], dict[str, Any]]:
    lines: list[str] = []
    q_start = time.perf_counter()
    hits = hybrid_search(
        query_text=query,
        top_k=20,
        pdf_name_filter=F16C_PDF_NAME,
    )
    candidates = expand_neighbors(hits, chunks_lookup)
    lines.append("[DEBUG] rerank前候选列表（chunk_id | title_h2）:")
    for cand in candidates:
        payload = cand.get("payload", {}) if isinstance(cand, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        cid = str(cand.get("chunk_id", ""))
        title_h2 = str(payload.get("title_h2", ""))
        lines.append(f"  - {cid} | {title_h2}")

    ranked = rerank(query_text=query, candidates=candidates, top_n=TOP_N)
    q_elapsed = time.perf_counter() - q_start

    lines.append("=" * 60)
    lines.append(f"Query: {query}")
    lines.append(f"检索候选数（rerank前）: {len(candidates)}")
    lines.append("-" * 60)

    selected_texts: list[str] = []
    for idx, item in enumerate(ranked, start=1):
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        rerank_score = float(item.get("rerank_score", 0.0))
        source = str(item.get("source", ""))
        chunk_id = str(item.get("chunk_id", ""))
        title_h1 = str(payload.get("title_h1", ""))
        title_h2 = str(payload.get("title_h2", ""))
        full_text = _full_text(payload.get("text", ""))
        if full_text:
            selected_texts.append(full_text)

        lines.append(
            f"[{idx}] rerank_score={rerank_score:.4f}  "
            f"source={source}  chunk_id={chunk_id}"
        )
        lines.append(f"    title_h1: {title_h1}  title_h2: {title_h2}")
        lines.append("    text全文:")
        if full_text:
            for row in full_text.split("\n"):
                lines.append(f"      {row}")
        else:
            lines.append("      ")
        lines.append("")

    assembled_context = "\n\n".join(selected_texts)
    cropped_context, total_tokens, removed_tokens = _apply_token_budget(
        assembled_context, TOKEN_BUDGET
    )
    removed_chars = len(assembled_context) - len(cropped_context)
    lines.append(
        f"[TOKEN_BUDGET] 上限={TOKEN_BUDGET} | 拼接后tokens={total_tokens} | "
        f"触顶={'是' if removed_tokens > 0 else '否'} | 删除tokens={removed_tokens} | 删除字符数={removed_chars}"
    )
    lines.append(f"耗时: {q_elapsed:.2f}s")
    lines.append("=" * 60)
    stat = {
        "query": query,
        "token_budget": TOKEN_BUDGET,
        "total_tokens_before_crop": total_tokens,
        "removed_tokens": removed_tokens,
        "removed_chars": removed_chars,
        "touched_budget": removed_tokens > 0,
        "elapsed_seconds": round(q_elapsed, 4),
    }
    return lines, stat


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    total_start = time.perf_counter()
    chunks_lookup = load_chunks_lookup(pdf_name=F16C_PDF_NAME)

    report_lines: list[str] = []
    stats: list[dict[str, Any]] = []
    for query in TEST_QUERIES:
        lines, stat = run_one_query(query, chunks_lookup)
        report_lines.extend(lines)
        stats.append(stat)

    total_elapsed = time.perf_counter() - total_start
    report_lines.append(f"总耗时: {total_elapsed:.2f}s")
    touched_count = sum(1 for s in stats if s["touched_budget"])
    removed_tokens_sum = sum(int(s["removed_tokens"]) for s in stats)
    removed_chars_sum = sum(int(s["removed_chars"]) for s in stats)
    report_lines.append("")
    report_lines.append("=== 6k Token 裁剪统计 ===")
    report_lines.append(f"测试问题数: {len(stats)}")
    report_lines.append(f"触及上限问题数: {touched_count}")
    report_lines.append(f"累计删除 tokens: {removed_tokens_sum}")
    report_lines.append(f"累计删除字符数: {removed_chars_sum}")
    for s in stats:
        report_lines.append(
            f"- {s['query']} | before={s['total_tokens_before_crop']} | "
            f"removed={s['removed_tokens']} | touched={s['touched_budget']}"
        )

    output_dir = ROOT / "scripts" / "temp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"retrieval_test_{timestamp}.txt"
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    stats_path = output_dir / f"retrieval_budget_stats_{timestamp}.json"
    stats_payload = {
        "token_budget": TOKEN_BUDGET,
        "total_queries": len(stats),
        "touched_queries": touched_count,
        "removed_tokens_sum": removed_tokens_sum,
        "removed_chars_sum": removed_chars_sum,
        "per_query": stats,
    }
    import json

    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"结果已写入: {output_path}")
    print(f"统计已写入: {stats_path}")


if __name__ == "__main__":
    main()

