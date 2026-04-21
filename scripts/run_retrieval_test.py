from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

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
    "FLCS故障如何处理",
]


def _full_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def run_one_query(query: str, chunks_lookup: dict[str, dict]) -> list[str]:
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

    ranked = rerank(query_text=query, candidates=candidates, top_n=5)
    q_elapsed = time.perf_counter() - q_start

    lines.append("=" * 60)
    lines.append(f"Query: {query}")
    lines.append(f"检索候选数（rerank前）: {len(candidates)}")
    lines.append("-" * 60)

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

    lines.append(f"耗时: {q_elapsed:.2f}s")
    lines.append("=" * 60)
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    total_start = time.perf_counter()
    chunks_lookup = load_chunks_lookup(pdf_name=F16C_PDF_NAME)

    report_lines: list[str] = []
    for query in TEST_QUERIES:
        report_lines.extend(run_one_query(query, chunks_lookup))

    total_elapsed = time.perf_counter() - total_start
    report_lines.append(f"总耗时: {total_elapsed:.2f}s")

    output_dir = ROOT / "scripts" / "temp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"retrieval_test_{timestamp}.txt"
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()

