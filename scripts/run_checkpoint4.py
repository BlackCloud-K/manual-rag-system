from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

ROOT = Path(__file__).resolve().parents[1]
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "dcs_manuals"

OUTPUT_DIR = ROOT / "scripts" / "temp_output"
OUTPUT_FILE_PREFIX = "checkpoint4_f16c_fine"

# 与 data/final/chunks_f16c_final.json 中 pdf_name 一致（弯引号 U+201C / U+201D）
F16C_PDF_NAME = "F-16C\u201c蝰蛇\u201d.pdf"
F16C_FINE_QUERIES = [
    "EPU是什么，什么时候会启动",
    "FLCS故障如何处理",
    "HSD页面怎么看",
    "空中加油对接步骤",
    "如何使用TGP瞄准吊舱锁定目标",
    "航炮EEGS模式怎么用",
    "发动机滑油压力正常范围是多少",
    "起飞速度和抬轮速度是多少",
    "HARM导弹有哪些攻击模式",
    "LANTIRN导航吊舱和瞄准吊舱的区别",
]
F16C_TOP_K = 3
TEXT_PREVIEW_LEN = 300


def _sparse_to_components(lexical_weights: Any) -> tuple[list[int], list[float]]:
    if isinstance(lexical_weights, dict):
        indices = lexical_weights.get("indices")
        values = lexical_weights.get("values")
        if isinstance(indices, list) and isinstance(values, list):
            return [int(i) for i in indices], [float(v) for v in values]
    return [], []


def _text_preview(text: Any, max_len: int = TEXT_PREVIEW_LEN) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len]


def _hybrid_query(
    client: QdrantClient,
    model: BGEM3FlagModel,
    query: str,
    *,
    limit: int,
    query_filter: models.Filter | None,
    with_payload: list[str],
) -> list[Any]:
    emb = model.encode(
        [query],
        batch_size=1,
        return_dense=True,
        return_sparse=True,
    )
    dense = [float(x) for x in emb["dense_vecs"][0]]
    sparse_indices, sparse_values = _sparse_to_components(emb["lexical_weights"][0])

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense,
                using="dense",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=with_payload,
        with_vectors=False,
    )
    return response.points


def _print_f16c_top3(query: str, points: list[Any]) -> None:
    print(f"\n=== [F-16C 精细验收] Query: {query} ===")
    print(f"    过滤: pdf_name == {F16C_PDF_NAME!r}")
    for rank, point in enumerate(points[:F16C_TOP_K], start=1):
        payload = point.payload or {}
        text = _text_preview(payload.get("text"))
        print(f"[{rank}] chunk_id={payload.get('chunk_id', '')}")
        print(f"     title_h2={payload.get('title_h2', '')}")
        print(f"     title_h3={payload.get('title_h3', '')}")
        print(f"     score={float(point.score):.6f}")
        print(f"     text(前{TEXT_PREVIEW_LEN}字): {text}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    start = time.perf_counter()
    client = QdrantClient(url=QDRANT_URL)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    results_for_file: dict[str, Any] = {
        "collection": COLLECTION_NAME,
        "qdrant_url": QDRANT_URL,
    }

    f16c_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="pdf_name",
                match=models.MatchValue(value=F16C_PDF_NAME),
            )
        ]
    )
    f16c_queries_out: list[dict[str, Any]] = []

    print("\n" + "=" * 60)
    print("=== F-16C 精细验收（pdf_name 固定过滤 + Hybrid Search Top3）===")
    print("=" * 60)

    for q in F16C_FINE_QUERIES:
        points = _hybrid_query(
            client,
            model,
            q,
            limit=F16C_TOP_K,
            query_filter=f16c_filter,
            with_payload=["chunk_id", "title_h2", "title_h3", "text"],
        )
        _print_f16c_top3(q, points)

        f16c_queries_out.append(
            {
                "query": q,
                "top3": [
                    {
                        "chunk_id": (p.payload or {}).get("chunk_id"),
                        "title_h2": (p.payload or {}).get("title_h2"),
                        "title_h3": (p.payload or {}).get("title_h3"),
                        "score": float(p.score),
                        "text_preview": _text_preview((p.payload or {}).get("text")),
                    }
                    for p in points[:F16C_TOP_K]
                ],
            }
        )

    results_for_file["f16c_fine_grained"] = {
        "pdf_name_filter": F16C_PDF_NAME,
        "prefetch_limit_per_vector": 20,
        "fusion": "RRF",
        "top_k": F16C_TOP_K,
        "text_preview_chars": TEXT_PREVIEW_LEN,
        "queries": f16c_queries_out,
    }

    elapsed = time.perf_counter() - start
    results_for_file["elapsed_seconds"] = round(elapsed, 3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_{timestamp}.json"
    output_path.write_text(
        json.dumps(results_for_file, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] 验收结果已写入: {output_path}")
    print(f"[DONE] 总耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
