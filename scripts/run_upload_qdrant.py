from __future__ import annotations

import json
import pickle
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "dcs_manuals"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_CACHE_PATH = ROOT / "data" / "embeddings" / "embeddings_cache.pkl"
FINAL_DIR = ROOT / "data" / "final"
FINAL_GLOB = "chunks_*_final.json"
BATCH_SIZE = 64

PAYLOAD_FIELDS = [
    "chunk_id",
    "pdf_name",
    "title_h1",
    "title_h2",
    "title_h3",
    "page_start",
    "page_end",
    "text",
    "image_paths",
    "prev_chunk_id",
    "next_chunk_id",
    "split_group_id",
]


def _load_pickle(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] embedding 缓存不存在: {path}")
        return []
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list):
        print("[WARN] embedding 缓存格式错误，期望 list。")
        return []
    return [item for item in raw if isinstance(item, dict)]


def _load_final_chunks() -> dict[str, dict[str, Any]]:
    chunk_map: dict[str, dict[str, Any]] = {}
    files = sorted(FINAL_DIR.glob(FINAL_GLOB))
    if not files:
        print(f"[WARN] 未找到 final 数据文件: {FINAL_DIR / FINAL_GLOB}")
        return chunk_map

    for file_path in files:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        print(f"[INFO] 读取 {file_path.name}: {len(raw)} chunks")
        for item in raw:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id.strip():
                chunk_map[chunk_id] = item
    return chunk_map


def _build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for field in PAYLOAD_FIELDS:
        value = chunk.get(field)
        if field == "image_paths" and not isinstance(value, list):
            value = []
        payload[field] = value
    return payload


def _ensure_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    existing_names = {c.name for c in collections}
    if COLLECTION_NAME in existing_names:
        print(f"[INFO] Collection 已存在，跳过创建: {COLLECTION_NAME}")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    print(f"[INFO] Collection 已创建: {COLLECTION_NAME}")


def _fetch_existing_chunk_ids(client: QdrantClient) -> set[str]:
    existing_ids: set[str] = set()
    offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=512,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            chunk_id = payload.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id.strip():
                existing_ids.add(chunk_id)
        if next_offset is None:
            break
        offset = next_offset

    print(f"[INFO] Qdrant 中已存在 chunk_id 数: {len(existing_ids)}")
    return existing_ids


def _upsert_points(client: QdrantClient, points: list[models.PointStruct]) -> None:
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def _point_id_from_chunk_id(chunk_id: str) -> str:
    # Qdrant point id must be uint or UUID; keep deterministic for idempotent upserts.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"dcs_manual::{chunk_id}"))


def _create_payload_indexes(client: QdrantClient) -> None:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="pdf_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title_h1",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title_h2",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="page_start",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    print("[INFO] Payload 索引已确保: pdf_name, title_h1, title_h2, page_start")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    start = time.perf_counter()

    embedding_records = _load_pickle(EMBEDDING_CACHE_PATH)
    if not embedding_records:
        print("[WARN] 无可上传 embedding 数据。")
        return
    print(f"[INFO] embedding 缓存条数: {len(embedding_records)}")

    final_chunks = _load_final_chunks()
    if not final_chunks:
        print("[WARN] 无可用 final chunk 数据。")
        return
    print(f"[INFO] final chunk 条数: {len(final_chunks)}")

    client = QdrantClient(url=QDRANT_URL)
    _ensure_collection(client)

    uploaded_chunk_ids = _fetch_existing_chunk_ids(client)

    merged_records: list[dict[str, Any]] = []
    missing_final = 0
    invalid_embedding = 0

    for item in embedding_records:
        chunk_id = item.get("chunk_id")
        dense = item.get("dense")
        sparse = item.get("sparse")
        if not isinstance(chunk_id, str) or not isinstance(dense, list) or not isinstance(sparse, dict):
            invalid_embedding += 1
            continue
        chunk = final_chunks.get(chunk_id)
        if chunk is None:
            missing_final += 1
            continue
        if chunk_id in uploaded_chunk_ids:
            continue

        merged_records.append(
            {
                "chunk_id": chunk_id,
                "dense": [float(x) for x in dense],
                "sparse": {int(k): float(v) for k, v in sparse.items()},
                "payload": _build_payload(chunk),
            }
        )

    print(f"[INFO] 待上传条数: {len(merged_records)}")
    if invalid_embedding:
        print(f"[WARN] 跳过无效 embedding 条数: {invalid_embedding}")
    if missing_final:
        print(f"[WARN] 未匹配到 final chunk 条数: {missing_final}")
    if not merged_records:
        _create_payload_indexes(client)
        elapsed = time.perf_counter() - start
        print(f"[DONE] 无新增上传。耗时: {elapsed:.2f}s")
        return

    uploaded = 0
    total = len(merged_records)

    for i in range(0, total, BATCH_SIZE):
        batch = merged_records[i : i + BATCH_SIZE]
        points: list[models.PointStruct] = []

        for row in batch:
            sparse_dict = row["sparse"]
            indices = list(sparse_dict.keys())
            values = list(sparse_dict.values())
            points.append(
                models.PointStruct(
                    id=_point_id_from_chunk_id(row["chunk_id"]),
                    payload=row["payload"],
                    vector={
                        "dense": row["dense"],
                        "sparse": models.SparseVector(indices=indices, values=values),
                    },
                )
            )

        _upsert_points(client, points)
        uploaded += len(batch)
        print(f"[PROGRESS] 已上传 {uploaded}/{total}")

    _create_payload_indexes(client)

    elapsed = time.perf_counter() - start
    print(f"[DONE] 上传总数: {uploaded}")
    print(f"[DONE] Collection: {COLLECTION_NAME}")
    print(f"[DONE] 总耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
