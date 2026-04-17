from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "final"
OUTPUT_PATH = ROOT / "data" / "embeddings" / "embeddings_cache.pkl"
INPUT_GLOB = "chunks_*_final.json"
BATCH_SIZE = 32
PROGRESS_EVERY = 100


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        with path.open("rb") as f:
            raw = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] 读取缓存失败，将从空缓存开始: {exc}")
        return []

    if not isinstance(raw, list):
        print("[WARN] 现有缓存格式不是 list，将从空缓存开始。")
        return []

    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        dense = item.get("dense")
        sparse = item.get("sparse")
        if isinstance(chunk_id, str) and isinstance(dense, list) and isinstance(sparse, dict):
            cleaned.append(item)
    return cleaned


def _save_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f)


def _sparse_to_dict(lexical_weights: Any) -> dict[int, float]:
    if isinstance(lexical_weights, dict):
        indices = lexical_weights.get("indices")
        values = lexical_weights.get("values")
        if isinstance(indices, list) and isinstance(values, list):
            return {int(idx): float(val) for idx, val in zip(indices, values)}
    return {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    start = time.perf_counter()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_records = _load_cache(OUTPUT_PATH)
    existing_ids = {
        item["chunk_id"] for item in existing_records if isinstance(item.get("chunk_id"), str)
    }
    print(f"[INFO] 已加载缓存条数: {len(existing_records)}")

    input_files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not input_files:
        print(f"[WARN] 未找到输入文件: {INPUT_DIR / INPUT_GLOB}")
        return

    all_chunks: list[tuple[str, str]] = []
    seen_input_ids: set[str] = set()
    for file_path in input_files:
        chunks = _load_chunks(file_path)
        print(f"[INFO] 读取文件 {file_path.name}: {len(chunks)} chunks")
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if not isinstance(text, str):
                text = ""
            if chunk_id in seen_input_ids:
                continue
            seen_input_ids.add(chunk_id)
            if chunk_id in existing_ids:
                continue
            all_chunks.append((chunk_id, text))

    total_to_process = len(all_chunks)
    print(f"[INFO] 待处理 chunks: {total_to_process}")
    if total_to_process == 0:
        elapsed = time.perf_counter() - start
        print(f"[INFO] 无新增数据。总缓存数: {len(existing_records)}，耗时: {elapsed:.2f}s")
        return

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    processed = 0
    for i in range(0, total_to_process, BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        batch_ids = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]

        outputs = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
        )
        dense_vecs = outputs.get("dense_vecs", [])
        lexical_weights = outputs.get("lexical_weights", [])

        for chunk_id, dense, sparse in zip(batch_ids, dense_vecs, lexical_weights):
            record = {
                "chunk_id": chunk_id,
                "dense": [float(x) for x in dense],
                "sparse": _sparse_to_dict(sparse),
            }
            existing_records.append(record)
            existing_ids.add(chunk_id)
            processed += 1

            if processed % PROGRESS_EVERY == 0:
                print(f"[PROGRESS] 已处理 {processed}/{total_to_process}")

        _save_cache(OUTPUT_PATH, existing_records)

    elapsed = time.perf_counter() - start
    print(f"[DONE] 新增处理总数: {processed}")
    print(f"[DONE] 缓存总数: {len(existing_records)}")
    print(f"[DONE] 输出文件: {OUTPUT_PATH}")
    print(f"[DONE] 总耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
