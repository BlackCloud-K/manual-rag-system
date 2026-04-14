"""Clean low-token garbage chunks from chunks_f16c.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "chunks_f16c.json"
OUTPUT_PATH = ROOT / "data" / "chunks_f16c_cleaned.json"


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _title_of(chunk: dict) -> str:
    h2 = _safe_str(chunk.get("title_h2"))
    h3 = _safe_str(chunk.get("title_h3"))
    if h2 and h3:
        return f"{h2} / {h3}"
    return h2 or h3 or _safe_str(chunk.get("title_h1")) or "-"


def _nearest_prev_kept(chunk_id: str, by_id: dict[str, dict], removed: set[str]) -> str | None:
    current = chunk_id
    seen: set[str] = set()
    while True:
        if current in seen:
            return None
        seen.add(current)
        chunk = by_id.get(current)
        if chunk is None:
            return None
        prev_id = chunk.get("prev_chunk_id")
        if prev_id is None:
            return None
        prev_id = _safe_str(prev_id)
        if not prev_id:
            return None
        if prev_id not in removed:
            return prev_id
        current = prev_id


def _nearest_next_kept(chunk_id: str, by_id: dict[str, dict], removed: set[str]) -> str | None:
    current = chunk_id
    seen: set[str] = set()
    while True:
        if current in seen:
            return None
        seen.add(current)
        chunk = by_id.get(current)
        if chunk is None:
            return None
        next_id = chunk.get("next_chunk_id")
        if next_id is None:
            return None
        next_id = _safe_str(next_id)
        if not next_id:
            return None
        if next_id not in removed:
            return next_id
        current = next_id


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"JSON root must be a list: {INPUT_PATH}")

    chunks: list[dict] = [c for c in raw if isinstance(c, dict)]
    by_id: dict[str, dict] = {}
    for chunk in chunks:
        cid = _safe_str(chunk.get("chunk_id"))
        if cid:
            by_id[cid] = chunk

    removed_infos: list[dict[str, object]] = []
    removed_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = _safe_str(chunk.get("chunk_id"))
        text = _safe_str(chunk.get("text"))
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count < 20:
            removed_ids.add(chunk_id)
            removed_infos.append(
                {
                    "chunk_id": chunk_id,
                    "title": _title_of(chunk),
                    "tokens": token_count,
                    "text": text,
                }
            )

    impacted_links: set[tuple[str | None, str | None]] = set()
    for info in removed_infos:
        removed_id = _safe_str(info["chunk_id"])
        prev_kept = _nearest_prev_kept(removed_id, by_id, removed_ids)
        next_kept = _nearest_next_kept(removed_id, by_id, removed_ids)
        impacted_links.add((prev_kept, next_kept))

    cleaned_chunks: list[dict] = []
    for chunk in chunks:
        chunk_id = _safe_str(chunk.get("chunk_id"))
        if chunk_id in removed_ids:
            continue
        new_chunk = dict(chunk)
        new_chunk["prev_chunk_id"] = _nearest_prev_kept(chunk_id, by_id, removed_ids)
        new_chunk["next_chunk_id"] = _nearest_next_kept(chunk_id, by_id, removed_ids)
        cleaned_chunks.append(new_chunk)

    OUTPUT_PATH.write_text(
        json.dumps(cleaned_chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"原始Chunk总数: {len(chunks)}")
    print(f"删除的Chunk数量: {len(removed_infos)}")
    if removed_infos:
        print("被删除Chunk明细:")
        for info in removed_infos:
            print(
                f"- chunk_id={info['chunk_id']} | 标题={info['title']} | "
                f"tokens={info['tokens']} | text={info['text']!r}"
            )
    else:
        print("被删除Chunk明细: 无")
    print(f"清理后Chunk总数: {len(cleaned_chunks)}")
    print("链表重建确认（受影响邻居新连接）:")
    filtered_links = sorted(
        impacted_links,
        key=lambda p: ((p[0] or ""), (p[1] or "")),
    )
    if filtered_links:
        for prev_id, next_id in filtered_links:
            print(f"- {prev_id or 'None'} -> {next_id or 'None'}")
    else:
        print("- 无（无删除）")

    print(f"已输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
