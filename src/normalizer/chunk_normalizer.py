"""Chunk cleanup and large-chunk splitting utilities."""
from __future__ import annotations

import re
from typing import Any

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
TOKENIZER.model_max_length = 10**9

SENT_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


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
        prev_id = str(prev_id)
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
        next_id = str(next_id)
        if next_id not in removed:
            return next_id
        current = next_id


def _rebuild_chain(chunks: list[dict]) -> list[dict]:
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        prev_id = None if idx == 0 else chunks[idx - 1].get("chunk_id")
        next_id = None if idx == total - 1 else chunks[idx + 1].get("chunk_id")
        chunk["prev_chunk_id"] = prev_id
        chunk["next_chunk_id"] = next_id
    return chunks


def remove_garbage_chunks(chunks: list) -> tuple[list, list]:
    """Remove chunks whose token count is less than 20, then reconnect chain."""
    dict_chunks = [c for c in chunks if isinstance(c, dict)]
    by_id = {
        str(c.get("chunk_id")): c
        for c in dict_chunks
        if c.get("chunk_id") is not None
    }

    removed_chunks: list[dict] = []
    removed_ids: set[str] = set()
    for chunk in dict_chunks:
        chunk_id = str(chunk.get("chunk_id"))
        text = _safe_text(chunk.get("text"))
        if _token_count(text) < 20:
            removed_chunks.append(chunk)
            removed_ids.add(chunk_id)

    kept_chunks: list[dict] = []
    for chunk in dict_chunks:
        chunk_id = str(chunk.get("chunk_id"))
        if chunk_id in removed_ids:
            continue
        new_chunk = dict(chunk)
        new_chunk["prev_chunk_id"] = _nearest_prev_kept(chunk_id, by_id, removed_ids)
        new_chunk["next_chunk_id"] = _nearest_next_kept(chunk_id, by_id, removed_ids)
        kept_chunks.append(new_chunk)

    return kept_chunks, removed_chunks


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENT_RE.findall(text)]
    sentences = [s for s in sentences if s]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _split_one_chunk(chunk: dict, target_size: int) -> list[dict]:
    chunk_id = str(chunk.get("chunk_id"))
    text = _safe_text(chunk.get("text"))
    total_tokens = _token_count(text)
    if total_tokens == 0:
        c = dict(chunk)
        c["split_group_id"] = None
        return [c]

    sentences = _split_sentences(text)
    pieces: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _token_count(sentence)
        if current_sentences and current_tokens + sent_tokens > target_size:
            pieces.append("".join(current_sentences).strip())
            current_sentences = [sentence]
            current_tokens = sent_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sent_tokens

    if current_sentences:
        pieces.append("".join(current_sentences).strip())

    out: list[dict] = []
    for idx, piece_text in enumerate(pieces):
        new_chunk = dict(chunk)
        new_chunk["chunk_id"] = f"{chunk_id}_part{idx}"
        new_chunk["text"] = piece_text
        new_chunk["split_group_id"] = chunk_id
        out.append(new_chunk)
    return out


def split_large_chunks(chunks: list, threshold: int = 3500, target_size: int = 1500) -> list:
    """Split chunks larger than threshold by greedy sentence accumulation."""
    dict_chunks = [c for c in chunks if isinstance(c, dict)]
    expanded: list[dict] = []

    for chunk in dict_chunks:
        text = _safe_text(chunk.get("text"))
        tokens = _token_count(text)
        if tokens > threshold:
            expanded.extend(_split_one_chunk(chunk, target_size=target_size))
        else:
            c = dict(chunk)
            c["split_group_id"] = None
            expanded.append(c)

    return _rebuild_chain(expanded)
