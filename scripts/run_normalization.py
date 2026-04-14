"""Run chunk normalization for all manual chunk files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normalizer.chunk_normalizer import (  # noqa: E402
    TOKENIZER,
    remove_garbage_chunks,
    split_large_chunks,
)

INPUT_FILES = [
    ROOT / "data" / "chunks_f16c.json",
    ROOT / "data" / "chunks_dcsjf17.json",
    ROOT / "data" / "chunks_f15c.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn.json",
]


def _token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def _output_path_for(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_normalized.json")


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _print_removed_chunks(removed_chunks: list[dict]) -> None:
    if not removed_chunks:
        print("被删除的垃圾Chunk: 无")
        return
    print("被删除的垃圾Chunk:")
    for c in removed_chunks:
        cid = c.get("chunk_id")
        text = _safe_text(c.get("text"))
        tokens = _token_count(text)
        print(f"- chunk_id={cid} | tokens={tokens} | text={text!r}")


def _print_split_chunks(before_split_chunks: list[dict], after_split_chunks: list[dict]) -> int:
    parent_token_map = {}
    for c in before_split_chunks:
        cid = str(c.get("chunk_id"))
        parent_token_map[cid] = _token_count(_safe_text(c.get("text")))

    groups: dict[str, list[dict]] = {}
    for c in after_split_chunks:
        group_id = c.get("split_group_id")
        if group_id is None:
            continue
        gid = str(group_id)
        groups.setdefault(gid, []).append(c)

    if not groups:
        print("被拆分的大Chunk: 无")
        return 0

    print("被拆分的大Chunk:")
    for gid, pieces in sorted(groups.items(), key=lambda kv: kv[0]):
        piece_tokens = [_token_count(_safe_text(p.get("text"))) for p in pieces]
        print(
            f"- 原chunk_id={gid} | 原tokens={parent_token_map.get(gid, 0)} | "
            f"拆分片数={len(pieces)} | 每片tokens={piece_tokens}"
        )
    return sum(len(v) - 1 for v in groups.values())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for input_path in INPUT_FILES:
        if not input_path.exists():
            print(f"跳过：文件不存在 {input_path}")
            print("")
            continue

        raw = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            print(f"跳过：JSON根节点不是list {input_path}")
            print("")
            continue

        original_chunks = [c for c in raw if isinstance(c, dict)]
        cleaned_chunks, removed_chunks = remove_garbage_chunks(original_chunks)
        normalized_chunks = split_large_chunks(cleaned_chunks, threshold=3500, target_size=1500)
        split_groups: dict[str, list[dict]] = {}
        for c in normalized_chunks:
            gid = c.get("split_group_id")
            if gid is None:
                continue
            split_groups.setdefault(str(gid), []).append(c)
        added_by_split = sum(len(v) - 1 for v in split_groups.values())

        output_path = _output_path_for(input_path)
        output_path.write_text(
            json.dumps(normalized_chunks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("=" * 88)
        print(f"文件: {input_path.name}")
        print(
            f"原始Chunk数={len(original_chunks)} -> 删除垃圾数={len(removed_chunks)} -> "
            f"拆分产生新增数={added_by_split} -> 最终Chunk数={len(normalized_chunks)}"
        )
        _print_removed_chunks(removed_chunks)
        _print_split_chunks(cleaned_chunks, normalized_chunks)
        print(f"已写入: {output_path}")
        print("")


if __name__ == "__main__":
    main()
