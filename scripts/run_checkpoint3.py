from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MANUAL_KEYS = [
    "f16c",
    "dcsjf17",
    "f15c",
    "dcsfa18cearlyaccessguidecn",
]

FINAL_FILES = {k: ROOT / "data" / f"chunks_{k}_final.json" for k in MANUAL_KEYS}
NORMALIZED_FILES = {k: ROOT / "data" / f"chunks_{k}_normalized.json" for k in MANUAL_KEYS}
CACHE_FILE = ROOT / "data" / "captions_cache.json"

IMG_CAPTION_PATTERN = re.compile(r"^\[IMG_CAPTION\]\s*(.*)$")
SAMPLE_TOTAL = 50
RANDOM_SEED = 42


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_image_paths(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if isinstance(v, str) and str(v).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def _extract_img_captions(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    captions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = IMG_CAPTION_PATTERN.match(stripped)
        if m:
            captions.append(m.group(1).strip())
    return captions


def _uniform_sample(groups: dict[str, list[dict[str, Any]]], total: int) -> list[tuple[str, dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED)
    manuals = list(groups.keys())
    if not manuals:
        return []

    target_per_manual = total // len(manuals)
    remainder = total % len(manuals)

    sampled: list[tuple[str, dict[str, Any]]] = []
    leftovers: list[tuple[str, dict[str, Any]]] = []

    for i, manual in enumerate(manuals):
        pool = groups.get(manual, [])
        take = min(len(pool), target_per_manual + (1 if i < remainder else 0))
        if take > 0:
            picked = rng.sample(pool, take) if len(pool) > take else pool[:]
            sampled.extend((manual, row) for row in picked)
            picked_ids = {id(x) for x in picked}
            leftovers.extend((manual, row) for row in pool if id(row) not in picked_ids)
        else:
            leftovers.extend((manual, row) for row in pool)

    if len(sampled) < total and leftovers:
        need = min(total - len(sampled), len(leftovers))
        sampled.extend(rng.sample(leftovers, need) if len(leftovers) > need else leftovers[:need])

    return sampled[:total]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    captioned_groups: dict[str, list[dict[str, Any]]] = {k: [] for k in MANUAL_KEYS}
    final_chunk_counts: dict[str, int] = {}
    normalized_chunk_counts: dict[str, int] = {}
    image_chunk_total = 0
    captioned_chunk_total = 0
    unique_image_paths: set[str] = set()

    for manual in MANUAL_KEYS:
        final_path = FINAL_FILES[manual]
        normalized_path = NORMALIZED_FILES[manual]

        final_chunks = _load_json(final_path)
        normalized_chunks = _load_json(normalized_path)
        if not isinstance(final_chunks, list) or not isinstance(normalized_chunks, list):
            raise ValueError(f"Invalid JSON structure for manual: {manual}")

        final_chunk_counts[manual] = len(final_chunks)
        normalized_chunk_counts[manual] = len(normalized_chunks)

        for chunk in final_chunks:
            if not isinstance(chunk, dict):
                continue
            image_paths = _extract_image_paths(chunk.get("image_paths"))
            if image_paths:
                image_chunk_total += 1
                for p in image_paths:
                    unique_image_paths.add(p.replace("\\", "/").strip())

            captions = _extract_img_captions(chunk.get("text"))
            if captions:
                captioned_chunk_total += 1
                captioned_groups[manual].append(
                    {
                        "chunk_id": str(chunk.get("chunk_id", "")),
                        "first_image_path": image_paths[0] if image_paths else "",
                        "captions": captions,
                    }
                )

    sampled_rows = _uniform_sample(captioned_groups, SAMPLE_TOTAL)

    print("=== 随机抽样（含[IMG_CAPTION]的Chunk） ===")
    print(f"抽样数量: {len(sampled_rows)} / 目标 {SAMPLE_TOTAL}")
    for idx, (_manual, row) in enumerate(sampled_rows, start=1):
        caption_text = " | ".join(row["captions"])
        print(f"\n[{idx}] chunk_id: {row['chunk_id']}")
        print(f"first_image_path: {row['first_image_path']}")
        print(f"caption: {caption_text}")

    total_final_chunks = sum(final_chunk_counts.values())
    total_normalized_chunks = sum(normalized_chunk_counts.values())
    caption_ratio = (captioned_chunk_total / image_chunk_total * 100.0) if image_chunk_total else 0.0

    cache_raw = _load_json(CACHE_FILE) if CACHE_FILE.exists() else {}
    cache = cache_raw if isinstance(cache_raw, dict) else {}
    cache_entries = len(cache)
    caption_failed_count = sum(1 for v in cache.values() if str(v) == "CAPTION_FAILED")

    print("\n=== 统计汇总 ===")
    print(f"四本final JSON的chunk总数: {total_final_chunks}")
    print(f"含摘要的chunk数: {captioned_chunk_total} / 含图片chunk数: {image_chunk_total} ({caption_ratio:.2f}%)")
    print(f"CAPTION_FAILED残留数: {caption_failed_count}")
    print(
        "四本final JSON的image_paths去重总图片数 vs captions_cache条目数: "
        f"{len(unique_image_paths)} vs {cache_entries}"
    )

    rule1_pass = (captioned_chunk_total / image_chunk_total) > 0.9 if image_chunk_total else False
    rule2_pass = caption_failed_count < (len(unique_image_paths) * 0.1) if unique_image_paths else False
    rule3_pass = total_final_chunks == total_normalized_chunks and all(
        final_chunk_counts[k] == normalized_chunk_counts[k] for k in MANUAL_KEYS
    )

    print("\n=== 验收判定 ===")
    print(
        f"{'PASS' if rule1_pass else 'FAIL'} | "
        "✅ 含摘要chunk占有图片chunk的比例 > 90%"
    )
    print(
        f"{'PASS' if rule2_pass else 'FAIL'} | "
        "✅ CAPTION_FAILED残留数 < 总图片数的10%"
    )
    print(
        f"{'PASS' if rule3_pass else 'FAIL'} | "
        "✅ 四本final JSON chunk总数与normalized JSON一致"
    )


if __name__ == "__main__":
    main()
