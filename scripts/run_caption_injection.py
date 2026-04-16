"""
执行顺序说明：
1) 先运行 scripts/run_vlm_captioning.py（TEST_MODE=True）生成前20张摘要缓存。
2) 人工检查输出质量后，将 run_vlm_captioning.py 的 TEST_MODE 改为 False 跑全量。
3) 全量完成后再运行本脚本，生成 chunks_*_final.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CAPTIONS_CACHE = ROOT / "data" / "captions_cache.json"
CHUNK_FILES = [
    ROOT / "data" / "chunks_f16c_normalized.json",
    ROOT / "data" / "chunks_dcsjf17_normalized.json",
    ROOT / "data" / "chunks_f15c_normalized.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn_normalized.json",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, content: Any) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_rel_path(path_str: str) -> str:
    normalized = path_str.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _extract_image_paths(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(x).strip() for x in raw_value if isinstance(x, str) and str(x).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def _inject_text(original_text: str, captions: list[str]) -> str:
    text = original_text if isinstance(original_text, str) else str(original_text or "")
    suffix = "".join(f"\n[IMG_CAPTION] {caption}" for caption in captions)
    return text + suffix


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not CAPTIONS_CACHE.exists():
        print(f"[ERROR] 缓存文件不存在: {CAPTIONS_CACHE}")
        return

    raw_cache = _load_json(CAPTIONS_CACHE)
    if not isinstance(raw_cache, dict):
        print(f"[ERROR] 缓存格式错误（应为字典）: {CAPTIONS_CACHE}")
        return
    cache = {str(k): str(v) for k, v in raw_cache.items()}

    for chunk_file in CHUNK_FILES:
        raw_chunks = _load_json(chunk_file)
        if not isinstance(raw_chunks, list):
            print(f"[WARN] 文件不是 list，跳过: {chunk_file}")
            continue

        injected_chunk_count = 0
        skipped_path_count = 0

        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue

            image_paths = _extract_image_paths(chunk.get("image_paths"))
            if not image_paths:
                continue

            captions: list[str] = []
            for image_path in image_paths:
                rel_path = _normalize_rel_path(image_path)
                caption = cache.get(rel_path)
                if caption is None:
                    skipped_path_count += 1
                    print(f"[WARN] 缓存中不存在图片路径，已跳过: {rel_path}")
                    continue
                if caption == "CAPTION_FAILED":
                    continue
                captions.append(caption)

            if captions:
                chunk["text"] = _inject_text(chunk.get("text", ""), captions)
                injected_chunk_count += 1

        output_name = chunk_file.name.replace("_normalized.json", "_final.json")
        output_path = chunk_file.with_name(output_name)
        _save_json(output_path, raw_chunks)

        print(f"\n=== {chunk_file.name} ===")
        print(f"注入摘要的Chunk数: {injected_chunk_count}")
        print(f"跳过的图片路径数: {skipped_path_count}")
        print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
