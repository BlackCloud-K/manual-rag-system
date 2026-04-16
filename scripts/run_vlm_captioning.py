"""
执行顺序说明：
1) 先运行本脚本（TEST_MODE=True）仅生成测试图片集的摘要缓存。
2) 人工检查输出质量后，将 TEST_MODE 改为 False 跑全量。
3) 全量完成后运行 scripts/run_caption_injection.py 生成 final JSON。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.image_preprocessor import preprocess_image
from src.vlm.vlm_client import get_image_caption

TEST_MODE = False
TEST_LIMIT = 10
TEST_PER_MANUAL = 3
TEST_GROUP_TOTAL = 12
CONCURRENCY = 2

CACHE_PATH = ROOT / "data" / "captions_cache.json"
CHUNK_FILES = [
    ROOT / "data" / "chunks_f16c_normalized.json",
    ROOT / "data" / "chunks_dcsjf17_normalized.json",
    ROOT / "data" / "chunks_f15c_normalized.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn_normalized.json",
]

INPUT_TOKEN_PRICE_PER_M = 0.15
OUTPUT_TOKEN_PRICE_PER_M = 0.60


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def collect_unique_existing_image_paths() -> list[str]:
    unique: set[str] = set()
    missing_count = 0

    for chunk_file in CHUNK_FILES:
        raw = _load_json(chunk_file)
        if not isinstance(raw, list):
            continue

        for chunk in raw:
            if not isinstance(chunk, dict):
                continue
            for image_path in _extract_image_paths(chunk.get("image_paths")):
                rel_path = _normalize_rel_path(image_path)
                abs_path = ROOT / rel_path
                if abs_path.exists():
                    unique.add(rel_path)
                else:
                    missing_count += 1
                    print(f"[WARN] 图片不存在，已跳过: {rel_path}")

    paths = sorted(unique)
    if missing_count:
        print(f"[WARN] 共发现 {missing_count} 条缺失图片路径引用。")
    return paths


def _manual_from_image_path(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    # Expected format: data/images/<manual>/<filename>
    if len(parts) >= 3:
        return parts[2]
    return ""


def build_test_paths(paths: list[str]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for rel_path in paths:
        manual = _manual_from_image_path(rel_path)
        if not manual:
            continue
        grouped.setdefault(manual, []).append(rel_path)

    selected: list[str] = []
    for manual in ("f16c", "dcsjf17", "f15c", "dcsfa18cearlyaccessguidecn"):
        selected.extend(grouped.get(manual, [])[:TEST_PER_MANUAL])

    return selected[:TEST_GROUP_TOTAL]


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    raw = _load_json(CACHE_PATH)
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def estimate_output_tokens(caption: str) -> int:
    # Rough estimate when only total_tokens is returned by client helper.
    return max(1, len(caption))


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI()

    all_paths = collect_unique_existing_image_paths()
    cache = load_cache()

    uncached = [p for p in all_paths if p not in cache or cache[p] == "CAPTION_FAILED"]
    selected = build_test_paths(uncached) if TEST_MODE else uncached

    if TEST_MODE:
        print(
            "[INFO] TEST_MODE=True，按手册分组抽样测试集："
            f"每组前{TEST_PER_MANUAL}张，合并后最多{TEST_GROUP_TOTAL}张；"
            f"TEST_LIMIT={TEST_LIMIT}（仅保留配置，不用于当前分组抽样）。"
        )
        print(f"[INFO] 本次实际处理测试图片: {len(selected)} 张。")
    else:
        print(f"[INFO] TEST_MODE=False，处理全部未缓存图片，共 {len(selected)} 张。")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    cache_lock = asyncio.Lock()

    processed = 0
    skipped = len(all_paths) - len(uncached)
    failed = 0
    total_tokens = 0
    input_tokens_est = 0
    output_tokens_est = 0

    async def worker(rel_path: str) -> None:
        nonlocal processed, failed, total_tokens, input_tokens_est, output_tokens_est
        abs_path = ROOT / rel_path
        caption = "CAPTION_FAILED"
        tokens = 0

        async with semaphore:
            try:
                base64_image = preprocess_image(str(abs_path))
                caption, tokens = await get_image_caption(base64_image, client)
            except FileNotFoundError:
                print(f"[WARN] 文件不存在（处理中）: {rel_path}")
            except Exception as exc:
                print(f"[WARN] 处理失败: {rel_path} | {exc}")

        async with cache_lock:
            cache[rel_path] = caption
            save_cache(cache)

            processed += 1
            total_tokens += tokens
            if caption == "CAPTION_FAILED":
                failed += 1
            else:
                out_est = estimate_output_tokens(caption)
                in_est = max(0, tokens - out_est)
                output_tokens_est += out_est
                input_tokens_est += in_est

            print(
                f"[PROGRESS] {processed}/{len(selected)} | "
                f"{rel_path} | tokens={tokens} | caption={caption[:60]}"
            )

    await asyncio.gather(*(worker(path) for path in selected))

    cost_usd = (
        (input_tokens_est / 1_000_000) * INPUT_TOKEN_PRICE_PER_M
        + (output_tokens_est / 1_000_000) * OUTPUT_TOKEN_PRICE_PER_M
    )

    print("\n=== VLM Captioning Report ===")
    print(f"处理张数: {processed}")
    print(f"跳过张数（已缓存）: {skipped}")
    print(f"失败张数: {failed}")
    print(f"总token消耗: {total_tokens}")
    print(
        "估算费用: "
        f"${cost_usd:.6f} USD "
        f"(input≈{input_tokens_est}, output≈{output_tokens_est})"
    )
    print(f"缓存文件: {CACHE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
