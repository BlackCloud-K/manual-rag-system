"""Image statistics diagnostics for four manual image folders."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data" / "images"
REPORT_PATH = ROOT / "scripts" / "temp_output" / "image_stats_diagnostic_report.txt"

MANUALS = [
    "f16c",
    "dcsjf17",
    "f15c",
    "dcsfa18cearlyaccessguidecn",
]

CHUNKS_NORMALIZED_FILES = [
    ROOT / "data" / "chunks_f16c_normalized.json",
    ROOT / "data" / "chunks_dcsjf17_normalized.json",
    ROOT / "data" / "chunks_f15c_normalized.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn_normalized.json",
]

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TOKEN_PER_IMAGE = 300
USD_PER_MILLION_IMAGE_TOKENS = 0.15


@dataclass
class ImageRecord:
    path: Path
    size_bytes: int
    width: int
    height: int
    sha256: str


def _fmt_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _manual_title(manual: str) -> str:
    return manual.upper()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_images_for_manual(manual: str) -> tuple[list[ImageRecord], list[str]]:
    manual_dir = IMAGE_ROOT / manual
    records: list[ImageRecord] = []
    errors: list[str] = []

    if not manual_dir.exists():
        errors.append(f"目录不存在: {manual_dir}")
        return records, errors

    for file_path in manual_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            size_bytes = file_path.stat().st_size
            with Image.open(file_path) as img:
                width, height = img.size
            sha256 = _hash_file(file_path)
            records.append(
                ImageRecord(
                    path=file_path,
                    size_bytes=size_bytes,
                    width=width,
                    height=height,
                    sha256=sha256,
                )
            )
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    return records, errors


def _distribution(records: list[ImageRecord]) -> tuple[int, int, int]:
    lt_5kb = 0
    between_5_50kb = 0
    gt_50kb = 0
    for record in records:
        if record.size_bytes < 5 * 1024:
            lt_5kb += 1
        elif record.size_bytes <= 50 * 1024:
            between_5_50kb += 1
        else:
            gt_50kb += 1
    return lt_5kb, between_5_50kb, gt_50kb


def _small_dimension_count(records: list[ImageRecord]) -> int:
    return sum(1 for r in records if r.width < 50 or r.height < 50)


def _normalize_image_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lower()


def _extract_image_path_values(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        result: list[str] = []
        for entry in raw_value:
            if isinstance(entry, str) and entry.strip():
                result.append(entry)
        return result
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value]
    return []


def _collect_json_reference_stats() -> tuple[int, int, list[str]]:
    chunks_with_images = 0
    unique_paths: set[str] = set()
    errors: list[str] = []

    for json_path in CHUNKS_NORMALIZED_FILES:
        if not json_path.exists():
            errors.append(f"文件不存在: {json_path}")
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                errors.append(f"根节点不是 list: {json_path}")
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                image_paths = _extract_image_path_values(item.get("image_paths"))
                if not image_paths:
                    continue
                chunks_with_images += 1
                for p in image_paths:
                    unique_paths.add(_normalize_image_path(p))
        except Exception as exc:
            errors.append(f"{json_path}: {exc}")

    return chunks_with_images, len(unique_paths), errors


def _estimate_cost_usd(unique_physical_images: int) -> tuple[int, float]:
    total_tokens = unique_physical_images * TOKEN_PER_IMAGE
    usd = (total_tokens / 1_000_000) * USD_PER_MILLION_IMAGE_TOKENS
    return total_tokens, usd


def _build_report() -> str:
    lines: list[str] = []
    all_records: list[ImageRecord] = []
    scan_errors: list[str] = []

    for manual in MANUALS:
        records, errors = _collect_images_for_manual(manual)
        all_records.extend(records)
        scan_errors.extend(errors)

        lt_5kb, between_5_50kb, gt_50kb = _distribution(records)
        small_px = _small_dimension_count(records)
        total_size = sum(r.size_bytes for r in records)

        lines.append(f"=== {_manual_title(manual)} ===")
        lines.append(f"图片总数: {len(records)} 张")
        lines.append(f"总大小: {_fmt_mb(total_size)}")
        lines.append(
            "大小分布: "
            f"<5KB: {lt_5kb}张 | 5-50KB: {between_5_50kb}张 | >50KB: {gt_50kb}张"
        )
        lines.append(f"尺寸<50px: {small_px}张")
        lines.append("")

    unique_hashes = {r.sha256 for r in all_records}
    physical_total_dedup = len(unique_hashes)

    chunks_with_images, json_unique_images, json_errors = _collect_json_reference_stats()
    total_tokens, estimated_usd = _estimate_cost_usd(physical_total_dedup)

    lines.append("=== 合计 ===")
    lines.append(f"物理图片总数: {physical_total_dedup} 张")
    lines.append(f"JSON引用Chunk数（image_paths非空）: {chunks_with_images} 个")
    lines.append(f"JSON引用图片数（去重）: {json_unique_images} 张")
    lines.append(
        f"估算API成本: ${estimated_usd:.6f} USD（按{TOKEN_PER_IMAGE} tokens/张, 共 {total_tokens} tokens）"
    )

    if scan_errors or json_errors:
        lines.append("")
        lines.append("=== 警告/错误 ===")
        for message in scan_errors + json_errors:
            lines.append(f"- {message}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    report = _build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\n已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
