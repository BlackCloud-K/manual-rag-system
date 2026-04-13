"""
Dependency:
pip install pymupdf pyyaml
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import fitz
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "documents"
OUTPUT_DIR = Path(__file__).resolve().parent / "temp_output"
CONFIG_PATH = ROOT_DIR / "config.yaml"


def normalize_manual_name(pdf_path: Path) -> str:
    # Keep manual key stable by using the filename stem directly.
    return pdf_path.stem


def is_bold(flags: int) -> bool:
    return bool(flags & (2**4))


def truncate_text(text: str, max_len: int = 40) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def analyze_pdf(pdf_path: Path) -> tuple[list[str], dict[str, Any]]:
    size_counts: dict[float, int] = defaultdict(int)
    size_values_for_median: list[float] = []
    size_samples: dict[float, list[tuple[str, bool]]] = defaultdict(list)
    total_image_blocks = 0

    with fitz.open(pdf_path) as doc:
        for page in doc:
            total_image_blocks += len(page.get_images())
            text_dict: dict[str, Any] = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        if not text:
                            continue
                        size = round(float(span.get("size", 0.0)), 2)
                        flags = int(span.get("flags", 0))
                        size_counts[size] += 1
                        size_values_for_median.append(size)
                        if len(size_samples[size]) < 5:
                            size_samples[size].append((truncate_text(text), is_bold(flags)))

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"PDF: {pdf_path.name}")
    lines.append("=" * 88)

    if not size_values_for_median:
        lines.append("No text spans found.")
        lines.append(f"Total image blocks (sum of page.get_images()): {total_image_blocks}")
        lines.append("")
        return lines, {
            "size_counts": dict(size_counts),
            "size_samples": dict(size_samples),
            "median_size": None,
            "total_image_blocks": total_image_blocks,
        }

    med_size = median(size_values_for_median)
    sorted_sizes_desc = sorted(size_counts.keys(), reverse=True)

    lines.append(f"Median font size: {med_size:.2f}")
    lines.append("")
    lines.append("All font sizes (desc) with span counts:")
    for size in sorted_sizes_desc:
        lines.append(f"  - {size:>6.2f}: {size_counts[size]} spans")

    lines.append("")
    lines.append("Samples for sizes > median:")
    larger_sizes = [s for s in sorted_sizes_desc if s > med_size]
    if not larger_sizes:
        lines.append("  (none)")
    else:
        for size in larger_sizes:
            lines.append(f"  Font size {size:.2f} (first up to 5 spans):")
            for idx, (sample_text, bold) in enumerate(size_samples[size], start=1):
                lines.append(f"    {idx}. bold={bold} | {sample_text}")

    lines.append("")
    lines.append(f"Total image blocks (sum of page.get_images()): {total_image_blocks}")
    lines.append("")
    return lines, {
        "size_counts": dict(size_counts),
        "size_samples": dict(size_samples),
        "median_size": med_size,
        "total_image_blocks": total_image_blocks,
    }


def infer_heading_levels(
    size_counts: dict[float, int], size_samples: dict[float, list[tuple[str, bool]]]
) -> dict[str, Any] | None:
    eligible_sizes = {size: cnt for size, cnt in size_counts.items() if cnt >= 10}
    if not eligible_sizes:
        return None

    body_size = max(eligible_sizes.items(), key=lambda item: item[1])[0]
    larger_sizes = sorted([s for s in eligible_sizes if s > body_size], reverse=True)
    heading_sizes = larger_sizes[:3]

    headings: dict[str, dict[str, Any]] = {}
    for idx, h_size in enumerate(heading_sizes, start=1):
        level = f"h{idx}"
        samples = [sample[0] for sample in size_samples.get(h_size, [])[:3]]
        headings[level] = {"size": h_size, "samples": samples}

    return {"body_size": body_size, "headings": headings}


def prompt_user_choice(pdf_name: str, inferred: dict[str, Any]) -> str:
    print(f"=== {pdf_name} 推断结果 ===")
    print(f"正文基准字号: {inferred['body_size']:.2f}pt")
    for level in ("h1", "h2", "h3"):
        if level not in inferred["headings"]:
            continue
        h_size = inferred["headings"][level]["size"]
        samples = inferred["headings"][level]["samples"]
        sample_text = " / ".join(samples) if samples else "(无样本)"
        print(f"{level.upper()} ({h_size:.2f}pt): {sample_text}")
    print()
    print("是否接受此推断？[y] 接受 / [e] 手动编辑 / [s] 跳过此PDF")

    while True:
        choice = input("请输入 y/e/s: ").strip().lower()
        if choice in {"y", "e", "s"}:
            return choice
        print("无效输入，请输入 y、e 或 s。")


def prompt_manual_edit(inferred: dict[str, Any]) -> dict[str, Any]:
    edited = {
        "body_size": inferred["body_size"],
        "headings": {k: dict(v) for k, v in inferred["headings"].items()},
    }

    raw = input(
        f"正文基准字号 body_size [{edited['body_size']:.2f}]，直接回车保留: "
    ).strip()
    if raw:
        try:
            edited["body_size"] = float(raw)
        except ValueError:
            print("输入无效，已保留原值。")

    for level in ("h1", "h2", "h3"):
        if level not in edited["headings"]:
            continue
        current = edited["headings"][level]["size"]
        raw = input(f"{level.upper()} 字号 [{current:.2f}]，直接回车保留: ").strip()
        if not raw:
            continue
        try:
            edited["headings"][level]["size"] = float(raw)
        except ValueError:
            print(f"{level.upper()} 输入无效，已保留原值。")

    return edited


def make_config_entry(confirmed: dict[str, Any]) -> dict[str, float]:
    body_size = float(confirmed["body_size"])
    entry: dict[str, float] = {"body_size": round(body_size, 2)}
    for level in ("h1", "h2", "h3"):
        if level in confirmed["headings"]:
            size_val = float(confirmed["headings"][level]["size"])
            entry[f"{level}_size_min"] = round(size_val - 1.0, 2)
    return entry


def merge_and_write_config(manual_entries: dict[str, dict[str, float]]) -> None:
    existing: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded

    yamlmanuals = existing.get("yamlmanuals")
    if not isinstance(yamlmanuals, dict):
        yamlmanuals = {}
    yamlmanuals.update(manual_entries)

    non_default_entries = [v for k, v in yamlmanuals.items() if k != "default"]
    if non_default_entries:
        default_entry: dict[str, float] = {
            "body_size": round(
                float(median([float(item["body_size"]) for item in non_default_entries])), 2
            )
        }
        for level in ("h1", "h2", "h3"):
            key = f"{level}_size_min"
            values = [float(item[key]) for item in non_default_entries if key in item]
            if values:
                default_entry[key] = round(float(median(values)), 2)
        yamlmanuals["default"] = default_entry

    existing["yamlmanuals"] = yamlmanuals
    CONFIG_PATH.write_text(
        yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"Documents directory not found: {DOCS_DIR}")
        return

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {DOCS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    confirmed_entries: dict[str, dict[str, float]] = {}

    for pdf_file in pdf_files:
        report_lines, stats = analyze_pdf(pdf_file)
        output_path = OUTPUT_DIR / f"{pdf_file.stem}_font_diagnostic.txt"
        output_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Wrote report: {output_path}")

        inferred = infer_heading_levels(stats["size_counts"], stats["size_samples"])
        if not inferred:
            print(f"=== {pdf_file.name} 推断结果 ===")
            print("可用字号不足（过滤后为空），跳过写入 config。")
            continue

        choice = prompt_user_choice(pdf_file.name, inferred)
        if choice == "s":
            print(f"已跳过: {pdf_file.name}")
            continue
        if choice == "e":
            inferred = prompt_manual_edit(inferred)

        manual_key = normalize_manual_name(pdf_file)
        confirmed_entries[manual_key] = make_config_entry(inferred)

    if confirmed_entries:
        merge_and_write_config(confirmed_entries)
        print(f"Config updated: {CONFIG_PATH}")
    else:
        print("No manuals confirmed. config.yaml was not modified.")


if __name__ == "__main__":
    main()
