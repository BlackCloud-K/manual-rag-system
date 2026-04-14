"""Diagnose internal structure of large chunks (>800 tokens)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "large_chunk_diagnostic_report.txt"

INPUT_FILES = [
    ROOT / "data" / "chunks_f16c.json",
    ROOT / "data" / "chunks_dcsjf17.json",
    ROOT / "data" / "chunks_f15c.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn.json",
]

STEP_RE = re.compile(r"^\s*(?:\d+[\.、]|步骤\s*\d+)", re.MULTILINE)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _preview(text: str, limit: int = 40) -> str:
    return text.replace("\n", " ").strip()[:limit]


def _chunk_title(chunk: dict) -> str:
    h3 = _safe_str(chunk.get("title_h3"))
    h2 = _safe_str(chunk.get("title_h2"))
    h1 = _safe_str(chunk.get("title_h1"))
    return h3 or h2 or h1 or "-"


def _token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _paragraphs_from_text(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _step_examples(text: str, limit: int = 5) -> list[str]:
    examples: list[str] = []
    for line in text.splitlines():
        if STEP_RE.match(line):
            examples.append(line.strip())
            if len(examples) >= limit:
                break
    return examples


def _analyze_manual(path: Path, tokenizer: AutoTokenizer) -> tuple[list[str], int, int]:
    lines: list[str] = []
    large_total = 0
    splittable = 0

    lines.append("=" * 88)
    lines.append(f"手册: {path.name}")
    lines.append("=" * 88)

    if not path.exists():
        lines.append(f"ERROR: 文件不存在: {path}")
        lines.append("")
        return lines, large_total, splittable

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        lines.append("ERROR: JSON 根节点不是 list，跳过")
        lines.append("")
        return lines, large_total, splittable

    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _safe_str(item.get("text"))
        total_tokens = _token_count(tokenizer, text)
        if total_tokens <= 800:
            continue

        large_total += 1
        chunk_id = _safe_str(item.get("chunk_id")) or "?"
        title = _chunk_title(item)
        lines.append(f"[{chunk_id}] {title} | {total_tokens} tokens")

        paragraphs = _paragraphs_from_text(text)
        lines.append(f"  \\n\\n段落数: {len(paragraphs)}")

        para_token_counts: list[int] = []
        for idx, para in enumerate(paragraphs, start=1):
            para_tokens = _token_count(tokenizer, para)
            para_token_counts.append(para_tokens)
            lines.append(
                f'  段落{idx}: {para_tokens} tokens | "{_preview(para, 40)}"'
            )

        line_count = len(text.splitlines())
        lines.append(f"  \\n行数: {line_count}")

        examples = _step_examples(text, limit=5)
        if examples:
            shown = ", ".join(f'"{e}"' for e in examples)
            lines.append(f"  含步骤编号: 是 | 示例: [{shown}]")
        else:
            lines.append("  含步骤编号: 否 | 示例: []")

        can_split = bool(para_token_counts) and max(para_token_counts) < 800
        if can_split:
            splittable += 1
        lines.append(
            f"  \\n\\n可有效切分(最大子段<800): {'是' if can_split else '否'}"
        )
        lines.append("")

    ratio = (splittable / large_total * 100.0) if large_total else 0.0
    lines.append(
        f"本手册大Chunk数量: {large_total} | "
        f"\\n\\n可有效切分数量: {splittable} ({ratio:.2f}%)"
    )
    lines.append("")
    return lines, large_total, splittable


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    # 诊断脚本只做分词计数，不走模型前向，超长文本无需触发长度告警。
    tokenizer.model_max_length = 10**9

    all_lines: list[str] = []
    grand_large = 0
    grand_splittable = 0

    for path in INPUT_FILES:
        lines, large_total, splittable = _analyze_manual(path, tokenizer)
        all_lines.extend(lines)
        grand_large += large_total
        grand_splittable += splittable

    all_lines.append("=" * 88)
    all_lines.append("四本手册汇总")
    all_lines.append("=" * 88)
    ratio = (grand_splittable / grand_large * 100.0) if grand_large else 0.0
    all_lines.append(f"总大Chunk数量: {grand_large}")
    all_lines.append(
        f"\\n\\n可有效切分数量: {grand_splittable} ({ratio:.2f}%)"
    )
    all_lines.append("")

    text = "\n".join(all_lines).rstrip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
