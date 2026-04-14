"""Token distribution diagnostics for selected chunk JSON files."""
from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "token_diagnostic_report.txt"

INPUT_FILES = [
    ROOT / "data" / "chunks_f16c.json",
    ROOT / "data" / "chunks_dcsjf17.json",
    ROOT / "data" / "chunks_f15c.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn.json",
]

RANGES = [
    ("< 100", lambda n: n < 100),
    ("100-200", lambda n: 100 <= n < 200),
    ("200-500 (舒适区)", lambda n: 200 <= n < 500),
    ("500-800", lambda n: 500 <= n <= 800),
    ("> 800", lambda n: n > 800),
]


@dataclass
class ChunkRow:
    chunk_id: str
    title_h1: str
    title_h2: str
    title_h3: str
    text: str
    tokens: int


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _preview(text: str, size: int) -> str:
    return text.replace("\n", " ").strip()[:size]


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_vals[lower])
    weight = rank - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def _fmt_ratio(count: int, total: int) -> str:
    pct = (count / total * 100.0) if total else 0.0
    return f"{count} ({pct:.2f}%)"


def _display_title(title_h2: str, title_h3: str) -> str:
    if title_h2 and title_h3:
        return f"{title_h2} / {title_h3}"
    return title_h2 or title_h3 or "-"


def _display_full_title(title_h1: str, title_h2: str, title_h3: str) -> str:
    parts = [p for p in (title_h1, title_h2, title_h3) if p]
    return " > ".join(parts) if parts else "-"


def _analyze_file(path: Path, tokenizer: AutoTokenizer) -> tuple[list[str], list[ChunkRow]]:
    lines: list[str] = []
    rows: list[ChunkRow] = []

    lines.append("=" * 80)
    lines.append(f"手册: {path.name}")
    lines.append("=" * 80)

    if not path.exists():
        lines.append(f"ERROR: 文件不存在: {path}")
        lines.append("")
        return lines, rows

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        lines.append("ERROR: JSON 根节点不是 list，跳过")
        lines.append("")
        return lines, rows

    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _safe_str(item.get("text"))
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        rows.append(
            ChunkRow(
                chunk_id=_safe_str(item.get("chunk_id", "?")),
                title_h1=_safe_str(item.get("title_h1", "")),
                title_h2=_safe_str(item.get("title_h2", "")),
                title_h3=_safe_str(item.get("title_h3", "")),
                text=text,
                tokens=token_count,
            )
        )

    token_values = [r.tokens for r in rows]
    total = len(rows)
    lines.append(f"Chunk 总数: {total}")

    if not token_values:
        lines.append("无可统计 chunk。")
        lines.append("")
        return lines, rows

    lines.append(
        "Token 统计: "
        f"min={min(token_values)}, "
        f"max={max(token_values)}, "
        f"median={statistics.median(token_values):.2f}, "
        f"mean={statistics.mean(token_values):.2f}, "
        f"p25={_percentile(token_values, 25):.2f}, "
        f"p75={_percentile(token_values, 75):.2f}"
    )
    lines.append("")
    lines.append("区间分布:")
    for label, rule in RANGES:
        count = sum(1 for n in token_values if rule(n))
        lines.append(f"- {label}: {_fmt_ratio(count, total)}")

    low_rows = [r for r in rows if r.tokens < 100]
    high_rows = [r for r in rows if r.tokens > 800]

    lines.append("")
    lines.append(f"< 100 token 的 chunk 明细 (共 {len(low_rows)} 条):")
    if low_rows:
        for r in low_rows:
            lines.append(
                f"- chunk_id={r.chunk_id} | 标题={_display_title(r.title_h2, r.title_h3)} "
                f"| tokens={r.tokens} | text[:80]={_preview(r.text, 80)!r}"
            )
    else:
        lines.append("- 无")

    lines.append("")
    lines.append(f"> 800 token 的 chunk 明细 (共 {len(high_rows)} 条):")
    if high_rows:
        for r in high_rows:
            lines.append(
                f"- chunk_id={r.chunk_id} | 标题={_display_full_title(r.title_h1, r.title_h2, r.title_h3)} "
                f"| tokens={r.tokens} | text[:120]={_preview(r.text, 120)!r}"
            )
    else:
        lines.append("- 无")

    lines.append("")
    return lines, rows


def _overall_lines(rows: list[ChunkRow]) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("四本手册合并汇总")
    lines.append("=" * 80)

    token_values = [r.tokens for r in rows]
    total = len(token_values)
    lines.append(f"Chunk 总数: {total}")
    if not token_values:
        lines.append("无可统计 chunk。")
        return lines

    lines.append(
        "Token 统计: "
        f"min={min(token_values)}, "
        f"max={max(token_values)}, "
        f"median={statistics.median(token_values):.2f}, "
        f"mean={statistics.mean(token_values):.2f}, "
        f"p25={_percentile(token_values, 25):.2f}, "
        f"p75={_percentile(token_values, 75):.2f}"
    )
    lines.append("")
    lines.append("区间分布:")
    for label, rule in RANGES:
        count = sum(1 for n in token_values if rule(n))
        lines.append(f"- {label}: {_fmt_ratio(count, total)}")
    lines.append("")
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    all_lines: list[str] = []
    all_rows: list[ChunkRow] = []

    for input_path in INPUT_FILES:
        file_lines, rows = _analyze_file(input_path, tokenizer)
        all_lines.extend(file_lines)
        all_rows.extend(rows)

    all_lines.extend(_overall_lines(all_rows))
    report_text = "\n".join(all_lines).rstrip() + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"\n已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
