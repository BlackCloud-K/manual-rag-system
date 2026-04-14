"""Evaluate multiple split strategies for large chunks (>800 tokens)."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "split_strategy_diagnostic_report.txt"

INPUT_FILES = [
    ROOT / "data" / "chunks_f16c.json",
    ROOT / "data" / "chunks_dcsjf17.json",
    ROOT / "data" / "chunks_f15c.json",
    ROOT / "data" / "chunks_dcsfa18cearlyaccessguidecn.json",
]

STEP_RE = re.compile(r"^\s*(?:\d+[\.、]|步骤\s*\d+)", re.MULTILINE)
TITLE_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十0-9]+[章节部分]|[0-9]+(?:\.[0-9]+){1,3})"
)
LIST_RE = re.compile(r"^\s*(?:[-•●]|[（(]?[0-9一二三四五六七八九十]+[)）])")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+")


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _nonempty(parts: list[str]) -> list[str]:
    return [p.strip() for p in parts if p and p.strip()]


def split_by_double_newline(text: str) -> list[str]:
    return _nonempty(text.split("\n\n"))


def split_by_single_line(text: str) -> list[str]:
    return _nonempty(text.split("\n"))


def split_by_step_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    out: list[str] = []
    current: list[str] = []
    for line in lines:
        if STEP_RE.match(line) and current:
            out.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        out.append("\n".join(current).strip())
    return _nonempty(out)


def split_by_title_and_list_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    out: list[str] = []
    current: list[str] = []
    for line in lines:
        if (TITLE_RE.match(line) or LIST_RE.match(line)) and current:
            out.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        out.append("\n".join(current).strip())
    return _nonempty(out)


def split_by_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
    return _nonempty(parts)


def _chunk_title(item: dict) -> str:
    return _safe_str(item.get("title_h3") or item.get("title_h2") or item.get("title_h1") or "-")


def _method_stats(tokenizer: AutoTokenizer, segments: list[str]) -> dict[str, float]:
    if not segments:
        return {"count": 0, "max_tokens": 0, "mean_tokens": 0.0, "effective": 0.0}
    lens = [_token_count(tokenizer, s) for s in segments]
    return {
        "count": float(len(lens)),
        "max_tokens": float(max(lens)),
        "mean_tokens": float(statistics.mean(lens)),
        "effective": 1.0 if max(lens) < 800 else 0.0,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer.model_max_length = 10**9

    methods = {
        "double_newline": split_by_double_newline,
        "single_line": split_by_single_line,
        "step_blocks": split_by_step_blocks,
        "title_list_blocks": split_by_title_and_list_blocks,
        "sentences": split_by_sentences,
    }

    lines: list[str] = []
    overall_totals = {k: {"chunks": 0, "effective": 0, "max_list": []} for k in methods}

    for path in INPUT_FILES:
        lines.append("=" * 88)
        lines.append(f"手册: {path.name}")
        lines.append("=" * 88)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            lines.append("ERROR: 非 list，跳过")
            lines.append("")
            continue

        manual_totals = {k: {"chunks": 0, "effective": 0, "max_list": []} for k in methods}
        large_items = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = _safe_str(item.get("text"))
            total = _token_count(tokenizer, text)
            if total > 800:
                large_items.append((item, text, total))

        lines.append(f"大Chunk数量: {len(large_items)}")
        lines.append("")

        for item, text, total in large_items:
            cid = _safe_str(item.get("chunk_id") or "?")
            title = _chunk_title(item)
            lines.append(f"[{cid}] {title} | {total} tokens")
            for name, fn in methods.items():
                segments = fn(text)
                st = _method_stats(tokenizer, segments)
                eff = "是" if st["effective"] > 0 else "否"
                lines.append(
                    f"  - {name}: 段数={int(st['count'])}, "
                    f"max={int(st['max_tokens'])}, mean={st['mean_tokens']:.1f}, "
                    f"可压到<800={eff}"
                )
                manual_totals[name]["chunks"] += 1
                manual_totals[name]["effective"] += int(st["effective"])
                manual_totals[name]["max_list"].append(st["max_tokens"])
                overall_totals[name]["chunks"] += 1
                overall_totals[name]["effective"] += int(st["effective"])
                overall_totals[name]["max_list"].append(st["max_tokens"])
            lines.append("")

        lines.append("本手册方法汇总:")
        for name in methods:
            total_chunks = manual_totals[name]["chunks"]
            eff_chunks = manual_totals[name]["effective"]
            ratio = (eff_chunks / total_chunks * 100.0) if total_chunks else 0.0
            avg_max = (
                statistics.mean(manual_totals[name]["max_list"])
                if manual_totals[name]["max_list"]
                else 0.0
            )
            lines.append(
                f"- {name}: 有效={eff_chunks}/{total_chunks} ({ratio:.2f}%), "
                f"平均最大子段token={avg_max:.1f}"
            )
        lines.append("")

    lines.append("=" * 88)
    lines.append("四本手册总体方法汇总")
    lines.append("=" * 88)
    ranking: list[tuple[str, float, float]] = []
    for name in methods:
        total_chunks = overall_totals[name]["chunks"]
        eff_chunks = overall_totals[name]["effective"]
        ratio = (eff_chunks / total_chunks * 100.0) if total_chunks else 0.0
        avg_max = (
            statistics.mean(overall_totals[name]["max_list"])
            if overall_totals[name]["max_list"]
            else 0.0
        )
        ranking.append((name, ratio, avg_max))
        lines.append(
            f"- {name}: 有效={eff_chunks}/{total_chunks} ({ratio:.2f}%), "
            f"平均最大子段token={avg_max:.1f}"
        )
    lines.append("")
    ranking.sort(key=lambda x: (-x[1], x[2]))
    lines.append("推荐顺序（按有效率优先，其次平均最大子段更小）:")
    for idx, (name, ratio, avg_max) in enumerate(ranking, start=1):
        lines.append(f"{idx}. {name} | 有效率={ratio:.2f}% | 平均最大子段={avg_max:.1f}")
    lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
