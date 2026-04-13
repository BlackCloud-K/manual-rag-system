"""Validate all data/chunks_*.json and write combined report to temp_output."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = ROOT / "scripts" / "temp_output" / "checkpoint1_validate.txt"

REQUIRED_FIELDS = (
    "chunk_id",
    "pdf_name",
    "title_h1",
    "page_start",
    "page_end",
    "text",
    "image_paths",
    "prev_chunk_id",
    "next_chunk_id",
)


def _stem_from_chunks_filename(chunks_path: Path) -> str:
    name = chunks_path.name
    if name.startswith("chunks_") and name.endswith(".json"):
        return name[len("chunks_") : -len(".json")]
    return chunks_path.stem


def _lines_for(chunks_path: Path, images_dir: Path) -> list[str]:
    out: list[str] = []

    if not chunks_path.exists():
        out.append(f"ERROR: chunks file not found: {chunks_path}")
        return out

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        out.append("ERROR: JSON root must be a list of chunks")
        return out
    chunks: list[dict] = raw
    total = len(chunks)
    out.append(f"文件: {chunks_path}")
    out.append(f"图片目录: {images_dir}")
    out.append(f"chunk 总数: {total}")
    out.append("")

    # --- 结构完整性 ---
    missing_any = 0
    for c in chunks:
        for k in REQUIRED_FIELDS:
            if k not in c:
                missing_any += 1
                break
    if missing_any == 0:
        out.append("✅ 结构完整性：通过")
    else:
        out.append(f"❌ 结构完整性：失败，缺失必填字段的 chunk 数: {missing_any}")
    out.append("")

    # --- text 非空 ---
    empty_text_ids: list[str] = []
    for c in chunks:
        t = c.get("text", "")
        if not isinstance(t, str) or not t.strip():
            empty_text_ids.append(str(c.get("chunk_id", "?")))
    if not empty_text_ids:
        out.append("✅ text非空率：通过")
    else:
        n = len(empty_text_ids)
        out.append(f"❌ text非空率：失败，共{n}个空chunk")
        for cid in empty_text_ids:
            out.append(f"  - {cid}")
    out.append("")

    # --- 链表 ---
    by_id = {c["chunk_id"]: c for c in chunks if "chunk_id" in c}
    broken = 0
    for c in chunks:
        cid = c.get("chunk_id")
        nxt = c.get("next_chunk_id")
        if nxt is not None:
            if nxt not in by_id or by_id[nxt].get("prev_chunk_id") != cid:
                broken += 1
        prv = c.get("prev_chunk_id")
        if prv is not None:
            if prv not in by_id or by_id[prv].get("next_chunk_id") != cid:
                broken += 1

    heads = [c for c in chunks if c.get("prev_chunk_id") is None]
    tails = [c for c in chunks if c.get("next_chunk_id") is None]
    if len(heads) != 1:
        broken += 1
    if len(tails) != 1:
        broken += 1

    if broken == 0:
        out.append("✅ 链表完整性：通过")
    else:
        out.append(
            f"❌ 链表完整性：失败，断链数量（含 next/prev 双向检查与头尾端点）: {broken}"
        )
        out.append(f"  prev 为 null 的 chunk 数: {len(heads)} (期望 1)")
        out.append(f"  next 为 null 的 chunk 数: {len(tails)} (期望 1)")
    out.append("")

    # --- 图片归属覆盖率（按磁盘文件）---
    referenced: set[str] = set()
    for c in chunks:
        paths = c.get("image_paths")
        if not isinstance(paths, list):
            continue
        for p in paths:
            if isinstance(p, str) and p.strip():
                referenced.add(Path(p).as_posix())

    total_paths_in_json = sum(
        len(c["image_paths"])
        for c in chunks
        if isinstance(c.get("image_paths"), list)
    )
    out.append(f"所有 chunk 的 image_paths 条目数（可重复引用）: {total_paths_in_json}")

    if not images_dir.is_dir():
        out.append(f"图片目录不存在: {images_dir}（跳过磁盘覆盖率）")
        out.append("")
    else:
        on_disk: set[str] = set()
        for f in sorted(images_dir.glob("*.png")):
            rel = f.relative_to(ROOT).as_posix()
            on_disk.add(rel)

        n_disk = len(on_disk)
        assigned = {p for p in on_disk if p in referenced}
        n_assigned = len(assigned)
        orphans = sorted(on_disk - referenced)
        missing_on_disk = sorted(referenced - on_disk)

        ratio = n_assigned / n_disk if n_disk else 1.0
        out.append(f"磁盘上 PNG 张数: {n_disk}")
        out.append(
            f"已被至少一个 chunk 引用的张数: {n_assigned}  "
            f"覆盖率: {ratio:.4f} ({n_assigned}/{n_disk})"
        )
        if orphans:
            out.append(f"未被任何 chunk 引用的图（共 {len(orphans)} 张）:")
            for p in orphans:
                out.append(f"  - {p}")
        else:
            out.append("未被任何 chunk 引用的图: 无")
        if missing_on_disk:
            out.append(
                f"chunk 中列出但磁盘上缺失的文件（共 {len(missing_on_disk)} 个）:"
            )
            for p in missing_on_disk:
                out.append(f"  - {p}")
        if n_disk and ratio > 0.95:
            out.append("✅ 图片归属覆盖率：通过（>95% 磁盘图已被引用）")
        elif n_disk:
            out.append("❌ 图片归属覆盖率：未达 95%（按磁盘图被引用比例）")
        else:
            out.append("（磁盘无 PNG，不判定覆盖率）")
    out.append("")

    # --- 人工抽查（random.seed(42) 每个文件独立可复现）---
    random.seed(42)
    out.append("=" * 72)
    out.append("人工抽查：随机 10 个 chunk")
    out.append("=" * 72)
    sample10 = random.sample(chunks, min(10, len(chunks)))
    for i, c in enumerate(sample10, 1):
        out.append(f"[{i}] chunk_id: {c.get('chunk_id')}")
        out.append(f"    title_h1: {c.get('title_h1', '')}")
        out.append(f"    title_h2: {c.get('title_h2', '')}")
        out.append(f"    title_h3: {c.get('title_h3', '')}")
        out.append(f"    page_start: {c.get('page_start')}  page_end: {c.get('page_end')}")
        tx = c.get("text") or ""
        if not isinstance(tx, str):
            tx = str(tx)
        preview = tx.replace("\n", " ")[:150]
        out.append(f"    text 前150字: {preview}")
        out.append(f"    image_paths: {c.get('image_paths', [])}")
        out.append("")

    random.seed(42)
    out.append("=" * 72)
    out.append("人工抽查：随机 5 个 image_paths 非空的 chunk")
    out.append("=" * 72)
    with_img = [
        c
        for c in chunks
        if isinstance(c.get("image_paths"), list) and len(c["image_paths"]) > 0
    ]
    sample5 = random.sample(with_img, min(5, len(with_img)))
    for i, c in enumerate(sample5, 1):
        out.append(f"[{i}] chunk_id: {c.get('chunk_id')}")
        out.append(f"    title_h1: {c.get('title_h1', '')}")
        out.append(f"    title_h2: {c.get('title_h2', '')}")
        out.append(f"    title_h3: {c.get('title_h3', '')}")
        out.append(f"    page_start: {c.get('page_start')}")
        out.append(f"    image_paths: {c.get('image_paths', [])}")
        tx = c.get("text") or ""
        if not isinstance(tx, str):
            tx = str(tx)
        preview = tx.replace("\n", " ")[:100]
        out.append(f"    text 前100字: {preview}")
        out.append("")

    out.append("=" * 72)
    out.append('已知章节：title_h1 包含 "驾驶舱仪表" 的 chunk')
    out.append("=" * 72)
    cockpit = [c for c in chunks if "驾驶舱仪表" in str(c.get("title_h1", ""))]
    if not cockpit:
        out.append("(无匹配)")
    else:
        for c in cockpit:
            out.append(
                f"chunk_id={c.get('chunk_id')}  page_start={c.get('page_start')}  "
                f"page_end={c.get('page_end')}  title_h2={c.get('title_h2', '')!r}  "
                f"title_h3={c.get('title_h3', '')!r}"
            )

    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    chunk_files = sorted(DATA_DIR.glob("chunks_*.json"))
    if not chunk_files:
        msg = f"No chunk JSON files under {DATA_DIR} (chunks_*.json)\n"
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(msg, encoding="utf-8")
        print(msg)
        print(f"已写入: {OUTPUT_PATH}")
        return

    all_lines: list[str] = []
    all_lines.append(f"共 {len(chunk_files)} 个 chunks_*.json\n")

    for idx, chunks_path in enumerate(chunk_files):
        stem = _stem_from_chunks_filename(chunks_path)
        images_dir = ROOT / "data" / "images" / stem
        all_lines.append("#" * 72)
        all_lines.append(f"[{idx + 1}/{len(chunk_files)}] stem={stem}")
        all_lines.append("#" * 72)
        all_lines.extend(_lines_for(chunks_path, images_dir))
        all_lines.append("")
        all_lines.append("")

    text = "\n".join(all_lines).rstrip() + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
