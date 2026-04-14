from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz
import yaml

try:
    from src.parser.toc_parser import parse_toc
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.parser.toc_parser import parse_toc


def _normalize_pdf_stem(pdf_name: str) -> str:
    stem = Path(pdf_name).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", stem)
    return normalized or "manual"


def build_chunks(
    toc_list: list[dict[str, Any]], doc: fitz.Document, pdf_name: str
) -> list[dict[str, Any]]:
    if not toc_list:
        return []

    pdf_stem = _normalize_pdf_stem(pdf_name)
    total_pages = len(doc)
    chunks: list[dict[str, Any]] = []
    level_context: dict[int, str] = {}

    for idx, item in enumerate(toc_list, start=1):
        level = int(item.get("level", 1))
        title = str(item.get("title", "")).strip()
        page_start = int(item.get("page", 1))

        # Reset deeper levels when moving up.
        for lv in (1, 2, 3):
            if lv > level:
                level_context.pop(lv, None)
        level_context[level] = title

        title_h1 = level_context.get(1, "")
        title_h2 = level_context.get(2, "")
        title_h3 = level_context.get(3, "")
        if level == 1:
            title_h1, title_h2, title_h3 = title, "", ""
        elif level == 2:
            title_h2, title_h3 = title, ""
        elif level == 3:
            title_h3 = title

        # page_end = next entry's page (not minus 1), so the last page of this
        # chunk is shared with the next chunk's first page. fill_text will cut
        # at the next title's Y coordinate on that shared page.
        if idx < len(toc_list):
            next_page = int(toc_list[idx].get("page", total_pages))
            page_end = next_page
        else:
            page_end = total_pages

        page_end = max(page_start, min(page_end, total_pages))
        chunk_id = f"{pdf_stem}_{idx:04d}"
        prev_chunk_id = None if idx == 1 else f"{pdf_stem}_{idx - 1:04d}"
        next_chunk_id = None if idx == len(toc_list) else f"{pdf_stem}_{idx + 1:04d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "pdf_name": pdf_name,
                "title_h1": title_h1,
                "title_h2": title_h2,
                "title_h3": title_h3,
                "page_start": page_start,
                "page_end": page_end,
                "text": "",
                "image_paths": [],
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
            }
        )

    return chunks


def load_margins_from_config(config_path: Path, pdf_name: str) -> tuple[float, float]:
    default_top = 55.0
    # Strip this many points from the bottom edge (not an absolute y coordinate).
    default_bottom = 55.0
    if not config_path.exists():
        return default_top, default_bottom

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return default_top, default_bottom

    # FIX: was "yamlmanuals" (typo), now correctly "manuals"
    manuals = loaded.get("manuals")
    if not isinstance(manuals, dict):
        return default_top, default_bottom

    manual_key = Path(pdf_name).stem
    manual_entry = manuals.get(manual_key)
    if not isinstance(manual_entry, dict):
        manual_entry = manuals.get("default", {})
    if not isinstance(manual_entry, dict):
        return default_top, default_bottom

    margin_top = float(manual_entry.get("margin_top", default_top))
    margin_bottom = float(manual_entry.get("margin_bottom", default_bottom))
    return margin_top, margin_bottom


def fill_text(
    chunks: list[dict[str, Any]],
    doc: fitz.Document,
    pdf_name: str,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    margin_top, margin_bottom = load_margins_from_config(config_path, pdf_name)
    return _fill_text_precise(chunks, doc, margin_top=margin_top, margin_bottom=margin_bottom)


def _pick_chunk_title(chunk: dict[str, Any]) -> str:
    return (
        str(chunk.get("title_h3", "")).strip()
        or str(chunk.get("title_h2", "")).strip()
        or str(chunk.get("title_h1", "")).strip()
    )


def _normalize_for_title_match(text: str) -> str:
    """Strip all whitespace so TOC titles match PDF lines like 'F-15 在战场'."""
    return "".join(ch for ch in (text or "") if not ch.isspace())


def _find_title_substring_index(block_text: str, title: str) -> int:
    """First index in *block_text* where normalized remainder starts with *title*."""
    title_clean = (title or "").strip()
    nt = _normalize_for_title_match(title_clean)
    if not nt:
        return -1
    block_text = block_text or ""
    for i in range(len(block_text) + 1):
        suffix = block_text[i:]
        if _normalize_for_title_match(suffix).startswith(nt):
            return i
    return -1


def find_title_y(
    page: fitz.Page,
    title: str,
    margin_top: float,
    margin_bottom: float,
) -> float:
    """Return the y0 of the line containing *title* on *page*.

    Searches at span/line granularity so that a title buried inside a mixed
    block is located precisely. Falls back to margin_top if not found.
    """
    title_clean = (title or "").strip()
    if not title_clean:
        return margin_top

    norm_title = _normalize_for_title_match(title_clean)
    if not norm_title:
        return margin_top

    body_bottom = page.rect.height - margin_bottom
    matches: list[tuple[float, float]] = []  # (line_y0, max_font_size)

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_bbox = line.get("bbox", [0, 0, 0, 0])
            if len(line_bbox) < 4:
                continue
            line_y0 = float(line_bbox[1])
            line_y1 = float(line_bbox[3])
            if line_y0 < margin_top or line_y1 > body_bottom:
                continue

            line_text = "".join(str(s.get("text", "")) for s in line.get("spans", [])).strip()
            max_size = max(
                (float(s.get("size", 0.0)) for s in line.get("spans", [])),
                default=0.0,
            )
            if norm_title in _normalize_for_title_match(line_text):
                matches.append((line_y0, max_size))

    if not matches:
        return margin_top

    # Prefer largest font size; break ties by smallest y0 (topmost).
    matches.sort(key=lambda item: (-item[1], item[0]))
    return matches[0][0]


def _extract_text_clip_y(
    page: fitz.Page,
    y_lo: float,
    y_hi: float,
    margin_top: float,
    margin_bottom: float,
    title: str | None = None,
    *,
    exclusive_end: bool = False,
) -> str:
    """Extract text in horizontal strip [y_lo, y_hi), clamped to body margins.

    Uses ``page.get_text(..., clip=...)`` so nothing below *body_bottom* is
    included (blocks that span body+footer no longer pull the footer in).

    *exclusive_end*: when *y_hi* is the next heading line's top (``find_title_y``),
    shrink slightly so that line is not included.
    """
    body_bottom = page.rect.height - margin_bottom
    y_lo = max(float(y_lo), float(margin_top))
    y_hi = min(float(y_hi), float(body_bottom))
    if exclusive_end:
        y_hi = max(y_lo, y_hi - 0.5)
    if y_lo >= y_hi:
        return ""

    rect = fitz.Rect(0, y_lo, page.rect.width, y_hi)
    raw = page.get_text("text", clip=rect)
    text = str(raw).strip()
    if not text:
        return ""

    title_clean = (title or "").strip()
    if title_clean:
        idx = _find_title_substring_index(text, title_clean)
        if idx >= 0:
            text = text[idx:].strip()
        else:
            return ""
    return text


def _fill_text_precise(
    chunks: list[dict[str, Any]], doc: fitz.Document, margin_top: float, margin_bottom: float
) -> list[dict[str, Any]]:
    total_pages = len(doc)

    for idx, chunk in enumerate(chunks):
        start_1based = int(chunk["page_start"])
        end_1based = int(chunk["page_end"])
        start_idx = max(0, start_1based - 1)
        end_idx = min(total_pages - 1, end_1based - 1)

        if end_idx < start_idx:
            chunk["text"] = ""
            continue

        current_title = _pick_chunk_title(chunk)
        next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else None
        next_title = _pick_chunk_title(next_chunk) if next_chunk else None

        parts: list[str] = []

        for page_idx in range(start_idx, end_idx + 1):
            page = doc[page_idx]
            body_bottom = page.rect.height - margin_bottom

            if start_idx == end_idx:
                # Single-page chunk: cut from current title to next title (or bottom).
                y_start = find_title_y(page, current_title, margin_top, margin_bottom)
                if (
                    next_chunk is not None
                    and next_title
                    and int(next_chunk["page_start"]) == int(chunk["page_start"])
                ):
                    y_end = find_title_y(page, next_title, margin_top, margin_bottom)
                    cut_exclusive = True
                else:
                    y_end = body_bottom
                    cut_exclusive = False
                page_text = _extract_text_clip_y(
                    page,
                    y_start,
                    y_end,
                    margin_top,
                    margin_bottom,
                    title=current_title,
                    exclusive_end=cut_exclusive,
                )

            elif page_idx == start_idx:
                # First page of a multi-page chunk: from current title to bottom.
                y_start = find_title_y(page, current_title, margin_top, margin_bottom)
                page_text = _extract_text_clip_y(
                    page,
                    y_start,
                    body_bottom,
                    margin_top,
                    margin_bottom,
                    title=current_title,
                    exclusive_end=False,
                )

            elif page_idx == end_idx:
                # Last page of a multi-page chunk: from top to next title (or bottom).
                should_cut = (
                    next_chunk is not None
                    and next_title
                    and int(next_chunk["page_start"]) == (page_idx + 1)
                )
                if should_cut:
                    y_end = find_title_y(page, next_title, margin_top, margin_bottom)
                    page_text = _extract_text_clip_y(
                        page,
                        margin_top,
                        y_end,
                        margin_top,
                        margin_bottom,
                        title=None,
                        exclusive_end=True,
                    )
                else:
                    page_text = _extract_text_clip_y(
                        page,
                        margin_top,
                        body_bottom,
                        margin_top,
                        margin_bottom,
                        title=None,
                        exclusive_end=False,
                    )

            else:
                # Middle pages: full body.
                page_text = _extract_text_clip_y(
                    page,
                    margin_top,
                    body_bottom,
                    margin_top,
                    margin_bottom,
                    title=None,
                    exclusive_end=False,
                )

            if page_text:
                parts.append(page_text)

        chunk["text"] = "\n".join(parts).strip()

    return chunks


def remove_short_same_page_stubs(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop chunks that are only heading scaffolding before a same-page child.

    Typical case: TOC has H2 then H3 on the same page; extraction yields a tiny
    chunk (titles only) and the real body starts in the next chunk.
    """
    if not chunks:
        return [], 0

    max_chars = 120
    max_lines = 3
    kept: list[dict[str, Any]] = []
    removed = 0

    for idx, chunk in enumerate(chunks):
        next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else None
        text = str(chunk.get("text", "")).strip()
        title_h3 = str(chunk.get("title_h3", "")).strip()

        if (
            next_chunk is not None
            and int(next_chunk["page_start"]) == int(chunk["page_start"])
            and not title_h3
            and len(text) <= max_chars
            and text.count("\n") < max_lines
        ):
            removed += 1
            continue

        kept.append(chunk)

    relink_chunk_neighbors(kept)
    return kept, removed


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def relink_chunk_neighbors(chunks: list[dict[str, Any]]) -> None:
    """Refresh prev_chunk_id / next_chunk_id after filtering or reordering."""
    for idx, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = chunks[idx - 1]["chunk_id"] if idx > 0 else None
        chunk["next_chunk_id"] = (
            chunks[idx + 1]["chunk_id"] if idx < len(chunks) - 1 else None
        )


def _text_matches_heading_only(text: str, chunk: dict[str, Any]) -> bool:
    """True if body *text* is only a repeat of one of the non-empty title fields."""
    raw = str(text or "").strip()
    if not raw:
        return False
    text_ws = _normalize_ws(raw)
    text_compact = _normalize_for_title_match(raw)
    for key in ("title_h3", "title_h2", "title_h1"):
        t = str(chunk.get(key, "")).strip()
        if not t:
            continue
        if text_ws == _normalize_ws(t):
            return True
        if text_compact == _normalize_for_title_match(t):
            return True
    return False


def remove_title_only_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove chunks whose body text is only a section heading (no extra prose).

    Matches when *text* equals any non-empty title_h1/h2/h3 (whitespace- or
    compact-normalized), not only the picked leaf title.
    """
    if not chunks:
        return [], 0

    kept: list[dict[str, Any]] = []
    removed = 0
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if _text_matches_heading_only(text, chunk):
            removed += 1
            continue
        kept.append(chunk)

    relink_chunk_neighbors(kept)
    return kept, removed


def merge_chunks_by_identical_title(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge chunks that share the same pick-title (h3 > h2 > h1) into one.

    All occurrences with identical title text are merged; text is concatenated in
    TOC order, page_start/page_end span the combined range. Chunk ids are not
    renumbered (first chunk keeps its chunk_id).
    """
    if not chunks:
        return [], 0

    from collections import defaultdict

    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        t = _normalize_ws(_pick_chunk_title(c))
        if not t:
            continue
        key_to_indices[t].append(i)

    merged_from: set[int] = set()
    out: list[dict[str, Any]] = []
    removed = 0

    for i, c in enumerate(chunks):
        if i in merged_from:
            continue
        t = _normalize_ws(_pick_chunk_title(c))
        if not t:
            out.append(c)
            continue
        idxs = key_to_indices[t]
        if len(idxs) == 1:
            out.append(c)
            continue

        group = [chunks[j] for j in idxs]
        for j in idxs[1:]:
            merged_from.add(j)
        removed += len(idxs) - 1

        first = dict(group[0])
        texts = [str(x.get("text", "")).strip() for x in group]
        first["text"] = "\n\n".join(tx for tx in texts if tx)
        first["page_start"] = min(int(x["page_start"]) for x in group)
        first["page_end"] = max(int(x["page_end"]) for x in group)
        out.append(first)

    for idx, chunk in enumerate(out):
        chunk["prev_chunk_id"] = out[idx - 1]["chunk_id"] if idx > 0 else None
        chunk["next_chunk_id"] = (
            out[idx + 1]["chunk_id"] if idx < len(out) - 1 else None
        )

    return out, removed


def remove_empty_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    filtered = [c for c in chunks if str(c.get("text", "")).strip()]
    removed_count = len(chunks) - len(filtered)
    relink_chunk_neighbors(filtered)
    return filtered, removed_count


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "documents" / "F-15C.pdf"
    config_path = project_root / "config.yaml"
    output_dir = project_root / "data"
    output_path = output_dir / "chunks_f15c.json"

    toc = parse_toc(str(pdf_path))

    print("\n=== F-15C 目录（TOC 分级）===")
    if toc:
        src = toc[0].get("source", "?")
        print(f"来源: {src}")
        for i, item in enumerate(toc, start=1):
            level = int(item.get("level", 0))
            title = str(item.get("title", "")).strip()
            page = item.get("page", "")
            indent = "  " * max(0, level - 1)
            print(f"{i:3d}. [H{level}] {indent}{title} → {page}")
    else:
        print("(无 TOC 条目)")

    with fitz.open(pdf_path) as document:
        chunks = build_chunks(toc, document, pdf_path.name)
        chunks = fill_text(chunks, document, pdf_name=pdf_path.name, config_path=config_path)
        chunks, removed_stub_count = remove_short_same_page_stubs(chunks)
        chunks, removed_title_only_count = remove_title_only_chunks(chunks)
        chunks, merged_same_title_count = merge_chunks_by_identical_title(chunks)
        chunks, removed_empty_count = remove_empty_chunks(chunks)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    margin_top, margin_bottom = load_margins_from_config(config_path, pdf_path.name)
    non_empty = sum(1 for c in chunks if str(c.get("text", "")).strip())
    min_page = min((int(c["page_start"]) for c in chunks), default=0)
    max_page = max((int(c["page_start"]) for c in chunks), default=0)

    print(f"Wrote: {output_path}")
    print(f"Chunk总数: {len(chunks)}")
    print(f"text非空Chunk数: {non_empty}")
    print(f"page_start范围: min={min_page}, max={max_page}")
    print(f"被删除的同页短标题占位Chunk数: {removed_stub_count}")
    print(f"被删除的仅标题无正文Chunk数: {removed_title_only_count}")
    print(f"因标题相同合并掉的Chunk数: {merged_same_title_count}")
    print(f"被删除的空Chunk数: {removed_empty_count}")
    print(f"margin_top={margin_top}, margin_bottom={margin_bottom}")

    if chunks:
        max_words = max(len(str(c.get("text", "")).split()) for c in chunks)
        max_chars = max(len(str(c.get("text", ""))) for c in chunks)
        longest = max(chunks, key=lambda c: len(str(c.get("text", "")).split()))
        print(f"最长text(按空白分词): {max_words} 词, chunk_id={longest['chunk_id']}")
        print(f"最长text(字符数): {max_chars}")
