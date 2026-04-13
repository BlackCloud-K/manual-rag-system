"""
Dependency:
pip install pymupdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import fitz

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "documents" / "F-15C.pdf"
TARGET_PAGES_1BASED = [10, 32, 50]


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def truncate_text(text: str, max_len: int = 60) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def extract_text_block_plain(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(str(span.get("text", "")))
    return "".join(parts)


def collect_image_placements(page: fitz.Page) -> list[tuple[int, fitz.Rect]]:
    """All (xref, rect) draws on this page; same xref may appear multiple times."""
    out: list[tuple[int, fitz.Rect]] = []
    for item in page.get_images():
        xref = int(item[0])
        for rect in page.get_image_rects(xref):
            out.append((xref, rect))
    return out


def pop_xref_for_bbox(
    placements: list[tuple[int, fitz.Rect]], bbox: tuple[float, ...], max_dist: float = 2.0
) -> int | None:
    """Take the closest unused placement to *bbox* (L∞), remove it, return xref."""
    if len(bbox) < 4 or not placements:
        return None
    bi = tuple(float(bbox[i]) for i in range(4))
    best_i: int | None = None
    best_d = float("inf")
    for i, (_xref, rect) in enumerate(placements):
        t = (rect.x0, rect.y0, rect.x1, rect.y1)
        d = max(abs(bi[j] - t[j]) for j in range(4))
        if d < best_d:
            best_d = d
            best_i = i
    if best_i is None or best_d > max_dist:
        return None
    xref, _ = placements.pop(best_i)
    return xref


def format_block_line(block: dict[str, Any], placements: list[tuple[int, fitz.Rect]]) -> str:
    btype = block.get("type")
    bb = block.get("bbox")
    if not isinstance(bb, (list, tuple)) or len(bb) < 4:
        return f"  [?] bbox missing | {block!r}"
    x0, y0, x1, y1 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))

    if btype == 0:
        plain = extract_text_block_plain(block)
        summary = truncate_text(plain)
        return (
            f"  [TEXT] y0={y0:.1f}  y1={y1:.1f} | {summary}"
        )

    if btype == 1:
        w = block.get("width", "?")
        h = block.get("height", "?")
        xref = pop_xref_for_bbox(placements, (x0, y0, x1, y1))
        xref_s = str(xref) if xref is not None else "?"
        return (
            f"  [IMAGE] y0={y0:.1f} y1={y1:.1f} | image xref={xref_s} w={w} h={h}"
        )

    return f"  [type={btype}] y0={y0:.1f}  y1={y1:.1f} | {block!r}"


def analyze_page(doc: fitz.Document, page_one_based: int) -> None:
    idx = page_one_based - 1
    if idx < 0 or idx >= len(doc):
        print(f"Page {page_one_based}: out of range (doc has {len(doc)} pages)")
        print()
        return

    page = doc[idx]
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
    blocks = text_dict.get("blocks", [])
    placements = collect_image_placements(page)

    def sort_key(b: dict[str, Any]) -> float:
        bb = b.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 2:
            return float(bb[1])
        return 0.0

    blocks_sorted = sorted(blocks, key=sort_key)

    print(safe_console_text(f"Page {page_one_based}:"))
    for block in blocks_sorted:
        print(safe_console_text(format_block_line(block, placements)))

    image_block_count = sum(1 for b in blocks if b.get("type") == 1)
    get_images_count = len(page.get_images())
    match = "是" if image_block_count == get_images_count else "否"

    print(safe_console_text(f"  该页图片 block 数 (dict, type==1): {image_block_count}"))
    print(safe_console_text(f"  该页 page.get_images() 数: {get_images_count}"))
    print(safe_console_text(f"  两种计数是否一致: {match}"))
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    with fitz.open(PDF_PATH) as doc:
        for p in TARGET_PAGES_1BASED:
            analyze_page(doc, p)


if __name__ == "__main__":
    main()
