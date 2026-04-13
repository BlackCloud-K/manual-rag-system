"""
Dependency:
pip install pymupdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "documents"


def fmt_rect(rect: fitz.Rect) -> str:
    return f"({rect.x0:.2f},{rect.y0:.2f},{rect.x1:.2f},{rect.y1:.2f})"


def detect_outline(doc: fitz.Document) -> list[str]:
    lines: list[str] = []
    toc = doc.get_toc()
    lines.append("检测一：PDF书签（Outline）")
    if not toc:
        lines.append("无书签")
        return lines

    for item in toc[:20]:
        level, title, page_no, *rest = item
        lines.append(f"[{level}] {title} → {page_no}")
    return lines


def page_has_toc_keyword(page: fitz.Page) -> bool:
    text = page.get_text("text")
    if not text:
        return False
    lowered = text.lower()
    return ("目录" in text) or ("contents" in lowered)


def link_target_page(link: dict[str, Any]) -> str:
    # Internal go-to links usually expose 0-based "page".
    if "page" in link and isinstance(link["page"], int) and link["page"] >= 0:
        return str(link["page"] + 1)
    if "to" in link and isinstance(link["to"], fitz.Point):
        # No reliable page number here; keep readable fallback.
        return "未知"
    return "未知"


def detect_toc_page_links(doc: fitz.Document) -> list[str]:
    lines: list[str] = []
    lines.append("检测二：目录页超链接")

    toc_pages: list[fitz.Page] = []
    max_scan = min(15, len(doc))
    for idx in range(max_scan):
        page = doc[idx]
        if page_has_toc_keyword(page):
            toc_pages.append(page)

    if not toc_pages:
        lines.append("前15页未发现目录页关键字（目录/Contents）")
        return lines

    for page in toc_pages:
        page_no = page.number + 1
        lines.append(f"目录候选页: 第{page_no}页")
        links = page.get_links()
        if not links:
            lines.append("目录页无超链接")
            continue

        for link in links[:20]:
            from_rect = link.get("from")
            if not isinstance(from_rect, fitz.Rect):
                continue
            target = link_target_page(link)
            lines.append(f"链接文字区域: {fmt_rect(from_rect)} → 目标页码: {target}")

    return lines


def analyze_pdf(pdf_path: Path) -> None:
    print("=" * 88)
    print(f"PDF: {pdf_path.name}")
    print("=" * 88)

    with fitz.open(pdf_path) as doc:
        for line in detect_outline(doc):
            print(line)
        print()
        for line in detect_toc_page_links(doc):
            print(line)
    print()


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"Documents directory not found: {DOCS_DIR}")
        return

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {DOCS_DIR}")
        return

    for pdf_file in pdf_files:
        analyze_pdf(pdf_file)


if __name__ == "__main__":
    main()
