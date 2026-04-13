from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


TOC_TRAILING_PAGE_PATTERN = re.compile(r"[\d\s\.。…]+$")
TOC_LABEL_TITLES = {"目录", "contents", "content"}


def _clean_toc_title(text: str) -> str:
    compact = " ".join((text or "").split())
    cleaned = TOC_TRAILING_PAGE_PATTERN.sub("", compact).strip()
    return cleaned


def _has_toc_keyword(page: fitz.Page) -> bool:
    text = page.get_text("text")
    if not text:
        return False
    lowered = text.lower()
    return ("目录" in text) or ("contents" in lowered)


def _is_toc_label_title(title: str) -> bool:
    normalized = " ".join((title or "").strip().lower().split())
    return normalized in TOC_LABEL_TITLES


def _normalize_link_target_page(link: dict[str, Any]) -> int | None:
    page_value = link.get("page")
    if isinstance(page_value, int) and page_value >= 0:
        return page_value + 1
    return None


def extract_toc_from_bookmarks(doc: fitz.Document) -> list[dict[str, Any]] | None:
    toc = doc.get_toc()
    if not toc:
        return None

    records: list[dict[str, Any]] = []
    for item in toc:
        if len(item) < 3:
            continue
        level, title, page = item[0], item[1], item[2]
        if not isinstance(level, int) or not isinstance(page, int):
            continue
        records.append({"level": level, "title": str(title).strip(), "page": page})

    # Filter out generic TOC labels and deduplicate adjacent identical titles
    # by keeping the larger page number.
    deduped: list[dict[str, Any]] = []
    for record in records:
        if _is_toc_label_title(record["title"]):
            continue
        if deduped and record["title"] == deduped[-1]["title"]:
            if int(record["page"]) >= int(deduped[-1]["page"]):
                deduped[-1] = record
            continue
        deduped.append(record)

    return deduped or None


def extract_toc_from_links(doc: fitz.Document) -> list[dict[str, Any]] | None:
    link_items: list[dict[str, Any]] = []
    max_scan_pages = min(15, len(doc))

    for page_index in range(max_scan_pages):
        page = doc[page_index]
        if not _has_toc_keyword(page):
            continue

        for link in page.get_links():
            from_rect = link.get("from")
            if not isinstance(from_rect, fitz.Rect):
                continue

            target_page = _normalize_link_target_page(link)
            if target_page is None:
                continue

            raw_text = page.get_textbox(fitz.Rect(from_rect))
            title = _clean_toc_title(raw_text)
            if not title:
                continue

            x0_bucket = round(from_rect.x0 / 10.0) * 10
            link_items.append(
                {
                    "title": title,
                    "page": target_page,
                    "x0_bucket": int(x0_bucket),
                }
            )

    if not link_items:
        return None

    buckets = sorted({item["x0_bucket"] for item in link_items})
    bucket_to_level = {bucket: min(idx + 1, 3) for idx, bucket in enumerate(buckets)}

    dedup_seen: set[tuple[str, int]] = set()
    records: list[dict[str, Any]] = []
    for item in link_items:
        dedup_key = (item["title"], item["page"])
        if dedup_key in dedup_seen:
            continue
        dedup_seen.add(dedup_key)
        records.append(
            {
                "level": bucket_to_level[item["x0_bucket"]],
                "title": item["title"],
                "page": item["page"],
            }
        )

    return records or None


def parse_toc(pdf_path: str) -> list[dict[str, Any]]:
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        from_bookmarks = extract_toc_from_bookmarks(doc)
        if from_bookmarks:
            return [{**item, "source": "bookmark"} for item in from_bookmarks]

        from_links = extract_toc_from_links(doc)
        if from_links:
            return [{**item, "source": "links"} for item in from_links]

    return []


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    documents_dir = project_root / "documents"
    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {documents_dir}")
    else:
        for pdf_file in pdf_files:
            toc_items = parse_toc(str(pdf_file))
            print("=" * 88)
            print(f"PDF: {pdf_file.name}")
            if not toc_items:
                print("No TOC found")
                print()
                continue

            source = toc_items[0].get("source", "unknown")
            print(f"Source: {source}")
            for item in toc_items[:20]:
                print(
                    f"[{item.get('level')}] {item.get('title')} -> page {item.get('page')}"
                )
            print()
