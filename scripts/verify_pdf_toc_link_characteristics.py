"""
Verify TOC/link behavior for manuals (e.g. F-16 / F-18): PDF internal links
only encode a *start* page, not an explicit section end. Compare with parse_toc
+ build_chunks span assignment.

Run from repo root:
  python scripts/verify_pdf_toc_link_characteristics.py
  python scripts/verify_pdf_toc_link_characteristics.py --pdf "D:\\path\\to\\manual.pdf"
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

# Repo root on sys.path for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.parser.chunk_builder import build_chunks  # noqa: E402
from src.parser.toc_parser import (  # noqa: E402
    _has_toc_keyword,
    _normalize_link_target_page,
    extract_toc_from_bookmarks,
    extract_toc_from_links,
    parse_toc,
)

# PyMuPDF link kinds (see fitz documentation)
_LINK_KIND_NAMES = {
    fitz.LINK_NONE: "NONE",
    fitz.LINK_GOTO: "GOTO",
    fitz.LINK_URI: "URI",
    fitz.LINK_LAUNCH: "LAUNCH",
    fitz.LINK_NAMED: "NAMED",
    fitz.LINK_GOTOR: "GOTOR",
}


def _normalize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem) or "manual"


def _default_pdf_candidates(documents_dir: Path) -> list[Path]:
    if not documents_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(documents_dir.glob("*.pdf")):
        s = _normalize_stem(p.name)
        # Heuristic: F-16C chunks stem f16c; FA-18 early access dcsfa18...
        if "f16" in s or "16" in s and "f" in s:
            out.append(p)
        elif "a18" in s or "f18" in s or "fa18" in s:
            out.append(p)
    return out


def _scan_toc_pages_link_stats(doc: fitz.Document, max_scan_pages: int = 15) -> dict:
    """Mirror toc_parser scan window; count link kinds and internal targets."""
    max_scan = min(max_scan_pages, len(doc))
    per_page: list[dict] = []
    kind_counter: Counter[int] = Counter()
    goto_with_page = 0
    goto_without_page = 0
    uri_only_on_toc_pages = 0

    for page_index in range(max_scan):
        page = doc[page_index]
        if not _has_toc_keyword(page):
            continue
        page_info = {
            "page_1based": page_index + 1,
            "links_total": 0,
            "goto_with_dest_page": 0,
            "uri": 0,
            "other": 0,
        }
        for link in page.get_links():
            page_info["links_total"] += 1
            k = link.get("kind", fitz.LINK_NONE)
            kind_counter[k] += 1
            if k == fitz.LINK_GOTO:
                tp = _normalize_link_target_page(link)
                if tp is not None:
                    page_info["goto_with_dest_page"] += 1
                    goto_with_page += 1
                else:
                    goto_without_page += 1
            elif k == fitz.LINK_URI:
                page_info["uri"] += 1
                uri_only_on_toc_pages += 1
            else:
                page_info["other"] += 1
        if page_info["links_total"]:
            per_page.append(page_info)

    return {
        "toc_keyword_pages": per_page,
        "kind_counts": { _LINK_KIND_NAMES.get(k, str(k)): v for k, v in kind_counter.items()},
        "goto_with_target_page": goto_with_page,
        "goto_without_target_page": goto_without_page,
        "uri_on_toc_keyword_pages": uri_only_on_toc_pages,
    }


def _report_one_pdf(pdf_path: Path) -> None:
    print("=" * 88)
    print(f"PDF: {pdf_path}")
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        print(f"Total pages: {total}")

        bm = extract_toc_from_bookmarks(doc)
        lk = extract_toc_from_links(doc)
        print(f"Bookmarks usable for TOC: {bool(bm)} ({len(bm) if bm else 0} entries)")
        print(f"Links-based TOC (first 15 pages, TOC keyword pages): {bool(lk)} ({len(lk) if lk else 0} entries)")

        toc = parse_toc(str(pdf_path))
        source = toc[0].get("source") if toc else None
        print(f"parse_toc() source: {source!r}, entries: {len(toc)}")

        stats = _scan_toc_pages_link_stats(doc, 15)
        print("\n--- Link scan (pages with 目录/Contents, first 15 doc pages) ---")
        print(f"Link kind counts on those pages: {stats['kind_counts']}")
        print(
            f"Internal GOTO with resolvable target page: {stats['goto_with_target_page']}; "
            f"GOTO without page: {stats['goto_without_target_page']}; "
            f"URI links: {stats['uri_on_toc_keyword_pages']}"
        )
        if stats["toc_keyword_pages"]:
            print("Per-page breakdown (1-based):")
            for row in stats["toc_keyword_pages"]:
                print(f"  page {row['page_1based']}: links={row['links_total']}, "
                      f"goto_dest={row['goto_with_dest_page']}, uri={row['uri']}, other={row['other']}")
        else:
            print("  (no page in first 15 matched TOC keyword — link TOC would be empty)")

        print(
            "\nNote: PDF link annotations describe a *destination* (start). "
            "There is no standard 'section end' in the link dict — only optional "
            "zoom/view; range end is imposed by the app (e.g. next TOC page or EOF)."
        )

        if not toc:
            print("\n(No TOC — skip chunk span simulation.)")
            return

        pages = [int(t["page"]) for t in toc]
        print(f"\nTOC target pages: min={min(pages)}, max={max(pages)}, unique_count={len(set(pages))}")
        print(f"Max TOC page vs last page: {max(pages)} vs {total} → "
              f"{total - max(pages)} pages after last TOC *start* with no new TOC anchor.")

        chunks = build_chunks(toc, doc, pdf_path.name)
        widths = [c["page_end"] - c["page_start"] for c in chunks]
        print(f"\nSimulated build_chunks spans (page_end - page_start), N={len(chunks)}:")
        print(f"  max width: {max(widths)} pages (chunk index {widths.index(max(widths)) + 1})")
        tail = chunks[-1]
        print(
            f"  LAST chunk: page_start={tail['page_start']}, page_end={tail['page_end']} "
            f"(width {tail['page_end'] - tail['page_start']}) title_h1={tail['title_h1']!r}"
        )

        print("\n--- 'Next start as previous end' ---")
        print(
            "build_chunks ALREADY sets page_end[i] = page_start of next TOC row "
            "(same as next link target page), except the last row, where there is "
            "no next row — so page_end is forced to total_pages. "
            "So the giant tail is not fixable by that rule alone; you need either "
            "more TOC anchors in the tail, or a synthetic split (heuristic / page cap / etc.)."
        )

        if len(set(pages)) <= 5 or max(pages) < total * 0.15:
            print(
                "\nObservation: many TOC entries share few target pages, or max target "
                "is still in the early part of the document — last chunk absorbs the rest."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Path to a PDF (repeatable). If omitted, tries documents/ heuristics.",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=_ROOT / "documents",
        help="Used when --pdf not given (default: <repo>/documents)",
    )
    args = parser.parse_args()

    paths: list[Path] = [Path(p).resolve() for p in args.pdf]
    if not paths:
        paths = _default_pdf_candidates(args.documents_dir)

    if not paths:
        print("No PDFs to analyze. Pass --pdf path/to.pdf or place F-16/F-18 PDFs under documents/.")
        print(f"documents dir: {args.documents_dir.resolve()}")
        return

    for p in paths:
        if not p.is_file():
            print(f"Skip (not a file): {p}")
            continue
        _report_one_pdf(p)


if __name__ == "__main__":
    main()
