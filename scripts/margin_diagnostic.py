"""
Dependency:
pip install pymupdf
"""

from __future__ import annotations

from pathlib import Path
import sys

import fitz


ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "documents" / "F-15C.pdf"
TARGET_PAGE_INDICES = [9, 29, 49]  # 0-based


def truncate_text(text: str, max_len: int = 30) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    with fitz.open(PDF_PATH) as doc:
        for page_idx in TARGET_PAGE_INDICES:
            if page_idx < 0 or page_idx >= len(doc):
                print(f"Page {page_idx + 1}: out of range")
                print()
                continue

            page = doc[page_idx]
            print(f"Page {page_idx + 1}:")
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                x0, y0, x1, y1, text, *_ = block
                content = truncate_text(str(text))
                if not content:
                    continue
                print(f"  y0={y0:.1f}  y1={y1:.1f}  | {safe_console_text(content)}")
            print()


if __name__ == "__main__":
    main()
