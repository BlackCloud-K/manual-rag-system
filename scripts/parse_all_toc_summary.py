"""One-off: summarize parse_toc() for every PDF in documents/."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.toc_parser import parse_toc  # noqa: E402


def main() -> None:
    docs = ROOT / "documents"
    for pdf in sorted(docs.glob("*.pdf")):
        items = parse_toc(str(pdf))
        print("=" * 72)
        print(pdf.name)
        if not items:
            print("  No TOC (no bookmarks and no usable directory-page links).")
            continue
        src = items[0].get("source", "?")
        levels = [int(x.get("level", 0)) for x in items]
        print(f"  source={src}, entries={len(items)}, max_level={max(levels)}")
        print("  first 10:")
        for it in items[:10]:
            lv, title, pg = it.get("level"), it.get("title", ""), it.get("page")
            t = title if len(title) <= 60 else title[:57] + "..."
            print(f"    [H{lv}] {t} -> p.{pg}")


if __name__ == "__main__":
    main()
