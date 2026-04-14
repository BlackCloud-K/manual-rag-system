"""Batch: TOC → chunks → text → drop empty → images for every PDF in documents/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.chunk_builder import (
    _normalize_pdf_stem,
    build_chunks,
    fill_text,
    load_margins_from_config,
    remove_empty_chunks,
    remove_short_same_page_stubs,
    remove_title_only_chunks,
)
from src.parser.image_extractor import assign_images_to_chunks
from src.parser.toc_parser import parse_toc


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    docs = ROOT / "documents"
    data_dir = ROOT / "data"
    config_path = ROOT / "config.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(docs.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {docs}")
        return

    for pdf_path in pdfs:
        pdf_name = pdf_path.name
        pdf_stem = _normalize_pdf_stem(pdf_name)
        out_json = data_dir / f"chunks_{pdf_stem}.json"
        img_dir = ROOT / "data" / "images" / pdf_stem

        print(f"\n=== {pdf_name} (stem={pdf_stem}) ===")

        toc = parse_toc(str(pdf_path))
        with fitz.open(pdf_path) as doc:
            chunks = build_chunks(toc, doc, pdf_name)
            chunks = fill_text(chunks, doc, pdf_name=pdf_name, config_path=config_path)
            chunks, _stub = remove_short_same_page_stubs(chunks)
            chunks, _title_only = remove_title_only_chunks(chunks)
            chunks, _removed = remove_empty_chunks(chunks)
            margin_top, margin_bottom = load_margins_from_config(config_path, pdf_name)
            chunks = assign_images_to_chunks(
                chunks,
                doc,
                pdf_stem,
                img_dir,
                margin_top,
                margin_bottom,
                verbose=False,
            )

        out_json.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"过滤: 同页短占位={_stub}, 仅标题无正文={_title_only}, 空text={_removed}"
        )
        print(f"Wrote: {out_json}")

        nonempty_img = sum(
            1
            for c in chunks
            if isinstance(c.get("image_paths"), list) and len(c["image_paths"]) > 0
        )
        total_paths = sum(
            len(c["image_paths"])
            for c in chunks
            if isinstance(c.get("image_paths"), list)
        )
        print(
            f"统计: chunk总数={len(chunks)}, "
            f"image_paths非空chunk数={nonempty_img}, "
            f"总图片路径数={total_paths}"
        )


if __name__ == "__main__":
    main()
