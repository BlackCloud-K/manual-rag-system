"""
Dependency:
pip install pymupdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "documents" / "F-15C.pdf"
OUTPUT_DIR = ROOT_DIR / "data" / "images" / "test"
# 1-based 页码列表（与 PDF 阅读器页码一致）
TARGET_PAGES_1BASED = [33, 42, 32, 31, 36, 37, 39, 113]

# get_pixmap 默认约 72 DPI；放大矩阵可提高导出清晰度（体积也会变大）。
RENDER_SCALE = 2.0

# 按 y0 排序后，若「下一块顶边 − 当前簇底边」< 该值（pt），则并入同一簇（bbox union）。
VERTICAL_GAP_MERGE_PT = 50.0


def process_page(doc: fitz.Document, page_one_based: int, mat: fitz.Matrix) -> None:
    idx = page_one_based - 1
    if idx < 0 or idx >= len(doc):
        print(f"Page {page_one_based}: out of range (doc has {len(doc)} pages)\n")
        return

    page = doc[idx]
    blocks = page.get_text("dict")["blocks"]

    image_blocks: list[dict[str, Any]] = [b for b in blocks if int(b.get("type", -1)) == 1]

    filtered_header_footer = 0
    filtered_solid = 0
    kept_count = 0

    print(f"Page {page_one_based} (image_blocks={len(image_blocks)}):")

    kept_bboxes: list[tuple[float, float, float, float]] = []
    for block_idx, block in enumerate(image_blocks):
        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        y0 = box[1]
        y1 = box[3]

        if y0 < 55 or y0 > 585:
            filtered_header_footer += 1
            print(
                f"  drop[{block_idx}] reason=header_footer "
                f"y0={y0:.1f} y1={y1:.1f}"
            )
            continue
        kept_bboxes.append(box)

    cluster_bboxes = _cluster_by_vertical_gap(kept_bboxes, max_gap_pt=VERTICAL_GAP_MERGE_PT)

    for merged_idx, box in enumerate(cluster_bboxes):
        clip = fitz.Rect(box)
        pix = page.get_pixmap(clip=clip, matrix=mat)

        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if arr.size == 0:
            std = 0.0
        else:
            arr = arr.reshape(pix.height, pix.width, pix.n)
            rgb = arr[:, :, :3]
            std = float(rgb.std())
        if std < 5:
            filtered_solid += 1
            print(
                f"  drop_merged[{merged_idx}] reason=solid_color std={std:.2f} "
                f"bbox=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}) "
                f"size={pix.width}x{pix.height}"
            )
            pix = None
            continue

        save_path = OUTPUT_DIR / f"page_{page_one_based}_img_{kept_count}.png"
        pix.save(save_path)
        print(
            f"  keep[{kept_count}] "
            f"bbox=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}) "
            f"size={pix.width}x{pix.height} -> {save_path}"
        )
        pix = None
        kept_count += 1

    print(
        f"  summary: saved={kept_count}, header_footer={filtered_header_footer}, "
        f"solid={filtered_solid}, clusters={len(cluster_bboxes)}\n"
    )


def _bbox_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _cluster_by_vertical_gap(
    bboxes: list[tuple[float, float, float, float]], max_gap_pt: float
) -> list[tuple[float, float, float, float]]:
    """按 y0 排序；竖直间隙 (next.y0 - cluster.y1) < max_gap_pt 则 union，否则新开簇。"""
    if not bboxes:
        return []
    ordered = sorted(bboxes, key=lambda b: (b[1], b[0]))
    clusters: list[tuple[float, float, float, float]] = []
    cur = ordered[0]
    for nxt in ordered[1:]:
        gap = nxt[1] - cur[3]
        if gap < max_gap_pt:
            cur = _bbox_union(cur, nxt)
        else:
            clusters.append(cur)
            cur = nxt
    clusters.append(cur)
    return clusters


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)

    with fitz.open(PDF_PATH) as doc:
        print(
            f"render scale: {RENDER_SCALE}x, "
            f"vertical gap merge: < {VERTICAL_GAP_MERGE_PT} pt\n"
        )
        for p in TARGET_PAGES_1BASED:
            process_page(doc, p, mat)

    print(f"done. pages: {TARGET_PAGES_1BASED}")


if __name__ == "__main__":
    main()
