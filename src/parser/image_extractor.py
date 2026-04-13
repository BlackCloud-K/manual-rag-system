"""从 PDF 页面提取 image cluster，并按章节 chunk 归属、写回 image_paths。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fitz
import numpy as np

try:
    from src.parser.chunk_builder import (
        find_title_y,
        load_margins_from_config,
        _pick_chunk_title,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.parser.chunk_builder import (
        find_title_y,
        load_margins_from_config,
        _pick_chunk_title,
    )

RENDER_SCALE = 2.0
VERTICAL_GAP_MERGE_PT = 50.0
SOLID_RGB_STD_MAX = 5.0


def _bbox_union(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _cluster_by_vertical_gap(
    bboxes: list[tuple[float, float, float, float]], max_gap_pt: float
) -> list[tuple[float, float, float, float]]:
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


def _save_pixmap_png(pix: fitz.Pixmap, path: Path) -> None:
    if pix.colorspace is not None and pix.colorspace.n == 4:
        rgb = fitz.Pixmap(fitz.csRGB, pix)
        rgb.save(path.as_posix())
        rgb = None
    else:
        pix.save(path.as_posix())


def extract_image_clusters(
    page: fitz.Page, margin_top: float, margin_bottom: float
) -> list[dict[str, Any]]:
    """按 type==1 块过滤页眉页脚、竖直间隙聚类、渲染 clip，过滤纯色后返回 cluster 列表。"""
    body_bottom = page.rect.height - margin_bottom
    blocks = page.get_text("dict")["blocks"]
    image_blocks = [b for b in blocks if int(b.get("type", -1)) == 1]

    kept: list[tuple[float, float, float, float]] = []
    for block in image_blocks:
        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        y0 = box[1]
        if y0 < margin_top or y0 > body_bottom:
            continue
        kept.append(box)

    clusters_bbox = _cluster_by_vertical_gap(kept, VERTICAL_GAP_MERGE_PT)
    mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    out: list[dict[str, Any]] = []

    for box in clusters_bbox:
        clip = fitz.Rect(box)
        pix = page.get_pixmap(clip=clip, matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if arr.size == 0:
            std = 0.0
        else:
            arr = arr.reshape(pix.height, pix.width, pix.n)
            std = float(arr[:, :, :3].std())
        if std < SOLID_RGB_STD_MAX:
            pix = None
            continue
        out.append({"bbox": box, "pixmap": pix})

    return out


def assign_images_to_chunks(
    chunks: list[dict[str, Any]],
    doc: fitz.Document,
    pdf_stem: str,
    output_dir: Path,
    margin_top: float,
    margin_bottom: float,
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_prefix = f"data/images/{pdf_stem}/"

    for c in chunks:
        c["image_paths"] = []

    chunk_index = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    total_clusters = 0
    assigned_ok = 0
    per_page_image_idx: dict[int, int] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        clusters = extract_image_clusters(page, margin_top, margin_bottom)
        total_clusters += len(clusters)

        markers: list[tuple[float, int, dict[str, Any]]] = []
        for c in chunks:
            ps = int(c["page_start"])
            pe = int(c["page_end"])
            idx = chunk_index[c["chunk_id"]]
            if ps == page_num:
                title = _pick_chunk_title(c)
                ty = find_title_y(page, title, margin_top, margin_bottom)
                markers.append((ty, idx, c))
            elif pe == page_num and ps != pe:
                markers.append((float(margin_top), idx, c))
            elif ps < page_num < pe:
                markers.append((float(margin_top), idx, c))
        markers.sort(key=lambda t: (t[0], t[1]))

        if not markers:
            for cl in clusters:
                cl.pop("pixmap", None)
            continue

        m = per_page_image_idx.get(page_num, 0)
        for cl in clusters:
            box = cl["bbox"]
            pix = cl["pixmap"]
            cy0 = float(box[1])

            candidates = [t for t in markers if t[0] < cy0]
            if candidates:
                owner = max(candidates, key=lambda t: (t[0], t[1]))[2]
            else:
                owner = min(markers, key=lambda t: (t[0], t[1]))[2]

            fname = f"page_{page_num}_img_{m}.png"
            abs_path = output_dir / fname
            _save_pixmap_png(pix, abs_path)
            rel_path = f"{rel_prefix}{fname}"
            owner.setdefault("image_paths", []).append(rel_path)
            assigned_ok += 1

            m += 1
            cl.pop("pixmap", None)
        per_page_image_idx[page_num] = m

    nonempty = sum(1 for c in chunks if len(c.get("image_paths", [])) > 0)
    max_chunk: dict[str, Any] | None = None
    max_len = 0
    for c in chunks:
        n = len(c.get("image_paths", []))
        if n > max_len:
            max_len = n
            max_chunk = c

    if verbose:
        print(f"总图片数（cluster）: {total_clusters}")
        print(f"成功归属图片数: {assigned_ok}")
        print(f"image_paths 非空的 chunk 数: {nonempty}")
        if max_chunk is not None and max_len > 0:
            print(
                f"image_paths 最多的 chunk: {max_chunk.get('chunk_id')} "
                f"({max_len} 张)"
            )
            for p in max_chunk.get("image_paths", []):
                print(f"  {p}")
        else:
            print("image_paths 最多的 chunk: (无)")

    return chunks


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    pdf_path = root / "documents" / "F-15C.pdf"
    chunks_path = root / "data" / "chunks_f15c.json"
    config_path = root / "config.yaml"
    out_dir = root / "data" / "images" / "f15c"
    pdf_stem = "f15c"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return
    if not chunks_path.exists():
        print(f"Chunks JSON not found: {chunks_path}")
        return

    margin_top, margin_bottom = load_margins_from_config(
        config_path, "F-15C.pdf"
    )

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks_data, list):
        print("chunks JSON must be a list")
        return

    with fitz.open(pdf_path) as doc:
        updated = assign_images_to_chunks(
            chunks_data,
            doc,
            pdf_stem,
            out_dir,
            margin_top,
            margin_bottom,
            verbose=True,
        )

    chunks_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写回: {chunks_path}")


if __name__ == "__main__":
    main()
