from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.normalizer.chunk_normalizer import remove_garbage_chunks, split_large_chunks
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
from src.vlm.image_preprocessor import preprocess_image
from src.vlm.vlm_client import get_image_caption

CAPTION_CONCURRENCY = 2


def ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def resolve_pdf(documents_dir: Path, pdf_arg: str) -> Path:
    name = pdf_arg.strip()
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    direct = documents_dir / name
    if direct.exists():
        return direct.resolve()
    matches = sorted(documents_dir.glob(name))
    if len(matches) == 1:
        return matches[0].resolve()
    cand = sorted(documents_dir.glob(f"*{name}*"))
    if len(cand) == 1:
        return cand[0].resolve()
    if not cand:
        raise FileNotFoundError(
            f"在 {documents_dir} 找不到 PDF：{pdf_arg!r}（尝试 {name!r}）"
        )
    lines = "\n".join(f"  - {p.name}" for p in cand[:25])
    more = "" if len(cand) <= 25 else f"\n  ... 另有 {len(cand) - 25} 个"
    raise FileNotFoundError(
        f"在 {documents_dir} 匹配到多个 PDF，请给出完整文件名（含中英文与符号）：\n{lines}{more}"
    )


def paths_for_stem(project_root: Path, stem: str) -> dict[str, Path]:
    data = project_root / "data"
    final_dir = project_root / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    return {
        "chunks": data / f"chunks_{stem}.json",
        "normalized": data / f"chunks_{stem}_normalized.json",
        "final": final_dir / f"chunks_{stem}_final.json",
    }


def stage_parse(project_root: Path, pdf_path: Path, pdf_name: str, stem: str) -> None:
    ensure_utf8_stdout()
    config_path = project_root / "config.yaml"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_json = paths_for_stem(project_root, stem)["chunks"]
    img_dir = project_root / "data" / "images" / stem

    print(f"[parse] PDF={pdf_name} stem={stem}")
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
            stem,
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
        f"[parse] 过滤: stub={_stub}, title_only={_title_only}, empty={_removed} -> wrote {out_json}"
    )


def stage_normalize(project_root: Path, stem: str) -> None:
    ensure_utf8_stdout()
    p = paths_for_stem(project_root, stem)["chunks"]
    outp = paths_for_stem(project_root, stem)["normalized"]
    if not p.exists():
        raise FileNotFoundError(f"缺少原始 chunks 文件：{p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"JSON 根不是 list：{p}")
    original_chunks = [c for c in raw if isinstance(c, dict)]
    cleaned_chunks, _removed_chunks = remove_garbage_chunks(original_chunks)
    normalized_chunks = split_large_chunks(cleaned_chunks, threshold=3500, target_size=1500)

    outp.write_text(
        json.dumps(normalized_chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[normalize] {p.name}: {len(original_chunks)} -> {len(normalized_chunks)} chunks -> {outp}"
    )


def _normalize_rel_path(path_str: str) -> str:
    normalized = path_str.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _extract_image_paths(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(x).strip() for x in raw_value if isinstance(x, str) and str(x).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def collect_image_paths_for_normalized(
    project_root: Path, norm_path: Path
) -> list[str]:
    raw = json.loads(norm_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    unique: set[str] = set()
    missing = 0
    for chunk in raw:
        if not isinstance(chunk, dict):
            continue
        for image_path in _extract_image_paths(chunk.get("image_paths")):
            rel_path = _normalize_rel_path(image_path)
            abs_path = project_root / rel_path
            if abs_path.exists():
                unique.add(rel_path)
            else:
                missing += 1
                print(f"[vlm][WARN] 图片不存在，跳过: {rel_path}")
    if missing:
        print(f"[vlm][WARN] 缺失图片路径条目数: {missing}")
    return sorted(unique)


def _caption_cache_paths(project_root: Path) -> Path:
    return project_root / "data" / "captions_cache.json"


def _load_caption_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _save_caption_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


async def stage_vlm_caption(project_root: Path, stem: str) -> None:
    ensure_utf8_stdout()
    norm_path = paths_for_stem(project_root, stem)["normalized"]
    if not norm_path.exists():
        raise FileNotFoundError(f"缺少 normalized 文件：{norm_path}")

    load_dotenv(project_root / ".env")
    client = AsyncOpenAI()

    paths = collect_image_paths_for_normalized(project_root, norm_path)
    cache_path = _caption_cache_paths(project_root)
    cache = _load_caption_cache(cache_path)
    uncached = [p for p in paths if p not in cache or cache[p] == "CAPTION_FAILED"]

    if not paths:
        print("[vlm] 无可处理图片路径，跳过 VLM")
        return

    print(
        f"[vlm] 图片路径数={len(paths)} | 待请求（无缓存或失败重试）={len(uncached)}"
    )
    if not uncached:
        print("[vlm] 已全部在 captions_cache.json 中有结果。")
        return

    semaphore = asyncio.Semaphore(CAPTION_CONCURRENCY)
    cache_lock = asyncio.Lock()
    processed = 0

    async def worker(rel_path: str) -> None:
        nonlocal processed
        caption = "CAPTION_FAILED"
        tokens = 0
        abs_path = project_root / rel_path
        async with semaphore:
            try:
                base64_image = preprocess_image(str(abs_path))
                caption, tokens = await get_image_caption(base64_image, client)
            except FileNotFoundError:
                print(f"[vlm][WARN] 不存在: {rel_path}")
            except Exception as exc:
                print(f"[vlm][WARN] 失败: {rel_path} | {exc}")

        async with cache_lock:
            cache[rel_path] = caption
            _save_caption_cache(cache_path, cache)
            processed += 1
            preview = caption[:80] + ("…" if len(caption) > 80 else "")
            print(f"[vlm] {processed}/{len(uncached)} {rel_path} | tokens={tokens} | {preview!r}")

    await asyncio.gather(*(worker(rel_p) for rel_p in uncached))
    print(f"[vlm] 完成本轮处理 {len(uncached)} 张，缓存: {cache_path}")


def _inject_text(original_text: str, captions: list[str]) -> str:
    text = original_text if isinstance(original_text, str) else str(original_text or "")
    suffix = "".join(f"\n[IMG_CAPTION] {caption}" for caption in captions)
    return text + suffix


def stage_inject(project_root: Path, stem: str) -> None:
    ensure_utf8_stdout()
    chunk_file = paths_for_stem(project_root, stem)["normalized"]
    cache_path = _caption_cache_paths(project_root)
    if not chunk_file.exists():
        raise FileNotFoundError(chunk_file)

    raw_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
    if not isinstance(raw_chunks, list):
        raise ValueError("normalized JSON 应为 list")

    cache: dict[str, str] = {}
    if cache_path.exists():
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw_cache, dict):
            cache = {str(k): str(v) for k, v in raw_cache.items()}

    injected = 0
    skipped_paths = 0
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        image_paths = _extract_image_paths(chunk.get("image_paths"))
        if not image_paths:
            continue
        captions: list[str] = []
        for image_path in image_paths:
            rel_path = _normalize_rel_path(image_path)
            cap = cache.get(rel_path)
            if cap is None:
                skipped_paths += 1
                continue
            if cap == "CAPTION_FAILED":
                continue
            captions.append(cap)
        if captions:
            chunk["text"] = _inject_text(chunk.get("text", ""), captions)
            injected += 1

    out_path = paths_for_stem(project_root, stem)["final"]
    out_path.write_text(
        json.dumps(raw_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[inject] 注入 chunk 数={injected}, 跳过(无缓存)路径数={skipped_paths} -> {out_path}"
    )


def stage_embed(project_root: Path) -> None:
    script = project_root / "scripts" / "run_embedding.py"
    print(f"[embed] 执行 {script}")
    r = subprocess.run([sys.executable, str(script)], cwd=str(project_root))
    if r.returncode != 0:
        raise RuntimeError(f"run_embedding.py 退出码 {r.returncode}")


def stage_upload(project_root: Path) -> None:
    script = project_root / "scripts" / "run_upload_qdrant.py"
    print(f"[upload] 执行 {script}")
    r = subprocess.run([sys.executable, str(script)], cwd=str(project_root))
    if r.returncode != 0:
        raise RuntimeError(f"run_upload_qdrant.py 退出码 {r.returncode}")


def stem_from_pdf_name(pdf_name: str) -> str:
    return _normalize_pdf_stem(pdf_name)
