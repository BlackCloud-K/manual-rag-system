from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src.pipeline.state import STAGE_ORDER, PipelineState, restart_from, state_path
from src.pipeline.stages import (
    paths_for_stem,
    resolve_pdf,
    stage_embed,
    stage_inject,
    stage_normalize,
    stage_parse,
    stage_upload,
    stage_vlm_caption,
    stem_from_pdf_name,
)


def _project_root() -> Path:
    # src/pipeline/import_manual.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _chunks_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(raw, list) and len(raw) > 0


def _should_skip_parse(state: PipelineState, chunks_path: Path) -> bool:
    if state.is_done("parse") and _chunks_nonempty(chunks_path):
        return True
    return False


def _should_skip_normalize(state: PipelineState, normalized_path: Path) -> bool:
    if state.is_done("normalize") and normalized_path.exists():
        try:
            return normalized_path.stat().st_size > 10
        except OSError:
            return False
    return False


def _should_skip_vlm(state: PipelineState) -> bool:
    return state.is_done("vlm")


def _should_skip_inject(state: PipelineState, final_path: Path) -> bool:
    return state.is_done("inject") and final_path.exists()


def _should_skip_embed(state: PipelineState) -> bool:
    return state.is_done("embed")


def _should_skip_upload(state: PipelineState) -> bool:
    return state.is_done("upload")


def main(argv: list[str] | None = None) -> int:
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))

    parser = argparse.ArgumentParser(
        description="将 documents/ 中的单本手册端到端接入 RAG（可断点续跑）。",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="PDF 文件名或部分名（须在 documents/ 下），例如 F-16C\"蝰蛇\".pdf",
    )
    parser.add_argument(
        "--restart-from",
        metavar="STAGE",
        default=None,
        help=(
            "从指定阶段重做并清除该阶段及之后的checkpoint。"
            f"取值: {', '.join(STAGE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有 checkpoint，仍按产物智能跳过（例如 VLM 已缓存的图片不会重复请求）。",
    )
    args = parser.parse_args(argv)

    root = _project_root()
    documents = root / "documents"
    if not documents.is_dir():
        print(f"[ERROR] 缺少目录: {documents}")
        return 1

    try:
        pdf_path = resolve_pdf(documents, args.pdf)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    pdf_name = pdf_path.name
    stem = stem_from_pdf_name(pdf_name)
    paths = paths_for_stem(root, stem)

    state = PipelineState.load(root, stem, pdf_name)
    if args.no_resume:
        state = PipelineState(project_root=root, stem=stem, pdf_name=pdf_name, stages={})
    if args.restart_from:
        try:
            rf = restart_from(args.restart_from)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 1
        state.mark_not_done_from(rf)
        state.save()
        print(f"[INFO] 已从 {rf!r} 起清除后续 checkpoint，状态文件: {state_path(root, stem)}")

    print(f"[INFO] pdf_name={pdf_name!r} stem={stem!r}")
    print(f"[INFO] 状态文件: {state_path(root, stem)}")

    try:
        # --- parse ---
        if _should_skip_parse(state, paths["chunks"]):
            print("[skip] parse（已完成且 chunks 存在）")
        else:
            stage_parse(root, pdf_path, pdf_name, stem)
            state.mark_done("parse")
            state.save()

        # --- normalize ---
        if _should_skip_normalize(state, paths["normalized"]):
            print("[skip] normalize（已完成且输出存在）")
        else:
            stage_normalize(root, stem)
            state.mark_done("normalize")
            state.save()

        # --- vlm ---
        if _should_skip_vlm(state):
            print("[skip] vlm（checkpoint 标记完成）")
        else:
            import asyncio

            asyncio.run(stage_vlm_caption(root, stem))
            state.mark_done("vlm")
            state.save()

        # --- inject ---
        if _should_skip_inject(state, paths["final"]):
            print("[skip] inject（已完成且 final 存在）")
        else:
            stage_inject(root, stem)
            state.mark_done("inject")
            state.save()

        # --- embed (global script) ---
        if _should_skip_embed(state):
            print("[skip] embed（checkpoint 标记完成）。若需强制更新向量请 --restart-from embed")
        else:
            stage_embed(root)
            state.mark_done("embed")
            state.save()

        # --- upload ---
        if _should_skip_upload(state):
            print("[skip] upload（checkpoint 标记完成）。若需重传请 --restart-from upload")
        else:
            stage_upload(root)
            state.mark_done("upload")
            state.save()

    except Exception as exc:
        print(f"[ERROR] 管道在运行中中断: {exc}")
        print(
            "       修复问题后重新执行同一命令即可从断点继续（已完成阶段会自动跳过）。"
        )
        traceback.print_exc()
        return 1

    print(
        "[DONE] 全流程完成。请将对应 PDF 放在 documents（或 frontend.pdf_dir）下以便前端枚举；"
        "并确认 config.yaml manuals 中为该文件名配置了页边距等。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
