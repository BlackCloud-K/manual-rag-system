from __future__ import annotations

import json
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def _project_root() -> Path:
    # Keep consistent with src modules' "find_project_root" behavior (search for config.yaml)
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "config.yaml").exists():
            return parent
    return current.parents[1]


ROOT = _project_root()

# Ensure running `python -m backend.main` from anywhere still resolves `src.*`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv(dotenv_path: Path) -> None:
    """
    Lightweight .env loader without extra dependencies.

    - Supports lines like KEY=VALUE / export KEY=VALUE
    - Ignores empty lines and comments
    - Does not overwrite already-exported environment variables
    """
    if not dotenv_path.exists():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        # Strip one-layer quotes: KEY="value" / KEY='value'
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _read_yaml_scalar(cfg_path: Path, dotted_key: str) -> str | None:
    """
    Minimal YAML reader (scalar only) for this repo's config shape.

    Constraint: backend 只依赖 fastapi + uvicorn，因此不能引入 PyYAML。
    """
    if not cfg_path.exists():
        return None

    keys = dotted_key.split(".")
    if not keys:
        return None

    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    # Track current section path based on indentation (2 spaces per level in this repo)
    stack: list[tuple[int, str]] = []

    def _current_path() -> list[str]:
        return [k for _, k in stack]

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        key_part, value_part = line.lstrip(" ").split(":", 1)
        key = key_part.strip().strip('"').strip("'")

        # Maintain stack by indentation
        while stack and stack[-1][0] >= indent:
            stack.pop()

        # Section node
        if value_part.strip() == "":
            stack.append((indent, key))
            continue

        # Scalar leaf
        path = _current_path() + [key]
        if path == keys:
            val = value_part.strip()
            # Strip inline comments (best-effort)
            if " #" in val:
                val = val.split(" #", 1)[0].rstrip()
            # Strip quotes
            val = val.strip().strip('"').strip("'")
            return val or None

    return None


def _read_yaml_list(cfg_path: Path, dotted_key: str) -> list[str]:
    """
    Minimal YAML list reader for scalar string lists.
    Example:
      frontend:
        pdf_options:
          - "F-15C.pdf"
    """
    if not cfg_path.exists():
        return []

    keys = dotted_key.split(".")
    if not keys:
        return []

    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    stack: list[tuple[int, str]] = []
    target_indent: int | None = None
    out: list[str] = []

    def _current_path() -> list[str]:
        return [k for _, k in stack]

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        # Collect list items while staying within target section indentation
        if target_indent is not None:
            if indent <= target_indent:
                break
            item = line.lstrip(" ")
            if item.startswith("- "):
                val = item[2:].strip()
                if " #" in val:
                    val = val.split(" #", 1)[0].rstrip()
                val = val.strip().strip('"').strip("'")
                if val:
                    out.append(val)
                continue
            # Ignore non-list lines nested inside the section.
            continue

        if ":" not in line:
            continue

        key_part, value_part = line.lstrip(" ").split(":", 1)
        key = key_part.strip().strip('"').strip("'")

        while stack and stack[-1][0] >= indent:
            stack.pop()

        if value_part.strip() == "":
            path = _current_path() + [key]
            stack.append((indent, key))
            if path == keys:
                target_indent = indent

    return out


def _label_from_pdf_name(pdf_name: str) -> str:
    base = str(pdf_name or "").strip()
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    if base == "DCS FA-18C Early Access Guide CN":
        return "FA-18C"
    if base == "DCS_ JF-17 _雷电_":
        return "JF-17"
    if base == "F-16C“蝰蛇”":
        return "F-16C蝰蛇"
    return base


def _frontend_pdf_directory(cfg_path: Path) -> Path:
    """Directory that holds downloadable PDFs; same default as `/pdf/` route."""
    pdf_dir_raw = _read_yaml_scalar(cfg_path, "frontend.pdf_dir")
    pdf_dir = (pdf_dir_raw or "documents").strip()
    p = Path(pdf_dir)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _manual_pdf_basenames(cfg_path: Path) -> list[str]:
    """
    Dropdown + filter values for retrieval. Prefer explicit `frontend.pdf_options`
    when set; otherwise list all *.pdf under `frontend.pdf_dir`.
    Names must match `pdf_name` in chunk payloads / Qdrant.
    """
    explicit = [
        s for s in (str(x).strip() for x in _read_yaml_list(cfg_path, "frontend.pdf_options")) if s
    ]
    if explicit:
        return explicit

    pdf_dir = _frontend_pdf_directory(cfg_path)
    if not pdf_dir.is_dir():
        return []

    names = [x.name for x in pdf_dir.glob("*.pdf") if x.is_file()]
    return sorted(names, key=str.casefold)


def _json_error(exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


def _safe_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def _images_serve_base() -> Path:
    return (ROOT / "data" / "images").resolve()


def _safe_resolve_under_images_dir(file_path: str) -> Path | None:
    """
    Resolve file_path under ROOT/data/images/, rejecting path traversal.
    file_path is the URL subpath (forward slashes only).
    """
    raw = str(file_path or "").strip()
    if not raw:
        return None

    norm = raw.replace("\\", "/").strip("/")
    for segment in norm.split("/"):
        if segment in ("", ".", ".."):
            return None

    base = _images_serve_base()
    try:
        target = (base / norm).resolve()
    except OSError:
        return None

    try:
        target.relative_to(base)
    except ValueError:
        return None

    if not target.is_file():
        return None
    return target


def _int_page_val(payload: dict[str, Any]) -> int:
    raw = payload.get("page_start", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _title_path_h1_h2(payload: dict[str, Any]) -> str:
    h1 = str(payload.get("title_h1", "") or "").strip()
    h2 = str(payload.get("title_h2", "") or "").strip()
    if h1 and h2:
        return f"{h1} › {h2}"
    return h1 or h2


class HistoryTurn(BaseModel):
    query: str = ""
    answer: str = ""


class ChatRequest(BaseModel):
    user_input: str
    history: list[HistoryTurn] = Field(default_factory=list)
    pdf_name: str = ""


class FeedbackRequest(BaseModel):
    message_id: str
    feedback: Literal["up", "down"]
    query: str
    answer: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    print("Manual RAG Backend started，访问 http://127.0.0.1:8000")


@app.get("/")
def index() -> Any:
    try:
        index_path = ROOT / "frontend" / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>Frontend not ready</h1>", status_code=200)
    except Exception as exc:
        return _json_error(exc)


@app.get("/config")
def get_config() -> Any:
    try:
        cfg_path = ROOT / "config.yaml"
        pdf_options = _manual_pdf_basenames(cfg_path)

        manuals = [{"label": _label_from_pdf_name(p), "pdf_name": p} for p in pdf_options]
        return {"manuals": manuals}
    except Exception as exc:
        return _json_error(exc)


def _sse_images_from_chunks(reranked: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen_src: set[str] = set()
    for row in reranked:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        paths = payload.get("image_paths", [])
        if not isinstance(paths, list):
            continue
        for p in paths:
            if not isinstance(p, str):
                continue
            t = p.strip()
            if not t or t in seen_src:
                continue
            if not _safe_exists(t):
                continue
            seen_src.add(t)
            ordered.append(t)
            if len(ordered) >= 5:
                return ordered
    return ordered


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    history_dicts = [t.model_dump() for t in (req.history or [])]
    pdf_name = (req.pdf_name or "").strip()
    pdf_filter: str | None = pdf_name if pdf_name else None

    def event_stream():  # type: ignore[misc]
        try:
            from src.router.router import route
            from src.generator import generator as gen

            yield (
                "event: status\ndata: "
                f"{json.dumps({'text': '正在思考...'}, ensure_ascii=False)}\n\n"
            )
            route_result = route(req.user_input, history_dicts, pdf_name_filter=pdf_filter)

            action = str(getattr(route_result, "action", "") or "").strip().lower()

            if action == "chat":
                reasoning = str(getattr(route_result, "reasoning", "") or "").strip()

                if reasoning == "injection_rule":
                    ans = (
                        "这个请求我不能按你的方式处理。请直接说明你想查询的手册条目或具体操作。"
                    )
                elif hasattr(gen, "generate_chat_response"):
                    answer_obj = gen.generate_chat_response(req.user_input, history_dicts)  # type: ignore[attr-defined]
                    ans = (
                        str(answer_obj.get("answer", "") if isinstance(answer_obj, dict) else answer_obj)
                        if answer_obj is not None
                        else ""
                    )
                else:
                    out = gen.generate_answer_with_history(req.user_input, "", {}, history_dicts)
                    ans = str(out.get("answer", "") or "")

                payload = {"sources": [], "images": [], "action": "chat"}
                yield f"event: answer_token\ndata: {json.dumps({'content': ans}, ensure_ascii=False)}\n\n"
                yield (
                    "event: sources\ndata: "
                    f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                )
                yield "event: done\ndata: {}\n\n"
                return

            if action == "reject":
                msg = "抱歉，我只能解答与当前已导入手册节选相关的问题。"
                payload = {"sources": [], "images": [], "action": "reject"}
                yield f"event: answer_token\ndata: {json.dumps({'content': msg}, ensure_ascii=False)}\n\n"
                yield (
                    "event: sources\ndata: "
                    f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                )
                yield "event: done\ndata: {}\n\n"
                return

            yield (
                "event: status\ndata: "
                f"{json.dumps({'text': '正在检索节选...'}, ensure_ascii=False)}\n\n"
            )
            query_q = str(getattr(route_result, "query", "") or "").strip() or req.user_input
            retr = gen.retrieve_for_stream(
                query=query_q,
                pdf_name_filter=pdf_filter,
                history=history_dicts,
            )
            reranked = retr["reranked"]
            if not isinstance(reranked, list):
                reranked = []
            chunk_map = retr["chunk_map"]
            if not isinstance(chunk_map, dict):
                chunk_map = {}
            ctx = str(retr["context_text"] or "")

            id_to_rank: dict[str, int] = {}
            cid_to_payload: dict[str, dict[str, Any]] = {}
            for idx, row in enumerate(reranked, start=1):
                cid = row.get("chunk_id")
                if not isinstance(cid, str) or not cid.strip():
                    continue
                if cid not in id_to_rank:
                    id_to_rank[cid] = idx
                pl = row.get("payload")
                if isinstance(pl, dict) and cid not in cid_to_payload:
                    cid_to_payload[cid] = pl

            stream_gen = gen.generate_answer_stream(
                query=query_q,
                context_text=ctx,
                chunk_map=chunk_map,
                history=history_dicts,
            )
            yield (
                "event: status\ndata: "
                f"{json.dumps({'text': '正在生成回答...'}, ensure_ascii=False)}\n\n"
            )

            sse_images = _sse_images_from_chunks(reranked)

            for item in stream_gen:
                it = item if isinstance(item, dict) else {}
                ty = str(it.get("type", "") or "").strip()

                if ty == "answer_token":
                    ct = str(it.get("content", "") or "")
                    yield (
                        "event: answer_token\ndata: "
                        f"{json.dumps({'content': ct}, ensure_ascii=False)}\n\n"
                    )
                    continue

                if ty == "sources_raw":
                    raw = str(it.get("content", "") or "")
                    normalized = raw.replace("\uff0c", ",").replace("，", ",")
                    keys = [x.strip() for x in normalized.split(",") if x.strip()]
                    sources_list: list[dict[str, Any]] = []
                    for k in keys:
                        real_id = chunk_map.get(k)
                        if not isinstance(real_id, str):
                            continue
                        pl = cid_to_payload.get(real_id)
                        if pl is None:
                            continue
                        pdf_n = str(pl.get("pdf_name", "") or "").strip()
                        sources_list.append(
                            {
                                "page": _int_page_val(pl),
                                "title_path": _title_path_h1_h2(pl),
                                "pdf_name": pdf_n if pdf_n else pdf_name,
                                "chunk_id": real_id,
                                "rank": int(id_to_rank.get(real_id, 0) or 0),
                            }
                        )
                    out_payload = {
                        "sources": sources_list,
                        "images": sse_images,
                        "action": "search",
                    }
                    yield (
                        "event: sources\ndata: "
                        f"{json.dumps(out_payload, ensure_ascii=False)}\n\n"
                    )
                    continue

                if ty == "done":
                    yield "event: done\ndata: {}\n\n"

        except Exception as exc:  # noqa: BLE001
            yield (
                "event: error\ndata: "
                f"{json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/pdf/{filename}")
def get_pdf(filename: str) -> Any:
    try:
        cfg_path = ROOT / "config.yaml"
        pdf_dir = _read_yaml_scalar(cfg_path, "frontend.pdf_dir")
        if not pdf_dir:
            pdf_dir = "documents"

        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.is_absolute():
            pdf_dir_path = ROOT / pdf_dir_path

        file_path = pdf_dir_path / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")

        return FileResponse(
            str(file_path),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _json_error(exc)


@app.get("/image/{file_path:path}")
def get_image_by_subpath(file_path: str) -> Any:
    try:
        resolved = _safe_resolve_under_images_dir(file_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Image not found")
        mime, _ = mimetypes.guess_type(str(resolved))
        return FileResponse(str(resolved), media_type=mime or "application/octet-stream")
    except HTTPException:
        raise
    except Exception as exc:
        return _json_error(exc)


@app.get("/image")
def get_image(path: str = Query(..., description="Image path from retrieval result")) -> Any:
    try:
        p = Path(path)
        file_path = p if p.is_absolute() else (ROOT / p)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(str(file_path))
    except HTTPException:
        raise
    except Exception as exc:
        return _json_error(exc)


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> Any:
    try:
        cfg_path = ROOT / "config.yaml"
        log_path = _read_yaml_scalar(cfg_path, "frontend.feedback_log_path") or "data/feedback_log.jsonl"

        path = Path(log_path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": req.message_id,
            "feedback": req.feedback,
            "query": req.query,
            "answer": req.answer,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return {"ok": True}
    except Exception as exc:
        return _json_error(exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

