from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


def _json_error(exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


def _safe_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


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
    print("DCS RAG Backend started，访问 http://127.0.0.1:8000")


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
        pdf_options = _read_yaml_list(cfg_path, "frontend.pdf_options")
        if not pdf_options:
            pdf_options = [
                "F-16C“蝰蛇”.pdf",
                "F-15C.pdf",
                "DCS_ JF-17 _雷电_.pdf",
                "DCS FA-18C Early Access Guide CN.pdf",
            ]

        manuals = [{"label": _label_from_pdf_name(p), "pdf_name": p} for p in pdf_options]
        return {"manuals": manuals}
    except Exception as exc:
        return _json_error(exc)


@app.post("/chat")
def chat(req: ChatRequest) -> Any:
    try:
        from src.router.router import route  # reuse existing module
        from src.generator import generator as gen

        history_dicts = [t.model_dump() for t in (req.history or [])]
        pdf_name = (req.pdf_name or "").strip()
        pdf_filter: str | None = pdf_name if pdf_name else None

        route_result = route(req.user_input, history_dicts, pdf_name_filter=pdf_filter)

        action = str(getattr(route_result, "action", "") or "").strip().lower()
        if action == "chat":
            # Spec: prefer generate_chat_response if exists, else fallback.
            if hasattr(gen, "generate_chat_response"):
                answer_obj = gen.generate_chat_response(req.user_input, history_dicts)  # type: ignore[attr-defined]
                answer = (
                    str(answer_obj.get("answer", "") if isinstance(answer_obj, dict) else answer_obj)
                    if answer_obj is not None
                    else ""
                )
            else:
                # Fallback per spec
                out = gen.generate_answer_with_history(req.user_input, "", {}, history_dicts)
                answer = str(out.get("answer", "") or "")

            return {"action": "chat", "answer": answer, "sources": [], "images": []}

        if action == "reject":
            return {
                "action": "reject",
                "answer": "抱歉，我只能回答 DCS 飞行手册相关的问题。",
                "sources": [],
                "images": [],
            }

        # action == "search" (default)
        query = str(getattr(route_result, "query", "") or "").strip() or req.user_input
        rag = gen.run_rag_pipeline_with_history(query=query, pdf_name_filter=pdf_filter, history=history_dicts)

        answer = str(rag.get("answer", "") or "")
        query_rewritten = bool(rag.get("query_rewritten", False))

        sources_in = rag.get("sources", [])
        if not isinstance(sources_in, list):
            sources_in = []

        # Note: generator.run_rag_pipeline_with_history() currently does NOT return `top_chunks`.
        # It returns `sources` from extract_sources(), each includes:
        # {chunk_id, page, title_path, image_paths}. We'll adapt images from that.
        sources: list[dict[str, Any]] = []
        image_candidates: list[str] = []
        for s in sources_in:
            if not isinstance(s, dict):
                continue
            chunk_id = str(s.get("chunk_id", "") or "")
            page = s.get("page", 0)
            try:
                page_i = int(page)
            except Exception:
                page_i = 0
            title_path = str(s.get("title_path", "") or "")
            rank = s.get("rank", 0)
            try:
                rank_i = int(rank)
            except Exception:
                rank_i = 0

            sources.append(
                {
                    "page": page_i,
                    "title_path": title_path,
                    "pdf_name": pdf_name,
                    "chunk_id": chunk_id,
                    "rank": rank_i,
                }
            )

            imgs = s.get("image_paths", [])
            if isinstance(imgs, list):
                for p in imgs:
                    if isinstance(p, str) and p.strip():
                        image_candidates.append(p)

        dedup_images: list[str] = []
        seen: set[str] = set()
        for p in image_candidates:
            if p in seen:
                continue
            seen.add(p)
            if _safe_exists(p):
                dedup_images.append(p)

        return {
            "action": "search",
            "answer": answer,
            "sources": sources,
            "images": dedup_images,
            "search_query": str(getattr(route_result, "query", "") or query),
            "query_rewritten": query_rewritten,
        }
    except Exception as exc:
        return _json_error(exc)


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

