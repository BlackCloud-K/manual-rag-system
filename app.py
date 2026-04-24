from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from dotenv import load_dotenv

from src.generator.generator import run_rag_pipeline_with_history
from src.retriever import retriever as retriever_mod


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def load_config() -> dict[str, Any]:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


@st.cache_resource
def init_models() -> str:
    """预热本地 embedding/reranker 模型，避免首次提问等待过长。"""
    try:
        retriever_mod.embed_query("warmup")
        retriever_mod.rerank(
            "warmup",
            candidates=[{"payload": {"text": "warmup"}}],
            top_n=1,
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def extract_unique_images(sources: list[dict] | None) -> list[str]:
    """从 sources 提取 image_paths，跨 chunk 去重并保持顺序。"""
    if not sources:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        paths = src.get("image_paths")
        if not isinstance(paths, list):
            continue
        for p in paths:
            if not isinstance(p, str):
                continue
            path = p.strip()
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(path)
    return out


def render_images(images: list[str] | None) -> None:
    """渲染回答相关图片；不存在的图片跳过。"""
    if not images:
        return

    existing: list[Path] = []
    for p in images:
        img_path = Path(p)
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        if img_path.exists():
            existing.append(img_path)
        else:
            print(f"[WARN] image not found: {img_path}")

    if not existing:
        return

    st.markdown("**相关图片**")
    if len(existing) <= 3:
        for img in existing:
            st.image(str(img), use_container_width=False, width=500)
        return

    for img in existing[:3]:
        st.image(str(img), use_container_width=False, width=500)
    with st.expander("查看更多图片"):
        for img in existing[3:]:
            st.image(str(img), use_container_width=False, width=500)


def render_sources(sources: list[dict] | None) -> None:
    """折叠展示来源列表（页码升序、chunk 去重）。"""
    if not sources:
        return

    dedup: dict[str, dict[str, Any]] = {}
    for src in sources:
        if not isinstance(src, dict):
            continue
        cid = str(src.get("chunk_id", "") or "").strip()
        if not cid or cid in dedup:
            continue
        dedup[cid] = src

    if not dedup:
        return

    def _page_value(item: dict[str, Any]) -> int:
        raw = item.get("page", item.get("page_start", 0))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    rows = sorted(dedup.values(), key=_page_value)
    with st.expander("📖 查看来源"):
        for src in rows:
            page = _page_value(src)
            title_path = str(src.get("title_path", "") or "").strip()
            if title_path:
                st.markdown(f"- 第 {page} 页 · {title_path}")
            else:
                st.markdown(f"- 第 {page} 页")


def _associated_query(messages: list[dict[str, Any]], msg_index: int) -> str:
    for i in range(msg_index - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user":
            return str(msg.get("content", "") or "")
    return ""


def _append_feedback_log(
    *,
    query: str,
    answer: str,
    search_query: str,
    query_rewritten: bool,
    feedback: str,
    sources: list[dict] | None,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "answer": answer,
        "search_query": search_query,
        "query_rewritten": bool(query_rewritten),
        "feedback": feedback,
        "sources": sources or [],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_feedback(msg_index: int, msg: dict[str, Any], query: str, feedback_log_path: Path) -> None:
    """渲染并记录 👍 / 👎 反馈。"""
    submitted = st.session_state.feedback_submitted
    if msg_index in submitted:
        st.caption("已反馈，感谢你的评价。")
        return

    cols = st.columns([1, 1, 6])
    with cols[0]:
        up = st.button("👍", key=f"fb_up_{msg_index}")
    with cols[1]:
        down = st.button("👎", key=f"fb_down_{msg_index}")

    if not up and not down:
        return

    feedback = "positive" if up else "negative"
    _append_feedback_log(
        query=query,
        answer=str(msg.get("content", "") or ""),
        search_query=str(msg.get("search_query", query) or query),
        query_rewritten=bool(msg.get("query_rewritten", False)),
        feedback=feedback,
        sources=msg.get("sources"),
        log_path=feedback_log_path,
    )
    submitted.add(msg_index)
    st.session_state.feedback_submitted = submitted
    st.success("反馈已记录")


def _friendly_error_message(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "qdrant" in lower or "connection" in lower or "refused" in lower:
        return f"检索服务连接失败，请确认 Qdrant 正在运行。详细信息：{text}"
    if "openai" in lower or "api key" in lower or "authentication" in lower:
        return f"OpenAI 调用失败，请检查 API 配置。详细信息：{text}"
    return f"生成回答时出错：{text}"


st.set_page_config(
    page_title="DCS 手册问答助手",
    page_icon="✈️",
    layout="centered",
)

cfg = load_config()
frontend_cfg = cfg.get("frontend")
if not isinstance(frontend_cfg, dict):
    frontend_cfg = {}

pdf_options = frontend_cfg.get("pdf_options")
if not isinstance(pdf_options, list) or not pdf_options:
    pdf_options = ["F-16C“蝰蛇”.pdf"]
pdf_options = [str(x) for x in pdf_options]

feedback_log_cfg = str(frontend_cfg.get("feedback_log_path", "data/feedback_log.jsonl"))
feedback_log_path = Path(feedback_log_cfg)
if not feedback_log_path.is_absolute():
    feedback_log_path = ROOT / feedback_log_path

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

with st.sidebar:
    st.header("设置")
    pdf_filter = st.selectbox("选择手册", options=pdf_options, index=0)
    if st.button("🔄 新对话"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.feedback_submitted = set()
        st.rerun()

st.title("DCS 手册问答助手")
st.caption("基于 F-16C 飞行手册的智能问答系统")
st.divider()

warmup_status = init_models()
if warmup_status != "ok":
    st.info("模型将在首次提问时加载，首次响应可能略慢。")

for idx, msg in enumerate(st.session_state.messages):
    role = str(msg.get("role", "assistant"))
    with st.chat_message(role):
        st.markdown(str(msg.get("content", "")))
        if role == "assistant":
            render_images(msg.get("images"))
            render_sources(msg.get("sources"))
            query_text = _associated_query(st.session_state.messages, idx)
            render_feedback(idx, msg, query_text, feedback_log_path)

prompt = st.chat_input("输入你的问题，例如：F-16 冷启动的步骤是什么？")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索手册并生成回答（首次启动可能需要加载模型）..."):
                result = run_rag_pipeline_with_history(
                    query=prompt,
                    pdf_name_filter=pdf_filter,
                    history=st.session_state.history,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(_friendly_error_message(exc))
            st.stop()

        answer = str(result.get("answer", "") or "")
        if answer.startswith("生成失败："):
            st.error(answer)
            st.stop()
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        search_query = str(result.get("search_query", prompt) or prompt)
        query_rewritten = bool(result.get("query_rewritten", False))
        images = extract_unique_images(sources)

        st.markdown(answer)
        render_images(images)
        render_sources(sources)

        pending_msg = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "images": images,
            "search_query": search_query,
            "query_rewritten": query_rewritten,
        }
        render_feedback(len(st.session_state.messages), pending_msg, prompt, feedback_log_path)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(pending_msg)
    st.session_state.history.append({"query": prompt, "answer": answer})
