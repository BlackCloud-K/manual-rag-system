from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from src.retriever import retriever as retriever_mod

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional until installed
    tiktoken = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "config.yaml").exists():
            return parent
    return current.parents[2]


def _load_yaml_config() -> dict[str, Any]:
    cfg_path = _find_project_root() / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _int_page_start(payload: dict[str, Any]) -> int:
    raw = payload.get("page_start", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _title_path_from_payload(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title_h1", "title_h2", "title_h3"):
        val = payload.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            parts.append(text)
    return " › ".join(parts)


def _format_chunk_block(chunk_label: str, payload: dict[str, Any]) -> str:
    page = _int_page_start(payload)
    title_path = _title_path_from_payload(payload)
    if title_path:
        header = f"---{chunk_label}（第{page}页 · {title_path}）---"
    else:
        header = f"---{chunk_label}（第{page}页）---"
    body = str(payload.get("text", "") or "")
    return f"{header}\n{body}"


def _encode_tokens(text: str, enc: Any) -> int:
    return len(enc.encode(text))


def _dedupe_by_chunk_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        cid = row.get("chunk_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append(row)
    return out


def _sort_reading_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        page = _int_page_start(payload)
        by_page.setdefault(page, []).append(row)

    ordered_rows: list[dict[str, Any]] = []
    for page in sorted(by_page.keys()):
        group = by_page[page]
        ids = {str(r.get("chunk_id", "")) for r in group if isinstance(r.get("chunk_id"), str)}
        id_to_row = {
            str(r["chunk_id"]): r for r in group if isinstance(r.get("chunk_id"), str)
        }

        heads: list[str] = []
        for cid in ids:
            payload = id_to_row[cid]["payload"]
            if not isinstance(payload, dict):
                continue
            prev = payload.get("prev_chunk_id")
            if not isinstance(prev, str) or not prev.strip() or prev not in ids:
                heads.append(cid)

        if not heads:
            heads = list(ids)

        heads = sorted(set(heads))
        consumed: set[str] = set()
        page_order: list[str] = []

        for h in heads:
            if h in consumed:
                continue
            cur: str | None = h
            while cur is not None and cur in ids and cur not in consumed:
                page_order.append(cur)
                consumed.add(cur)
                nxt = id_to_row[cur]["payload"].get("next_chunk_id")
                if not isinstance(nxt, str) or nxt not in ids:
                    break
                cur = nxt

        for cid in sorted(ids):
            if cid not in consumed:
                page_order.append(cid)

        for cid in page_order:
            ordered_rows.append(id_to_row[cid])

    return ordered_rows


def assemble_context(
    ranked_chunks: list[dict[str, Any]],
    max_context_tokens: int = 6000,
) -> tuple[dict[str, str], str]:
    """按 rerank 分数做 token 预算筛选，再按阅读顺序排版并编号，返回 (chunk_map, context_text)。"""
    if tiktoken is None:
        raise ImportError("assemble_context 需要 tiktoken，请先安装：pip install tiktoken")

    enc = tiktoken.get_encoding("cl100k_base")
    deduped = _dedupe_by_chunk_id(ranked_chunks)

    selected: list[dict[str, Any]] = []
    total_tokens = 0
    placeholder = "chunk_0"
    for row in deduped:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        block = _format_chunk_block(placeholder, payload)
        ntok = _encode_tokens(block, enc)
        if total_tokens + ntok <= max_context_tokens:
            selected.append(row)
            total_tokens += ntok
        else:
            break

    ordered = _sort_reading_order(selected)
    chunk_map: dict[str, str] = {}
    blocks: list[str] = []
    for idx, row in enumerate(ordered, start=1):
        label = f"chunk_{idx}"
        cid = row.get("chunk_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        chunk_map[label] = cid
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        blocks.append(_format_chunk_block(label, payload))

    context_text = "\n\n".join(blocks)
    return chunk_map, context_text


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def generate_answer(
    query: str,
    context_text: str,
    chunk_map: dict[str, str],
    ranked_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """调用 GPT 生成回答；ranked_chunks 用于填充 sources（可选）。"""
    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    model = str(gen_cfg.get("model", "gpt-4o-mini"))
    temperature = float(gen_cfg.get("temperature", 0))

    system_prompt = (
        "你是 DCS 飞行模拟器手册问答助手。你的唯一知识来源是用户消息中提供的手册片段。\n\n"
        "规则：\n"
        "1. 仅根据提供的片段内容回答。你可以对片段内容进行归纳和总结，但不允许补充片段中完全没有提及的信息。\n"
        "2. 如果片段中找不到相关信息，但是你的预训练知识中有与问题相关的补充信息（例如解释为什么手册中没有、或提供背景知识），可以补充说明，但必须明确标注『以下为模型补充，不来自手册，请自行辨别』。\n"
        "3. 如果只能部分回答，据实回答能确认的部分，并说明哪些信息在提供的片段中未涉及。\n"
        "4. 回答步骤类问题时，保持原文的步骤编号和顺序。\n"
        "你必须以 JSON 格式回复，不要包含 markdown 代码围栏，结构如下：\n"
        "{\n"
        '  "answer": "你的回答内容",\n'
        '  "used_chunks": ["实际使用的片段编号列表，如 chunk_1, chunk_3"]\n'
        "}\n"
    )
    user_prompt = f"{context_text}\n\n用户问题：{query}"

    if OpenAI is None:
        return {
            "answer": "生成失败：未安装 openai 库（pip install openai）",
            "used_chunks": [],
            "sources": [],
            "usage": {},
        }

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            stream=False,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - 用户要求不向外抛出
        return {
            "answer": f"生成失败：{exc}",
            "used_chunks": [],
            "sources": [],
            "usage": {},
        }

    raw_content = (response.choices[0].message.content or "").strip()
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }

    to_parse = _strip_json_fence(raw_content)
    try:
        parsed = json.loads(to_parse)
        if not isinstance(parsed, dict):
            raise ValueError("parsed JSON is not an object")
    except Exception:
        warnings.warn(
            f"LLM 返回非 JSON，已回退为纯文本。原始内容前 200 字：{raw_content[:200]!r}",
            stacklevel=2,
        )
        parsed = {"answer": raw_content, "used_chunks": []}

    answer = str(parsed.get("answer", "") or "")
    raw_used = parsed.get("used_chunks", [])
    if not isinstance(raw_used, list):
        raw_used = []

    used_real_ids: list[str] = []
    for item in raw_used:
        label = str(item).strip()
        if label in chunk_map:
            used_real_ids.append(chunk_map[label])
        elif label in {v for v in chunk_map.values()}:
            used_real_ids.append(label)

    sources = (
        extract_sources(ranked_chunks, used_real_ids)
        if ranked_chunks is not None
        else []
    )

    return {
        "answer": answer,
        "used_chunks": used_real_ids,
        "sources": sources,
        "usage": usage,
    }


def extract_sources(
    ranked_chunks: list[dict[str, Any]],
    used_chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """从 rerank 结果中提取 LLM 声明使用的 chunk 的来源信息。"""
    lookup: dict[str, dict[str, Any]] = {}
    for row in ranked_chunks:
        cid = row.get("chunk_id")
        if isinstance(cid, str) and cid.strip() and cid not in lookup:
            lookup[cid] = row

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in used_chunk_ids:
        if cid in seen:
            continue
        row = lookup.get(cid)
        if row is None:
            continue
        seen.add(cid)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        imgs = payload.get("image_paths", [])
        if not isinstance(imgs, list):
            imgs = []
        image_paths = [str(p) for p in imgs if isinstance(p, str) and p.strip()]
        out.append(
            {
                "chunk_id": cid,
                "page": _int_page_start(payload),
                "title_path": _title_path_from_payload(payload),
                "image_paths": image_paths,
            }
        )
    return out


def run_rag_pipeline(
    query: str,
    pdf_name_filter: str | None = None,
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    """检索 → 扩展 → rerank → 组装上下文 → 生成。"""
    cfg = _load_yaml_config()
    retrieval_cfg = cfg.get("retrieval")
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}

    top_k = int(retrieval_cfg.get("hybrid_prefetch", 50))
    rerank_top_n = int(retrieval_cfg.get("rerank_top_n", 5))

    if max_context_tokens is None:
        max_context_tokens = int(gen_cfg.get("max_context_tokens", 6000))

    chunks_lookup = retriever_mod.load_chunks_lookup(pdf_name_filter)
    hits = retriever_mod.hybrid_search(
        query,
        top_k=top_k,
        pdf_name_filter=pdf_name_filter,
    )
    candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
    reranked = retriever_mod.rerank(query, candidates, top_n=rerank_top_n)

    chunk_map, context_text = assemble_context(reranked, max_context_tokens=max_context_tokens)
    return generate_answer(query, context_text, chunk_map, ranked_chunks=reranked)
