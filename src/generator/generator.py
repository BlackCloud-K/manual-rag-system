from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

from src.generator.llm_client import chat_complete
from src.retriever import retriever as retriever_mod

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional until installed
    tiktoken = None  # type: ignore[assignment]


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


def _debug_used_chunks_enabled() -> bool:
    cfg = _load_yaml_config()
    generation_cfg = cfg.get("generation")
    if not isinstance(generation_cfg, dict):
        return False
    return bool(generation_cfg.get("debug_used_chunks", False))


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


FORMAT_SUFFIX_STREAM = '''
请严格按照以下格式输出，不要输出任何其他内容：
ANSWER_START
（你的回答正文）
ANSWER_END
SOURCES_START
（逗号分隔的 chunk key 列表，仅包含实际引用的 chunk，如 chunk_1,chunk_2）
SOURCES_END
'''

ANSWER_MARKER = "ANSWER_START\n"


def _answer_stream_chunk_system_prompt() -> str:
    """System prompt for generate_answer-style (no history); must match generate_answer wording."""
    return (
        '你是 DCS 飞行模拟器手册问答助手。应该热情、全面地相应用户的关于飞行手册的问题。\n\n'
        "规则：\n"
        "1. 根据提供的片段内容回答。请结合片段内容给出详尽的解答，注意答案可能来自多个不同段落。\n"
        "2. 如果片段中找不到相关信息，但是你的预训练知识中有与问题相关的补充信息（例如解释为什么手册中没有、或提供背景知识），可以补充说明，但必须明确标注信息为模型补充，不来自手册，请自行辨别』。\n"
        "3. 如果只能部分回答，据实回答能确认的部分，并说明哪些信息在提供的片段中未涉及。\n"
        "4. 回答步骤类问题时，保持原文的步骤编号和顺序。\n"
        "5. 引用手册内容时使用“手册”或者章节标题进行指代，而非“chunk x”。\n"
    )


def _generation_system_prompt() -> str:
    return (
        "你是 DCS 飞行模拟器手册问答助手。应该热情、全面地相应用户的关于飞行手册的问题。\n\n"
        "规则：\n"
        "1. 根据提供的片段内容回答。请结合片段内容给出详尽的解答，注意答案可能来自多个不同段落。\n"
        "2. 如果片段中找不到相关信息，但是你的预训练知识中有与问题相关的补充信息（例如解释为什么手册中没有、或提供背景知识），可以补充说明，但必须明确标注信息为模型补充，不来自手册，请自行辨别』。\n"
        "3. 如果只能部分回答，据实回答能确认的部分，并说明哪些信息在提供的片段中未涉及。\n"
        "4. 回答步骤类问题时，保持原文的步骤编号和顺序。\n"
        "5. 引用手册内容时使用“手册”或者章节标题进行指代，而非“chunk x”。\n"
    )


def _chat_completion_json(
    *,
    model: str,
    temperature: float,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, int], str | None]:
    try:
        raw_content = str(chat_complete(messages=messages, stream=False, json_mode=True) or "")
    except Exception as exc:  # noqa: BLE001 - 用户要求不向外抛出
        return "", {}, f"生成失败：{exc}"

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    return raw_content.strip(), usage, None


def assemble_context_only(ranked_chunks: list[dict[str, Any]], max_context_tokens: int = 6000) -> tuple[dict[str, str], str]:
    """
    仅封装 assemble_context（与 run_rag_pipeline_with_history / run_rag_pipeline 内一致），供后端流式管线复用组装 context。
    """
    return assemble_context(ranked_chunks, max_context_tokens=max_context_tokens)


def retrieve_for_stream(
    query: str,
    pdf_name_filter: str | None = None,
    max_context_tokens: int = 6000,
    history: list[dict] | None = None,
    max_history_tokens: int = 2000,
) -> dict[str, Any]:
    """
    与 run_rag_pipeline_with_history 相同的检索 → 上下文组装，不包含生成调用。
    返回 reranked / chunk_map / context_text / search_query 等，供流式接口使用。
    """
    cfg = _load_yaml_config()
    retrieval_cfg = cfg.get("retrieval")
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}

    top_k = int(retrieval_cfg.get("hybrid_prefetch", 50))
    rerank_top_n = int(retrieval_cfg.get("rerank_top_n", 5))
    effective_max_context_tokens = int(
        gen_cfg.get("max_context_tokens", 6000)
        if max_context_tokens == 6000
        else max_context_tokens
    )
    effective_max_history_tokens = int(
        gen_cfg.get("max_history_tokens", 2000)
        if max_history_tokens == 2000
        else max_history_tokens
    )

    trimmed_history = trim_history(history or [], max_history_tokens=effective_max_history_tokens)
    search_query = query
    if trimmed_history:
        search_query = rewrite_query(query, trimmed_history)

    chunks_lookup = retriever_mod.load_chunks_lookup(pdf_name_filter)
    hits = retriever_mod.hybrid_search(
        search_query,
        top_k=top_k,
        pdf_name_filter=pdf_name_filter,
    )
    candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
    reranked = retriever_mod.rerank(search_query, candidates, top_n=rerank_top_n)

    chunk_map, context_text = assemble_context(
        reranked,
        max_context_tokens=effective_max_context_tokens,
    )
    return {
        "reranked": reranked,
        "chunk_map": chunk_map,
        "context_text": context_text,
        "original_query": query,
        "search_query": search_query,
        "query_rewritten": search_query != query,
        "trimmed_history": trimmed_history,
    }


def _parse_generation_json(raw_content: str, chunk_map: dict[str, str]) -> tuple[str, list[str]]:
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
    if _debug_used_chunks_enabled():
        print(f"[DEBUG used_chunks] raw_used={raw_used!r}", file=sys.stderr)

    chunk_values = set(chunk_map.values())
    used_real_ids: list[str] = []
    for item in raw_used:
        label = str(item).strip()
        if label in chunk_map:
            used_real_ids.append(chunk_map[label])
        elif label in chunk_values:
            used_real_ids.append(label)
    if _debug_used_chunks_enabled():
        print(f"[DEBUG used_chunks] mapped_used_real_ids={used_real_ids!r}", file=sys.stderr)

    return answer, used_real_ids


def trim_history(history: list[dict], max_history_tokens: int = 2000) -> list[dict]:
    """
    对历史对话进行 token 裁剪，从最早的轮次开始丢弃，直到总 token 数 <= max_history_tokens。
    """
    if not history or max_history_tokens <= 0:
        return []
    if tiktoken is None:
        raise ImportError("trim_history 需要 tiktoken，请先安装：pip install tiktoken")

    enc = tiktoken.get_encoding("cl100k_base")
    kept_reversed: list[dict] = []
    used_tokens = 0

    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        query = str(turn.get("query", "") or "")
        answer = str(turn.get("answer", "") or "")
        turn_tokens = _encode_tokens(query, enc) + _encode_tokens(answer, enc)

        if turn_tokens > max_history_tokens:
            continue
        if used_tokens + turn_tokens > max_history_tokens:
            break

        kept_reversed.append({"query": query, "answer": answer})
        used_tokens += turn_tokens

    return list(reversed(kept_reversed))


def rewrite_query(
    query: str,
    history: list[dict],
) -> str:
    """
    根据对话历史判断当前 query 是否依赖上文语境，如果是则改写为自包含问题。
    """
    if not history:
        return query

    # 只使用最近 1-2 轮，控制 token 成本
    recent = [h for h in history if isinstance(h, dict)][-2:]
    if not recent:
        return query

    lines: list[str] = []
    for turn in recent:
        hq = str(turn.get("query", "") or "").strip()
        ha = str(turn.get("answer", "") or "").strip()
        if not hq and not ha:
            continue
        lines.append(f"用户：{hq}")
        lines.append(f"助手：{ha[:200]}")
    history_text = "\n".join(lines).strip()
    if not history_text:
        return query

    prompt = (
        "你是一个查询改写助手。根据对话历史判断用户的当前问题是否依赖上文语境。\n\n"
        "如果当前问题包含指代（如\"它\"、\"这个\"、\"那个步骤\"）、省略了主语或对象、"
        "或者脱离上文就无法理解，请将其改写为一个独立的、自包含的完整问题。\n\n"
        "如果当前问题本身就是完整的、不依赖上文的独立问题，请原样输出。\n\n"
        "只输出最终的问题文本，不要输出任何解释。\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"当前问题：{query}\n\n"
        "输出："
    )

    def _needs_context_rewrite(text: str) -> bool:
        q = text.strip()
        if not q:
            return False
        patterns = [
            r"^第\s*\d+\s*步",
            r"\b这(个|一步|步)\b",
            r"\b那(个|一步|步)\b",
            r"\b它\b",
            r"具体是什么",
            r"什么意思",
            r"怎么做",
        ]
        return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)

    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    model = str(gen_cfg.get("model", "gpt-4o-mini"))

    try:
        rewritten = str(
            chat_complete(
                messages=[
                    {"role": "system", "content": "你是查询改写助手。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                json_mode=False,
            )
            or ""
        )
        rewritten = rewritten.strip("`\"' \n\t")
        if rewritten and rewritten != query:
            return rewritten
    except Exception:
        return query

    # 兜底：LLM 未有效改写且当前问题显著依赖上下文时，拼接最近问题形成可检索的自包含 query
    if _needs_context_rewrite(query):
        last_query = str(recent[-1].get("query", "") or "").strip()
        if last_query:
            return f"关于“{last_query}”，{query}"
    return query


def generate_answer_with_history(
    query: str,
    context_text: str,
    chunk_map: dict,
    history: list[dict] | None = None,
    max_history_tokens: int = 2000,
) -> dict:
    """
    支持多轮对话历史的生成函数。功能与 generate_answer() 完全相同，
    区别仅在于 messages 数组中插入了历史对话轮次。
    """
    if not history:
        return generate_answer(query, context_text, chunk_map)

    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    model = str(gen_cfg.get("model", "gpt-4o-mini"))
    temperature = float(gen_cfg.get("temperature", 0))
    effective_max_history_tokens = int(
        gen_cfg.get("max_history_tokens", 2000)
        if max_history_tokens == 2000
        else max_history_tokens
    )

    trimmed_history = trim_history(history, max_history_tokens=effective_max_history_tokens)
    user_prompt = f"{context_text}\n\n用户问题：{query}"

    messages: list[dict[str, str]] = [{"role": "system", "content": _generation_system_prompt()}]
    for turn in trimmed_history:
        hq = str(turn.get("query", "") or "")
        ha = str(turn.get("answer", "") or "")
        messages.append({"role": "user", "content": hq})
        messages.append({"role": "assistant", "content": ha})
    messages.append({"role": "user", "content": user_prompt})

    raw_content, usage, err = _chat_completion_json(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    if err is not None:
        return {
            "answer": err,
            "used_chunks": [],
            "sources": [],
            "usage": {},
        }

    answer, used_real_ids = _parse_generation_json(raw_content, chunk_map)
    return {
        "answer": answer,
        "used_chunks": used_real_ids,
        "sources": [],
        "usage": usage,
    }


def run_rag_pipeline_with_history(
    query: str,
    pdf_name_filter: str | None = None,
    max_context_tokens: int = 6000,
    history: list[dict] | None = None,
    max_history_tokens: int = 2000,
) -> dict:
    """
    支持多轮对话历史的端到端 RAG pipeline。
    功能与 run_rag_pipeline() 相同，但生成阶段会注入历史对话。
    """
    cfg = _load_yaml_config()
    retrieval_cfg = cfg.get("retrieval")
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}

    top_k = int(retrieval_cfg.get("hybrid_prefetch", 50))
    rerank_top_n = int(retrieval_cfg.get("rerank_top_n", 5))
    effective_max_context_tokens = int(
        gen_cfg.get("max_context_tokens", 6000)
        if max_context_tokens == 6000
        else max_context_tokens
    )
    effective_max_history_tokens = int(
        gen_cfg.get("max_history_tokens", 2000)
        if max_history_tokens == 2000
        else max_history_tokens
    )

    trimmed_history = trim_history(history or [], max_history_tokens=effective_max_history_tokens)
    search_query = query
    if trimmed_history:
        search_query = rewrite_query(query, trimmed_history)

    chunks_lookup = retriever_mod.load_chunks_lookup(pdf_name_filter)
    hits = retriever_mod.hybrid_search(
        search_query,
        top_k=top_k,
        pdf_name_filter=pdf_name_filter,
    )
    candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
    reranked = retriever_mod.rerank(search_query, candidates, top_n=rerank_top_n)

    chunk_map, context_text = assemble_context(
        reranked,
        max_context_tokens=effective_max_context_tokens,
    )
    generated = generate_answer_with_history(
        query=query,
        context_text=context_text,
        chunk_map=chunk_map,
        history=trimmed_history,
        max_history_tokens=effective_max_history_tokens,
    )
    generated["sources"] = extract_sources(reranked, generated.get("used_chunks", []))
    generated["original_query"] = query
    generated["search_query"] = search_query
    generated["query_rewritten"] = search_query != query
    return generated


def generate_answer_stream(
    query: str,
    context_text: str,
    chunk_map: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    max_history_tokens: int = 2000,
):
    """两段式流式输出：先流式回答，再非流式抽取 used_chunks。"""
    yield from _generate_answer_stream_two_pass(
        query=query,
        context_text=context_text,
        chunk_map=chunk_map,
        history=history,
        max_history_tokens=max_history_tokens,
    )


def _generate_answer_stream_two_pass(
    query: str,
    context_text: str,
    chunk_map: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    max_history_tokens: int = 2000,
):
    """两段式流式输出实现（保持 answer_token/sources_raw/done 事件格式不变）。"""
    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    model = str(gen_cfg.get("model", "gpt-4o-mini"))
    temperature = float(gen_cfg.get("temperature", 0))
    effective_max_history_tokens = int(
        gen_cfg.get("max_history_tokens", 2000)
        if max_history_tokens == 2000
        else max_history_tokens
    )

    user_prompt = f"{context_text}\n\n用户问题：{query}"

    trimmed_history_l: list[dict[str, Any]] = []
    hist_in = history or []
    if hist_in:
        trimmed_history_l = trim_history(hist_in, max_history_tokens=effective_max_history_tokens)

    if not trimmed_history_l:
        system_prompt = _answer_stream_chunk_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        system_prompt = _generation_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for turn in trimmed_history_l:
            hq = str(turn.get("query", "") or "")
            ha = str(turn.get("answer", "") or "")
            messages.append({"role": "user", "content": hq})
            messages.append({"role": "assistant", "content": ha})
        messages.append({"role": "user", "content": user_prompt})

    answer_parts: list[str] = []

    try:
        stream = chat_complete(
            messages=messages,
            stream=True,
            json_mode=False,
        )
        for part in stream:
            piece = str(part or "")
            if not piece:
                continue
            answer_parts.append(piece)
            yield {"type": "answer_token", "content": piece}

        full_answer = "".join(answer_parts).strip()
        raw_sources = _infer_used_chunks_json(
            model=model,
            query=query,
            answer_text=full_answer,
            context_text=context_text,
            chunk_map=chunk_map,
        )
        if raw_sources:
            yield {"type": "sources_raw", "content": raw_sources}

    except Exception as exc:  # noqa: BLE001
        yield {"type": "answer_token", "content": f"生成失败：{exc}"}
    yield {"type": "done"}


def _infer_used_chunks_json(
    *,
    model: str,
    query: str,
    answer_text: str,
    context_text: str,
    chunk_map: dict[str, Any],
) -> str:
    """第二段调用：非流式、json_object，返回逗号分隔的 chunk key。"""
    chunk_keys = [str(k).strip() for k in chunk_map.keys() if str(k).strip()]
    if not chunk_keys:
        return ""

    system_prompt = (
        "你是来源抽取器。请根据用户问题、回答内容和参考 chunks，判断回答实际使用了哪些 chunk key。\n"
        "你必须返回 JSON 对象，格式：{\"used_chunks\": [\"chunk_1\", \"chunk_2\"]}。\n"
        "仅允许返回给定候选 keys 中的值，不要新增不存在的 key。"
    )
    user_prompt = (
        f"用户问题：{query}\n\n"
        f"回答内容：{answer_text}\n\n"
        f"候选 chunk keys：{', '.join(chunk_keys)}\n\n"
        f"参考 chunks：\n{context_text}\n"
    )

    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    src_extract_cfg = gen_cfg.get("source_extraction")
    if not isinstance(src_extract_cfg, dict):
        src_extract_cfg = {}
    extract_model = str(src_extract_cfg.get("model", "gpt-4o-mini"))
    extract_provider_raw = src_extract_cfg.get("provider")
    extract_provider = (
        str(extract_provider_raw).strip().lower()
        if extract_provider_raw is not None
        else ""
    )
    extract_temperature = float(src_extract_cfg.get("temperature", 0))
    extract_max_tokens = int(src_extract_cfg.get("max_tokens", 1000))

    try:
        raw = str(
            chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                json_mode=True,
                model_name=extract_model,
                provider=extract_provider if extract_provider else None,
                temperature=extract_temperature,
                max_tokens=extract_max_tokens,
            )
            or ""
        ).strip()
        if _debug_used_chunks_enabled():
            print(f"[DEBUG used_chunks] infer_raw_json={raw[:500]!r}", file=sys.stderr)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return ""
        used = parsed.get("used_chunks", [])
        if not isinstance(used, list):
            return ""
        if _debug_used_chunks_enabled():
            print(f"[DEBUG used_chunks] infer_used_list={used!r}", file=sys.stderr)
        valid_set = set(chunk_keys)
        final_keys: list[str] = []
        for x in used:
            k = str(x).strip()
            if not k or k not in valid_set:
                continue
            if k in final_keys:
                continue
            final_keys.append(k)
        if _debug_used_chunks_enabled():
            print(f"[DEBUG used_chunks] infer_final_keys={final_keys!r}", file=sys.stderr)
        return ",".join(final_keys)
    except Exception:
        return ""


def _try_extract_sources_block(buffer: str) -> str | None:
    marker = "SOURCES_START"
    end_marker = "SOURCES_END"
    s0 = buffer.find(marker)
    if s0 == -1:
        return None
    ptr = s0 + len(marker)
    while ptr < len(buffer) and buffer[ptr] in " \t\r":
        ptr += 1
    if ptr < len(buffer) and buffer[ptr] == "\n":
        ptr += 1

    content_start = ptr
    s1 = buffer.find(end_marker, content_start)
    if s1 == -1:
        return None

    inner = buffer[content_start:s1]
    lines_out: list[str] = []
    for line in inner.splitlines():
        stripped = line.strip()
        if "chunk key" in stripped and ("逗号分隔" in stripped or "chunk_1,chunk_2" in stripped):
            continue
        lines_out.append(line)
    joined = "\n".join(lines_out).strip()
    return joined


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

    try:
        raw_content = str(
            chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                json_mode=True,
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001 - 用户要求不向外抛出
        return {
            "answer": f"生成失败：{exc}",
            "used_chunks": [],
            "sources": [],
            "usage": {},
        }

    raw_content = raw_content.strip()
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
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
    id_to_rank: dict[str, int] = {}
    for row in ranked_chunks:
        cid = row.get("chunk_id")
        if isinstance(cid, str) and cid.strip() and cid not in lookup:
            lookup[cid] = row
            id_to_rank[cid] = len(id_to_rank) + 1

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
                "rank": int(id_to_rank.get(cid, 0) or 0),
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
