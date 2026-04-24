from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tiktoken
import yaml

from src.generator.generator import (
    assemble_context,
    generate_answer,
    generate_answer_with_history,
    run_rag_pipeline_with_history,
    trim_history,
)
import src.generator.generator as generator_mod
from src.retriever import retriever as retriever_mod


_PDF_ALIASES: dict[str, str] = {
    "f16c": "F-16C“蝰蛇”.pdf",
    "dcsjf17": "DCS_ JF-17 _雷电_.pdf",
    "jf17": "DCS_ JF-17 _雷电_.pdf",
    "f15c": "F-15C.pdf",
    "fa18c": "DCS FA-18C Early Access Guide CN.pdf",
    "dcsfa18c": "DCS FA-18C Early Access Guide CN.pdf",
}


def _load_yaml_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _resolve_pdf_name(user_value: str | None) -> str | None:
    if user_value is None or not str(user_value).strip():
        return None
    key = str(user_value).strip().lower()
    if key in _PDF_ALIASES:
        return _PDF_ALIASES[key]
    return str(user_value).strip()


def _history_tokens(history: list[dict]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for turn in history:
        total += len(enc.encode(str(turn.get("query", "") or "")))
        total += len(enc.encode(str(turn.get("answer", "") or "")))
    return total


def _safe_text_for_console(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return text
    except Exception:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _run_test_1(pdf_filter: str | None, max_context_tokens: int) -> None:
    query = "F-16 冷启动的步骤是什么？"
    chunks_lookup = retriever_mod.load_chunks_lookup(pdf_filter)
    hits = retriever_mod.hybrid_search(query, top_k=50, pdf_name_filter=pdf_filter)
    candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
    reranked = retriever_mod.rerank(query, candidates, top_n=10)
    chunk_map, context_text = assemble_context(reranked, max_context_tokens=max_context_tokens)

    base = generate_answer(query, context_text, chunk_map, ranked_chunks=reranked)
    with_none = generate_answer_with_history(
        query=query,
        context_text=context_text,
        chunk_map=chunk_map,
        history=None,
    )

    required_keys = {"answer", "used_chunks", "sources", "usage"}
    assert required_keys.issubset(base.keys()), "generate_answer 返回结构不完整"
    assert required_keys.issubset(with_none.keys()), "generate_answer_with_history(history=None) 返回结构不完整"
    assert isinstance(with_none.get("used_chunks"), list), "used_chunks 类型应为 list"
    assert isinstance(with_none.get("sources"), list), "sources 类型应为 list"
    assert isinstance(with_none.get("usage"), dict), "usage 类型应为 dict"

    print("[PASS] 测试1：无历史时结构一致（answer 内容允许不同）")


def _run_test_2a(pdf_filter: str | None, max_history_tokens: int) -> None:
    q1 = "F-16 冷启动的步骤是什么？"
    q2 = "第5步具体是什么？"

    r1 = run_rag_pipeline_with_history(
        query=q1,
        pdf_name_filter=pdf_filter,
        history=None,
        max_history_tokens=max_history_tokens,
    )
    a1 = str(r1.get("answer", "") or "")
    history = [{"query": q1, "answer": a1}]

    r2 = run_rag_pipeline_with_history(
        query=q2,
        pdf_name_filter=pdf_filter,
        history=history,
        max_history_tokens=max_history_tokens,
    )
    a2 = str(r2.get("answer", "") or "")
    r2_no_history = run_rag_pipeline_with_history(
        query=q2,
        pdf_name_filter=pdf_filter,
        history=None,
        max_history_tokens=max_history_tokens,
    )
    a2_no_history = str(r2_no_history.get("answer", "") or "")
    search_query = str(r2.get("search_query", "") or "")
    rewritten = bool(r2.get("query_rewritten", False))
    source_ids = [str(s.get("chunk_id", "")) for s in (r2.get("sources") or []) if isinstance(s, dict)]

    print("[INFO] 测试2a（需要改写的追问）")
    print("Q1:", q1)
    print("A1:", _safe_text_for_console(a1))
    print("Q2:", q2)
    print("search_query:", _safe_text_for_console(search_query))
    print("query_rewritten:", rewritten)
    print("A2（带历史）:", _safe_text_for_console(a2))
    print("A2（不带历史）:", _safe_text_for_console(a2_no_history))

    assert "冷启动" in search_query, "测试2a失败：search_query 未补全到冷启动语境"
    assert rewritten is True, "测试2a失败：query_rewritten 应为 True"
    assert "f16c_0109" not in source_ids, "测试2a失败：来源仍命中着陆章节 f16c_0109"
    assert ("EPU GEN" in a2 or "EPU PMG" in a2 or "确认熄灭" in a2), "测试2a失败：回答未体现冷启动第5步要点"

    out_dir = ROOT / "scripts" / "temp_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"history_test_result_2a_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_payload = {
        "pdf_name_filter": pdf_filter,
        "q1": q1,
        "a1": a1,
        "q2": q2,
        "search_query": search_query,
        "query_rewritten": rewritten,
        "a2_with_history": a2,
        "a2_without_history": a2_no_history,
        "with_history_used_chunks": r2.get("used_chunks", []),
        "with_history_sources": r2.get("sources", []),
        "without_history_used_chunks": r2_no_history.get("used_chunks", []),
        "without_history_sources": r2_no_history.get("sources", []),
        "with_history_usage": r2.get("usage", {}),
        "without_history_usage": r2_no_history.get("usage", {}),
    }
    out_path.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] 测试2结果已保存：{out_path}")
    print("[PASS] 测试2a：追问改写与检索命中验证通过")


def _run_test_2b(pdf_filter: str | None, max_history_tokens: int) -> None:
    q1 = "F-16 冷启动的步骤是什么？"
    q2 = "RWS的搜索范围是多少？"

    r1 = run_rag_pipeline_with_history(
        query=q1,
        pdf_name_filter=pdf_filter,
        history=None,
        max_history_tokens=max_history_tokens,
    )
    a1 = str(r1.get("answer", "") or "")
    history = [{"query": q1, "answer": a1}]

    r2 = run_rag_pipeline_with_history(
        query=q2,
        pdf_name_filter=pdf_filter,
        history=history,
        max_history_tokens=max_history_tokens,
    )
    a2 = str(r2.get("answer", "") or "")
    search_query = str(r2.get("search_query", "") or "")
    rewritten = bool(r2.get("query_rewritten", False))

    print("[INFO] 测试2b（不需要改写的独立问题）")
    print("Q2:", q2)
    print("search_query:", _safe_text_for_console(search_query))
    print("query_rewritten:", rewritten)
    print("A2:", _safe_text_for_console(a2))

    assert search_query == q2, "测试2b失败：独立问题不应被改写"
    assert rewritten is False, "测试2b失败：query_rewritten 应为 False"
    assert ("RWS" in a2 or "搜索范围" in a2 or "扫描" in a2), "测试2b失败：回答未体现 RWS 搜索范围主题"
    print("[PASS] 测试2b：独立问题不改写验证通过")


def _run_test_2c() -> None:
    original = "第5步具体是什么？"
    history = [{"query": "F-16 冷启动的步骤是什么？", "answer": "..."},]

    saved_openai = generator_mod.OpenAI

    class _BrokenOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("mock api failure")

    try:
        generator_mod.OpenAI = _BrokenOpenAI  # type: ignore[assignment]
        rewritten = generator_mod.rewrite_query(original, history)
    finally:
        generator_mod.OpenAI = saved_openai  # type: ignore[assignment]

    assert rewritten == original, "测试2c失败：API失败时应返回原始 query"
    print("[PASS] 测试2c：改写 API 失败容错验证通过")


def _run_test_3() -> None:
    long_history = []
    for i in range(10):
        long_history.append(
            {
                "query": f"Q{i}: " + ("token " * 500),
                "answer": f"A{i}: " + ("token " * 500),
            }
        )

    trimmed = trim_history(long_history, max_history_tokens=1000)
    total = _history_tokens(trimmed)
    assert total <= 1000, f"裁剪后 token 超限: {total}"

    if trimmed:
        first_idx = int(str(trimmed[0]["query"]).split(":")[0].replace("Q", ""))
        assert first_idx >= len(long_history) - len(trimmed), "裁剪结果未优先保留最近轮次"

    print("[PASS] 测试3：历史裁剪满足 token 上限，且优先保留最近轮次")


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮历史支持测试脚本")
    parser.add_argument(
        "--pdf",
        dest="pdf",
        default="F-16C“蝰蛇”.pdf",
        help="pdf_name_filter（支持短名如 f16c，默认 F-16C 手册）",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    cfg = _load_yaml_config()
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    max_context_tokens = int(gen_cfg.get("max_context_tokens", 6000))
    max_history_tokens = int(gen_cfg.get("max_history_tokens", 2000))
    pdf_filter = _resolve_pdf_name(args.pdf)

    print("========== 历史对话支持测试 ==========")
    print(f"pdf_name_filter: {pdf_filter}")
    print(f"max_context_tokens: {max_context_tokens}")
    print(f"max_history_tokens: {max_history_tokens}")

    _run_test_1(pdf_filter=pdf_filter, max_context_tokens=max_context_tokens)
    _run_test_2a(pdf_filter=pdf_filter, max_history_tokens=max_history_tokens)
    _run_test_2b(pdf_filter=pdf_filter, max_history_tokens=max_history_tokens)
    _run_test_2c()
    _run_test_3()

    print("全部测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
