from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.generator.generator import assemble_context, generate_answer
from src.retriever import retriever as retriever_mod


# CLI 短名 -> data/final 中 chunk 的 pdf_name 字段（需与向量库 payload 一致）
_PDF_ALIASES: dict[str, str] = {
    "f16c": "F-16C\u201c蝰蛇\u201d.pdf",
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


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1_000_000 * 0.15 + completion_tokens / 1_000_000 * 0.60


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 端到端手动测试：检索 + LLM 生成")
    parser.add_argument("query", help="用户问题")
    parser.add_argument(
        "--pdf",
        dest="pdf",
        default=None,
        help="手册过滤：短名如 f16c、dcsjf17，或完整 pdf_name",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    cfg = _load_yaml_config()
    retrieval_cfg = cfg.get("retrieval")
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
    gen_cfg = cfg.get("generation")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}

    top_k = int(retrieval_cfg.get("hybrid_prefetch", 50))
    rerank_top_n = int(retrieval_cfg.get("rerank_top_n", 5))
    max_context_tokens = int(gen_cfg.get("max_context_tokens", 6000))

    pdf_filter = _resolve_pdf_name(args.pdf)

    print("========== RAG 端到端测试 ==========")
    print(f"问题：{args.query}")
    print(f"手册过滤：{pdf_filter or '-'}")

    t0 = time.perf_counter()
    chunks_lookup = retriever_mod.load_chunks_lookup(pdf_filter)
    hits = retriever_mod.hybrid_search(
        args.query,
        top_k=top_k,
        pdf_name_filter=pdf_filter,
    )
    candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
    reranked = retriever_mod.rerank(args.query, candidates, top_n=rerank_top_n)
    t_retrieve_end = time.perf_counter()

    chunk_map, context_text = assemble_context(reranked, max_context_tokens=max_context_tokens)
    t_assemble_end = time.perf_counter()

    result = generate_answer(args.query, context_text, chunk_map, ranked_chunks=reranked)
    t_end = time.perf_counter()

    retrieve_s = t_retrieve_end - t0
    gen_s = t_end - t_assemble_end
    total_s = t_end - t0

    print("\n---------- 回答 ----------")
    print(result.get("answer", ""))

    print("\n---------- 来源 ----------")
    sources = result.get("sources") or []
    if not sources:
        print("（无）")
    else:
        for i, src in enumerate(sources, start=1):
            page = src.get("page", "")
            title_path = (src.get("title_path") or "").strip()
            cid = src.get("chunk_id", "")
            if title_path:
                head = f"[{i}] 第{page}页 · {title_path}（chunk: {cid}）"
            else:
                head = f"[{i}] 第{page}页（chunk: {cid}）"
            print(head)
            imgs = src.get("image_paths") or []
            if isinstance(imgs, list) and imgs:
                names = ", ".join(Path(p).name for p in imgs if isinstance(p, str))
                print(f"    图片：{names}")
            else:
                print("    图片：无")

    usage = result.get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", 0) or 0)
    cost = _estimate_cost_usd(pt, ct)

    print("\n---------- 统计 ----------")
    print(f"检索耗时：{retrieve_s:.2f}s")
    print(f"生成耗时：{gen_s:.2f}s")
    print(f"端到端耗时：{total_s:.2f}s")
    print(f"Token 消耗：prompt={pt}, completion={ct}, total={tt}")
    print(f"预估成本：${cost:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
