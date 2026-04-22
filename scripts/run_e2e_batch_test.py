from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.generator import assemble_context, generate_answer
from src.retriever import retriever as retriever_mod


def _load_yaml_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1_000_000 * 0.15 + completion_tokens / 1_000_000 * 0.60


def _read_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] 跳过第 {line_no} 行（JSON 解析失败）：{exc}")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _run_one_question(
    row: dict,
    top_k: int,
    rerank_top_n: int,
    max_context_tokens: int,
) -> dict:
    qid = row.get("id", "")
    query = str(row.get("query", "") or "")
    pdf_filter = row.get("pdf_filter")
    pdf_name = str(pdf_filter).strip() if pdf_filter is not None else ""
    qtype = str(row.get("type", "") or "")
    notes = str(row.get("notes", "") or "")

    base: dict = {
        "id": qid,
        "query": query,
        "type": qtype,
        "notes": notes,
    }

    retrieval_seconds = 0.0
    generation_seconds = 0.0
    reranked: list = []

    t0 = time.perf_counter()
    try:
        chunks_lookup = retriever_mod.load_chunks_lookup(pdf_name or None)
        hits = retriever_mod.hybrid_search(
            query,
            top_k=top_k,
            pdf_name_filter=pdf_name or None,
        )
        candidates = retriever_mod.expand_neighbors(hits, chunks_lookup)
        reranked = retriever_mod.rerank(query, candidates, top_n=rerank_top_n)
    except Exception as exc:  # noqa: BLE001
        t1 = time.perf_counter()
        retrieval_seconds = t1 - t0
        base["error"] = str(exc)
        base["answer"] = ""
        base["used_chunks"] = []
        base["sources"] = []
        base["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        base["cost_usd"] = 0.0
        base["timing"] = {
            "retrieval_seconds": round(retrieval_seconds, 2),
            "generation_seconds": 0.0,
            "total_seconds": round(retrieval_seconds, 2),
        }
        return base

    t1 = time.perf_counter()
    retrieval_seconds = t1 - t0

    t2 = time.perf_counter()
    try:
        chunk_map, context_text = assemble_context(
            reranked,
            max_context_tokens=max_context_tokens,
        )
        result = generate_answer(
            query,
            context_text,
            chunk_map,
            ranked_chunks=reranked,
        )
    except Exception as exc:  # noqa: BLE001
        t3 = time.perf_counter()
        generation_seconds = t3 - t2
        base["error"] = str(exc)
        base["answer"] = ""
        base["used_chunks"] = []
        base["sources"] = []
        base["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        base["cost_usd"] = 0.0
        base["timing"] = {
            "retrieval_seconds": round(retrieval_seconds, 2),
            "generation_seconds": round(generation_seconds, 2),
            "total_seconds": round(retrieval_seconds + generation_seconds, 2),
        }
        return base

    t3 = time.perf_counter()
    generation_seconds = t3 - t2

    usage = result.get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", 0) or 0)
    cost = _cost_usd(pt, ct)

    answer = str(result.get("answer", "") or "")
    err_text: str | None = None
    if answer.startswith("生成失败："):
        err_text = answer.removeprefix("生成失败：").strip() or answer

    out = {
        **base,
        "answer": answer,
        "used_chunks": result.get("used_chunks") or [],
        "sources": result.get("sources") or [],
        "timing": {
            "retrieval_seconds": round(retrieval_seconds, 2),
            "generation_seconds": round(generation_seconds, 2),
            "total_seconds": round(retrieval_seconds + generation_seconds, 2),
        },
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
        },
        "cost_usd": round(cost, 4),
    }
    if err_text is not None:
        out["error"] = err_text
    return out


def _build_summary(results: list[dict]) -> dict:
    by_type: dict[str, dict[str, float | int]] = {}
    type_stats: dict[str, list[tuple[float, float]]] = defaultdict(list)

    total_prompt = 0
    total_completion = 0
    total_all = 0
    total_cost = 0.0

    for r in results:
        qtype = str(r.get("type", "") or "") or "（未标注）"
        timing = r.get("timing") or {}
        total_s = float(timing.get("total_seconds", 0) or 0)
        cost = float(r.get("cost_usd", 0) or 0)
        type_stats[qtype].append((total_s, cost))

        u = r.get("usage") or {}
        total_prompt += int(u.get("prompt_tokens", 0) or 0)
        total_completion += int(u.get("completion_tokens", 0) or 0)
        total_all += int(u.get("total_tokens", 0) or 0)
        total_cost += float(r.get("cost_usd", 0) or 0)

    for qtype, pairs in type_stats.items():
        n = len(pairs)
        avg_time = sum(p[0] for p in pairs) / n if n else 0.0
        avg_cost = sum(p[1] for p in pairs) / n if n else 0.0
        by_type[qtype] = {
            "count": n,
            "avg_time": round(avg_time, 2),
            "avg_cost": round(avg_cost, 4),
        }

    return {
        "by_type": by_type,
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_all,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="批量 RAG 端到端测试")
    parser.add_argument(
        "--input",
        default=str(ROOT / "scripts" / "temp_output" / "e2e_test_questions.jsonl"),
        help="JSONL 问题文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果 JSON 路径；默认 scripts/temp_output/e2e_test_results_<时间戳>.json（%%Y%%m%%d_%%H%%M%%S）",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / "scripts" / "temp_output" / f"e2e_test_results_{stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    questions = _read_questions(input_path)
    n = len(questions)
    if n == 0:
        print(f"[ERROR] 未从 {input_path} 读取到有效题目。")
        return 1

    print("[INFO] 首题可能触发 embedding / reranker 冷加载（约数秒），summary 中的 avg_time 含首题。")

    run_started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict] = []

    for i, row in enumerate(questions, start=1):
        qid = row.get("id", f"line{i}")
        qtype = str(row.get("type", "") or "")
        one = _run_one_question(row, top_k, rerank_top_n, max_context_tokens)
        results.append(one)

        total_s = one.get("timing", {}).get("total_seconds", 0)
        status = "FAIL" if "error" in one else "OK"
        print(f"[{i}/{n}] {qid} - {qtype} - {total_s}s - {status}")

    wall_total = time.perf_counter() - run_started
    avg_time = wall_total / n if n else 0.0

    payload = {
        "test_run": {
            "timestamp": started_at,
            "total_questions": n,
            "total_time_seconds": round(wall_total, 2),
            "avg_time_seconds": round(avg_time, 2),
        },
        "results": results,
        "summary": _build_summary(results),
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] 已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
