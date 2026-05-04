"""Build F-16C retrieval testset: stratified sampling + GPT-4o-mini questions + out-of-scope."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKS_PATH = ROOT / "data" / "final" / "chunks_f16c_final.json"
OUTPUT_PATH = ROOT / "data" / "testset_f16c.json"
TARGET_SAMPLE = 13
MIN_TEXT_LEN = 150
FRAGMENT_LEN = 800

OUT_OF_SCOPE = [
    "F-16C的最大航程是多少海里",
    "如何在DCS多人模式中申请服务器管理员权限",
]

SYSTEM_PROMPT = """你是一个 handbook / manual 节选问答测试集的构造助手。
你的任务是根据给定的节选片段，生成一个真实用户可能提出的中文问题。

判断好问题的标准：
- 问题能且只能从给定片段中找到答案
- 使用玩家/飞行员视角自然提问，不照抄原文结构
- 不要问"第X步是什么"这类步骤编号问题
- 不要在问题里直接出现答案关键词

以下是好问题和坏问题的对比示例：

【例1 - 步骤类片段】
片段：「起飞前检查第12步：滑油压力 检查 正常指示为15-65 psi」
坏问题：第12步检查什么？
好问题：起飞前滑油压力的正常范围是多少？

【例2 - 描述类片段】
片段：「TWS模式是一种多目标跟踪模式。鉴于扫描范围过大，跟踪文件会存在较长的刷新时间...目标进行规避机动时雷达可能脱锁」
坏问题：TWS模式的跟踪文件刷新时间是多少？
好问题：使用TWS模式跟踪目标时有什么缺点？

【例3 - 系统介绍类片段】
片段：「EPU为肼动力独立装置，可为飞机提供大约10到15分钟的应急液压和电力。如果飞行员失去了发动机，通常将会使用EPU」
坏问题：EPU是什么装置？
好问题：发动机失效后飞机靠什么维持液压和电力，能撑多久？

只输出问题本身，不要输出任何解释或其他内容。"""


def _load_chunks() -> list[dict]:
    raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def _filter_by_text_len(chunks: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in chunks:
        text = c.get("text", "")
        if isinstance(text, str) and len(text) >= MIN_TEXT_LEN:
            out.append(c)
    return out


def _sample_chunks(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    """Stratified sample: prefer title_h1 groups with >=3 chunks, 1-2 per group, 13 total, avoid duplicate title_h2 when possible."""
    by_h1: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        h1 = str(c.get("title_h1", ""))
        by_h1[h1].append(c)

    large_h1 = [h1 for h1, lst in by_h1.items() if len(lst) >= 3]
    small_h1 = [h1 for h1, lst in by_h1.items() if len(lst) < 3]

    random.seed(42)
    random.shuffle(large_h1)
    random.shuffle(small_h1)

    selected: list[dict] = []
    seen_h2: set[str] = set()

    def _pick_ids(picked: list[dict]) -> set[str]:
        return {str(x.get("chunk_id", "")) for x in picked if x.get("chunk_id")}

    def pick_from_group(h1_key: str, max_take: int) -> None:
        nonlocal selected
        if len(selected) >= TARGET_SAMPLE:
            return
        pool = list(by_h1[h1_key])
        random.shuffle(pool)
        take = min(max_take, TARGET_SAMPLE - len(selected))
        if take <= 0:
            return

        picked: list[dict] = []
        picked_ids = _pick_ids(picked)

        for c in pool:
            if len(picked) >= take:
                break
            h2 = str(c.get("title_h2", ""))
            if h2 in seen_h2:
                continue
            picked.append(c)
            picked_ids.add(str(c.get("chunk_id", "")))
            seen_h2.add(h2)

        if len(picked) < take:
            for c in pool:
                if len(picked) >= take:
                    break
                cid = str(c.get("chunk_id", ""))
                if cid in picked_ids:
                    continue
                picked.append(c)
                picked_ids.add(cid)
                seen_h2.add(str(c.get("title_h2", "")))

        selected.extend(picked)

    for h1 in large_h1:
        if len(selected) >= TARGET_SAMPLE:
            break
        max_take = random.randint(1, 2)
        pick_from_group(h1, max_take)

    for h1 in small_h1:
        if len(selected) >= TARGET_SAMPLE:
            break
        max_take = random.randint(1, 2)
        pick_from_group(h1, max_take)

    if len(selected) < TARGET_SAMPLE:
        remaining: list[dict] = []
        used_ids = _pick_ids(selected)
        for c in chunks:
            cid = str(c.get("chunk_id", ""))
            if cid and cid not in used_ids:
                remaining.append(c)
        random.shuffle(remaining)
        for c in remaining:
            if len(selected) >= TARGET_SAMPLE:
                break
            h2 = str(c.get("title_h2", ""))
            if h2 in seen_h2:
                continue
            selected.append(c)
            seen_h2.add(h2)
        for c in remaining:
            if len(selected) >= TARGET_SAMPLE:
                break
            cid = str(c.get("chunk_id", ""))
            if any(str(x.get("chunk_id", "")) == cid for x in selected):
                continue
            selected.append(c)

    selected = selected[:TARGET_SAMPLE]
    involved_h1 = sorted({str(c.get("title_h1", "")) for c in selected})
    return selected, involved_h1


def _generate_question(client: OpenAI, chunk: dict) -> str | None:
    text = chunk.get("text", "")
    if not isinstance(text, str):
        text = str(text or "")
    fragment = text[:FRAGMENT_LEN]
    user_content = f"""以下为节选片段：

{fragment}

请据此生成一个自然的中文问题。"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        q = (response.choices[0].message.content or "").strip()
        return q if q else None
    except Exception as exc:
        print(f"[WARN] GPT 调用失败 chunk_id={chunk.get('chunk_id')!r}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    chunks = _load_chunks()
    filtered = _filter_by_text_len(chunks)
    sampled, involved_h1 = _sample_chunks(filtered)

    print("抽样结果（人工核查）：")
    for c in sampled:
        print(
            f"  chunk_id={c.get('chunk_id')}  "
            f"title_h1={c.get('title_h1')!r}  "
            f"title_h2={c.get('title_h2')!r}"
        )

    load_dotenv(ROOT / ".env")
    client = OpenAI()

    records: list[dict] = []
    for chunk in sampled:
        question = _generate_question(client, chunk)
        if question is None:
            continue
        next_id = len(records) + 1
        records.append(
            {
                "id": f"t{next_id:03d}",
                "question": question,
                "source_chunk_id": str(chunk.get("chunk_id", "")),
                "title_h1": chunk.get("title_h1"),
                "title_h2": chunk.get("title_h2"),
                "expected_in_topn": True,
                "note": "",
            }
        )
        time.sleep(0.5)

    k = len(records)
    for qtext in OUT_OF_SCOPE:
        k += 1
        records.append(
            {
                "id": f"t{k:03d}",
                "question": qtext,
                "source_chunk_id": None,
                "title_h1": None,
                "title_h2": None,
                "expected_in_topn": False,
                "note": "手册外问题，预期不召回相关内容",
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n已写入: {OUTPUT_PATH}")
    in_scope_n = len(records) - len(OUT_OF_SCOPE)
    print(
        f"汇总：共生成 {len(records)} 题（手册内 {in_scope_n} 题，手册外 {len(OUT_OF_SCOPE)} 题）"
    )
    print(f"抽样涉及的 title_h1 大类（共 {len(involved_h1)} 个）：")
    for h1 in involved_h1:
        print(f"  - {h1!r}")


if __name__ == "__main__":
    main()
