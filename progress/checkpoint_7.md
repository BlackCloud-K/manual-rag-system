# Checkpoint 7 验收总结

**项目**：DCS 飞行手册多模态 RAG 系统  
**阶段**：阶段七 — Streamlit MVP 前端与多轮对话支持  
**验收日期**：2026 年 4 月  
**状态**：✅ 通过

---

## 交付物

| 交付物 | 说明 |
|--------|------|
| `app.py` | Streamlit 前端入口，对话式界面，集成 RAG pipeline |
| `src/generator/generator.py`（新增函数） | 新增 `trim_history`、`generate_answer_with_history`、`run_rag_pipeline_with_history`、`rewrite_query` 四个函数，原有函数未修改 |
| `scripts/test_history.py` | 多轮对话与 query 改写验证脚本 |
| `config.yaml`（更新） | 追加 `max_history_tokens` 和 `frontend` 配置块 |
| `data/feedback_log.jsonl`（运行时生成） | 用户反馈日志文件 |

---

## 验收结果

### 前端功能验收

| 验收场景 | 结果 | 说明 |
|---------|------|------|
| 启动加载 | ✅ | `streamlit run app.py` 正常启动，页面渲染无报错 |
| 单轮问答 | ✅ | 输入问题后回答正确展示，图片和来源信息正常渲染 |
| 追问（需改写） | ✅ | "第5步具体是什么？"结合冷启动历史，正确命中冷启动章节 |
| 独立问题（不改写） | ✅ | 冷启动追问后输入"RWS的搜索范围是多少"，未被历史干扰 |
| 手册外问题 | ✅ | 拒答时不展示无关图片 |
| 新对话 | ✅ | 侧边栏"新对话"按钮清空历史并重置页面 |
| 反馈按钮 | ✅ | 👍/👎 点击后 `data/feedback_log.jsonl` 正确写入 |
| 图片缺失容错 | ✅ | 图片路径不存在时跳过，不影响回答展示 |

### 多轮对话与 Query 改写验收

| 测试场景 | 结果 | 详情 |
|---------|------|------|
| 无历史行为一致性 | ✅ | `generate_answer_with_history(history=None)` 与 `generate_answer()` 输出结构一致 |
| 追问改写有效性 | ✅ | Q1="冷启动步骤" → Q2="第5步具体是什么" → search_query 包含"冷启动"，命中 `f16c_0099`（程序 › 冷启动，第113页），回答为 EPU GEN/PMG 指示灯确认熄灭 |
| 独立问题不误改写 | ✅ | 历史为冷启动相关 → Q2="RWS的搜索范围是多少" → `query_rewritten=False`，回答关于 RWS |
| 历史裁剪 | ✅ | 构造超长历史（10轮）设置 `max_history_tokens=1000`，裁剪后总 token 数 ≤ 1000，保留的是最近轮次 |
| 改写 API 容错 | ✅ | `rewrite_query` API 调用失败时返回原始 query，不抛异常 |

**对比数据（追问场景 "第5步具体是什么？"）：**

| 条件 | 命中 Chunk | 来源章节 | 回答内容 |
|------|-----------|---------|---------|
| 有历史 + 改写 | `f16c_0099` | 程序 › 冷启动（第113页） | EPU GEN 和 EPU PMG 指示灯确认熄灭 ✅ |
| 无历史 | `f16c_0163` | 空对空武器使用 › AIM-120（第237页） | 按下武器投放按钮发射导弹 ❌ |

---

## 技术方案

### 一、多轮对话历史架构

在 `src/generator/generator.py` 中新增四个函数，**不修改任何现有函数**（`assemble_context`、`generate_answer`、`extract_sources`、`run_rag_pipeline` 全部保持原样）。

#### `trim_history(history, max_history_tokens=2000)`

对历史对话进行 token 裁剪。从最近的轮次向前累积，当总 token 数超过 `max_history_tokens` 时停止，丢弃更早的轮次。使用 `tiktoken` 的 `cl100k_base` encoding 计数，与现有代码一致。单轮超限时整轮丢弃，不截断单条消息。

#### `generate_answer_with_history(query, context_text, chunk_map, history, max_history_tokens)`

支持多轮对话的生成函数。与 `generate_answer()` 的区别仅在于 messages 构造方式：

```
system:    System Prompt（与 generate_answer 完全相同）
user:      Q1（历史轮，不带 chunks）
assistant: A1
user:      Q2（历史轮，不带 chunks）
assistant: A2
...
user:      Q_current + context_text（当前轮，带 chunks）
```

关键设计决策：**历史轮只传 Q&A 文本，不传 chunk context**。原因是历史轮的 chunks 已经"消化"成了 answer，重传会浪费大量 token 且稀释注意力。当前轮的 user message 构造方式与 `generate_answer()` 完全一致。

当 `history` 为 `None` 或空列表时，行为与 `generate_answer()` 完全一致。

#### `rewrite_query(query, history)`

根据对话历史判断当前 query 是否依赖上文语境。调用 GPT-4o-mini（`temperature=0`），只传最近 1-2 轮历史（answer 截断到前 200 字），判断当前问题是否包含指代、省略或承接关系：

- **依赖上文**：改写为自包含的完整问题（如 "第5步具体是什么？" → "F-16 冷启动第5步的具体操作是什么？"）
- **不依赖上文**：原样返回（如 "RWS的搜索范围是多少？" → 原样返回）

API 调用失败时返回原始 query，不阻塞主流程。

#### `run_rag_pipeline_with_history(query, pdf_name_filter, max_context_tokens, history, max_history_tokens)`

端到端 pipeline，在检索前调用 `rewrite_query`：

```
有 history → rewrite_query(query, history) → 用改写后的 query 检索
无 history → 原样检索
```

**改写后的 query 只用于检索，不传给生成层**。生成层的 user message 仍使用原始 query + context_text，因为 LLM 有历史对话上下文，能自行理解指代。

返回 dict 中追加 `original_query`、`search_query`、`query_rewritten` 三个字段供前端调试和日志使用。

### 二、Token 预算控制

| 组件 | Token 上限 | 配置来源 |
|------|-----------|---------|
| Chunk context（当前轮） | 6000 | `config.yaml` → `generation.max_context_tokens` |
| 对话历史 | 2000 | `config.yaml` → `generation.max_history_tokens` |
| System prompt | ~300 | 固定 |
| 总 input 上限 | ~8500 | 各组件之和 |

2000 token 的历史预算约可容纳 3-4 轮精简的 Q&A（每轮约 500-700 token）。裁剪策略为从最早轮次丢弃，保留最近的对话。

### 三、Streamlit 前端架构

**布局方案**：对话式布局（`st.chat_message` + `st.chat_input`），非左右分栏。选择对话式的原因：Streamlit 的 chat 组件原生支持成熟，实现简单，移动端天然适配；图片和来源信息在回答下方嵌入展示。

**Session State 设计**：

- `messages`：完整对话记录（含图片路径、来源信息、改写标记），用于页面渲染
- `history`：精简的 Q&A 列表（`[{"query": ..., "answer": ...}]`），传给 `run_rag_pipeline_with_history()`

**答案渲染流程**：

1. 用户输入 → 立即显示用户消息
2. 调用 `run_rag_pipeline_with_history()` → 显示 spinner
3. 渲染 LLM 回答文本（`st.markdown`）
4. 提取并去重图片路径 → `st.image()` 渲染（限制最大宽度，超过 3 张折叠展示）
5. 来源信息以 `st.expander("📖 查看来源")` 折叠展示，格式为 `第 X 页 · 标题路径`
6. 👍/👎 反馈按钮 → 点击写入 `data/feedback_log.jsonl`

**错误处理**：Qdrant 连接失败、API 超时等异常以 `st.error` 友好提示，不加入对话历史。图片文件不存在时跳过，打印 warning 到控制台。

**侧边栏**：手册选择下拉框（从 `config.yaml` 读取 `frontend.pdf_options`）、"新对话"按钮（清空 session state）。

---

## 阶段七设计决策（相对原技术方案的调整）

| 原计划 | 实际决策 | 原因 |
|--------|---------|------|
| 左右分栏布局（左文字右图片） | 对话式布局（图片内嵌回答下方） | Streamlit 的 `st.columns` 不支持固定侧边栏，对话式组件更成熟，实现更简洁 |
| 流式输出 + 打字机效果 | 非流式 JSON 一次性渲染 | JSON 结构化输出与流式互斥（阶段六已确认），4-8 秒端到端延迟 MVP 可接受 |
| PDF 页码跳转链接 | 仅显示页码和标题路径 | `file://` 在浏览器中因安全策略不可用，内嵌 PDF viewer 增加复杂度，MVP 阶段显示页码即足够定位 |
| 检索层独立（不感知历史） | 检索前加 query 改写 | 测试验证了不改写时追问命中完全错误的章节；改写仅在有历史时触发，无历史时零额外开销 |
| 多轮历史传完整 chunks | 历史只传 Q&A 文本 | 历史轮的 chunks 已消化为 answer，重传浪费 token 并稀释 LLM 注意力 |

---

## 成本分析

| 场景 | 额外开销 | 说明 |
|------|---------|------|
| 无历史的首轮查询 | $0（无额外调用） | 不触发 rewrite_query，与原 pipeline 成本一致 |
| 有历史的追问（需改写） | ~$0.0001（改写调用） + 历史 token 增量 | 改写 prompt 极短（< 200 token），生成层多 ~500-1500 token 历史 |
| 有历史的独立问题（不改写） | ~$0.0001（改写调用） | 仍触发改写调用但结果为原样返回，生成层多历史 token |
| 单次问答总成本（含改写） | ~$0.0009-$0.0012 | 相比阶段六的 $0.0007-$0.0010 增加约 20% |

---

## 已知局限（不影响 MVP）

- **无关问题仍触发检索**：用户输入与手册完全无关的闲聊（如"今天天气怎么样"）时，系统仍会执行完整的检索+生成流程。需要 Agentic RAG 路由能力（LLM 先判断"是否需要检索"）解决，已列为迭代优化项
- **术语/同义词鸿沟**：用户说"超视距"但手册只有"TWS"时，embedding 相似度不足导致检索命中率低。需同义词映射表或术语扩展解决，已 deferred
- **上下文过长偶尔答偏**：多轮对话中历史越积越长，LLM 有时被历史中的无关信息干扰。follow up 可修正，根本改善靠更精准的 history 裁剪和未来的 query 路由
- **图文并排阅读体验受限**：对话式布局中图片和文字纵向排列，用户"一边看步骤一边对照图"的体验不如左右分栏。未来可迁移至更灵活的前端框架（Gradio 或 FastAPI + HTML）实现分栏布局

---

## 下一步：迭代优化方向

| 优先级 | 优化项 | 解决的问题 | 实现复杂度 |
|--------|--------|-----------|-----------|
| P1 | Agentic RAG 路由 | 无关问题不触发检索，节省 token 和时间；支持 LLM 主动追问用户 | 中 |
| P1 | 同义词/术语映射表 | 玩家俗语与手册术语的鸿沟（"超视距"→"TWS"） | 低 |
| P2 | 流式输出 + 引用分离 | 减少用户等待体感；先流式输出文字，末尾追加结构化引用 | 中 |
| P2 | PDF 内嵌查看器 | 点击来源直接在页面内查看原文 | 中 |
| P2 | 查询分解（复合问题） | "从冷启动到滑行完整流程"类跨章节问题 | 中 |
| P3 | 前端框架升级 | 图文并排布局、更灵活的交互控件 | 高 |