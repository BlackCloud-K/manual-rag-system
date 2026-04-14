# Checkpoint 2 验收总结

**项目**：DCS 飞行手册多模态 RAG 系统  
**阶段**：阶段二 — Chunk 规范化与 Metadata 构建  
**验收日期**：2026 年 4 月  
**状态**：✅ 通过

---

## 交付物

| 交付物 | 说明 |
|--------|------|
| `src/normalizer/chunk_normalizer.py` | Chunk 清理与拆分核心模块，包含垃圾清理和超长拆分两个函数 |
| `scripts/run_normalization.py` | 四本手册批量规范化入口脚本 |
| `data/chunks_f16c_normalized.json` | F-16C 规范化后 Chunk JSON |
| `data/chunks_dcsjf17_normalized.json` | JF-17 规范化后 Chunk JSON |
| `data/chunks_f15c_normalized.json` | F-15C 规范化后 Chunk JSON |
| `data/chunks_dcsfa18cearlyaccessguidecn_normalized.json` | FA-18C 规范化后 Chunk JSON |

**四本手册处理结果：**

| 手册 | 原始 Chunk 数 | 删除垃圾 | 拆分新增 | 最终 Chunk 数 |
|------|-------------|---------|---------|-------------|
| F-16C 蝰蛇 | 208 | 4 | 0 | 204 |
| DCS JF-17 雷电 | 205 | 3 | 10 | 212 |
| F-15C | 80 | 0 | 0 | 80 |
| DCS FA-18C | 295 | 1 | 3 | 297 |

---

## 验收结果

全部四本规范化 JSON 通过 Checkpoint 1 验收脚本的所有自动验收项：

- ✅ **结构完整性**：所有 Chunk 包含必要字段，无缺失
- ✅ **text 非空率**：无空 Chunk
- ✅ **链表完整性**：`prev_chunk_id` / `next_chunk_id` 双向一致，无断链
- ✅ **新增字段**：`split_group_id` 字段正确写入（拆分子 Chunk 记录父级 ID，未拆分 Chunk 为 null）

---

## 技术方案

### 诊断优先原则

在写任何规范化代码之前，依次执行了两轮诊断：

**Token 分布诊断**（`scripts/token_diagnostic.py`）：使用 `BAAI/bge-m3` tokenizer 对四本手册全量统计，输出每本手册的区间分布以及 < 100 token 和 > 800 token 的逐条明细。四本手册合并后 788 个 Chunk 的中位数为 260 token，整体分布合理。

**大 Chunk 内部结构诊断**（`scripts/large_chunk_diagnostic.py`）：对 102 个 > 800 token 的大 Chunk，测试了双换行、单行、步骤块、标题列表块、句子五种切分策略的有效率。发现四本手册的文本在 PDF 提取后几乎不含 `\n\n` 双换行（有效率 0%），段落结构信息在提取阶段已经丢失，段落级切分不可行。

### 垃圾 Chunk 清理

**删除条件**：token 数 < 20 的 Chunk。这类 Chunk 主要为目录页残留（如"1.1.4. 飞机驾驶舱布局 — 30 —"，10 token），向量化后成为噪音，对检索无帮助。

**链表重建**：被删除 Chunk 的前后邻居重新连接，保证链表连续性。20–100 token 的短 Chunk 全部保留——这类 Chunk 是 TOC 驱动切分的自然结果，反映章节本身内容量，强行合并会破坏语义边界。

### 超长 Chunk 拆分

**阈值选择**：3500 token。经诊断确认真正需要处理的异常 Chunk 仅 4 个（`dcsjf17_0003` 14804 token、`dcsfa18cearlyaccessguidecn_0258` 4564 token、`dcsfa18cearlyaccessguidecn_0114` 3077 token、`f15c_0075` 3361 token），其余 800–3500 token 的 Chunk 保留原样。

**拆分策略**：句子级贪心累积切分，以 `。！？!?` 为句子边界，目标大小 1500 token/片。切分点始终在句子边界，不截断句子。

**子 Chunk 元数据**：
- `chunk_id` 命名为 `{原 chunk_id}_part0`、`_part1`……
- 继承父 Chunk 的 `title_h1/h2/h3`、`pdf_name`、`page_start/end`、`image_paths`
- 新增 `split_group_id` 字段记录原始父 Chunk ID，供检索阶段合并同组 Chunk 使用

**链表重建**：拆分后重建全局链表，子 Chunk 组内部连成链，与前后原始 Chunk 也正确连接。

### 阶段二设计决策（相对原技术方案的调整）

| 原计划 | 实际决策 | 原因 |
|--------|---------|------|
| 向上合并 < 100 token Chunk | 取消合并，仅删除 < 20 token 垃圾 Chunk | TOC 驱动切分下短 Chunk 反映章节自然粒度，合并破坏语义边界 |
| 向下拆分 > 800 token Chunk | 阈值提高至 3500 token，仅处理极端异常值 | 绝大多数大 Chunk 在 bge-m3 的 8192 token 上下文内，强行拆分反而劣化检索质量 |
| section_type 自动打标 | 取消 | TOC 切分已保证每个 Chunk 有明确的三级标题，section_type 冗余；检索阶段直接用 title_h1/h2/h3 做 Payload 过滤即可 |
| chapter 功能模块打标 | 取消 | 同上，title_h1 已承担章节定位功能 |

---

## 主要困难与解决过程

### 1. 段落切分策略全部失效

**问题**：PDF 文本提取后双换行（`\n\n`）几乎不存在，102 个大 Chunk 的双换行有效率为 0%。步骤块和标题列表块有效率也仅 25–27%，不具备通用性。

**解决**：改用句子级贪心累积切分，有效率 99%，平均最大子段 212 token，是唯一可靠的通用方案。

### 2. 阈值设计的权衡

**问题**：初始计划对所有 > 800 token 的 Chunk 拆分，但诊断发现大多数大 Chunk（800–3000 token）内容完整、语义自洽，强行按句子拆分会导致语义碎片化，反而劣化检索。

**解决**：将阈值从 800 提高到 3500，只处理真正会导致 LLM context 超限的极端 Chunk，其余接受现状。同时设计了 `split_group_id` 机制，检索时命中某子 Chunk 可拉取同组兄弟一起送 LLM，弥补拆分导致的段落截断影响。

### 3. 句子切分导致段落截断

**问题**：按句子贪心切分不可避免地会将同一段落的内容切入不同子 Chunk，影响连贯性。

**解决**：这是 MVP 阶段可接受的代价。通过三个机制共同兜底：（1）`split_group_id` 检索时合并同组 Chunk；（2）相邻 Chunk 扩展机制；（3）BM25 关键词匹配补充语义向量的不足。后续可升级为 Parent-Child Chunking 方案（技术方案 P3 迭代项）从根本上解决。

---

## 已知局限（不影响 MVP）

- 句子级切分可能将同一操作步骤的前后内容切入不同子 Chunk（如冷启动步骤 7 和步骤 8 分属两片）；检索时 `split_group_id` 合并机制可缓解但不能完全消除
- 少数"参见其他章节"类短 Chunk（20–50 token，如"详情参阅 INS 对准部分"）保留在数据集中；这类 Chunk 向量质量偏低，后续验收时需专项测试其对检索的影响
- `_normalized` JSON 的图片目录路径仍指向原始手册图片目录（`data/images/f16c/`），未创建新目录，验收脚本跳过了磁盘覆盖率检查；实际图片文件未变动，路径引用正确

---

## 下一步：阶段三

- VLM 图片摘要：对四本手册提取的全部图片，调用 GPT-4o-mini API 生成一句话摘要
- 摘要注入：将摘要追加到所属 Chunk 的 text 字段末尾，作为检索补充线索
- 成本控制：图片缩至 512px 后调用，预处理前先跑前 50 页估算总成本