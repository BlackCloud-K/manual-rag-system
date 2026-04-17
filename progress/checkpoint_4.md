# Checkpoint 4 验收总结

**项目**：DCS 飞行手册多模态 RAG 系统  
**阶段**：阶段四 — 向量化与向量库构建  
**验收日期**：2026 年 4 月  
**状态**：✅ 通过

---

## 交付物

| 交付物 | 说明 |
|--------|------|
| `scripts/run_embedding.py` | 批量向量化脚本，使用 bge-m3 生成 Dense + Sparse 双路向量，支持断点续传 |
| `scripts/run_upload_qdrant.py` | Qdrant 写入脚本，创建 Collection、批量上传向量与 Payload、建立字段索引 |
| `scripts/run_checkpoint4.py` | 自动验收脚本，含跨手册 Hybrid Search 测试与 F-16C 精细召回测试 |
| `data/embeddings/embeddings_cache.pkl` | 787 条 Chunk 的 Dense + Sparse 向量缓存 |

**向量化统计：**

| 指标 | 数值 |
|------|------|
| 处理 Chunk 总数 | 787 |
| Dense 向量维度 | 1024 |
| Sparse 向量格式 | `dict[int, float]`（非零词 index → 权重） |
| 向量化耗时 | 105 秒（RTX 5070，use_fp16=True，batch_size=32） |
| 缓存文件大小 | < 15 MB |

**Qdrant Collection 统计：**

| 指标 | 数值 |
|------|------|
| Collection 名称 | `dcs_manuals` |
| 实际 Point 数 | 787 |
| 上传耗时 | 11 秒（batch_size=64） |
| Payload 索引字段 | `pdf_name`、`title_h1`、`title_h2`、`page_start` |

---

## 验收结果

全部自动验收项通过：

- ✅ **Point 总数**：787，与阶段三输出 JSON 条目数完全一致
- ✅ **Payload 索引**：4 个字段索引均已建立，无缺失
- ✅ **Hybrid Search 可用**：5 个跨手册 Query 均返回语义相关结果，响应时间 3.6 秒
- ✅ **F-16C 精细召回**：14 个测试 Query 中 10 个召回准确，2 个部分准确，2 个明确失败（英文缩写类）

---

## 技术方案

### 基础设施

Qdrant 通过 Docker 部署在本地，版本 1.17.1，存储卷挂载至 `C:\Users\Keven\qdrant_storage`，配置 `--restart unless-stopped` 实现开机自启。服务监听 `localhost:6333`（HTTP API）和 `localhost:6334`（gRPC）。

Qdrant 作为独立通用服务运行，不绑定单一项目目录，通过 Collection 名称隔离不同项目数据，便于后续复用。

### 向量化（bge-m3）

使用 `FlagEmbedding` 库的 `BGEM3FlagModel` 加载 `BAAI/bge-m3`，`use_fp16=True` 启用半精度加速。对每个 Chunk 的 `text` 字段（原文 + `[IMG_CAPTION]` 摘要）批量推理，batch_size=32，同时输出：

- **Dense Vector**：1024 维 float32，表征语义
- **Sparse Vector**：从 `lexical_weights` 字段提取非零词权重，转换为 `dict[int, float]` 格式

向量化结果持久化到 `data/embeddings/embeddings_cache.pkl`，实现断点续传，同时避免 Qdrant 重建索引时重复推理。

### Qdrant Collection 配置

Collection `dcs_manuals` 配置双路向量：

```
Dense:  name="dense",  size=1024, distance=Cosine
Sparse: name="sparse", SparseVectorParams
```

每个 Point 的 Payload 包含完整 Chunk 元数据：`chunk_id`、`pdf_name`、`title_h1/h2/h3`、`page_start/end`、`text`、`image_paths`、`prev_chunk_id`、`next_chunk_id`、`split_group_id`。

对 `pdf_name`、`title_h1`、`title_h2`、`page_start` 四个字段建立 Payload 索引，支持阶段五检索时的高效过滤。

### Hybrid Search 配置

查询时 Dense 和 Sparse 各自 prefetch top20，通过 RRF（倒数排名融合）合并后返回 Top K。单次 Hybrid Search 响应时间约 3.6 秒（含 Python 端处理）。

### 单 Collection 多手册设计

四本手册统一存入 `dcs_manuals` 单一 Collection，以 `pdf_name` 字段区分来源。该设计支持：
- 跨手册语义检索（默认行为）
- 按手册过滤检索（加 `pdf_name` filter）
- 后续追加新手册无需重建索引，直接 upsert 新 Chunk

---

## 召回质量分析（F-16C 精细测试，14 题）

| 类型 | 题目示例 | 结果 |
|------|---------|------|
| 系统说明类 | HSD页面怎么看、HARM攻击模式、TGP锁定目标 | ✅ 准确 |
| 操作步骤类 | 空中加油对接、冷启动、CCRP投弹、AIM-9使用 | ✅ 准确 |
| 参数数值类 | 发动机滑油压力范围（返回含25-65 psi的chunk） | ✅ 准确 |
| 英文缩写类 | FLCS故障处理、航炮EEGS模式 | ❌ 失败 |
| 语义描述类 | TWS模式有什么限制 | ❌ 失败 |

**失败原因分析：**

- **英文缩写召回失败**：用户 Query 使用缩写（FLCS、EEGS），手册原文使用全称或中文译名，Sparse 向量词汇不匹配，Dense 语义距离不足以弥补。
- **语义描述型失败**：Query 包含"限制"一词，但原文直接展开描述限制内容而未使用"限制"关键词，且 TWS 章节标题未参与向量化，标题信号缺失。

两类失败均已记录为阶段五典型测试用例。

---

## 主要设计决策（相对原技术方案的调整）

| 原计划 | 实际决策 | 原因 |
|--------|---------|------|
| 未明确 Collection 数量 | 单 Collection 存四本手册 | 支持跨手册检索；后续追加手册无需重建；`pdf_name` filter 保留按手册查询能力 |
| 图片缩至 512px 后向量化 | 向量化输入为纯文本（原文 + IMG_CAPTION） | 图片不参与向量计算，Text-Anchored 架构，VLM 摘要已在阶段三注入文本 |
| 标题前置拼入向量化文本 | 暂缓，维持原始 text 字段 | 各手册标题层级深度不一致，定死 h2 前置可能引入噪音；推迟至阶段五测试集验证后决策 |
| Qdrant 存储在项目目录 | 存储在用户目录（`C:\Users\Keven\qdrant_storage`） | Qdrant 作为通用服务，不绑定单一项目，便于跨项目复用 |

---

## 已知局限（不影响 MVP）

- 英文缩写查询召回率偏低，需阶段五通过 Query 改写或缩写扩展词典解决
- 语义描述型查询（无关键词重叠）在纯文本向量化下存在召回盲区，标题前置方案待测试集验证后决策
- Sparse 向量基于 bge-m3 学习权重，非标准 BM25，对极低频专有名词（如 DCS 特有缩写）的权重可能不足

---

## 下一步：阶段五

- 实现完整检索链路：Hybrid Search + 相邻 Chunk 扩展 + bge-reranker-v2-m3 精排
- 构建测试集（10 题以上），覆盖精确参数、步骤定位、英文缩写、连续流程、手册外问题各类型
- Query 改写模块：针对英文缩写召回失败问题，在检索前展开缩写并补充中文语义
- BM25/Dense 权重 A/B 测试，确定最终检索参数
- Checkpoint 5：召回率终极测试，精确类 100% 命中 Top N，端到端检索耗时 < 3 秒