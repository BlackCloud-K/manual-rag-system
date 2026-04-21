# manual-rag-system

面向飞行手册的多模态 RAG 系统（Text-Anchored Image Retrieval + Hybrid Search）。

## 项目文件结构与职责说明

> 说明：以下按当前仓库实际文件整理。对数量巨大的自动生成文件（如图片）采用“目录 + 命名规则”说明。

```text
manual-rag-system/
├─ .env
├─ .gitignore
├─ config.yaml
├─ RAG technical proposal.pdf
├─ README.md
├─ documents/
├─ progress/
├─ scripts/
├─ src/
└─ data/
```

---

## 根目录文件

- `README.md`：项目总览（本文件），用于快速理解目录结构与各文件职责。
- `.env`：本地环境变量（主要用于 OpenAI API Key 等敏感配置，不应提交远端）。
- `.gitignore`：Git 忽略规则（忽略 `documents/`、`data/images/`、缓存、虚拟环境等）。
- `config.yaml`：PDF 文本/图片提取裁切参数配置（按手册配置 `margin_top` / `margin_bottom`）。
- `RAG technical proposal.pdf`：项目技术方案文档（设计背景与路线说明）。

---

## `documents/`（原始手册输入）

- `documents/DCS FA-18C Early Access Guide CN.pdf`：FA-18C 手册原始 PDF。
- `documents/DCS_ JF-17 _雷电_.pdf`：JF-17 手册原始 PDF。
- `documents/F-15C.pdf`：F-15C 手册原始 PDF。
- `documents/F-16C“蝰蛇”.pdf`：F-16C 手册原始 PDF。

> 这些 PDF 是全流程输入源，`process_all.py` / `toc_parser.py` / `chunk_builder.py` 等都基于它们工作。

---

## `src/`（核心业务代码）

### `src/parser/`

- `src/parser/toc_parser.py`：解析 TOC（优先书签，回退目录页链接），输出层级化目录项。
- `src/parser/chunk_builder.py`：基于 TOC 构建 chunk、提取正文、按标题 Y 坐标精细切分、清理空/占位 chunk。
- `src/parser/image_extractor.py`：提取页面图片 cluster，渲染并按章节归属，回填 `image_paths`。

### `src/normalizer/`

- `src/normalizer/chunk_normalizer.py`：Chunk 规范化工具：删除垃圾 chunk、拆分超长 chunk、重建链表关系。

### `src/vlm/`

- `src/vlm/image_preprocessor.py`：图片预处理（转 RGB JPEG 并编码为 base64）。
- `src/vlm/vlm_client.py`：VLM 调用封装（`gpt-4o-mini` 生成图片摘要，含重试退避）。
- `src/vlm/__init__.py`：VLM 子模块说明与包初始化。

---

## `scripts/`（流程入口与诊断脚本）

### 1) 主流程 / Checkpoint 脚本

- `scripts/process_all.py`：批处理入口（TOC -> chunk -> 文本填充 -> 清理 -> 图片归属）。
- `scripts/checkpoint1_validate.py`：阶段一验收（结构完整性、文本非空、链表完整性、图片覆盖率）。
- `scripts/run_normalization.py`：阶段二批量规范化（删除垃圾 chunk + 拆分超长 chunk）。
- `scripts/run_vlm_captioning.py`：阶段三批量生成图片摘要并写入 `captions_cache.json`（支持 TEST_MODE）。
- `scripts/run_caption_injection.py`：将图片摘要注入 normalized chunk 文本，生成 final JSON。
- `scripts/run_checkpoint3.py`：阶段三验收（摘要注入覆盖率、失败率、数量一致性）。
- `scripts/run_embedding.py`：用 `bge-m3` 为 final chunks 生成 dense+sparse 向量缓存。
- `scripts/run_upload_qdrant.py`：上传向量与 payload 到 Qdrant，自动建 collection 与 payload index。
- `scripts/run_checkpoint4.py`：阶段四检索验收（Hybrid Search + F-16C 精细 query 评估）。

### 2) 诊断 / 实验脚本

- `scripts/clean_chunks.py`：早期单文件清理脚本（针对 `chunks_f16c.json` 的低 token 垃圾 chunk）。
- `scripts/font_diagnostic.py`：字体统计与标题层级推断诊断（早期方案验证用，可写入配置）。
- `scripts/toc_diagnostic.py`：TOC 能力诊断（书签/目录关键词页/链接目标）。
- `scripts/parse_all_toc_summary.py`：批量汇总每本手册的 TOC 解析结果。
- `scripts/verify_pdf_toc_link_characteristics.py`：验证 PDF 目录链接“只有起始页”特性及 chunk 尾部吸收行为。
- `scripts/margin_diagnostic.py`：按页输出文本块 Y 坐标，辅助调 `margin_top` / `margin_bottom`。
- `scripts/image_diagnostic.py`：对指定页检查文本块/图片块计数与位置对应关系。
- `scripts/image_extract_test.py`：图片提取与聚类策略实验（单页/多页导出测试图）。
- `scripts/image_stats_diagnostic.py`：统计图片体量、尺寸、引用覆盖、重复与成本估算。
- `scripts/token_diagnostic.py`：统计 chunk token 分布（区间占比、超长/过短明细）。
- `scripts/large_chunk_diagnostic.py`：分析 >800 token chunk 的内部结构与可切分性。
- `scripts/split_strategy_diagnostic.py`：比较多种切分策略有效率并给出推荐顺序。

### 3) 输出与缓存目录

- `scripts/temp_output/checkpoint1_validate.txt`：阶段一验收报告文本。
- `scripts/temp_output/checkpoint4_f16c_fine_*.json`：阶段四 F-16C 精细检索验收结果。
- `scripts/temp_output/checkpoint4_hybrid_search_*.json`：阶段四混合检索样例结果。
- `scripts/temp_output/*_font_diagnostic.txt`：各手册字体诊断报告。
- `scripts/temp_output/image_stats_diagnostic_report.txt`：图片统计诊断报告。
- `scripts/temp_output/large_chunk_diagnostic_report.txt`：大 chunk 诊断报告。
- `scripts/temp_output/split_strategy_diagnostic_report.txt`：切分策略对比报告。
- `scripts/temp_output/token_diagnostic_report.txt`：token 诊断报告。
- `scripts/__pycache__/`：Python 字节码缓存（运行时自动生成）。

---

## `data/`（处理产物与中间数据）

### 1) Chunk JSON（阶段化）

- `data/chunks_f16c.json`：F-16C 原始切分结果（含 text、image_paths、链表字段）。
- `data/chunks_dcsjf17.json`：JF-17 原始切分结果。
- `data/chunks_f15c.json`：F-15C 原始切分结果。
- `data/chunks_dcsfa18cearlyaccessguidecn.json`：FA-18C 原始切分结果。
- `data/chunks_f16c_normalized.json`：F-16C 规范化后结果。
- `data/chunks_dcsjf17_normalized.json`：JF-17 规范化后结果。
- `data/chunks_f15c_normalized.json`：F-15C 规范化后结果。
- `data/chunks_dcsfa18cearlyaccessguidecn_normalized.json`：FA-18C 规范化后结果。

### 2) 图片摘要与向量化

- `data/captions_cache.json`：图片路径 -> VLM 摘要缓存（支持断点续跑）。
- `data/embeddings/embeddings_cache.pkl`：chunk_id -> dense+sparse 向量缓存。

### 3) 最终检索输入

- `data/final/chunks_f16c_final.json`：F-16C 最终检索 chunk（正文 + `[IMG_CAPTION]`）。
- `data/final/chunks_dcsjf17_final.json`：JF-17 最终检索 chunk。
- `data/final/chunks_f15c_final.json`：F-15C 最终检索 chunk。
- `data/final/chunks_dcsfa18cearlyaccessguidecn_final.json`：FA-18C 最终检索 chunk。

### 4) 图片文件

- `data/images/f16c/*.png`：F-16C 提取图片（按 `page_{页码}_img_{序号}.png` 命名）。
- `data/images/dcsjf17/*.png`：JF-17 提取图片。
- `data/images/f15c/*.png`：F-15C 提取图片。
- `data/images/dcsfa18cearlyaccessguidecn/*.png`：FA-18C 提取图片。
- `data/images/test/`：图片提取实验输出（测试页导出）。

---

## `progress/`（阶段总结）

- `progress/checkpoint_1.md`：阶段一（解析与图文归属）验收总结。
- `progress/checkpoint_2.md`：阶段二（规范化）验收总结。
- `progress/checkpoint_3.md`：阶段三（VLM 摘要注入）验收总结。
- `progress/checkpoint_4.md`：阶段四（向量化与向量库）验收总结。

---

## 推荐阅读顺序（新成员快速上手）

1. `progress/checkpoint_1.md` -> `progress/checkpoint_4.md`（先看全流程演进）
2. `src/parser/`（理解数据从 PDF 到 chunk 的核心逻辑）
3. `scripts/process_all.py`、`scripts/run_normalization.py`、`scripts/run_vlm_captioning.py`
4. `scripts/run_embedding.py`、`scripts/run_upload_qdrant.py`
