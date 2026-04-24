# manual-rag-system

面向飞行手册的多模态 RAG 系统（Text-Anchored Image Retrieval + Hybrid Search）。

## 项目文件结构与职责说明

> 说明：以下按当前仓库可见文件整理；对数量较大的自动产物使用“目录 + 命名规则”描述。

```text
manual-rag-system/
├─ .env
├─ .gitignore
├─ config.yaml
├─ RAG technical proposal.pdf
├─ README.md
├─ documents/                     # 原始 PDF（通常被 .gitignore 忽略）
├─ src/                           # 核心模块
│  ├─ parser/
│  ├─ normalizer/
│  ├─ vlm/
│  ├─ retriever/
│  └─ generator/
├─ scripts/                       # 流程入口、验收与诊断脚本
│  └─ temp_output/                # 各类测试/验收输出
├─ data/                          # 中间数据与最终数据
│  ├─ final/
│  └─ embeddings/
└─ progress/                      # 阶段总结文档（checkpoint_1~6）
```

---

## 根目录文件

- `README.md`：项目总览与目录职责说明（本文件）。
- `.env`：本地环境变量（如 OpenAI Key、Qdrant 配置等，勿提交）。
- `.gitignore`：忽略规则（如 `documents/`、`data/images/`、缓存等）。
- `config.yaml`：PDF 解析相关配置（如页边距裁切参数）。
- `RAG technical proposal.pdf`：技术方案文档。

---

## `src/`（核心业务代码）

- `src/parser/`
  - `toc_parser.py`：解析目录结构（书签优先，必要时回退目录页）。
  - `chunk_builder.py`：基于 TOC 进行 chunk 切分、文本填充与清理。
  - `image_extractor.py`：提取/聚类页面图片并回填 `image_paths`。
- `src/normalizer/`
  - `chunk_normalizer.py`：去噪、拆分超长 chunk、重建链表关系。
- `src/vlm/`
  - `image_preprocessor.py`：图片预处理（RGB/JPEG/base64）。
  - `vlm_client.py`：VLM 摘要调用封装（含重试退避）。
- `src/retriever/`
  - `retriever.py`：检索模块（Hybrid/BM25/向量检索相关逻辑）。
- `src/generator/`
  - `generator.py`：生成模块（基于检索上下文组织回答）。

---

## `scripts/`（入口、验收、诊断）

### 主流程与阶段验收

- `process_all.py`：全量处理入口（解析 -> 切分 -> 图文归属）。
- `checkpoint1_validate.py`：阶段一结构与质量校验。
- `run_normalization.py`：阶段二规范化批处理。
- `run_vlm_captioning.py`：阶段三图片摘要生成（写入 `captions_cache.json`）。
- `run_caption_injection.py`：将摘要注入 chunk，生成 final JSON。
- `run_checkpoint3.py`：阶段三验收。
- `run_embedding.py`：生成 dense+sparse embedding 缓存。
- `run_upload_qdrant.py`：上传向量与 payload 到 Qdrant。
- `run_checkpoint4.py`：阶段四检索验收。
- `run_checkpoint5.py`：阶段五相关验收流程。

### 检索/生成/E2E 脚本

- `run_retrieval_test.py`：检索测试与统计输出。
- `build_testset.py`：构建测试集（如 `data/testset_f16c.json`）。
- `run_e2e_test.py`：单次 E2E 测试。
- `run_e2e_batch_test.py`：批量 E2E 测试。

### 诊断/实验脚本

- `toc_diagnostic.py`、`parse_all_toc_summary.py`、`verify_pdf_toc_link_characteristics.py`：TOC 能力诊断。
- `font_diagnostic.py`、`margin_diagnostic.py`：版式/页边距诊断。
- `image_diagnostic.py`、`image_extract_test.py`、`image_stats_diagnostic.py`：图片提取与统计诊断。
- `token_diagnostic.py`、`large_chunk_diagnostic.py`、`split_strategy_diagnostic.py`：chunk 长度与切分策略诊断。
- `clean_chunks.py`：早期清理脚本（保留作参考）。

### `scripts/temp_output/`（典型输出）

- `checkpoint1_validate.txt`：阶段一验收报告。
- `checkpoint4_hybrid_search_*.json`、`checkpoint4_f16c_fine_*.json`：阶段四检索结果。
- `checkpoint5_result_*.json`：阶段五结果。
- `e2e_test_questions.jsonl`、`e2e_test_results_*.json`：E2E 问题集与结果。
- `retrieval_test_*.txt`、`retrieval_budget_stats_*.json`：检索评测与预算统计。
- `*_font_diagnostic.txt`、`token_diagnostic_report.txt`、`split_strategy_diagnostic_report.txt` 等：诊断报告。

---

## `data/`（处理中间产物与最终输入）

- 原始 chunk：
  - `chunks_f16c.json`
  - `chunks_f15c.json`
  - `chunks_dcsjf17.json`
  - `chunks_dcsfa18cearlyaccessguidecn.json`
- 规范化 chunk：
  - `chunks_f16c_normalized.json`
  - `chunks_f15c_normalized.json`
  - `chunks_dcsjf17_normalized.json`
  - `chunks_dcsfa18cearlyaccessguidecn_normalized.json`
- 最终检索输入（`data/final/`）：
  - `chunks_f16c_final.json`
  - `chunks_f15c_final.json`
  - `chunks_dcsjf17_final.json`
  - `chunks_dcsfa18cearlyaccessguidecn_final.json`
- 其它：
  - `captions_cache.json`：图片摘要缓存。
  - `embeddings/embeddings_cache.pkl`：向量缓存。
  - `testset_f16c.json`：测试集。
- 图片目录（通常不入库）：
  - `data/images/<manual_name>/*.png`

---

## `progress/`（阶段总结）

- `checkpoint_1.md`：阶段一总结。
- `checkpoint_2.md`：阶段二总结。
- `checkpoint_3.md`：阶段三总结。
- `checkpoint_4.md`：阶段四总结。
- `checkpoint_5.md`：阶段五总结。
- `checkpoint_6.md`：阶段六总结。

---

## 推荐阅读顺序（新成员快速上手）

1. `progress/checkpoint_1.md` -> `progress/checkpoint_6.md`（先看演进全貌）
2. `src/parser/`、`src/normalizer/`（理解数据生产链路）
3. `scripts/process_all.py`、`scripts/run_normalization.py`、`scripts/run_vlm_captioning.py`
4. `src/retriever/retriever.py`、`scripts/run_retrieval_test.py`
5. `src/generator/generator.py`、`scripts/run_e2e_test.py`
