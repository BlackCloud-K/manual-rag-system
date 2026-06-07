# manual-rag-system

面向飞行手册的多模态 RAG 系统（Text-Anchored Image Retrieval + Hybrid Search）。

---

## 快速开始

### 前置条件

| 依赖 | 说明 |
|------|------|
| Python 3.10+ | 建议项目根目录建虚拟环境 |
| Qdrant | `config.yaml` 默认 `http://localhost:6333`，检索前必须可连 |
| `.env` | 至少配置问答/路由用的 `GOOGLE_API_KEY`（或改 `config.yaml` 用 OpenAI 时配 `OPENAI_API_KEY`）；VLM 配图摘要需要 `OPENAI_API_KEY` |

启动 Qdrant（Docker 示例）：

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

安装 Python 依赖（仓库无 `requirements.txt`，按实际模块安装，示例）：

```powershell
cd D:\Programs\manual-rag-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn pyyaml python-dotenv openai google-generativeai qdrant-client FlagEmbedding pymupdf tiktoken
```

---

### 运行 Web 前端（推荐：自写后端 + `frontend/index.html`）

`frontend/index.html` 通过相对路径请求 `/config`、`/chat`、`/feedback`，**必须由 FastAPI 后端托管**，不能直接用浏览器打开本地 HTML 文件（`file://` 会跨域失败）。

```powershell
cd D:\Programs\manual-rag-system
.\.venv\Scripts\Activate.ps1

# 启动后端（默认 http://127.0.0.1:8000，并返回 frontend/index.html）
python -m backend.main
```

浏览器访问：

- 问答界面：`http://127.0.0.1:8000/`
- 手册列表 API：`http://127.0.0.1:8000/config`

手册下拉列表来源（二选一，见 `config.yaml` 的 `frontend` 段）：

- 未配置 `frontend.pdf_options`：自动扫描 `frontend.pdf_dir`（默认 `documents/`）下所有 `*.pdf`
- 配置了 `frontend.pdf_options`：仅显示列表中的文件名（须与 chunk 里的 `pdf_name`、Qdrant payload 一致）

---

### 运行 Streamlit 界面（可选）

与上面 Web 前端是**另一套 UI**，入口为根目录 `app.py`：

```powershell
cd D:\Programs\manual-rag-system
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

默认地址一般为 `http://localhost:8501`。同样需要 Qdrant 与 `.env` 中的模型 Key。

---

### 增量添加一本全新 PDF

适用于**已有库上新增手册**，不重建其它手册的 chunk。流水线入口：`src/pipeline/import_manual.py`（断点续跑，状态写在 `data/.pipeline_import/<stem>.json`）。

#### 1. 放入 PDF

将文件放到 `documents/`（或与 `config.yaml` 中 `frontend.pdf_dir` 一致的目录），例如：

```text
documents/MyNewManual.pdf
```

文件名会写入 chunk 的 `pdf_name` 字段，**前端筛选与检索过滤都依赖该完整文件名**，请勿随意改名。

#### 2. 配置页边距（解析裁切）

在 `config.yaml` 的 `manuals` 下增加一项，**键名为 PDF 文件名去掉 `.pdf` 后的 stem**（与 `load_margins_from_config` 一致），例如：

```yaml
manuals:
  "MyNewManual":
    margin_top: 55
    margin_bottom: 55
  default:
    margin_top: 55
    margin_bottom: 55
```

页眉页脚仍残留时，可适当增大 `margin_top` / `margin_bottom`。

#### 3. 确认环境

- Qdrant 已启动（见上文 Docker 命令）
- `.env` 中 `OPENAI_API_KEY` 可用（VLM 配图摘要）
- 生成/路由用的 Google 或 OpenAI Key 与 `config.yaml` 中 `generation` / `router` 的 `provider` 一致

#### 4. 执行端到端接入（单本）

```powershell
cd D:\Programs\manual-rag-system
.\.venv\Scripts\Activate.ps1

# --pdf 可为完整文件名、无 .pdf 后缀、或 documents/ 下唯一匹配的部分名
python -m src.pipeline.import_manual --pdf "MyNewManual.pdf"
```

该命令按顺序执行（已完成阶段会自动跳过）：

| 阶段 | 作用 | 主要产物 |
|------|------|----------|
| parse | TOC 切分、正文与图片 | `data/chunks_<stem>.json`、`data/images/...` |
| normalize | 去噪、超长拆分 | `data/chunks_<stem>_normalized.json` |
| vlm | 图片摘要（写入全局缓存） | `data/captions_cache.json` |
| inject | 摘要注入 | `data/final/chunks_<stem>_final.json` |
| embed | **仅对未入缓存的 chunk** 向量化 | `data/embeddings/embeddings_cache.pkl` |
| upload | **仅上传 Qdrant 中尚不存在的 chunk_id** | Qdrant collection `dcs_manuals` |

`<stem>` 由 PDF 文件名规范化得到：

- 含英文字母/数字时：小写并去掉标点（与旧行为一致），如 `F-16C“蝰蛇”.pdf` → `f16c`，`MyNewManual.pdf` → `mynewmanual`
- **纯中文文件名**：保留汉字作为 stem，如 `余钊焕量子场论讲义.pdf` → `余钊焕量子场论讲义`（避免多本中文 PDF 都落到 `manual` 互相覆盖）

#### 5. 验证前端可见

1. 重启或刷新后端：`python -m backend.main`
2. 打开 `http://127.0.0.1:8000/`，侧栏应出现新手册
3. 若未出现：确认 PDF 在 `documents/` 且未被 `frontend.pdf_options` 白名单排除

#### 6. 中断后续跑 / 从某阶段重做

```powershell
# 从中断处继续（默认）
python -m src.pipeline.import_manual --pdf "MyNewManual.pdf"

# 从 embed 起重做（会清除 embed、upload 的 checkpoint）
python -m src.pipeline.import_manual --pdf "MyNewManual.pdf" --restart-from embed

# 从 parse 起全量重做该手册
python -m src.pipeline.import_manual --pdf "MyNewManual.pdf" --restart-from parse
```

`--restart-from` 可选值：`parse`, `normalize`, `vlm`, `inject`, `embed`, `upload`。

> **说明**：`embed` / `upload` 步骤会扫描**所有** `data/final/chunks_*_final.json`，但只对**新增 chunk_id** 写缓存或上传 Qdrant，因此增量添加一本 PDF 不会影响已上传的其它手册向量。

---

## 项目文件结构与职责说明

> 说明：以下按当前仓库可见文件整理；对数量较大的自动产物使用“目录 + 命名规则”描述。

```text
manual-rag-system/
├─ .env
├─ .gitignore
├─ app.py                         # Streamlit 入口（可选 UI）
├─ backend/
│  └─ main.py                     # FastAPI 后端 + 托管 frontend/index.html
├─ frontend/
│  └─ index.html                  # Web 问答界面（须由 backend 提供）
├─ config.yaml
├─ RAG technical proposal.pdf
├─ README.md
├─ documents/                     # 原始 PDF（通常被 .gitignore 忽略）
├─ src/                           # 核心模块
│  ├─ parser/
│  ├─ normalizer/
│  ├─ vlm/
│  ├─ retriever/
│  ├─ generator/
│  ├─ router/
│  └─ pipeline/
│     └─ import_manual.py         # 单本 PDF 增量接入（推荐）
├─ scripts/                       # 流程入口、验收与诊断脚本
│  └─ temp_output/                # 各类测试/验收输出
├─ data/                          # 中间数据与最终数据
│  ├─ final/
│  ├─ embeddings/
│  └─ .pipeline_import/           # 每本手册 import 断点状态
└─ progress/                      # 阶段总结文档（checkpoint_1~7）
```

---

## 根目录文件

- `README.md`：项目总览与目录职责说明（本文件）。
- `.env`：本地环境变量（如 OpenAI Key、Qdrant 配置等，勿提交）。
- `.gitignore`：忽略规则（如 `documents/`、`data/images/`、缓存等）。
- `config.yaml`：PDF 解析相关配置（如页边距裁切参数）。
- `RAG technical proposal.pdf`：技术方案文档。
- `app.py`：Streamlit 应用入口（可选 UI）。
- `backend/main.py`：FastAPI 后端；托管 `frontend/index.html`，提供 `/chat`（SSE）、`/config`、`/pdf`、`/image`、`/feedback`。
- `frontend/index.html`：Web 问答前端（须配合 `python -m backend.main` 使用）。
- `src/pipeline/import_manual.py`：单本 PDF 端到端增量接入（见上文「增量添加一本全新 PDF」）。

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
- `src/router/`
  - `router.py`：查询路由模块（问题分类与检索策略分流）。

---

## `scripts/`（入口、验收、诊断）

### 主流程与阶段验收

- `process_all.py`：全量处理入口（解析 -> 切分 -> 图文归属，偏早期批处理）。
- **推荐** `python -m src.pipeline.import_manual --pdf <文件名>`：单本手册增量接入（含 normalize / VLM / embed / upload）。
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
- `test_history.py`：历史问答样例回放测试脚本。

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
  - `feedback_log.jsonl`：反馈与问答历史日志。
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
- `checkpoint_7.md`：阶段七总结。

---

## 推荐阅读顺序（新成员快速上手）

1. `progress/checkpoint_1.md` -> `progress/checkpoint_7.md`（先看演进全貌）
2. `src/parser/`、`src/normalizer/`（理解数据生产链路）
3. `scripts/process_all.py`、`scripts/run_normalization.py`、`scripts/run_vlm_captioning.py`
4. `src/retriever/retriever.py`、`scripts/run_retrieval_test.py`
5. `src/generator/generator.py`、`scripts/run_e2e_test.py`
