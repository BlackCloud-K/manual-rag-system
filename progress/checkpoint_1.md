# Checkpoint 1 验收总结

**项目**：DCS 飞行手册多模态 RAG 系统  
**阶段**：阶段一 — 文档解析与图文归属  
**验收日期**：2026 年 4 月  
**状态**：✅ 通过

---

## 交付物

| 交付物 | 说明 |
|--------|------|
| `src/parser/toc_parser.py` | TOC 解析模块，支持书签和目录页超链接两种来源 |
| `src/parser/chunk_builder.py` | Chunk 构建模块，包含文字提取、页内精细切分、空 Chunk 清理 |
| `src/parser/image_extractor.py` | 图片提取与归属模块，cluster 合并后区域渲染 |
| `data/chunks_*.json` | 四本手册的结构化 Chunk JSON，共 4 份 |
| `data/images/` | 提取的图片文件，按手册分目录存储 |
| `scripts/checkpoint1_validate.py` | 自动验收脚本 |

**四本手册处理结果：**

| 手册 | Chunk 数 | 图片数 | 归属覆盖率 |
|------|---------|-------|-----------|
| DCS FA-18C Early Access Guide CN | 295 | 360 | 100% |
| DCS JF-17 雷电 | 205 | 317 | 99.68% |
| F-15C | 66 | 95 | 97.9% |
| F-16C 蝰蛇 | 208 | 313 | 100% |

---

## 验收结果

全部四本手册通过 Checkpoint 1 的所有自动验收项：

- ✅ **结构完整性**：所有 Chunk 包含九个必要字段，无缺失
- ✅ **text 非空率**：无空 Chunk（空 Chunk 在处理阶段已自动清理）
- ✅ **链表完整性**：`prev_chunk_id` / `next_chunk_id` 双向一致，无断链
- ✅ **图片归属覆盖率**：全部 > 95%

---

## 技术方案

### TOC 解析（标题层级来源）

放弃了原计划的字体大小阈值检测方案，改为直接解析 PDF 自带的结构化导航信息：

- **有书签的手册**（JF-17、F-15C）：使用 `doc.get_toc()` 直接获取 `[层级, 标题, 页码]` 列表
- **无书签的手册**（FA-18C、F-16C）：扫描前 15 页找到目录页，通过 `page.get_links()` 提取超链接，用链接区域的 x0 坐标聚类推断 H1/H2/H3 层级

字体大小方案被放弃的原因：四本手册字体体系差异较大，且全部标题均为非加粗（`bold=False`），无法用加粗属性区分标题层级，阈值调试成本高且不稳定。

### Chunk 切分

以 TOC 条目为切分锚点，每个 TOC 条目对应一个 Chunk。`page_end` 设置为下一个 TOC 条目的页码（不减 1），使相邻 Chunk 共享边界页，由 `fill_text` 在页内用标题 Y 坐标精确截断。

文字提取使用 `page.get_text("blocks")`，按 Y 坐标范围过滤正文区域，过滤规则基于 `margin_top` / `margin_bottom` 两个可配置参数（写入 `config.yaml`，按手册单独配置）。

### 页内精细切分（`find_title_y`）

为解决同页多标题的切分问题，实现了 `find_title_y` 函数：在页面内遍历所有 line（span 级别粒度），找到包含标题文字的 line，返回其 y0 坐标作为切分点。匹配到多个候选时优先选字号最大的，字号相同时选最靠上的。

文字提取使用 `y1 > y_start`（而非 `y0 >= y_start`）作为 block 包含条件，确保标题所在 block 不因 block 顶部略高于 y_start 而被整体过滤。

### 图片提取

放弃了直接提取 PDF 内嵌图片数据的方案，改为**区域渲染**：

1. 从 `page.get_text("dict")` 获取 type==1 的 image block，过滤页眉页脚（y0 < margin_top 或 y0 > margin_bottom）
2. 按 Y 坐标做 cluster 合并（相邻 block 间距 < 50pt 则合并），避免同一张逻辑图片的多个 PDF layer 被分开提取
3. 对每个 cluster 的 bbox 加横向 padding（左右各 140px），再调用 `page.get_pixmap(clip=rect)` 渲染该区域
4. 过滤纯色图（numpy 标准差 < 5）

区域渲染方案的优势：渲染结果是所有图层合并后的最终视觉效果，标注文字、箭头、图例框均完整保留，与用户在 PDF 阅读器中看到的完全一致。

### 图片归属

对每张图片 cluster，在该页收集所有 chunk 的标题 Y 坐标，找到小于图片 y0 的最大标题 Y 坐标，对应 chunk 即为归属目标。该规则天然把图片归入其上方最近标题所在的 chunk，不依赖图注文字（兼容无图注的手册如 JF-17）。

---

## 主要困难与解决过程

### 1. 字体阈值方案失效

**问题**：原计划用字体大小和加粗属性识别标题层级，但诊断发现四本手册标题全为非加粗，字号分布差异大，无法建立稳定的通用规则。

**解决**：改用 TOC 解析（书签 + 目录页超链接），利用 PDF 自带的结构化信息，完全绕开字体启发式判断。

### 2. 同页多标题导致 Chunk 文字重复

**问题**：同一页有多个标题时，相邻 Chunk 的 `page_start` 相同，原始的按页码提取逻辑导致它们提取完全相同的文字。

**解决**：实现 `find_title_y` 精确定位标题在页面内的 Y 坐标，对 `page_start` 页从当前标题 Y 坐标提取到下一个标题 Y 坐标，实现页内精细切分。

### 3. 标题所在 block 被整体过滤

**问题**：`find_title_y` 返回的是标题 line 的 y0，但 block 的 y0 可能略小于 line 的 y0，导致用 `y0 >= y_start` 过滤时标题 block 被整体丢弃，text 中缺少标题文字。

**解决**：将 `page_start` 页的 block 过滤条件改为 `y1 > y_start`，只要 block 的底部超过截取起点即纳入，确保包含标题 line 的 block 不被误排除。

### 4. 跨页内容归属错误

**问题**：`page_end = 下一个 TOC 页码 - 1` 导致某些节的内容延伸到下一页的上半段时被截断，这部分内容错误地归入了下一个 chunk。

**解决**：将 `page_end` 改为等于下一个 TOC 条目的页码（不减 1），`fill_text` 在 `page_end` 页用 `find_title_y` 找到下一个 chunk 标题的 Y 坐标作为截止点，正确处理跨页边界。

### 5. 图片被拆分为多个 PDF layer

**问题**：PDF 制作时将带标注的图片拆成多个 image block（主图 + 标注箭头 + 标注文字框），直接提取内嵌图片数据只能得到其中一层，标注信息丢失。

**解决**：改用区域渲染方案，对合并后的 cluster bbox 调用 `page.get_pixmap(clip=rect)` 整体渲染，所有图层自动合并，标注完整保留。同时加入横向 padding（左右各 140px）以覆盖伸出 bbox 外的标注箭头。

### 6. `config.yaml` key typo 导致 margin 配置失效

**问题**：代码读取 `loaded.get("manuals")`，但 config 文件顶层 key 写成了 `yamlmanuals`，导致始终使用默认值，margin_bottom 配置无效。

**解决**：统一 config 文件和代码的 key 名称为 `manuals`。

---

## 已知局限（不影响 MVP）

- 部分带外部标注的图片（如驾驶舱面板图，标注数字在图片 bbox 外侧）横向 padding 后仍可能有少量标注被裁切
- 少数 `page_end` 值偏大（等于下一个 TOC 锚点页码而非内容实际结束页），不影响 `page_start` 的准确性和文字提取正确性
- FA-18C 部分 `title_h2` 含目录页残留的省略号和页码字符串，不影响检索功能

---

## 下一步：阶段二

- Chunk 大小统计与规范化（合并 < 100 token 的 Chunk，拆分 > 800 token 的 Chunk）
- `section_type` 自动打标（步骤类 / 参数检查类 / 警告类 / 描述类）
- 完整 Metadata 结构构建