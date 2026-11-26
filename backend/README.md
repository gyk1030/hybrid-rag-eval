# hybrid-rag-eval · 后端

本目录是 hybrid-rag-eval 的后端。以下所有命令都在 `backend/` 目录下执行。

面向**中文年报问答**的本地混合检索 RAG 系统，附带一套可复现的三阶段对比评估。

全部组件本地运行，不依赖任何云服务：解析用 PyMuPDF + pdfplumber，向量与精排用
BGE 系列模型，生成用 llama.cpp，评估用 RAGAS（裁判也是本地模型）。

## 为什么做这个

单靠向量检索在财报场景会漏掉一半问题：问「2023 年末个人贷款不良率」，
向量路返回的是语义相近但年份不对的段落，而真正含「1.37%」的那一块排在第 9 名。
本项目把「向量 + BM25 → RRF → 精排」串起来，并用三阶段对比量化每一步的贡献。

实测（23 题、7 份年报）：

| 指标 | baseline（纯向量） | hybrid（+BM25/RRF） | rerank（+精排） |
|---|---|---|---|
| faithfulness | 0.5507 | 0.5303 | **0.8043** |
| answer_relevancy | 0.6068 | 0.6726 | **0.7137** |
| context_precision | 0.5922 | 0.7873 | **0.8380** |
| context_recall | 0.8043 | 1.0000 | 1.0000 |
| id_based_context_recall | 0.4783 | 0.7826 | **0.8696** |
| 拒答数 | 5 | 3 | **1** |

两个不经 LLM 裁判的硬指标也同向改善，说明提升不是评分偏差：
`id_based_context_recall` 直接比对 chunk_id，与人工数的「参考块命中」完全一致。

## 架构

```
PDF ──解析──> chunks.jsonl ──┬──> BGE-M3 编码 ──> PostgreSQL + pgvector
                             └──> jieba 分词 ──> BM25 索引（本地缓存）

问题 ──┬──> 向量 Top20 ──┐
       └──> BM25  Top20 ──┴──> RRF 融合 ──> BGE-Reranker 精排 ──> 上下文
                                                                    │
                                      llama.cpp（Qwen2.5-3B）<── 提示词
                                                                    │
                                                              AnswerResult
                                                        （答案 / 上下文 / 追踪线索）
```

## 目录结构

```
app/
├── config.py                # 所有参数集中在这里，支持环境变量覆盖
├── main.py                  # FastAPI：GET /ask、GET /health
├── ingestion/               # PDF 解析与分块（含 Pydantic 数据契约）
├── retrieval/
│   ├── vector_store.py      # BGE-M3 稠密+稀疏入库与检索
│   ├── bm25_index.py        # jieba + BM25Okapi，带本地缓存
│   ├── rrf.py               # 倒数排名融合
│   └── reranker.py          # BGE-Reranker 交叉编码器精排
├── generation/
│   ├── answer.py            # 上下文组装、提示词、LLM 调用、降级
│   └── trace.py             # 各阶段追踪线索
├── eval/                    # build_golden / run_pipeline / ragas_eval
└── tests/                   # 降级与追踪的回归用例（不依赖模型）
data/
├── golden_set.json          # 23 题评测集
└── eval_results/            # 三阶段 records 与评分结果
docs/执行流程.md              # 11 张流程图 + 参数表 + 性能基线 + 踩坑表
```

## 快速开始

### 1. 依赖

需要 Python 3.12、PostgreSQL（安装 pgvector 扩展）、llama.cpp。

```bash
uv sync
```

### 2. 配置

数据库连接通过环境变量注入（代码里的默认值只是占位符）：

```powershell
$env:RAG_PG_DSN = "postgresql://user:password@localhost:5432/rag"
```

### 3. 准备模型

| 模型 | 放置路径 | 说明 |
|---|---|---|
| BGE-M3 | `models/bge-m3` | 稠密 1024 维 + 稀疏权重 |
| BGE-Reranker-v2-m3 | `models/bge-reranker-v2-m3` | 交叉编码器 |
| Qwen2.5-3B-Q5_K_M.gguf | 任意路径 | 由 llama.cpp 加载 |

模型权重不入库（见 `.gitignore`）。下载可用 hf-mirror：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

### 4. 建库

```bash
# 解析 PDF → chunks.jsonl
python -m app.ingestion.pdf_parser --input data/corpus --output data/parsed/chunks.jsonl

# 编码入库（5,573 块，GPU bf16 约 6 分钟）
python -m app.retrieval.vector_store --input data/parsed/chunks.jsonl

# 建 BM25 索引
python -m app.retrieval.bm25_index --rebuild
```

### 5. 启动服务

```bash
# 本地 LLM（-ngl 999 必须有，否则退回 CPU 只有 4 tok/s）
llama-server -m /path/to/qwen2.5-3b-instruct-q5_k_m.gguf -c 8192 -ngl 999 --port 8080 --host 0.0.0.0 -fa auto

# 应用（三种写法等价，都在 backend/ 目录下执行）
python -m app.main
uv run python -m app.main
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ `uvicorn` 的 app 参数必须是**模块路径**，用点号分隔：
> `app.main:app` 可以，`.\app\main:app` 会报
> `TypeError: the 'package' argument is required to perform a relative import`
> （以 `.` 开头会被当成相对导入）。

```bash
GET http://127.0.0.1:8000/ask?q=万科2024年的研发投入金额是多少
GET http://127.0.0.1:8000/health
```

响应包含 `answer`、`contexts`、`degraded`、`trace` 与各阶段耗时。

#### 启动排错

| 报错 | 原因 | 处理 |
|---|---|---|
| `Expected a Python module at: src\hybrid_rag_eval\__init__.py` | 项目被当成可安装的库去构建，但代码在 `app/` 下 | `pyproject.toml` 已声明 `[tool.uv] package = false`；若你本地是旧版本，先 `git pull` |
| `uv trampoline failed to canonicalize script path` | `.venv/Scripts/*.exe` 这些 uv 生成的启动器损坏 | `uv pip install --force-reinstall --no-deps uvicorn`，或直接用 `uv run python -m app.main` |
| `Failed to spawn: uvicorn / program not found` | 同一 venv 里并发跑了两条 `uv run`，环境正在被改写 | 一次只跑一条，等前一条结束 |
| `TypeError: ... relative import` | app 参数写成了 `.\app\main:app` | 改成 `app.main:app` |


### 解析的两个坑（都已在 `pdf_parser.py` 里处理）

**跨页表格会丢表头。** pdfplumber 按页识别表格，一张跨页表会被拆成两张，下一页那张
起始行直接是数据。平安银行 2024 年报第 37 页就是这样：表头「项 目 / 2024年12月31日 /
2023年12月31日」留在第 36 页，第 37 页只剩「贴现 | 185,729 | 5.5% …」和
「个人贷款 | … | 1.39% | … | 1.37% | …」——两个数字在块内无法归年。

处理：`link_continued_tables()` 在列数一致且表格贴着页边（上一页底边 ≥85% 页高，或
本页顶边 ≤15% 页高）时，把上一页的表头接过来，并在块首标注「表头沿用第 N 页」。
重新解析后共接回 73 个表格块。

**两级表头会被当成数据。** 财报表格常见「年度行 + 余额/占比/不良率 行」两行表头。
`header_row_count()` 按「前导非数值行」识别表头行数，切分时整组重复，不再把第二行
表头当正文。

另外，整表含财税数值却没有任何年份标记的块，会在标题后标注「无表头·数值不可归年」，
避免模型把无年份的数值硬安到问题所问的年份上。

> ⚠️ 重新解析后语料从 5,656 块变为 5,573 块（两级表头不再计入正文），
> `data/eval_results/` 里的评分结果是**旧语料**上跑的，需要重跑三阶段采集与评分。

### 6. 跑评估

```bash
python -m app.eval.build_golden --generate 30     # 造题（需人工复核）
python -m app.eval.run_pipeline --stage rerank    # 三阶段各跑一次
python -m app.eval.ragas_eval --input data/eval_results/rerank_records.json
python -m app.eval.ragas_eval --compare data/eval_results
```

#### 五个指标，其中一个是零成本的

| 指标 | 输入 | 算多少 LLM 调用 | 衡量什么 |
|---|---|---|---|
| `faithfulness` | answer + contexts | 2（拆陈述 + NLI） | 答案是否都有资料支撑 |
| `answer_relevancy` | answer | 3（正反向生成问题） | 答案是否切题 |
| `context_precision` | question + reference + 每条 context | N=5（每条一次） | 有用的资料是否排在前面 |
| `context_recall` | reference + contexts | 1（标准答案逐句归因） | 事实是否被召回 |
| `id_based_context_recall` | chunk_id 两列 | **0** | 参考块是否被召回 |

**为什么加第五个。** 实测 `context_recall` 在 hybrid / rerank 上恒为 1.0（23/23 满分），
已经无法区分阶段——标准答案多是单句单事实，加上 3B 裁判偏「Yes」。
`id_based_context_recall` 直接比对 chunk_id 集合，不经过裁判：

```text
baseline 11/23 = 0.4783   hybrid 18/23 = 0.7826   rerank 20/23 = 0.8696
```

与手工数出来的「参考块命中」完全一致，且能拉开 hybrid 和 rerank 的差距。

#### 两个常用参数

```bash
# 只补算 id 召回（秒级），并合并进已有结果，不重跑要裁判的四个指标
python -m app.eval.ragas_eval --input data/eval_results/rerank_records.json \
       --metrics id_based_context_recall --merge

# 只跑指定指标
python -m app.eval.ragas_eval --input ... --metrics faithfulness,context_precision
```

`--merge` 按行序对齐已有结果文件（同一条 records、同一顺序），行数不一致会放弃合并并打印提示。
合并记录的 `merge_history` 会写进结果 JSON，便于追溯哪些指标是哪次补算的。

## 设计取舍

**为什么 RRF 权重是 (1.0, 3.0)。** 财报问题里精确 token（金额、比率、编号）
的区分度远高于语义相似度，实测把 BM25 权重提到 3 倍后参考块命中从 11/23 升到 19/23。

**为什么精排用 fp16、嵌入用 bf16。** 同一张卡（GTX 1650 SUPER，无 Tensor Core）上
两者结论相反：双编码器 fp16 比 bf16 慢 5 倍；交叉编码器 fp16 反而快 15%–40%。
瓶颈不同（算力 vs 显存带宽），结论记录在 `config.py` 注释里。

**为什么提示词不要求行内引用。** 3B 模型会把 `[编号]` 写错（引用到别家公司的资料），
或把资料头当正文续写。出处改由 `contexts` 字段提供，更准确。

**为什么降级而不用 500。** 四个依赖都可能挂（PG、BM25 缓存、精排模型、LLM）。
阶梯式降级保证「能用什么用什么」，但**必须**把降级写进 `degraded` 字段——
未标注的兜底答案比报错更危险。

**为什么要做「答案年份校验」。** 实测 3B 模型会拿 2024 年报表格里的 1.39%
回答「2023 年末」的问题，而正确的 1.37% 当时就排在上下文第一位。这类错误比幻觉
更危险：数字是真的，只是年份配错了，看起来完全可信。

因此 `generation/consistency.py` 在生成后做一次确定性校验（不依赖模型）：

1. 把答案里的数字逐个拿到上下文比对，确定「这些数字来自哪一条资料」；
2. 看那条资料的文件名年份是否在问题所问年份里（区间题会展开，如 2021–2024）；
3. 结果写进响应的 `answer_check` 字段，`status=mismatch` 时前端显示红色告警条。

不采用「答案里出现别的年份就算错」是因为财报答案经常要对比（「较 2022 年末上升
0.05 个百分点」），那种引用是合法的，不能误伤。

## 已知限制

- **评测集规模小**：23 题，单条就能让 faithfulness 跳动 0.04。表里的方向可信，
  具体数值不宜当精度看。
- **题型偏精确型**（约 70%），对 BM25 有利，可能高估 hybrid 的收益。
  需要补充语义型题目才能公平比较。
- **精排是检索侧瓶颈**：约 730 ms/条（390 字、512 token），20 条候选约 15 s。
  降 `RERANK_MAX_LENGTH` 到 256 约可减半。
- **表格数值提取不全**：部分报表页只有表头没有单元格值，这类页面出的题不可作答，
  是 Golden Set 里主要的脏数据来源。
- **仓库历史说明**：本项目开发时未使用 git，提交历史是在收尾时按模块拆分的
  **逻辑重构**，不是逐日回放。每个提交的代码状态都是当前实现。

## 测试

```bash
python -m pytest app -q      # 66 个用例，约 5 秒
```

降级与追踪的用例全部使用桩对象，不加载模型，可在没有 GPU/数据库的机器上运行。
