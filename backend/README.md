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
| faithfulness | 0.4674 | 0.5290 | **0.7329** |
| answer_relevancy | 0.3782 | 0.5778 | **0.7136** |
| context_precision | 0.6063 | 0.7745 | **0.8464** |
| context_recall | 0.7826 | 0.9348 | **0.9565** |
| 参考块命中 | 11/23 | 19/23 | **20/23** |
| 拒答数 | 11 | 5 | **3** |

harness 无关的两个硬指标（不经 LLM 裁判）也同向改善，说明提升不是评分偏差。

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

# 编码入库（5,656 块，GPU bf16 约 6 分钟）
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


### 6. 跑评估

```bash
python -m app.eval.build_golden --generate 30     # 造题（需人工复核）
python -m app.eval.run_pipeline --stage rerank    # 三阶段各跑一次
python -m app.eval.ragas_eval --input data/eval_results/rerank_records.json
python -m app.eval.ragas_eval --compare data/eval_results
```

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
python -m pytest app -q      # 27 个用例，约 15 秒
```

降级与追踪的用例全部使用桩对象，不加载模型，可在没有 GPU/数据库的机器上运行。
