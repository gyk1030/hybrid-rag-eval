# hybrid-rag-eval

面向**中文年报问答**的混合检索 RAG 系统与可复现评估框架。

> 状态：后端 + Angular 前端均已可用，完成一轮三阶段评估。

## 仓库结构

```
hybrid-rag-eval/
├── backend/          # 后端：解析、索引、检索、生成、评估
│   ├── app/
│   │   ├── config.py         # 所有参数集中在这里（支持 .env）
│   │   ├── main.py           # FastAPI：GET /ask、GET /health
│   │   ├── ingestion/        # PDF 解析与分块
│   │   ├── retrieval/        # 向量 / BM25 / RRF / 精排
│   │   ├── generation/       # 提示词组装、LLM 调用、降级、追踪
│   │   ├── eval/             # Golden Set、三阶段采集、RAGAS 评分
│   │   └── tests/            # 降级与追踪的回归用例
│   ├── data/                 # 评测集与评估结果（语料不入库）
│   ├── docs/执行流程.md       # 11 张流程图 + 参数表 + 性能基线
│   └── README.md             # 后端的详细说明与快速开始
└── frontend/         # Angular 20 对话框
    ├── src/app/
    │   ├── app.ts / app.html / app.css
    │   ├── chat.service.ts   # /ask、/health 封装
    │   ├── models.ts         # 与后端响应对应的类型
    │   └── pipeline-status.ts # 「最终结果来自哪一步」的判定
    └── README.md
```

## 它解决什么问题

单靠向量检索在财报问答里会漏掉大量问题：问「2023 年末个人贷款不良率」，
向量路返回语义相近但年份不对的段落，真正含「1.37%」的那一块排在第 9 名。
本项目串联「向量 + BM25 → RRF → 精排」，并用三阶段对比量化每一步的贡献。

实测（23 题、7 份年报）：

| 指标 | baseline（纯向量） | hybrid（+BM25/RRF） | rerank（+精排） |
|---|---|---|---|
| faithfulness | 0.5507 | 0.5303 | **0.8043** |
| context_recall | 0.8043 | 1.0000 | 1.0000 |
| id_based_context_recall | 0.4783 | 0.7826 | **0.8696** |

两组不经过 LLM 裁判的硬指标（参考块命中、拒答数）同向改善，
说明提升不是评分偏差。

## 快速开始

```powershell
# 1. 后端
cd backend
uv sync
Copy-Item .env.example .env      # 编辑 .env 填入 RAG_PG_DSN（真实口令不入库）
python -m app.main               # 监听 127.0.0.1:8000

# 2. 前端（另开终端）
cd frontend
pnpm install
pnpm start                       # http://localhost:4200
```

前端请求 `/api/*`，由开发服务器代理到后端，浏览器不涉及跨域。

完整步骤（模型准备、建库、评估）见 [backend/README.md](backend/README.md)，
前端说明见 [frontend/README.md](frontend/README.md)，
架构与逐环节细节见 [backend/docs/执行流程.md](backend/docs/执行流程.md)。

## 设计要点

- **全本地运行**：嵌入与精排用 BGE 系列，生成用 llama.cpp，评测裁判也是本地模型，
  不依赖任何云服务。
- **降级而非崩溃**：数据库、BM25 缓存、精排模型、LLM 任一不可用都会阶梯降级，
  并把降级写进 `degraded` 字段——未标注的兜底答案比报错更危险。
- **可追踪**：每次问答记录 vector / bm25 / rrf / rerank / context / timing / fallback
  七个阶段，用来回答「正确片段是在哪一步丢的」。
- **可复现评估**：Golden Set 与检索解耦，评分可反复重跑，不需要重新检索。

## 测试

```powershell
cd backend
python -m pytest app -q      # 27 个用例，约 15 秒
```

降级与追踪的用例全部使用桩对象，不加载模型，可在无 GPU、无数据库的机器上运行。

## 已知限制

- 评测集仅 23 题，单条样本就能让 faithfulness 跳动 0.04；方向可信，数值不宜当精度看。
- 题型偏精确型（约 70%），对 BM25 有利，可能高估 hybrid 的收益。
- 精排是检索侧瓶颈（约 670 ms/条，20 条候选约 14 s），降 `RERANK_MAX_LENGTH`
  收益有限（实测 1.14x）且会截断 22% 的长块，不建议。
- 部分报表页只有表头没有单元格值，这类页面生成的题目不可作答，是评测集的主要脏数据来源。
