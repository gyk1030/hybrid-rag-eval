"""入库与检索的集中配置。

``app/retrieval`` 下的各模块只从本文件读取参数——数据库连接、模型与数据路径、
批大小、召回/融合/精排的默认条数——不再各自定义常量；调参只需要改这里一处。

需要随环境变化的项支持两种注入方式（都可，优先环境变量）：

1. 环境变量，例如把 ``RAG_PG_DSN`` 指向另一套库::

       $env:RAG_PG_DSN = "postgresql://user:pass@host:5432/other"

2. 项目根目录（``backend/``）下的 ``.env`` 文件，适合放本地数据库口令::

       RAG_PG_DSN=postgresql://postgres:123456@localhost:5432/rag

   ``.env`` 已在 ``.gitignore`` 里，不会进版本库；把 ``.env.example``
   复制成 ``.env`` 再改即可。这里只做极简解析（``KEY=VALUE``），不引入额外依赖，
   且**不覆盖已存在的环境变量**。
"""

from __future__ import annotations

import os
from pathlib import Path

# config.py 位于 app/ 下，上一级就是项目根（backend/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | str) -> int:
    """把 ``KEY=VALUE`` 逐行读进 ``os.environ``，返回实际注入的条目数。

    规则与常见实现保持一致：跳过空行与 ``#`` 注释、去掉值两端的引号、
    已存在的环境变量不覆盖（命令行/系统变量优先）。
    """
    target = Path(path)
    if not target.is_file():
        return 0

    injected = 0
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            injected += 1
    return injected


# 先读 .env（项目根目录优先，其次当前工作目录），再读取下面的常量
for _candidate in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
    load_dotenv(_candidate)

# --------------------------------------------------------------------------- #
# 数据库
# --------------------------------------------------------------------------- #
# vector_store（写入/向量检索）与 bm25_index（词法检索）共用同一个库，
# 库中需要已安装 pgvector 扩展。
#
# 默认值只是占位符：真实密码放在环境变量或 .env 里，不要写进代码。
DSN = os.environ.get("RAG_PG_DSN", "postgresql://postgres:postgres@localhost:5432/rag")

# --------------------------------------------------------------------------- #
# 路径（相对项目根目录，请在根目录下运行）
# --------------------------------------------------------------------------- #
# 嵌入模型：BGE-M3，产出 1024 维稠密向量 + 稀疏权重
EMBEDDING_MODEL_PATH = "models/bge-m3"
# 精排模型：BGE-Reranker-v2-m3（交叉编码器）
RERANKER_MODEL_PATH = "models/bge-reranker-v2-m3"
# 解析产物：每行一个 chunk
CHUNKS_PATH = "data/parsed/chunks.jsonl"
# BM25 缓存（分词结果 + 元数据），新增 chunk 后需要 --rebuild
BM25_INDEX_PATH = "data/index/bm25_index.pkl"

# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #
# 每积累多少行提交一次：太小则 fsync 次数多，太大则崩溃时丢的已编码结果多
DB_BATCH_SIZE = 200

# --------------------------------------------------------------------------- #
# 召回
# --------------------------------------------------------------------------- #
# 单路检索接口（vector_store.search / bm25_index.search）的默认返回条数
SEARCH_TOP_K = 10
# 混合检索时每一路召回的条数，也就是送进 RRF 的候选池大小
HYBRID_TOP_K = 20
# RRF 融合后保留的条数
HYBRID_TOP_N = 10
# 送进精排的候选数（reranker CLI 的默认值），一般 >= HYBRID_TOP_N
RERANK_CANDIDATES = 20

# --------------------------------------------------------------------------- #
# 融合（RRF）
# --------------------------------------------------------------------------- #
# 平滑常数：越大越看重「被几路同时召回」，越小越看重「单路名次靠前」
RRF_K = 60
# 两路权重，顺序为 (向量, BM25)；如 (2.0, 1.0) 表示更信任语义检索，
# (1.0, 2.0) 表示更信任关键词精确匹配
RRF_WEIGHTS = (1.0, 3.0)

# --------------------------------------------------------------------------- #
# 精度策略
# --------------------------------------------------------------------------- #
# 嵌入模型（BGE-M3）精度。None 表示按设备自动：GPU 用 bf16、CPU 用 fp32。
# 本机 GTX 1650 SUPER 实测 fp16 比 fp32 慢 5 倍，所以 auto 不选 fp16。
EMBED_USE_FP16: bool | None = None
EMBED_USE_BF16: bool | None = None
# 精排模型（交叉编码器）精度。None 表示有 CUDA 时用 fp16：
# 实测同一张卡上精排 fp16 比 fp32 快 15%~40%，且排序结果一致。
RERANK_USE_FP16: bool | None = None

# --------------------------------------------------------------------------- #
# 精排
# --------------------------------------------------------------------------- #
# 精排后返回的条数
RERANK_TOP_M = 5
# 交叉编码器一次前向的批大小；4 GB 显存下 8 比较稳
RERANK_BATCH_SIZE = 8
# 单对 (query, passage) 的最大 token 数；调到 256 大约快 1.5 倍
RERANK_MAX_LENGTH = 512

# --------------------------------------------------------------------------- #
# 生成（本地 llama.cpp 服务）
# --------------------------------------------------------------------------- #
# llama.cpp 的 OpenAI 兼容接口地址，对应 `llama-server --host 127.0.0.1 --port 8080`
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
# 实测（GTX 1650 SUPER，-ngl 999）：qwen2.5-3b 解码 34 tok/s、裁判单次 0.9s；
# Qwen3.5-4B 只有 19 tok/s、1.9s，且需要关闭思考才有正文。评测用前者。
LLM_MODEL = "qwen2.5-3b-instruct-q5_k_m.gguf"
# 单次请求超时（秒）；3B 模型在 CPU/GPU 上生成 512 token 一般几十秒内
LLM_TIMEOUT = 120
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# --------------------------------------------------------------------------- #
# 提示词组装
# --------------------------------------------------------------------------- #
# 送进提示词的资料文本：单条最多保留多少字符（表格块可能上千字符）
RAG_MAX_CHUNK_CHARS = 800
# 所有资料加起来的字符预算，避免超出模型上下文
RAG_MAX_CONTEXT_CHARS = 4000

# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
# Golden Set 与评估结果目录
EVAL_GOLDEN_PATH = "data/golden_set.json"
EVAL_RESULTS_DIR = "data/eval_results"
# 三阶段的召回参数（baseline 只走向量；hybrid 融合后取 5；rerank 精排取 5）
EVAL_BASELINE_TOP_K = 5
EVAL_HYBRID_TOP_N = 5
EVAL_RERANK_CANDIDATES = 10
EVAL_RERANK_TOP_M = 5
# 裁判打分温度：评分要可复现，固定为 0
JUDGE_TEMPERATURE = 0.0
# Qwen3.5 是思考模型：llama.cpp 会把推理写进 reasoning_content，content 可能一直为空。
# 实测关闭思考后同一请求 2.2s 返回、且直接给 JSON；不关则 300 token 也拿不到正文。
LLM_DISABLE_THINKING = True

# --------------------------------------------------------------------------- #
# 降级策略
# --------------------------------------------------------------------------- #
# 单路检索失败（PG 断连 / 模型加载失败）时，是否降级为「另一路直通」。
# 关掉则还原成「任一路失败即整体报错」的行为。
DEGRADE_ON_ROUTE_FAILURE = True
# 精排不可用（模型缺失 / 显存不足）时，是否退回 RRF 融合后的原始顺序。
DEGRADE_ON_RERANK_FAILURE = True
# LLM 不可用时，是否返回抽取式兜底答案（会显式把 llm 记进 degraded 字段）。
DEGRADE_ON_LLM_FAILURE = True
# 抽取式兜底答案里，最多带多少字符的原文
FALLBACK_ANSWER_CHARS = 300

# --------------------------------------------------------------------------- #
# 问题追踪线索
# --------------------------------------------------------------------------- #
# 各阶段预览文本的截断长度（vector / bm25 / rrf 的 text[:N]）
TRACE_PREVIEW_CHARS = 80
# /ask 是否默认返回追踪线索；关掉可减小响应体
TRACE_ENABLED = True
