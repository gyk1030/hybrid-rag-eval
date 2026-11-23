"""FastAPI 后端：把用户问题交给 RAG 链路生成答案。

接口::

    GET /ask?q=国华网安2024年的研发投入是多少        → 答案 + 问题 + 检索结果
    GET /health                                      → 服务与本地 LLM 的状态
    GET /                                            → 健康探针

启动::

    python -m app.main                      # 或 python app/main.py
    uvicorn app.main:app --reload           # 开发模式

注意：这里的接口用同步 ``def`` 定义，FastAPI 会把它们丢到线程池执行，
因此阻塞的检索/生成调用不会卡住事件循环；首次请求会触发模型懒加载（几十秒）。
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query

# 允许用脚本方式直接运行（python app/main.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    LLM_BASE_URL,
    RERANK_CANDIDATES,
    RERANK_TOP_M,
    TRACE_ENABLED,
)
from app.generation.answer import (  # noqa: E402
    LLMError,
    RetrievalUnavailable,
    answer_question,
    pipeline_status,
)

app = FastAPI(
    title="hybrid-rag-eval",
    description="向量 + BM25 混合检索 → RRF 融合 → BGE 精排 → 本地 llama.cpp 生成",
    version="0.1.0",
)


@app.get("/")
def root() -> dict:
    """最简单的存活探针。"""
    return {"message": True}


@app.get("/health")
def health() -> dict:
    """返回服务状态、本地 LLM 是否可达，以及检索栈各组件是否已就绪。

    ``pipeline`` 只反映「已经尝试加载过的组件」，不会为了健康检查去加载模型。
    """
    reachable = False
    detail = "未检查"
    try:
        response = requests.get(f"{LLM_BASE_URL.rstrip('/')}/models", timeout=3)
        reachable = response.status_code == 200
        detail = f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        detail = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok",
        "llm": {"base_url": LLM_BASE_URL, "reachable": reachable, "detail": detail},
        "pipeline": pipeline_status(),
    }


@app.get("/ask")
def ask(
    q: str = Query(..., min_length=1, description="用户问题"),
    top_m: int = Query(RERANK_TOP_M, ge=1, le=20, description="送进提示词的资料条数"),
    candidates: int = Query(RERANK_CANDIDATES, ge=1, le=100, description="精排候选数"),
    trace: bool = Query(TRACE_ENABLED, description="是否返回各阶段追踪线索"),
) -> dict:
    """GET /ask?q=... → ``{question, answer, contexts, degraded, warnings, trace, ...}``。

    降级契约：``degraded`` 为空表示全链路正常；非空时 ``answer`` 的可信度下降，
    客户端应当把它展示出来（例如在 UI 上加一条提示）。

    追踪：``trace`` 记录 vector / bm25 / rrf / rerank / context / timing / fallback
    七个阶段，用来回答「哪一步把正确片段弄丢了」。``trace=false`` 可关掉以减小响应体。
    """
    try:
        result = answer_question(q, top_m=top_m, candidates=candidates, trace=trace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RetrievalUnavailable as exc:
        # 两路检索都挂了：没有任何证据可给，只能明确失败
        raise HTTPException(status_code=503, detail=f"检索不可用：{exc}") from exc

    payload = asdict(result)
    # LLM 也不可用且没检索到任何资料时，兜底答案没有信息量，按失败返回更诚实
    if "llm" in result.degraded and not result.contexts:
        raise HTTPException(
            status_code=503,
            detail="本地 LLM 不可用，且未检索到任何资料：" + "；".join(result.warnings),
        )
    return payload


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
