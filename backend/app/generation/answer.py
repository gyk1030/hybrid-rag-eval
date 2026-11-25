"""RAG 回答生成：检索 + 精排 → 组装提示词 → 调用本地 llama.cpp 生成答案。

完整链路::

    问题
      → 混合检索（向量 + BM25 → RRF，取 TopN 候选）
      → BGE-Reranker 精排（取 TopM）
      → 组装提示词（参考资料 + 问题）
      → llama.cpp 的 OpenAI 兼容接口 /chat/completions
      → AnswerResult（答案 + 问题 + 检索结果 + 用量 + 耗时）

用法::

    python -m app.generation.answer -q "国华网安2024年的研发投入是多少？"
    python -m app.generation.answer -q "..." --top-m 3 --json

作为库使用::

    from app.generation.answer import answer_question

    result = answer_question("研发投入是多少？", top_m=5)
    print(result.answer)
    for ctx in result.contexts:
        print(ctx.index, ctx.chunk_id, ctx.score)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

# 允许用脚本方式直接运行（python app/generation/answer.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    CHUNKS_PATH,
    DEGRADE_ON_LLM_FAILURE,
    DEGRADE_ON_RERANK_FAILURE,
    FALLBACK_ANSWER_CHARS,
    LLM_BASE_URL,
    LLM_DISABLE_THINKING,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    RAG_MAX_CHUNK_CHARS,
    RAG_MAX_CONTEXT_CHARS,
    RERANK_CANDIDATES,
    RERANK_TOP_M,
    TRACE_ENABLED,
    TRACE_PREVIEW_CHARS,
)

from app.generation.consistency import check_year_consistency  # noqa: E402

logger = logging.getLogger("app.generation.answer")

# 注意：这里刻意**不**要求模型在答案里标注 [编号]。实测 3B 模型在这种情况下会把
# 编号写错（例如引用到别家公司的资料）或复述资料头部，反而降低可信度；出处由
# AnswerResult.contexts 里的 index/source/page 提供，更准确。
#
# 「年份规则」写成了可执行的判断（跳过哪条、什么情况可以跨年），而不是「请核对」：
# 实测只写「请核对年份」时，3B 模型会拿 2024 年报表格里的 1.39% 回答「2023 年末」，
# 而正确的 1.37% 当时就在上下文里。生成后另有 consistency.check_year_consistency
# 做确定性兜底。
SYSTEM_PROMPT = (
    "你是中文财报问答助手。\n"
    "回答步骤：先在【参考资料】中逐条查找与问题直接相关的数字或事实，"
    "找到就把该数字原样写进答案（保留单位）。"
    "资料里的“□适用 √不适用”是勾选项，可以忽略。"
    "答案控制在 3 句以内。\n"
    "每条参考资料开头都标了来源文件名，文件名里的年份就是这份年报的年度。\n"
    "【年份规则】先确认问题问的年份：\n"
    "1. 问题只有一个年份（如“2023 年末”）时，只能用该年份年报的资料；"
    "文件名年份不同的资料一律跳过，即使里面的数字看起来更接近问题。\n"
    "2. 问题同时提到多个年份（如“2025 相对 2024 年”）时，这几年的年报都可以用。\n"
    "3. 问题没有年份时不受此限。\n"
    "【公司规则】只能用与问题所指公司一致的资料；拿别家公司的数字作答属于严重错误。\n"
    "如果没有任何资料同时满足公司和年份要求，回答“根据提供的资料无法回答该问题”。"
)


class LLMError(RuntimeError):
    """调用本地 LLM 失败（服务未启动、超时、响应异常）。"""


class RetrievalUnavailable(RuntimeError):
    """向量与 BM25 两路都不可用，没有任何可用的检索手段。"""


@dataclass(slots=True)
class RetrievedContext:
    """送进提示词的一条参考资料。"""

    index: int
    chunk_id: str
    score: float
    text: str
    source: str
    page: int
    block_type: str
    rrf_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
    """一次问答的完整结果。"""

    question: str
    answer: str
    contexts: list[RetrievedContext]
    model: str
    messages: list[dict]
    usage: dict | None = None
    bootstrap_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    elapsed_ms: float = 0.0
    # 本次问答中发生降级的环节，可能包含 "vector" / "bm25" / "rerank" / "llm"。
    # 空列表表示全链路正常。调用方必须检查这个字段——非空时 answer 的可信度下降。
    degraded: list[str] = field(default_factory=list)
    # 每个降级环节的具体原因（异常类型 + 消息），便于排查
    warnings: list[str] = field(default_factory=list)
    # 各阶段追踪线索（vector / bm25 / rrf / rerank / context / timing / fallback）。
    # trace=False 时为 None。
    trace: dict | None = None
    # 年份一致性校验结果：
    # {"status": "ok"|"mismatch"|"skipped", "asked_years": [...], "answer_years": [...],
    #  "source_year": 2024, "source_index": 4, "detail": "..."}
    # status=mismatch 时 answer 不可信（数字是真的，但年份配错了）。
    answer_check: dict | None = None


# --------------------------------------------------------------------------- #
# 检索栈（懒加载，进程内只建一次）
# --------------------------------------------------------------------------- #
# 注意：这里缓存的是「组件对象」，不缓存「失败状态」——加载失败的组件不会写入，
# 因此下次调用会重新尝试。PG 重启、显存释放后无需重启服务即可自愈。
_PIPELINE: dict[str, Any] = {}
_PIPELINE_ERRORS: dict[str, str] = {}
_PIPELINE_LOCK = threading.Lock()


def _build_store() -> Any:
    """构造向量库并**预热嵌入模型**。

    预热是为了让追踪里的 ``vector_ms`` 只反映「编码查询 + SQL」的耗时：
    否则首次请求的 30 秒模型加载会被算进检索，看不出真实查询有多快。
    """
    from app.retrieval.vector_store import VectorStore, get_model

    store = VectorStore(CHUNKS_PATH)
    get_model()  # 触发 BGE-M3 加载
    return store


def _build_index() -> Any:
    """构造 BM25 索引并**预热 Okapi 结构**（否则首次查询要额外几百毫秒）。"""
    from app.retrieval.bm25_index import BM25Index

    index = BM25Index().build_or_load()
    index._ensure_ready()
    return index


def _build_reranker() -> Any:
    """构造精排器并**预热交叉编码器**，理由同 :func:`_build_store`。"""
    from app.retrieval.reranker import Reranker

    reranker = Reranker()
    reranker.model  # 触发 BGE-Reranker 加载
    return reranker


def _load_component(name: str, factory: Callable[[], Any]) -> Any:
    """加载单个组件；失败时记录原因并返回 None，不向外抛异常。"""
    started = time.perf_counter()
    try:
        component = factory()
    except Exception as exc:  # noqa: BLE001 - 任一组件不可用都不该让进程起不来
        reason = f"{type(exc).__name__}: {exc}"
        _PIPELINE_ERRORS[name] = reason
        logger.warning("检索组件 %s 加载失败，本次将降级运行：%s", name, reason)
        return None
    _PIPELINE_ERRORS.pop(name, None)
    # 记录各组件加载耗时，供追踪区分「加载慢」与「查询慢」
    _PIPELINE.setdefault("load_ms", {})[name] = round((time.perf_counter() - started) * 1000, 1)
    return component


def get_pipeline() -> dict[str, Any]:
    """懒加载向量库、BM25 索引与精排模型。

    单个组件失败不会影响其他组件：失败的组件值为 None，并在
    :data:`_PIPELINE_ERRORS` 里留下原因。调用方按「谁可用用谁」的方式降级。

    三个组件都会在加载时预热各自的模型/结构，因此后续各阶段的耗时只包含查询本身。
    """
    with _PIPELINE_LOCK:
        if _PIPELINE.get("store") is None:
            _PIPELINE["store"] = _load_component("store", _build_store)
        if _PIPELINE.get("index") is None:
            _PIPELINE["index"] = _load_component("index", _build_index)
        if _PIPELINE.get("reranker") is None:
            _PIPELINE["reranker"] = _load_component("reranker", _build_reranker)
    return _PIPELINE


def pipeline_status() -> dict[str, Any]:
    """当前检索栈的可用性（不触发加载），供 /health 使用。"""
    components = {name: _PIPELINE.get(name) is not None for name in ("store", "index", "reranker")}
    return {
        "loaded": components,
        "errors": dict(_PIPELINE_ERRORS),
        "load_ms": dict(_PIPELINE.get("load_ms") or {}),
        "ready": all(components.values()),
    }


# --------------------------------------------------------------------------- #
# 提示词组装
# --------------------------------------------------------------------------- #
def build_contexts(hits: Sequence[Any], *, top_m: int | None = None) -> list[RetrievedContext]:
    """把精排结果转成资料列表，并按字符预算截断。

    逐行清洗（保留 Markdown 表格的换行结构），单条超过 ``RAG_MAX_CHUNK_CHARS``
    截断，全部资料合计不超过 ``RAG_MAX_CONTEXT_CHARS``。
    """
    selected = list(hits[:top_m]) if top_m else list(hits)
    contexts: list[RetrievedContext] = []
    budget = RAG_MAX_CONTEXT_CHARS

    for position, hit in enumerate(selected, start=1):
        if budget <= 0:
            break

        raw = str(getattr(hit, "text", "") or "")
        text = "\n".join(" ".join(line.split()) for line in raw.splitlines() if line.strip())
        if len(text) > RAG_MAX_CHUNK_CHARS:
            text = text[:RAG_MAX_CHUNK_CHARS] + "…"
        if len(text) > budget:
            text = text[:budget] + "…"
        budget -= len(text)

        fused = getattr(hit, "fused_score", None)
        contexts.append(
            RetrievedContext(
                index=position,
                chunk_id=str(getattr(hit, "chunk_id", "")),
                score=float(getattr(hit, "score", 0.0)),
                text=text,
                source=str(getattr(hit, "source", "")),
                page=int(getattr(hit, "page", 0) or 0),
                block_type=str(getattr(hit, "block_type", "")),
                rrf_score=float(fused) if fused is not None else None,
                ranks=dict(getattr(hit, "ranks", {}) or {}),
            )
        )
    return contexts


def build_messages(question: str, contexts: Sequence[RetrievedContext]) -> list[dict[str, str]]:
    """把资料和问题拼成 chat completions 需要的消息列表。

    资料头部用 ``<资料N｜文件：…｜第X页｜正文>`` 这种尖括号格式：实测比
    ``[N] 来源：xxx.pdf 第X页`` 更不容易被模型当成正文续写（后者会导致答案末尾
    原样复述来源行）。
    """
    blocks = []
    for ctx in contexts:
        kind = "表格" if ctx.block_type == "table" else "正文"
        name = Path(ctx.source).name or "未知来源"
        blocks.append(f"<资料{ctx.index}｜文件：{name}｜第{ctx.page}页｜{kind}>\n{ctx.text}")

    material = "\n\n".join(blocks) if blocks else "（未检索到相关资料）"
    # 问题放在资料前面：3B 小模型对「先看问题、再找资料」的指令遵循明显更好
    user_content = (
        f"【问题】\n{question.strip()}\n\n"
        f"【参考资料】\n{material}\n\n"
        "请按上述步骤作答。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
# 调用本地 LLM
# --------------------------------------------------------------------------- #
def call_llm(
    messages: Sequence[dict[str, str]],
    *,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: int = LLM_TIMEOUT,
    base_url: str = LLM_BASE_URL,
) -> tuple[str, dict | None]:
    """调用 OpenAI 兼容的 ``/chat/completions``，返回 ``(答案文本, usage)``。

    Qwen3.5 这类思考模型会把推理写进 ``reasoning_content``，正文可能整个为空
    （实测 1200 token 预算也拿不到 content）。因此默认按
    :data:`app.config.LLM_DISABLE_THINKING` 关闭思考：
    ``chat_template_kwargs={"enable_thinking": False}``。
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if LLM_DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(
            f"无法连接本地 LLM 服务（{url}）：{type(exc).__name__}: {exc}\n"
            "请确认 llama.cpp 已启动：llama-server -m <模型.gguf> --host 127.0.0.1 --port 8080"
        ) from exc

    if response.status_code != 200:
        raise LLMError(f"LLM 返回 HTTP {response.status_code}：{response.text[:300]}")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"LLM 响应格式异常：{response.text[:300]}") from exc

    return str(content).strip(), data.get("usage")


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def _extractive_answer(contexts: Sequence[RetrievedContext]) -> str:
    """抽取式兜底答案：原样取最相关的一段原文，并明确标注「未经改写」。

    这个函数存在的唯一理由是 LLM 不可用时仍然能让调用方拿到证据，
    因此必须一眼看出它**不是**生成结果，不能伪装成正常回答。
    """
    if not contexts:
        return "未检索到相关资料，且本地 LLM 不可用。"

    head = contexts[0]
    text = " ".join(head.text.split())[:FALLBACK_ANSWER_CHARS]
    name = Path(head.source).name or "未知来源"
    return (
        "（本地 LLM 不可用，以下为检索到的最相关原文，未经改写）\n"
        f"{text}\n"
        f"—— 出处：{name} 第 {head.page} 页"
    )


def answer_question(
    question: str,
    *,
    top_m: int = RERANK_TOP_M,
    candidates: int = RERANK_CANDIDATES,
    include_messages: bool = True,
    trace: bool = TRACE_ENABLED,
) -> AnswerResult:
    """完整链路：检索 → 精排 → 提示词 → LLM → :class:`AnswerResult`。

    ``trace=True`` 时在结果里带上 :class:`app.generation.trace.TraceRecorder`
    记录的各阶段线索（vector / bm25 / rrf / rerank / context / timing / fallback）。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    if top_m < 1:
        raise ValueError("top_m 必须 >= 1")
    if candidates < 1:
        raise ValueError("candidates 必须 >= 1")

    from app.generation.trace import TraceRecorder  # 延迟导入，避免追踪影响纯提示词测试
    from app.retrieval.rrf import hybrid_search

    recorder = TraceRecorder(question) if trace else None

    # 首次请求要加载两个模型 + BM25 索引（几十秒），单独计时，避免混进检索耗时
    bootstrap_started = time.perf_counter()
    pipeline = get_pipeline()
    bootstrap_ms = (time.perf_counter() - bootstrap_started) * 1000
    if recorder is not None:
        recorder.add_timing("bootstrap_ms", bootstrap_ms)
        # 把各组件的加载耗时单列出来：慢在加载还是慢在查询，一眼可分
        for name, elapsed in (pipeline.get("load_ms") or {}).items():
            recorder.add_timing(f"load_{name}_ms", elapsed)

    degraded: list[str] = []
    warnings: list[str] = []

    store = pipeline.get("store")
    index = pipeline.get("index")
    reranker = pipeline.get("reranker")

    # 两路都不可用就没有任何检索手段——这时必须报错，不能假装降级成功
    if store is None and index is None:
        detail = "；".join(f"{k}: {v}" for k, v in _PIPELINE_ERRORS.items()) or "未知原因"
        raise RetrievalUnavailable(f"向量检索与 BM25 均不可用（{detail}）")

    started = time.perf_counter()

    route_failures: list[str] = []
    hits = hybrid_search(
        question,
        top_k=candidates,
        top_n=candidates,
        store=store,
        index=index,
        failures=route_failures,
        build_missing=False,
        trace=recorder,
    )
    for failure in route_failures:
        name = failure.split(":", 1)[0].strip()
        if name not in degraded:
            degraded.append(name)
        warnings.append(failure)
        logger.warning("检索路径降级 —— %s", failure)

    # 精排：失败就退回 RRF 融合顺序（FusedHit 已经按相关性排好）
    # 追踪单独记录精排「前 M 条」的原始顺序，才能判断是不是重排把片段调出了窗口
    rerank_ms: float | None = None
    before_rerank = list(hits[:top_m]) if recorder is not None else []

    if hits:
        if reranker is None:
            reason = _PIPELINE_ERRORS.get("reranker", "未初始化")
            if DEGRADE_ON_RERANK_FAILURE:
                degraded.append("rerank")
                warnings.append(f"rerank: 不可用（{reason}）")
                reranked = list(hits[:top_m])
                if recorder is not None:
                    recorder.mark_fallback("rerank", f"不可用（{reason}）")
            else:
                raise RuntimeError(f"精排不可用：{reason}")
        else:
            rerank_started = time.perf_counter()
            try:
                reranked = reranker.rerank(question, hits, top_m=top_m)
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
            except Exception as exc:  # noqa: BLE001 - 显存不足/模型损坏都走这里
                if not DEGRADE_ON_RERANK_FAILURE:
                    raise
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
                degraded.append("rerank")
                warnings.append(f"rerank: {type(exc).__name__}: {exc}")
                logger.warning("精排失败，退回 RRF 顺序：%s", exc)
                reranked = list(hits[:top_m])
                if recorder is not None:
                    recorder.mark_fallback("rerank", f"{type(exc).__name__}: {exc}")
    else:
        reranked = []
        if recorder is not None:
            recorder.mark_fallback("retrieval", "没有召回到任何片段")

    retrieval_ms = (time.perf_counter() - started) * 1000
    if recorder is not None:
        recorder.add_timing("retrieval_ms", retrieval_ms)
        recorder.record_rerank(before_rerank, reranked, rerank_ms)

    context_started = time.perf_counter()
    contexts = build_contexts(reranked)
    if recorder is not None:
        recorder.record_context(contexts, (time.perf_counter() - context_started) * 1000)

    messages = build_messages(question, contexts)

    generation_started = time.perf_counter()
    usage: dict | None = None
    try:
        answer, usage = call_llm(messages)
    except LLMError as exc:
        if not DEGRADE_ON_LLM_FAILURE:
            raise
        # 抽取式兜底：把最相关的一段原文原样返回，并在 degraded 里明确标注。
        # 绝不能伪装成生成答案——调用方靠 degraded 字段区分。
        degraded.append("llm")
        warnings.append(f"llm: {type(exc).__name__}: {exc}")
        logger.warning("LLM 不可用，返回抽取式兜底答案：%s", exc)
        answer = _extractive_answer(contexts)
        if recorder is not None:
            recorder.mark_fallback("llm", f"{type(exc).__name__}: {exc}")
    generation_ms = (time.perf_counter() - generation_started) * 1000
    if recorder is not None:
        recorder.add_timing("generation_ms", generation_ms)
        recorder.add_timing("total_ms", bootstrap_ms + retrieval_ms + generation_ms)

    # 年份一致性校验：确定性兜底，不依赖模型自觉。
    # 3B 模型实测会拿别年度资料作答（数字是真的、年份是错的），这里把它拦下来。
    check = check_year_consistency(question, answer, contexts)
    if check.status == "mismatch":
        warnings.append(f"year_mismatch: {check.detail}")
        logger.warning("年份校验未通过 —— %s", check.detail)

    return AnswerResult(
        question=question,
        answer=answer,
        contexts=contexts,
        model=LLM_MODEL,
        messages=list(messages) if include_messages else [],
        usage=usage,
        bootstrap_ms=round(bootstrap_ms, 1),
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        elapsed_ms=round(bootstrap_ms + retrieval_ms + generation_ms, 1),
        degraded=degraded,
        warnings=warnings,
        trace=recorder.to_dict() if recorder is not None else None,
        answer_check=check.to_dict(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.generation.answer",
        description="检索 + 精排 + 本地 LLM 生成答案。",
    )
    parser.add_argument("-q", "--question", required=True, help="用户问题")
    parser.add_argument(
        "-m", "--top-m", type=int, default=RERANK_TOP_M, help=f"送进提示词的资料条数，默认 {RERANK_TOP_M}"
    )
    parser.add_argument(
        "-c",
        "--candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help=f"精排候选数，默认 {RERANK_CANDIDATES}",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON（含 messages 与 trace）")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="打印各阶段追踪线索（vector / bm25 / rrf / rerank / context / timing / fallback）",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=TRACE_PREVIEW_CHARS,
        help=f"追踪线索里每条预览的字符数，默认 {TRACE_PREVIEW_CHARS}",
    )
    args = parser.parse_args(argv)

    try:
        result = answer_question(
            args.question,
            top_m=args.top_m,
            candidates=args.candidates,
            trace=args.json or args.trace,
        )
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    except RetrievalUnavailable as exc:
        print(f"检索不可用：{exc}", file=sys.stderr)
        return 4

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0

    if args.trace:
        from app.generation.trace import TraceRecorder

        recorder = TraceRecorder(args.question, preview_chars=args.preview)
        # 复用已算好的 trace 数据渲染，避免为打印重跑一遍检索
        recorder.vector = _rehydrate(result.trace, "vector")
        recorder.bm25 = _rehydrate(result.trace, "bm25")
        recorder.rrf = _rehydrate(result.trace, "rrf")
        recorder.rerank_before = _rehydrate(result.trace, "rerank.before")
        recorder.rerank_after = _rehydrate(result.trace, "rerank.after")
        recorder.context = _rehydrate_context(result.trace)
        recorder.timing = dict((result.trace or {}).get("timing") or {})
        recorder.fallback = list((result.trace or {}).get("fallback") or [])
        recorder.warnings = list((result.trace or {}).get("warnings") or [])
        print(recorder.render())
        print()

    print(f"问题：{result.question}")
    print(f"检索到 {len(result.contexts)} 条资料（{result.retrieval_ms:.0f} ms）：")
    for ctx in result.contexts:
        print(
            f"  [{ctx.index}] {Path(ctx.source).name} 第{ctx.page}页 "
            f"{ctx.block_type} 精排分={ctx.score:.4f}"
        )
    print(f"\n回答（{result.generation_ms:.0f} ms，模型 {result.model}）：\n{result.answer}")
    if result.usage:
        print(f"\n用量：{result.usage}")
    if result.answer_check:
        status = result.answer_check.get("status")
        detail = result.answer_check.get("detail", "")
        if status == "mismatch":
            print(f"\n⚠ 年份校验未通过：{detail}")
        elif status == "ok":
            print(f"\n年份校验：通过（{detail}）")
        else:
            print(f"\n年份校验：跳过（{detail}）")
    if result.degraded:
        print(f"\n⚠ 本次发生降级：{', '.join(result.degraded)}")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


def _rehydrate(trace: dict | None, path: str):
    """把 to_dict() 出来的普通字典还原成 StageHit，供 CLI 复用渲染逻辑。"""
    from app.generation.trace import StageHit

    node: Any = trace or {}
    for key in path.split("."):
        node = (node or {}).get(key) or []
    return [
        StageHit(
            rank=int(item.get("rank", 0)),
            chunk_id=str(item.get("chunk_id", "")),
            score=float(item.get("score", 0.0)),
            preview=str(item.get("preview", "")),
            ranks=dict(item.get("ranks") or {}),
            pool_rank=int(item["pool_rank"]) if item.get("pool_rank") is not None else None,
        )
        for item in (node or [])
    ]


def _rehydrate_context(trace: dict | None):
    from app.generation.trace import ContextEntry

    return [
        ContextEntry(
            index=int(item.get("index", 0)),
            chunk_id=str(item.get("chunk_id", "")),
            source=str(item.get("source", "")),
            page=int(item.get("page", 0)),
            block_type=str(item.get("block_type", "")),
            chars=int(item.get("chars", 0)),
            score=float(item.get("score", 0.0)),
        )
        for item in ((trace or {}).get("context") or [])
    ]


if __name__ == "__main__":
    sys.exit(main())
