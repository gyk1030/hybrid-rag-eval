"""问题追踪线索（query trace）：记录一次问答在每个检索阶段看到了什么。

设计目标：出问题时能回答「哪一步把正确片段弄丢了」，而不是只看最终上下文猜。

七个阶段对应 :class:`TraceRecorder` 的七个区块：

===========  ==================================================  ==============================
阶段          记录内容                                            用途
===========  ==================================================  ==============================
vector       Top-K 的 chunk_id / score / text[:80]               判断向量路是否召回了正确片段
bm25         Top-K 的 chunk_id / score / text[:80]               判断关键词路是否召回了正确片段
rrf          融合后有序 chunk_id + RRF score + 各路名次           判断融合是否把正确片段压下去
rerank       精排前后各 Top-M 的 chunk_id + 分数                   判断重排是否把正确片段调出窗口
context      最终进入 Prompt 的 chunk_id 列表                     模型实际看到什么
timing       各阶段耗时（ms）                                      定位慢查询
fallback     降级标记与原因                                        解释「为什么这次结果和平时不一样」
===========  ==================================================  ==============================

用法::

    recorder = TraceRecorder(question)
    hits = hybrid_search(question, trace=recorder)   # 记录 vector / bm25 / rrf
    reranked = reranker.rerank(question, hits, top_m=5)
    recorder.record_rerank(hits[:5], reranked)
    contexts = build_contexts(reranked)
    recorder.record_context(contexts)
    trace = recorder.to_dict()

:mod:`app.retrieval.rrf` 只是按鸭子类型调用 ``record_vector`` / ``record_bm25`` /
``record_rrf``，不 import 本模块，因此两者之间没有循环依赖。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import TRACE_PREVIEW_CHARS


def preview(text: str, limit: int = TRACE_PREVIEW_CHARS) -> str:
    """把文本压成一行短预览：折叠空白后截断，超长加省略号。"""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


@dataclass(slots=True)
class StageHit:
    """某个阶段的一条命中记录。"""

    rank: int
    chunk_id: str
    score: float
    preview: str
    # 仅 rrf 阶段有值：该片段在各路的原始名次，如 {"vector": 4, "bm25": 1}
    ranks: dict[str, int] = field(default_factory=dict)
    # 仅精排阶段有值：该片段在「融合后完整候选池」里的名次。
    # 判断「重排是否把片段调出/调进窗口」靠的就是它，而不是前后各 Top-M。
    pool_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "preview": self.preview,
        }
        if self.ranks:
            payload["ranks"] = dict(self.ranks)
        if self.pool_rank is not None:
            payload["pool_rank"] = self.pool_rank
        return payload


@dataclass(slots=True)
class ContextEntry:
    """最终进入 Prompt 的一条资料。"""

    index: int
    chunk_id: str
    source: str
    page: int
    block_type: str
    chars: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": Path(self.source).name,
            "page": self.page,
            "block_type": self.block_type,
            "chars": self.chars,
            "score": round(self.score, 6),
        }


def _to_stage_hits(
    hits: Iterable[Any],
    *,
    limit: int | None = None,
    keep_ranks: bool = False,
    preview_chars: int = TRACE_PREVIEW_CHARS,
) -> list[StageHit]:
    selected = list(hits)
    if limit is not None:
        selected = selected[:limit]
    records: list[StageHit] = []
    for rank, hit in enumerate(selected, start=1):
        records.append(
            StageHit(
                rank=rank,
                chunk_id=str(getattr(hit, "chunk_id", "")),
                score=float(getattr(hit, "score", 0.0) or 0.0),
                preview=preview(getattr(hit, "text", ""), preview_chars),
                ranks=dict(getattr(hit, "ranks", {}) or {}) if keep_ranks else {},
            )
        )
    return records


class TraceRecorder:
    """按阶段收集一次问答的追踪线索。

    所有 ``record_*`` 方法都是「记了就不抛错」：追踪本身不能影响主链路。
    """

    def __init__(self, query: str, *, preview_chars: int = TRACE_PREVIEW_CHARS) -> None:
        self.query = query
        self.preview_chars = preview_chars
        self.vector: list[StageHit] = []
        self.bm25: list[StageHit] = []
        self.rrf: list[StageHit] = []
        self.rerank_before: list[StageHit] = []
        self.rerank_after: list[StageHit] = []
        self.context: list[ContextEntry] = []
        self.timing: dict[str, float] = {}
        self.fallback: list[str] = []
        self.warnings: list[str] = []
        # 融合后完整候选池里的名次，用于给精排结果标注「原来排第几」
        self.pool_rank: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 记录
    # ------------------------------------------------------------------ #
    def record_vector(self, hits: Sequence[Any], elapsed_ms: float) -> None:
        self.vector = _to_stage_hits(hits, preview_chars=self.preview_chars)
        self.timing["vector_ms"] = round(elapsed_ms, 1)

    def record_bm25(self, hits: Sequence[Any], elapsed_ms: float) -> None:
        self.bm25 = _to_stage_hits(hits, preview_chars=self.preview_chars)
        self.timing["bm25_ms"] = round(elapsed_ms, 1)

    def record_rrf(self, hits: Sequence[Any], elapsed_ms: float) -> None:
        self.rrf = _to_stage_hits(hits, keep_ranks=True, preview_chars=self.preview_chars)
        self.timing["rrf_ms"] = round(elapsed_ms, 1)
        # 记住完整候选池的名次：精排是在池内重排，不是只看前 M 条
        self.pool_rank = {hit.chunk_id: hit.rank for hit in self.rrf}

    def record_route_failure(self, name: str, detail: str) -> None:
        """某一路检索失败：该阶段留空，并记下降级原因。"""
        setattr(self, name, [])
        self.timing[f"{name}_ms"] = 0.0
        self.mark_fallback(name, detail)

    def record_rerank(
        self,
        before: Sequence[Any],
        after: Sequence[Any],
        elapsed_ms: float | None = None,
    ) -> None:
        """记录精排前后的顺序对比。

        ``before`` 是进入精排前的前 M 条（按 RRF 顺序），``after`` 是精排结果。
        两者都带 ``pool_rank``（该片段在融合池里的名次），因此即使某条不在
        ``before`` 里，也能看出它是从池内第几名被提上来的。
        """
        self.rerank_before = _to_stage_hits(before, keep_ranks=True, preview_chars=self.preview_chars)
        self.rerank_after = _to_stage_hits(after, keep_ranks=True, preview_chars=self.preview_chars)
        for hit in self.rerank_before:
            hit.pool_rank = self.pool_rank.get(hit.chunk_id)
        for hit in self.rerank_after:
            hit.pool_rank = self.pool_rank.get(hit.chunk_id)
        if elapsed_ms is not None:
            self.timing["rerank_ms"] = round(elapsed_ms, 1)

    def record_context(self, contexts: Sequence[Any], elapsed_ms: float | None = None) -> None:
        self.context = [
            ContextEntry(
                index=int(getattr(ctx, "index", position)),
                chunk_id=str(getattr(ctx, "chunk_id", "")),
                source=str(getattr(ctx, "source", "")),
                page=int(getattr(ctx, "page", 0) or 0),
                block_type=str(getattr(ctx, "block_type", "")),
                chars=len(str(getattr(ctx, "text", "") or "")),
                score=float(getattr(ctx, "score", 0.0) or 0.0),
            )
            for position, ctx in enumerate(contexts, start=1)
        ]
        if elapsed_ms is not None:
            self.timing["context_ms"] = round(elapsed_ms, 1)

    def mark_fallback(self, name: str, reason: str = "") -> None:
        if name not in self.fallback:
            self.fallback.append(name)
        if reason:
            self.warnings.append(f"{name}: {reason}")

    def add_timing(self, name: str, elapsed_ms: float) -> None:
        self.timing[name] = round(elapsed_ms, 1)

    def elapsed_since(self, name: str, started: float) -> float:
        """计时辅助：记录 ``name`` 且返回耗时（ms）。"""
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.add_timing(name, elapsed_ms)
        return elapsed_ms

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "vector": [hit.to_dict() for hit in self.vector],
            "bm25": [hit.to_dict() for hit in self.bm25],
            "rrf": [hit.to_dict() for hit in self.rrf],
            "rerank": {
                "before": [hit.to_dict() for hit in self.rerank_before],
                "after": [hit.to_dict() for hit in self.rerank_after],
            },
            "context": [entry.to_dict() for entry in self.context],
            "timing": dict(self.timing),
            "fallback": list(self.fallback),
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        """渲染成可读的树状文本，供 CLI 使用。"""
        lines: list[str] = [f"问题：{self.query}"]
        lines.extend(self._render_stage("vector", self.vector))
        lines.extend(self._render_stage("bm25", self.bm25))
        lines.extend(self._render_stage("rrf", self.rrf, with_ranks=True))

        if self.rerank_before or self.rerank_after:
            ms = self.timing.get("rerank_ms")
            head = f"─ rerank{f'（{ms}ms）' if ms is not None else ''}"
            lines.append(f"{head} 前 {len(self.rerank_before)} → 后 {len(self.rerank_after)}")
            for hit in self.rerank_before:
                lines.append(f"   前 #{hit.rank}  {hit.chunk_id}  rrf={hit.score:.6f}")
            for hit in self.rerank_after:
                if hit.pool_rank:
                    span = f"（池内 #{hit.pool_rank} → 精排 #{hit.rank}，{hit.pool_rank - hit.rank:+d} 位）"
                else:
                    span = ""
                lines.append(f"   后 #{hit.rank}  {hit.chunk_id}  rerank={hit.score:.4f}{span}")

        ms = self.timing.get("context_ms")
        lines.append(f"─ context{f'（{ms}ms）' if ms is not None else ''} {len(self.context)} 条")
        for entry in self.context:
            lines.append(
                f"   [{entry.index}] {entry.chunk_id}  {Path(entry.source).name} 第{entry.page}页  "
                f"{entry.block_type}  {entry.chars} 字  分={entry.score:.4f}"
            )

        if self.timing:
            rendered = "  ".join(f"{name}={value}ms" for name, value in self.timing.items())
            lines.append(f"─ timing  {rendered}")

        if self.fallback:
            lines.append(f"─ fallback  {', '.join(self.fallback)}")
            for warning in self.warnings:
                lines.append(f"   {warning}")
        return "\n".join(lines)

    def _render_stage(
        self, name: str, hits: Sequence[StageHit], *, with_ranks: bool = False
    ) -> list[str]:
        ms = self.timing.get(f"{name}_ms")
        head = f"─ {name}{f'（{ms}ms）' if ms is not None else ''} {len(hits)} 条"
        if not hits:
            head += "（空 / 不可用）"
        lines = [head]
        for hit in hits:
            suffix = ""
            if with_ranks and hit.ranks:
                suffix = "  (" + " ".join(f"{key}={value}" for key, value in sorted(hit.ranks.items())) + ")"
            lines.append(f"   {hit.rank:>2}. {hit.score:9.6f}  {hit.chunk_id}{suffix}")
            lines.append(f"       {hit.preview}")
        return lines
