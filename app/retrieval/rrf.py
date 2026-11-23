"""RRF（Reciprocal Rank Fusion）：把向量检索与 BM25 检索的 TopK 融合成 TopN。

RRF 只看排名、不看分数，因此天然适合融合「余弦相似度」和「BM25 分数」这种
量纲完全不同、甚至不可比的检索结果::

    score(d) = Σ_i  w_i / (k + rank_i(d))

其中 ``rank_i(d)`` 是文档 d 在第 i 路结果里的名次（从 1 开始），``k`` 是平滑常数
（论文取 60）。某个文档只被一路召回时，另一路的排名视为无穷大（不计分）。

本模块的融合逻辑是纯函数，不依赖数据库和模型，便于单测；需要端到端跑通时用
:func:`hybrid_search`（内部才去加载两个检索器）。

用法::

    python -m app.retrieval.rrf -q "研发投入" -k 20 -n 10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# 允许用脚本方式直接运行（python app/retrieval/rrf.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里，
# hybrid_search() 里延迟导入的 `app.retrieval.*` 会报 ModuleNotFoundError。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    CHUNKS_PATH,
    DEGRADE_ON_ROUTE_FAILURE,
    HYBRID_TOP_K,
    HYBRID_TOP_N,
    RRF_K,
    RRF_WEIGHTS,
)


@dataclass(slots=True)
class FusedHit:
    """一条融合结果。"""

    chunk_id: str
    score: float
    text: str
    source: str
    page: int
    block_type: str
    ranks: dict[str, int] = field(default_factory=dict)

    def snippet(self, length: int = 80) -> str:
        """返回用于展示的短摘要。"""
        flat = " ".join(self.text.split())
        return flat[:length] + ("…" if len(flat) > length else "")

    def rank_text(self) -> str:
        """把各路名次渲染成 ``bm25=3 vector=1`` 这样的短串。"""
        return " ".join(f"{name}={rank}" for name, rank in sorted(self.ranks.items()))


def _chunk_id_of(item: Any) -> str:
    """从检索结果里取出 chunk_id；也允许直接传 chunk_id 字符串。"""
    if isinstance(item, str):
        return item
    chunk_id = getattr(item, "chunk_id", None)
    if chunk_id is None:
        raise TypeError(f"无法从 {type(item).__name__} 上取到 chunk_id")
    return str(chunk_id)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    top_n: int | None = None,
) -> list[FusedHit]:
    """把多路「已按相关性排好序」的结果融合成一个排名。

    参数
    ----
    ranked_lists:
        多路检索结果，每路是一个有序序列，元素可以是带 ``chunk_id`` 属性的对象
        （``VectorHit`` / ``Bm25Hit`` / ``FusedHit``）或直接的 chunk_id 字符串。
    k:
        RRF 平滑常数，默认 :data:`RRF_K`。
    weights:
        各路权重，默认全 1.0；长度必须与 ``ranked_lists`` 一致。
    names:
        各路的名称，用于结果里的 ``ranks`` 字段，默认 ``rank1``、``rank2``……
    top_n:
        只返回前 N 条；``None`` 表示全部返回。

    说明
    ----
    同一路结果里如果出现重复的 chunk_id，只按它最靠前的那次名次计分（避免人为
    刷高分数）；分数相同时按 chunk_id 升序排列，保证结果可复现。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not ranked_lists:
        return []

    count = len(ranked_lists)
    if weights is None:
        weights = [1.0] * count
    if len(weights) != count:
        raise ValueError(f"weights 长度({len(weights)}) 与结果路数({count}) 不一致")
    if names is None:
        names = [f"rank{index}" for index in range(1, count + 1)]
    if len(names) != count:
        raise ValueError(f"names 长度({len(names)}) 与结果路数({count}) 不一致")

    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    payload: dict[str, Any] = {}

    for name, hits, weight in zip(names, ranked_lists, weights):
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            chunk_id = _chunk_id_of(hit)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] += weight / (k + position)
            ranks[chunk_id][name] = position
            payload.setdefault(chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        ordered = ordered[:top_n]

    fused: list[FusedHit] = []
    for chunk_id, score in ordered:
        source = payload[chunk_id]
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                score=score,
                text=getattr(source, "text", "") or "",
                source=getattr(source, "source", "") or "",
                page=int(getattr(source, "page", 0) or 0),
                block_type=getattr(source, "block_type", "") or "",
                ranks=dict(ranks[chunk_id]),
            )
        )
    return fused


def fuse_hybrid(
    vector_hits: Sequence[Any],
    bm25_hits: Sequence[Any],
    *,
    k: int = RRF_K,
    weights: Sequence[float] = RRF_WEIGHTS,
    top_n: int | None = None,
) -> list[FusedHit]:
    """两路融合的便捷封装：``vector`` + ``bm25``。"""
    return reciprocal_rank_fusion(
        [vector_hits, bm25_hits],
        k=k,
        weights=weights,
        names=["vector", "bm25"],
        top_n=top_n,
    )


def hybrid_search(
    query: str,
    *,
    top_k: int = HYBRID_TOP_K,
    top_n: int = HYBRID_TOP_N,
    rrf_k: int = RRF_K,
    weights: Sequence[float] = RRF_WEIGHTS,
    store: Any = None,
    index: Any = None,
    failures: list[str] | None = None,
    build_missing: bool = True,
    degrade: bool | None = None,
    trace: Any = None,
) -> list[FusedHit]:
    """跑完整的「向量 TopK + BM25 TopK → RRF → TopN」。

    两个检索器按需加载；重复调用时建议把 ``store`` / ``index`` 传进来复用，
    避免反复加载模型、重建 BM25 索引。

    降级
    ----
    传入 ``failures`` 列表时启用「单路降级」：某一路抛异常（PG 断连、模型加载
    失败等）只把原因追加进 ``failures``，该路按空结果参与融合，另一路照常返回。
    这是安全的——:func:`reciprocal_rank_fusion` 本来就允许某一路为空。

    不传 ``failures`` 时默认仍然抛错（CLI 直接调用时应当看得见故障）；
    传了列表但想强制抛错，把 ``degrade=False``。

    参数 ``build_missing=False`` 表示「``store`` / ``index`` 为 None 就是不可用」，
    不要再去构造——供上层已经做过可用性探测的场景使用（如 :mod:`app.generation.answer`）。

    追踪
    ----
    传入 ``trace``（:class:`app.generation.trace.TraceRecorder`）时，会把两路原始
    结果与融合明细按阶段记录下来。这里只按鸭子类型调用对方的方法，不 import 该
    模块，避免循环依赖；``trace`` 为 None 时完全不产生额外开销。
    """
    if not query or not query.strip():
        return []

    if degrade is None:
        degrade = DEGRADE_ON_ROUTE_FAILURE and failures is not None

    if build_missing and (store is None or index is None):
        # 延迟导入：纯函数部分不应该被迫加载 torch / jieba
        from app.retrieval.bm25_index import BM25Index
        from app.retrieval.vector_store import VectorStore

        store = store or VectorStore(CHUNKS_PATH)
        index = index or BM25Index().build_or_load()

    routed: list[tuple[str, Any, Any]] = [
        ("vector", store, lambda: store.search(query, top_k=top_k)),
        ("bm25", index, lambda: index.search(query, top_k=top_k)),
    ]

    vector_hits: list[Any] = []
    bm25_hits: list[Any] = []
    for name, searcher, run in routed:
        if searcher is None:
            message = f"{name}: 不可用（未初始化）"
            if not degrade:
                raise RuntimeError(f"检索路径不可用 —— {message}")
            if failures is not None:
                failures.append(message)
            if trace is not None:
                trace.record_route_failure(name, "不可用（未初始化）")
            continue
        try:
            started = time.perf_counter()
            hits = run()
            elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - 降级时任何异常都不该拖垮整条链路
            message = f"{name}: {type(exc).__name__}: {exc}"
            if not degrade:
                raise
            if failures is not None:
                failures.append(message)
            if trace is not None:
                trace.record_route_failure(name, f"{type(exc).__name__}: {exc}")
            continue
        if trace is not None:
            recorder = trace.record_vector if name == "vector" else trace.record_bm25
            recorder(hits, elapsed_ms)
        if name == "vector":
            vector_hits = list(hits)
        else:
            bm25_hits = list(hits)

    fusion_started = time.perf_counter()
    fused = fuse_hybrid(vector_hits, bm25_hits, k=rrf_k, weights=weights, top_n=top_n)
    if trace is not None:
        trace.record_rrf(fused, (time.perf_counter() - fusion_started) * 1000)
    return fused


def _format_hit(rank: int, hit: FusedHit) -> str:
    return (
        f"{rank:>2}. {hit.score:.6f}  {hit.chunk_id}  "
        f"[{hit.block_type} p{hit.page}]  {Path(hit.source).name}\n"
        f"     {hit.rank_text()}\n"
        f"     {hit.snippet()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.rrf",
        description="向量检索 + BM25 检索的 RRF 融合。",
    )
    parser.add_argument("-q", "--query", required=True, help="查询语句")
    parser.add_argument(
        "-k", "--top-k", type=int, default=HYBRID_TOP_K, help=f"每路召回的条数，默认 {HYBRID_TOP_K}"
    )
    parser.add_argument(
        "-n", "--top-n", type=int, default=HYBRID_TOP_N, help=f"融合后返回条数，默认 {HYBRID_TOP_N}"
    )
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help=f"RRF 平滑常数，默认 {RRF_K}")
    args = parser.parse_args(argv)

    hits = hybrid_search(args.query, top_k=args.top_k, top_n=args.top_n, rrf_k=args.rrf_k)
    if not hits:
        print(f"没有命中：{args.query!r}")
        return 0

    print(f"查询 {args.query!r}（每路 top{args.top_k} → RRF k={args.rrf_k} → top{args.top_n}）：")
    for position, hit in enumerate(hits, 1):
        print(_format_hit(position, hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
