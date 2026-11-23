"""问题追踪线索测试：各阶段是否被如实记录。

全部用桩对象，不加载真实模型，几秒内跑完。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation import answer as answer_module  # noqa: E402
from app.generation.trace import (  # noqa: E402
    ContextEntry,
    StageHit,
    TraceRecorder,
    preview,
)
from app.retrieval.rrf import hybrid_search  # noqa: E402

STAGE_KEYS = {"query", "vector", "bm25", "rrf", "rerank", "context", "timing", "fallback", "warnings"}


# --------------------------------------------------------------------------- #
# 桩对象
# --------------------------------------------------------------------------- #
class StubHit:
    def __init__(self, chunk_id: str, text: str = "一段用于预览的原文内容", score: float = 1.0) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.source = "data/corpus/示例.pdf"
        self.page = 12
        self.block_type = "text"


class StubSearcher:
    def __init__(self, ids: list[str], *, fail: bool = False) -> None:
        self.ids = ids
        self.fail = fail

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[StubHit]:
        if self.fail:
            raise ConnectionError("模拟：检索不可用")
        return [StubHit(chunk_id=chunk_id, score=1.0 / (position + 1)) for position, chunk_id in enumerate(self.ids[:top_k])]


class StubReranker:
    """假精排：把顺序反转，方便验证「前后顺序确实被记录」。"""

    def rerank(self, query, hits, top_m=5):
        return list(reversed(list(hits)[:top_m]))


@pytest.fixture(autouse=True)
def clean_pipeline() -> Iterator[None]:
    answer_module._PIPELINE.clear()
    answer_module._PIPELINE_ERRORS.clear()
    yield
    answer_module._PIPELINE.clear()
    answer_module._PIPELINE_ERRORS.clear()


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def test_preview_collapses_whitespace_and_truncates() -> None:
    assert preview("a\n\nb   c") == "a b c"
    long_text = "字" * 200
    result = preview(long_text, 80)
    assert len(result) == 81  # 80 个字符 + 省略号
    assert result.endswith("…")


# --------------------------------------------------------------------------- #
# TraceRecorder 各阶段
# --------------------------------------------------------------------------- #
def test_recorder_captures_all_stages() -> None:
    recorder = TraceRecorder("问题")
    recorder.record_vector([StubHit("v1"), StubHit("v2")], 12.5)
    recorder.record_bm25([StubHit("b1")], 3.25)
    recorder.record_rrf([StubHit("r1"), StubHit("r2")], 0.2)
    recorder.record_rerank([StubHit("r1"), StubHit("r2")], [StubHit("r2"), StubHit("r1")], 633.0)
    recorder.record_context(
        [
            answer_module.RetrievedContext(
                index=1, chunk_id="c1", score=0.99, text="正文", source="data/corpus/x.pdf",
                page=3, block_type="text",
            )
        ]
    )
    recorder.add_timing("generation_ms", 1000.0)

    payload = recorder.to_dict()
    assert set(payload) == STAGE_KEYS
    assert [hit["chunk_id"] for hit in payload["vector"]] == ["v1", "v2"]
    assert [hit["chunk_id"] for hit in payload["bm25"]] == ["b1"]
    assert [hit["chunk_id"] for hit in payload["rrf"]] == ["r1", "r2"]
    assert [hit["chunk_id"] for hit in payload["rerank"]["before"]] == ["r1", "r2"]
    assert [hit["chunk_id"] for hit in payload["rerank"]["after"]] == ["r2", "r1"]
    assert payload["context"][0]["chunk_id"] == "c1"
    assert payload["context"][0]["source"] == "x.pdf"  # 只留文件名
    assert payload["timing"]["vector_ms"] == 12.5
    assert payload["timing"]["rerank_ms"] == 633.0
    assert payload["timing"]["generation_ms"] == 1000.0
    assert payload["fallback"] == []


def test_recorder_keeps_each_route_rank_on_rrf() -> None:
    """RRF 阶段必须保留各路名次，才能判断是「哪一路没召回」。"""
    from app.retrieval.rrf import fuse_hybrid

    fused = fuse_hybrid([StubHit("a")], [StubHit("b"), StubHit("a")])
    recorder = TraceRecorder("问题")
    recorder.record_rrf(fused, 0.1)

    top = recorder.to_dict()["rrf"][0]
    assert top["chunk_id"] == "a"
    assert top["ranks"] == {"vector": 1, "bm25": 2}


def test_fallback_marker_deduplicates() -> None:
    recorder = TraceRecorder("问题")
    recorder.mark_fallback("rerank", "第一次")
    recorder.mark_fallback("rerank", "第二次")
    payload = recorder.to_dict()
    assert payload["fallback"] == ["rerank"]
    assert len(payload["warnings"]) == 2


def test_render_contains_every_stage() -> None:
    recorder = TraceRecorder("问题")
    recorder.record_vector([StubHit("v1")], 1.0)
    recorder.record_bm25([StubHit("b1")], 2.0)
    recorder.record_rrf([StubHit("r1")], 0.1)
    recorder.record_rerank([StubHit("r1")], [StubHit("r1")], 3.0)
    recorder.record_context(
        [answer_module.RetrievedContext(index=1, chunk_id="c1", score=1.0, text="t",
                                        source="a.pdf", page=1, block_type="text")]
    )
    recorder.mark_fallback("rerank", "超时")

    rendered = recorder.render()
    for token in ("vector", "bm25", "rrf", "rerank", "context", "timing", "fallback", "问题："):
        assert token in rendered


# --------------------------------------------------------------------------- #
# hybrid_search 集成
# --------------------------------------------------------------------------- #
def test_hybrid_search_records_both_routes_and_fusion() -> None:
    recorder = TraceRecorder("问题")
    hits = hybrid_search(
        "问题",
        top_k=3,
        top_n=3,
        store=StubSearcher(["v1", "v2"]),
        index=StubSearcher(["b1", "v1"]),
        trace=recorder,
        build_missing=False,
    )
    payload = recorder.to_dict()
    assert len(hits) == 3
    assert [hit["chunk_id"] for hit in payload["vector"]] == ["v1", "v2"]
    assert [hit["chunk_id"] for hit in payload["bm25"]] == ["b1", "v1"]
    assert {hit["chunk_id"] for hit in payload["rrf"]} == {"v1", "v2", "b1"}
    assert payload["timing"]["vector_ms"] >= 0
    assert payload["timing"]["rrf_ms"] >= 0


def test_hybrid_search_records_route_failure_as_fallback() -> None:
    failures: list[str] = []
    recorder = TraceRecorder("问题")
    hybrid_search(
        "问题",
        store=StubSearcher([], fail=True),
        index=StubSearcher(["b1"]),
        failures=failures,
        trace=recorder,
        build_missing=False,
    )
    payload = recorder.to_dict()
    assert payload["vector"] == []
    assert "vector" in payload["fallback"]
    assert any("ConnectionError" in warning for warning in payload["warnings"])


# --------------------------------------------------------------------------- #
# answer_question 集成
# --------------------------------------------------------------------------- #
def test_answer_question_returns_trace() -> None:
    answer_module._PIPELINE.update(
        {
            "store": StubSearcher(["v1"]),
            "index": StubSearcher(["b1"]),
            "reranker": StubReranker(),
        }
    )
    # 直接打桩 LLM，避免依赖 llama.cpp
    original = answer_module.call_llm
    answer_module.call_llm = lambda *args, **kwargs: ("答案", {"completion_tokens": 2})
    try:
        result = answer_module.answer_question("问题", top_m=2, include_messages=False)
    finally:
        answer_module.call_llm = original

    assert result.trace is not None
    assert set(result.trace) == STAGE_KEYS
    assert result.trace["query"] == "问题"
    assert [hit["chunk_id"] for hit in result.trace["context"]] == [c.chunk_id for c in result.contexts]
    assert "retrieval_ms" in result.trace["timing"]
    assert "generation_ms" in result.trace["timing"]
    assert "total_ms" in result.trace["timing"]


def test_answer_question_can_disable_trace() -> None:
    answer_module._PIPELINE.update(
        {
            "store": StubSearcher(["v1"]),
            "index": StubSearcher(["b1"]),
            "reranker": StubReranker(),
        }
    )
    original = answer_module.call_llm
    answer_module.call_llm = lambda *args, **kwargs: ("答案", None)
    try:
        result = answer_module.answer_question("问题", trace=False)
    finally:
        answer_module.call_llm = original

    assert result.trace is None


def test_trace_records_rerank_fallback() -> None:
    class DeadReranker:
        def rerank(self, query, hits, top_m=5):
            raise RuntimeError("模拟：显存不足")

    answer_module._PIPELINE.update(
        {"store": StubSearcher(["v1"]), "index": StubSearcher(["b1"]), "reranker": DeadReranker()}
    )
    original = answer_module.call_llm
    answer_module.call_llm = lambda *args, **kwargs: ("答案", None)
    try:
        result = answer_module.answer_question("问题", top_m=2)
    finally:
        answer_module.call_llm = original

    assert "rerank" in result.trace["fallback"]
    assert any("显存不足" in warning for warning in result.trace["warnings"])


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #
def test_ask_can_disable_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from app import main as api

    captured: dict = {}

    def fake_answer(question, **kwargs):
        captured.update(kwargs)
        return answer_module.AnswerResult(
            question=question, answer="答案", contexts=[], model="stub", messages=[]
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    client = TestClient(api.app)

    client.get("/ask", params={"q": "问题", "trace": "false"})
    assert captured["trace"] is False

    client.get("/ask", params={"q": "问题"})
    assert captured["trace"] is True  # 默认开启（TRACE_ENABLED）
