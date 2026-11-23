"""降级链路测试：不需要加载真实模型，全部用桩对象。

覆盖：
- ``hybrid_search`` 单路失败降级 / 不降级时抛错
- ``get_pipeline`` 单组件失败不影响其他组件，且失败会被重试
- ``answer_question`` 的精排降级、LLM 抽取式兜底、两路全挂时明确报错
- HTTP 层 ``/ask`` 与 ``/health`` 的降级契约
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
from app.retrieval.rrf import fuse_hybrid, hybrid_search  # noqa: E402


# --------------------------------------------------------------------------- #
# 桩对象
# --------------------------------------------------------------------------- #
class StubHit:
    """最小可用的检索结果：融合只需要 chunk_id 和 text。"""

    def __init__(self, chunk_id: str, text: str = "片段") -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.source = "data/corpus/stub.pdf"
        self.page = 1
        self.block_type = "text"
        self.score = 1.0


class StubSearcher:
    def __init__(self, ids: list[str], *, fail: bool = False) -> None:
        self.ids = ids
        self.fail = fail

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[StubHit]:
        if self.fail:
            raise ConnectionError("模拟：检索不可用")
        return [StubHit(chunk_id) for chunk_id in self.ids[:top_k]]


class DeadReranker:
    def rerank(self, query, hits, top_m=5):
        raise RuntimeError("模拟：CUDA out of memory")


class StubReranker:
    """假精排：保持原顺序截断，避免测试里加载真实交叉编码器。"""

    def rerank(self, query, hits, top_m=5):
        return list(hits)[:top_m]


@pytest.fixture(autouse=True)
def clean_pipeline() -> Iterator[None]:
    """每个用例前后都清空进程级单例，避免用例间互相污染。"""
    answer_module._PIPELINE.clear()
    answer_module._PIPELINE_ERRORS.clear()
    yield
    answer_module._PIPELINE.clear()
    answer_module._PIPELINE_ERRORS.clear()


# --------------------------------------------------------------------------- #
# hybrid_search
# --------------------------------------------------------------------------- #
def test_vector_failure_degrades_to_bm25() -> None:
    failures: list[str] = []
    hits = hybrid_search(
        "问题",
        top_k=5,
        top_n=5,
        store=StubSearcher([], fail=True),
        index=StubSearcher(["a", "b"]),
        failures=failures,
        build_missing=False,
    )
    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert any("vector" in item for item in failures)
    assert all("vector" not in hit.ranks for hit in hits)


def test_bm25_failure_degrades_to_vector() -> None:
    failures: list[str] = []
    hits = hybrid_search(
        "问题",
        top_k=5,
        top_n=5,
        store=StubSearcher(["a", "b"]),
        index=StubSearcher([], fail=True),
        failures=failures,
        build_missing=False,
    )
    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert any("bm25" in item for item in failures)


def test_failure_propagates_without_failures_list() -> None:
    """不传 failures 时保持严格语义：异常必须抛出去（CLI 需要看见故障）。"""
    with pytest.raises(ConnectionError):
        hybrid_search(
            "问题",
            store=StubSearcher([], fail=True),
            index=StubSearcher(["a"]),
            build_missing=False,
        )


def test_both_routes_empty_returns_empty() -> None:
    assert fuse_hybrid([], [], top_n=5) == []


# --------------------------------------------------------------------------- #
# get_pipeline
# --------------------------------------------------------------------------- #
def test_single_component_failure_does_not_block_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """store 加载失败时，index / reranker 仍应被尝试（这里用标记值代替真加载，避免拉模型）。"""

    def flaky(name, factory):
        if name == "store":
            answer_module._PIPELINE_ERRORS["store"] = "模拟：加载失败"
            return None
        return f"stub-{name}"

    monkeypatch.setattr(answer_module, "_load_component", flaky)
    pipeline = answer_module.get_pipeline()
    status = answer_module.pipeline_status()

    assert pipeline.get("store") is None
    assert pipeline.get("index") == "stub-index"
    assert pipeline.get("reranker") == "stub-reranker"
    assert status["ready"] is False
    assert "store" in status["errors"]


def test_failed_component_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """失败不会被缓存成「永久不可用」：下一次调用会重新尝试。"""
    attempts = {"store": 0}

    def flaky(name, factory):
        if name == "store":
            attempts["store"] += 1
            if attempts["store"] == 1:
                answer_module._PIPELINE_ERRORS["store"] = "模拟：第一次失败"
                return None
            # 真实实现成功后会清除错误记录，桩也要保持一致
            answer_module._PIPELINE_ERRORS.pop("store", None)
            return "stub-store"
        return f"stub-{name}"

    monkeypatch.setattr(answer_module, "_load_component", flaky)
    assert answer_module.get_pipeline().get("store") is None
    assert answer_module.get_pipeline().get("store") == "stub-store"
    assert "store" not in answer_module._PIPELINE_ERRORS


# --------------------------------------------------------------------------- #
# answer_question
# --------------------------------------------------------------------------- #
def test_rerank_failure_falls_back_to_rrf_order(monkeypatch: pytest.MonkeyPatch) -> None:
    answer_module._PIPELINE.update(
        {
            "store": StubSearcher(["z"]),
            "index": StubSearcher(["a", "b", "c"]),
            "reranker": DeadReranker(),
        }
    )
    result = answer_module.answer_question("问题", top_m=2, include_messages=False)

    assert "rerank" in result.degraded
    assert any("rerank" in item for item in result.warnings)
    # 精排挂掉后仍然拿到 RRF 前 top_m 条
    assert len(result.contexts) == 2


def test_llm_failure_returns_extractive_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    answer_module._PIPELINE.update(
        {
            "store": StubSearcher(["z"]),
            "index": StubSearcher(["a"]),
            "reranker": StubReranker(),
        }
    )

    def dead_llm(*args, **kwargs):
        raise answer_module.LLMError("模拟：连接被拒绝")

    monkeypatch.setattr(answer_module, "call_llm", dead_llm)
    result = answer_module.answer_question("问题", top_m=1, include_messages=False)

    assert "llm" in result.degraded
    # 兜底答案必须一眼看出不是生成结果
    assert result.answer.startswith("（本地 LLM 不可用")
    assert "出处：" in result.answer
    assert result.contexts  # 证据仍然返回


def test_both_routes_down_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        answer_module, "_load_component", lambda name, factory: None
    )
    with pytest.raises(answer_module.RetrievalUnavailable):
        answer_module.answer_question("问题")


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app import main as api

    return TestClient(api.app)


def test_health_does_not_trigger_loading(client) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert set(payload["pipeline"]) == {"loaded", "errors", "load_ms", "ready"}
    # 健康检查不应该把模型加载起来
    assert payload["pipeline"]["loaded"] == {"store": False, "index": False, "reranker": False}
    assert payload["pipeline"]["load_ms"] == {}


def test_ask_returns_200_with_degraded_field(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as api

    def fake_answer(question, **kwargs):
        return answer_module.AnswerResult(
            question=question,
            answer="（本地 LLM 不可用，以下为检索到的最相关原文，未经改写）\n原文\n—— 出处：x.pdf 第 1 页",
            contexts=[
                answer_module.RetrievedContext(
                    index=1,
                    chunk_id="c1",
                    score=0.5,
                    text="原文",
                    source="data/corpus/x.pdf",
                    page=1,
                    block_type="text",
                )
            ],
            model="stub",
            messages=[],
            degraded=["llm"],
            warnings=["llm: LLMError: 模拟"],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = client.get("/ask", params={"q": "问题"})
    payload = response.json()

    assert response.status_code == 200  # 降级不该变成 5xx
    assert payload["degraded"] == ["llm"]
    assert payload["contexts"]


def test_ask_returns_503_when_nothing_useful(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as api

    def fake_answer(question, **kwargs):
        return answer_module.AnswerResult(
            question=question,
            answer="未检索到相关资料，且本地 LLM 不可用。",
            contexts=[],
            model="stub",
            messages=[],
            degraded=["llm"],
            warnings=["llm: 模拟"],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = client.get("/ask", params={"q": "问题"})
    assert response.status_code == 503


def test_ask_maps_errors_to_status_codes(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as api

    def unavailable(question, **kwargs):
        raise answer_module.RetrievalUnavailable("模拟")

    monkeypatch.setattr(api, "answer_question", unavailable)
    assert client.get("/ask", params={"q": "问题"}).status_code == 503

    def bad_request(question, **kwargs):
        raise ValueError("问题不能为空")

    monkeypatch.setattr(api, "answer_question", bad_request)
    assert client.get("/ask", params={"q": "问题"}).status_code == 400
