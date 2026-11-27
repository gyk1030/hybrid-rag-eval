"""评测采集侧的年份校验：generate_answer 是否把 answer_check 写进 records。

不调真实模型：把 call_llm 打桩，检索侧用桩对象。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval import run_pipeline  # noqa: E402


class StubHit:
    """最小可用的检索结果：build_contexts 只读取这几个属性。"""

    def __init__(self, chunk_id: str, source: str, text: str, page: int = 37) -> None:
        self.chunk_id = chunk_id
        self.source = source
        self.text = text
        self.page = page
        self.block_type = "text"
        self.score = 0.99
        self.ranks = {"bm25": 1}
        self.fused_score = 0.06


HIT_2023 = StubHit(
    "c2023",
    "data/corpus/000001_平安银行_2023年年度报告.pdf",
    "个人贷款2023 年末，本行个人贷款不良率1.37%，较上年末上升0.05 个百分点。",
)
HIT_2024 = StubHit(
    "c2024",
    "data/corpus/000001_平安银行_2024年年度报告.pdf",
    "个人贷款方面，2024 年末，本行个人贷款不良率1.39%，较上年末上升0.02 个百分点。",
)

QUESTION = "平安银行2023年末个人贷款不良率是多少？"


def test_generate_answer_records_ok_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(run_pipeline, "call_llm", lambda messages, **kw: ("1.37%", None))
    payload = run_pipeline.generate_answer(QUESTION, [HIT_2023])

    assert payload["answer_check"]["status"] == "ok"
    assert payload["answer_check"]["asked_years"] == [2023]
    assert payload["answer_check"]["source_year"] == 2023


def test_generate_answer_records_mismatch(monkeypatch: pytest.MonkeyPatch):
    """模型用 2024 年的 1.39% 回答 2023 年的问题 → records 里要标出来。"""
    monkeypatch.setattr(
        run_pipeline, "call_llm", lambda messages, **kw: ("2024年末个人贷款不良率为1.39%", None)
    )
    payload = run_pipeline.generate_answer(QUESTION, [HIT_2024])

    assert payload["answer_check"]["status"] == "mismatch"
    assert payload["answer_check"]["source_year"] == 2024
    assert "2024" in payload["answer_check"]["detail"]


def test_run_stage_persists_answer_check(monkeypatch: pytest.MonkeyPatch):
    """run_stage 要把校验结果落到 record 上，而不是只留在 payload 里。"""

    class StubStore:
        def search(self, question, top_k=10):
            return [HIT_2024]

    class StubReranker:
        def rerank(self, question, hits, top_m=5):
            return list(hits)[:top_m]

    monkeypatch.setattr(
        run_pipeline,
        "get_pipeline",
        lambda: {"store": StubStore(), "index": object(), "reranker": StubReranker()},
    )
    monkeypatch.setattr(
        run_pipeline, "call_llm", lambda messages, **kw: ("2024年末个人贷款不良率为1.39%", None)
    )

    golden = [
        {
            "id": "gs_test",
            "question": QUESTION,
            "ground_truth": "1.37%",
            "relevant_chunk_ids": ["c2023"],
        }
    ]
    records = run_pipeline.run_stage("baseline", golden)

    assert len(records) == 1
    assert records[0]["failed"] is False
    assert records[0]["answer_check"]["status"] == "mismatch"
