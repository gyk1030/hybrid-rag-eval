"""年份一致性校验的单元测试。

这个模块的存在理由：3B 模型会拿 2024 年报的 1.39% 回答「2023 年末」的问题，
而正确资料（1.37%）当时就在上下文里。数字是真的、年份是错的，光看答案看不出来。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation.consistency import (  # noqa: E402
    check_year_consistency,
    extract_numbers,
    extract_years,
    question_years,
    source_year,
)


@dataclass
class StubContext:
    index: int
    source: str
    text: str


def ctx(index: int, year: int, text: str) -> StubContext:
    return StubContext(
        index=index,
        source=f"data/corpus/000001_平安银行_{year}年年度报告.pdf",
        text=text,
    )


# --------------------------------------------------------------------------- #
# 基础抽取
# --------------------------------------------------------------------------- #
def test_source_year_from_filename():
    assert source_year("data/corpus/000001_平安银行_2023年年度报告.pdf") == 2023
    # 万科的文件名里带全角空格与全角字母，年份仍应解析正确
    assert source_year("data/corpus/000002_万  科Ａ_2024年年度报告.pdf") == 2024
    assert source_year("data/corpus/没有年份.pdf") is None


def test_extract_years_dedup_and_order():
    assert extract_years("2023 年末为 1.37%，较 2022 年末上升，2023 年再次确认") == [2023, 2022]


def test_extract_numbers_skips_year_token():
    values = extract_numbers("2023 年末为 1.37%，较上年末上升 0.05 个百分点")
    assert 1.37 in values
    assert 0.05 in values
    # 「2023 年」是年份，不应被当成数值
    assert 2023 not in values


# --------------------------------------------------------------------------- #
# 核心场景
# --------------------------------------------------------------------------- #
def test_mismatch_when_answer_uses_other_year_table():
    """用户实际遇到的 bug：问 2023，答案取自 2024 年报表格（1.39%）。"""
    contexts = [
        ctx(1, 2023, "2023 年末，本行个人贷款不良率1.37%，较上年末上升0.05 个百分点。"),
        ctx(4, 2024, "| 个人贷款 | 1,767,168 | 52.4% | 1.39% | 1,977,719 | 58.0% | 1.37% |"),
    ]
    check = check_year_consistency(
        "平安银行2023年末个人贷款不良率是多少？",
        "根据资料4，平安银行2024年末个人贷款不良率是1.39%。",
        contexts,
    )
    assert check.status == "mismatch"
    assert check.asked_years == [2023]
    assert check.source_year == 2024
    assert "2024 年报" in check.detail


def test_ok_when_answer_uses_asked_year():
    contexts = [
        ctx(1, 2023, "个人贷款2023 年末，本行个人贷款不良率1.37%，较上年末上升0.05 个百分点。"),
    ]
    check = check_year_consistency(
        "平安银行2023年末个人贷款不良率是多少？",
        "平安银行2023年末个人贷款不良率为1.37%。",
        contexts,
    )
    assert check.status == "ok"
    assert check.source_year == 2023


def test_skipped_when_question_has_no_year():
    contexts = [ctx(1, 2024, "研发投入金额为5.08亿元。")]
    check = check_year_consistency("万科研发投入是多少？", "5.08亿元", contexts)
    assert check.status == "skipped"


def test_multi_year_question_allows_either_year():
    """「2025 相对 2024 年」这类对比问题，两年的年报都合法。"""
    contexts = [ctx(1, 2024, "对公房地产贷款不良率2.22%，较上年末上升0.43 个百分点。")]
    check = check_year_consistency(
        "2025 相对 2024年，对公房地产贷款不良率是上升还是下降？",
        "上升 0.43 个百分点。",
        contexts,
    )
    assert check.status == "ok"


def test_comparative_year_in_answer_does_not_false_positive():
    """答案里出现「较 2022 年末」是对比引用，不能误判成年份错配。"""
    contexts = [
        ctx(1, 2023, "2023 年末个人贷款不良率1.37%，较上年末（2022 年末）上升0.05 个百分点。"),
    ]
    check = check_year_consistency(
        "平安银行2023年末个人贷款不良率是多少？",
        "2023 年末个人贷款不良率为1.37%，较2022 年末上升0.05 个百分点。",
        contexts,
    )
    assert check.status == "ok"


def test_fallback_catches_year_in_answer_without_numbers():
    """答案没有可归属的数值时，用「答案里明写的年份」兜底。"""
    contexts = [ctx(1, 2024, "管理层并未采取利率对冲政策。")]
    check = check_year_consistency(
        "国华网安2023年是否采取利率对冲政策？",
        "根据2024年年报，管理层未采取利率对冲政策。",
        contexts,
    )
    assert check.status == "mismatch"
    assert check.answer_years == [2024]


def test_to_dict_shape():
    contexts = [ctx(1, 2023, "1.37%")]
    payload = check_year_consistency("2023 年末不良率？", "1.37%", contexts).to_dict()
    assert set(payload) == {
        "status",
        "asked_years",
        "answer_years",
        "source_year",
        "source_index",
        "detail",
    }
    assert isinstance(payload["asked_years"], list)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("2023 年营业收入", [2023]),
        ("2021-2024 年住宅物业换手率", [2021, 2024]),
        ("营业收入是多少", []),
    ],
)
def test_question_year_extraction(question: str, expected: list[int]):
    assert extract_years(question) == expected


def test_question_years_expands_range():
    """「2021-2024 年」区间题往往需要中间年份的数据，逐年展开。"""
    assert question_years("2021-2024 年住宅物业换手率由多少提升至多少？") == [
        2021,
        2022,
        2023,
        2024,
    ]
    assert question_years("2023 年末不良率") == [2023]


# --------------------------------------------------------------------------- #
# 与 answer_question 的集成：即使模型答错，check 也必须报出来
# --------------------------------------------------------------------------- #
def test_answer_question_records_mismatch(monkeypatch: pytest.MonkeyPatch):
    """把模型打桩成「用 2024 年报的 1.39% 回答 2023 年问题」，校验必须拦住。"""
    import app.retrieval.rrf as rrf_module
    from app.generation import answer as answer_module

    class Hit:
        chunk_id = "stub_p037_tbl001"
        score = 0.98
        text = "| 个人贷款 | 1,767,168 | 52.4% | 1.39% | 1,977,195 | 58.0% | 1.37% |"
        source = "data/corpus/000001_平安银行_2024年年度报告.pdf"
        page = 37
        block_type = "table"
        ranks = {"bm25": 13}

    class StubReranker:
        def rerank(self, question, hits, top_m=5):
            return list(hits)[:top_m]

    monkeypatch.setattr(rrf_module, "hybrid_search", lambda *a, **k: [Hit()])
    monkeypatch.setattr(
        answer_module,
        "call_llm",
        lambda *a, **k: ("根据资料1，平安银行2024年末个人贷款不良率是1.39%。", None),
    )
    monkeypatch.setattr(
        answer_module,
        "_PIPELINE",
        {"store": object(), "index": object(), "reranker": StubReranker()},
    )

    result = answer_module.answer_question(
        "平安银行2023年末个人贷款不良率是多少？", top_m=1, trace=False
    )

    check = result.answer_check
    assert check is not None
    assert check["status"] == "mismatch"
    assert check["source_year"] == 2024
    assert any("year_mismatch" in warning for warning in result.warnings)


def test_answer_question_check_ok_when_model_answers_correctly(monkeypatch: pytest.MonkeyPatch):
    import app.retrieval.rrf as rrf_module
    from app.generation import answer as answer_module

    class Hit:
        chunk_id = "stub_p037_b002"
        score = 0.99
        text = "个人贷款2023 年末，本行个人贷款不良率1.37%，较上年末上升0.05 个百分点。"
        source = "data/corpus/000001_平安银行_2023年年度报告.pdf"
        page = 37
        block_type = "text"
        ranks = {"bm25": 3}

    class StubReranker:
        def rerank(self, question, hits, top_m=5):
            return list(hits)[:top_m]

    monkeypatch.setattr(rrf_module, "hybrid_search", lambda *a, **k: [Hit()])
    monkeypatch.setattr(
        answer_module, "call_llm", lambda *a, **k: ("平安银行2023年末个人贷款不良率是1.37%。", None)
    )
    monkeypatch.setattr(
        answer_module,
        "_PIPELINE",
        {"store": object(), "index": object(), "reranker": StubReranker()},
    )

    result = answer_module.answer_question(
        "平安银行2023年末个人贷款不良率是多少？", top_m=1, trace=False
    )
    assert result.answer_check is not None
    assert result.answer_check["status"] == "ok"
    assert not any("year_mismatch" in warning for warning in result.warnings)
