"""评估侧测试：Dataset 组装、id 召回、指标合并。

`id_based_context_recall` 不调裁判 LLM，因此这些用例既快又不依赖 llama.cpp。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.ragas_eval import (  # noqa: E402
    DEFAULT_METRICS,
    METRIC_NAMES,
    build_dataset,
    make_metrics,
    merge_with_existing,
)


def make_record(**overrides):
    record = {
        "stage": "hybrid",
        "id": "gs_001",
        "question": "平安银行2023年末个人贷款不良率是多少？",
        "answer": "1.37%",
        "contexts": ["2023 年末个人贷款不良率1.37%", "无关内容"],
        "ground_truth": "1.37%",
        "context_chunk_ids": ["c1", "c2"],
        "reference_chunk_ids": ["c2"],
        "failed": False,
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Dataset 组装
# --------------------------------------------------------------------------- #
def test_build_dataset_maps_chunk_ids():
    dataset = build_dataset([make_record()])
    sample = dataset.samples[0]
    assert sample.retrieved_context_ids == ["c1", "c2"]
    assert sample.reference_context_ids == ["c2"]
    assert sample.user_input.startswith("平安银行")
    assert sample.reference == "1.37%"


def test_build_dataset_tolerates_missing_ids():
    record = make_record()
    record.pop("context_chunk_ids")
    record.pop("reference_chunk_ids")
    sample = build_dataset([record]).samples[0]
    assert sample.retrieved_context_ids is None
    assert sample.reference_context_ids is None


# --------------------------------------------------------------------------- #
# id 基召回
# --------------------------------------------------------------------------- #
def test_id_based_recall_counts_reference_hits():
    """命中 1 / 参考 1 → 1.0；完全不命中 → 0.0。"""
    import asyncio

    from ragas.metrics import IDBasedContextRecall

    metric = IDBasedContextRecall()
    hit = build_dataset([make_record(answer="1.37%")]).samples[0]
    miss = build_dataset(
        [make_record(context_chunk_ids=["c9"], reference_chunk_ids=["c2"])]
    ).samples[0]

    assert asyncio.run(metric.single_turn_ascore(hit)) == 1.0
    assert asyncio.run(metric.single_turn_ascore(miss)) == 0.0


def test_id_based_recall_partial():
    """参考块 2 个、只命中 1 个 → 0.5。"""
    import asyncio

    from ragas.metrics import IDBasedContextRecall

    record = make_record(context_chunk_ids=["c1"], reference_chunk_ids=["c1", "c2"])
    sample = build_dataset([record]).samples[0]
    assert asyncio.run(IDBasedContextRecall().single_turn_ascore(sample)) == 0.5


# --------------------------------------------------------------------------- #
# 指标选择
# --------------------------------------------------------------------------- #
def test_make_metrics_selects_subset():
    metrics = make_metrics(["id_based_context_recall"])
    assert [m.name for m in metrics] == ["id_based_context_recall"]


def test_make_metrics_rejects_unknown():
    with pytest.raises(SystemExit):
        make_metrics(["not_a_metric"])


def test_default_metrics_exclude_saturated_recall():
    """context_recall 实测恒为 1.0、已饱和，默认不再消耗裁判调用。"""
    assert "context_recall" not in DEFAULT_METRICS
    assert "id_based_context_recall" in DEFAULT_METRICS


def test_saturated_recall_still_selectable():
    """需要与历史结果对齐时，仍可显式启用。"""
    assert "context_recall" in METRIC_NAMES
    metrics = make_metrics(["context_recall"])
    assert [m.name for m in metrics] == ["context_recall"]


def test_default_metrics_are_valid():
    """默认集里的每个名字都必须存在于可选集合中。"""
    assert set(DEFAULT_METRICS) <= set(METRIC_NAMES)
    assert len(make_metrics()) == len(DEFAULT_METRICS)


# --------------------------------------------------------------------------- #
# 合并进已有结果
# --------------------------------------------------------------------------- #
def test_merge_keeps_old_metrics_and_adds_new(tmp_path: Path):
    output = tmp_path / "hybrid.json"
    output.write_text(
        json.dumps(
            {
                "stage": "hybrid",
                "evaluated_at": "2026-11-26T15:00:00",
                "rows": [
                    {"user_input": "q1", "faithfulness": 1.0},
                    {"user_input": "q2", "faithfulness": 0.0},
                ],
                "mean": {"faithfulness": 0.5},
            }
        ),
        encoding="utf-8",
    )

    payload = {
        "metrics": ["id_based_context_recall"],
        "rows": [
            {
                "user_input": "q1",
                "id_based_context_recall": 1.0,
                "retrieved_context_ids": ["c1"],
                "reference_context_ids": ["c1"],
            },
            {
                "user_input": "q2",
                "id_based_context_recall": 0.0,
                "retrieved_context_ids": ["c9"],
                "reference_context_ids": ["c2"],
            },
        ],
        "mean": {},
        "failures": [],
    }

    merged = merge_with_existing(payload, output)

    assert merged["rows"][0]["faithfulness"] == 1.0  # 旧指标保留
    assert merged["rows"][0]["id_based_context_recall"] == 1.0  # 新指标补上
    assert merged["rows"][0]["reference_context_ids"] == ["c1"]  # chunk_id 一起带上
    assert merged["mean"]["faithfulness"] == 0.5
    assert merged["mean"]["id_based_context_recall"] == 0.5
    assert merged["merge_history"][0]["metrics"] == ["id_based_context_recall"]


def test_merge_gives_up_when_row_count_differs(tmp_path: Path):
    output = tmp_path / "hybrid.json"
    output.write_text(json.dumps({"rows": [{"user_input": "q1"}]}), encoding="utf-8")

    payload = {"metrics": ["id_based_context_recall"], "rows": [], "mean": {}, "failures": []}
    merged = merge_with_existing(payload, output)

    # 对不齐时原样返回，不写坏文件
    assert merged["rows"] == []
    assert "merge_history" not in merged


def test_merge_noop_when_output_missing(tmp_path: Path):
    payload = {"metrics": ["id_based_context_recall"], "rows": [], "mean": {}, "failures": []}
    merged = merge_with_existing(payload, tmp_path / "missing.json")
    assert merged is payload
