"""组装 RAGAS Dataset、注入本地裁判、执行评估并保存（评估流程第 2–5 步）。

它读 :mod:`app.eval.run_pipeline` 产出的 ``{stage}_records.json``，做四件事：

1. 组装 ``SingleTurnSample``（user_input / response / retrieved_contexts / reference）
   并包成 ``EvaluationDataset``；
2. 注入本地裁判：LLM 用 llama.cpp 的 OpenAI 兼容接口（:data:`app.config.LLM_*`），
   embedding 用本地 ``models/bge-m3``——两者都**显式传入**，不走任何联网默认值；
3. 执行 ``evaluate(dataset, metrics=[...], llm=..., embeddings=...)``，
   四个指标：faithfulness / answer_relevancy / context_precision / context_recall；
4. 保存逐条结果到 ``{stage}.json``，并把均值、裁判模型、温度等写入 ``eval_meta.jsonl``。

它不跑检索，所以可以反复调评分参数、反复评估同一批 records。

用法::

    python -m app.eval.ragas_eval --input data/eval_results/rerank_records.json
    python -m app.eval.ragas_eval --input ... --output data/eval_results/rerank.json
    python -m app.eval.ragas_eval --compare data/eval_results/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402
    EVAL_RESULTS_DIR,
    JUDGE_TEMPERATURE,
    LLM_BASE_URL,
    LLM_DISABLE_THINKING,
    LLM_MODEL,
    LLM_TIMEOUT,
)

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
META_FILE = "eval_meta.jsonl"


# --------------------------------------------------------------------------- #
# 本地 embedding（BGE-M3）
# --------------------------------------------------------------------------- #
def make_embeddings():
    """把项目里已在用的 BGE-M3 包成 Langchain Embeddings，供 RAGAS 计算相似度。"""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class BgeM3Embeddings(Embeddings):
        def __init__(self, batch_size: int = 8) -> None:
            self.batch_size = batch_size
            self._encoder = None

        def _ensure_encoder(self):
            # 注意：不要把属性命名成 model——ragas 会把 embeddings.model 当成模型名
            # 写进 usage 事件，那里需要字符串，传一个 M3Embedder 对象会直接报 ValidationError。
            if self._encoder is None:
                from app.retrieval.vector_store import get_model

                self._encoder = get_model()
            return self._encoder

        def _encode(self, texts: Sequence[str]) -> list[list[float]]:
            output = self._ensure_encoder().encode(
                list(texts),
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return [vec.tolist() for vec in output["dense_vecs"]]

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return self._encode(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._encode([text])[0]

    return LangchainEmbeddingsWrapper(BgeM3Embeddings())


# --------------------------------------------------------------------------- #
# 本地裁判（llama.cpp）
# --------------------------------------------------------------------------- #
def make_judge():
    """用 llama.cpp 的 OpenAI 兼容接口做裁判。"""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}} if LLM_DISABLE_THINKING else None
    )
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="not-needed",  # llama.cpp 不校验
        model=LLM_MODEL,
        temperature=JUDGE_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        max_tokens=1024,
        extra_body=extra_body,
    )
    return LangchainLLMWrapper(llm)


def make_metrics():
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return [faithfulness, answer_relevancy, context_precision, context_recall]


# --------------------------------------------------------------------------- #
# Dataset 组装与评估
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"records 文件不存在：{path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"records 文件为空或格式不对：{path}")
    return data


def build_dataset(records: Sequence[dict[str, Any]]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=list(record["contexts"]),
            reference=record.get("ground_truth", ""),
        )
        for record in records
    ]
    return EvaluationDataset(samples=samples)


def run_eval(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """跑四个指标。

    降级策略：整批 ``evaluate`` 失败时**不放弃**，改成逐个指标单独跑——
    一个指标（例如 answer_relevancy 依赖 embedding）挂掉，其余三个仍然出分。
    单条样本失败由 ragas 自己记成 NaN，不在这里处理。
    """
    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = build_dataset(records)
    judge = make_judge()
    embeddings = make_embeddings()
    # llama.cpp 是单实例推理，并发只会排队并触发超时：worker 限 1，超时放宽到 10 分钟
    run_config = RunConfig(timeout=600, max_workers=1)

    failures: list[str] = []
    try:
        result = evaluate(
            dataset=dataset,
            metrics=make_metrics(),
            llm=judge,
            embeddings=embeddings,
            run_config=run_config,
        )
        frame = result.to_pandas()
    except Exception as exc:  # noqa: BLE001 - 整批失败时降级为逐指标重跑
        failures.append(f"批量评估失败：{type(exc).__name__}: {exc}")
        print(f"  ⚠ 批量评估失败，改为逐个指标重跑：{type(exc).__name__}: {exc}")
        frame = _evaluate_one_by_one(dataset, judge, embeddings, run_config, failures)

    rows = json.loads(frame.to_json(orient="records", force_ascii=False))
    mean = {}
    for name in METRIC_NAMES:
        if name in frame.columns:
            value = pd.to_numeric(frame[name], errors="coerce").mean()
            mean[name] = None if pd.isna(value) else round(float(value), 4)
        else:
            mean[name] = None
    return {"rows": rows, "mean": mean, "failures": failures}


def _evaluate_one_by_one(dataset, judge, embeddings, run_config, failures: list[str]):
    """逐个指标跑，收集成功的列；全部失败则返回只有原始字段的空表。"""
    from ragas import evaluate

    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    available = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    frame = None
    for name in METRIC_NAMES:
        try:
            single = evaluate(
                dataset=dataset,
                metrics=[available[name]],
                llm=judge,
                embeddings=embeddings,
                run_config=run_config,
            ).to_pandas()
        except Exception as exc:  # noqa: BLE001 - 单个指标的失败不影响其他指标
            message = f"{name} 评估失败：{type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"    ✗ {message}")
            continue

        if frame is None:
            frame = single
        elif name in single.columns:
            frame[name] = single[name]
        print(f"    ✓ {name} 完成")

    if frame is None:
        # 四个指标全挂：保留原始字段，均值留空，至少把失败原因记进 eval_meta
        base = dataset.to_pandas()
        for name in METRIC_NAMES:
            base[name] = None
        return base
    return frame


# --------------------------------------------------------------------------- #
# 保存与对比
# --------------------------------------------------------------------------- #
def save_result(
    stage: str,
    payload: dict[str, Any],
    output: Path,
    *,
    samples: int,
    extra: dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "stage": stage,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "samples": samples,
        "metrics": METRIC_NAMES,
        "mean": payload["mean"],
        "judge": {"model": LLM_MODEL, "base_url": LLM_BASE_URL, "temperature": JUDGE_TEMPERATURE},
        "failures": list(payload.get("failures") or []),
        "rows": payload["rows"],
    }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    meta = {
        "stage": stage,
        "evaluated_at": document["evaluated_at"],
        "samples": samples,
        "mean": payload["mean"],
        "judge_model": LLM_MODEL,
        "judge_base_url": LLM_BASE_URL,
        "temperature": JUDGE_TEMPERATURE,
        "failed_metrics": list(payload.get("failures") or []),
    }
    meta.update(extra or {})
    with (output.parent / META_FILE).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")


def compare(results_dir: Path) -> int:
    """读三个 stage 的 json，拼一张 Markdown 对比表。"""
    documents: dict[str, dict[str, Any]] = {}
    for stage in ("baseline", "hybrid", "rerank"):
        path = results_dir / f"{stage}.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                documents[stage] = json.load(handle)

    if not documents:
        print(f"{results_dir} 下没有 baseline.json / hybrid.json / rerank.json", file=sys.stderr)
        return 2

    stages = list(documents)
    header = "| 指标 | " + " | ".join(stages) + " |"
    separator = "|---" * (len(stages) + 1) + "|"
    print(f"# 三阶段对比（{results_dir.as_posix()}）\n")
    print(header)
    print(separator)
    for metric in METRIC_NAMES:
        cells = []
        for stage in stages:
            value = documents[stage].get("mean", {}).get(metric)
            cells.append("—" if value is None else f"{value:.4f}")
        print(f"| {metric} | " + " | ".join(cells) + " |")
    samples = [str(documents[s].get("samples", "?")) for s in stages]
    print("| 样本数 | " + " | ".join(samples) + " |")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.ragas_eval",
        description="用 RAGAS + 本地裁判评估一个 stage 的 records。",
    )
    parser.add_argument("--input", help="run_pipeline 产出的 {stage}_records.json")
    parser.add_argument("--output", help="逐条结果输出路径，默认 {EVAL_RESULTS_DIR}/<stage>.json")
    parser.add_argument("--stage", help="覆盖 stage 名（默认取 records 里的 stage 字段）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（冒烟用）")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="把采集阶段失败的样本（answer 为空）也纳入评分",
    )
    parser.add_argument("--compare", help="只做对比：传入 eval_results 目录")
    args = parser.parse_args(argv)

    if args.compare:
        return compare(Path(args.compare))

    if not args.input:
        parser.error("要么给 --input，要么用 --compare 指定结果目录")

    records = load_records(Path(args.input))

    # 采集阶段失败的题目（answer 为空）默认跳过：它们没有可评的回答，
    # 放进去只会把均值拉成 0，掩盖真实水平。要评分就加 --include-failed。
    failed = [r for r in records if r.get("failed")]
    if failed and not args.include_failed:
        records = [r for r in records if not r.get("failed")]
        print(f"跳过 {len(failed)} 条采集失败样本（用 --include-failed 可一并评估）")

    if args.limit:
        records = records[: args.limit]
    if not records:
        print("没有可评估的样本", file=sys.stderr)
        return 2

    stage = args.stage or records[0].get("stage") or Path(args.input).stem.replace("_records", "")
    output = Path(args.output) if args.output else Path(EVAL_RESULTS_DIR) / f"{stage}.json"

    print(f"stage={stage}  样本={len(records)}  裁判={LLM_MODEL} @ {LLM_BASE_URL}")
    payload = run_eval(records)
    save_result(
        stage,
        payload,
        output,
        samples=len(records),
        extra={"skipped_failed": len(failed) if not args.include_failed else 0},
    )

    print("\n平均分：")
    for metric in METRIC_NAMES:
        value = payload["mean"][metric]
        print(f"  {metric:20s} {'—' if value is None else f'{value:.4f}'}")
    if payload.get("failures"):
        print(f"\n⚠ 有 {len(payload['failures'])} 个指标评估失败（已记入 {META_FILE}）：")
        for failure in payload["failures"]:
            print(f"  - {failure}")
    print(f"\n逐条结果：{output.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
