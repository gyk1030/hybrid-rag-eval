"""跑管线、收集评估原料（评估流程第 1 步）。

这个文件**只做采集，不做评估**：对 Golden Set 里的每个问题，按 ``--stage`` 指定的
检索策略跑一遍「检索 → 组装提示词 → LLM 生成」，把当次实际进入 LLM 的
``contexts`` 和最终 ``answer`` 落盘，供 :mod:`app.eval.ragas_eval` 反复评分。

三个阶段共用同一个生成函数、同一个提示词、同一个 LLM，**唯一变量是 contexts 的来源**：

============  ==========================================================
stage         检索策略
============  ==========================================================
baseline      只用向量检索，取 top5
hybrid        向量 top20 + BM25 top20 → RRF → top5
rerank        向量 top20 + BM25 top20 → RRF → top10 → BGE 精排 → top5
============  ==========================================================

用法::

    python -m app.eval.run_pipeline --stage baseline
    python -m app.eval.run_pipeline --stage hybrid --limit 5
    python -m app.eval.run_pipeline --stage rerank --output data/eval_results/rerank_records.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402
    CHUNKS_PATH,
    EVAL_BASELINE_TOP_K,
    EVAL_GOLDEN_PATH,
    EVAL_HYBRID_TOP_N,
    EVAL_RERANK_CANDIDATES,
    EVAL_RERANK_TOP_M,
    EVAL_RESULTS_DIR,
    HYBRID_TOP_K,
)
from app.generation.answer import call_llm, get_pipeline  # noqa: E402

STAGES = ("baseline", "hybrid", "rerank")


# --------------------------------------------------------------------------- #
# 三种检索策略
# --------------------------------------------------------------------------- #
def make_retriever(
    stage: str,
    pipeline: dict[str, Any],
) -> Callable[[str, Any | None], list[Any]]:
    """返回一个 ``(question, recorder) -> hits`` 的函数；三阶段只有这一处不同。

    评估场景不做降级：某个 stage 需要的组件缺失时直接报错退出，
    否则会产出一批「降级后的 records」，把三阶段对比的前提（只换检索策略）破坏掉。

    ``recorder`` 为 None 时不记录任何追踪信息；传入
    :class:`app.generation.trace.TraceRecorder` 则记录该阶段的线索，
    便于事后回答「正确片段是在哪一步丢的」。
    """
    store = pipeline.get("store")
    index = pipeline.get("index")
    reranker = pipeline.get("reranker")

    if stage == "baseline":
        if store is None:
            raise SystemExit("baseline 需要向量检索，但 VectorStore 不可用")

        def retrieve(question: str, recorder: Any | None) -> list[Any]:
            started = time.perf_counter()
            hits = store.search(question, top_k=EVAL_BASELINE_TOP_K)
            if recorder is not None:
                recorder.record_vector(hits, (time.perf_counter() - started) * 1000)
            return hits

        return retrieve

    if stage == "hybrid":
        if store is None or index is None:
            raise SystemExit("hybrid 需要向量与 BM25 两路，但有组件不可用")
        from app.retrieval.rrf import hybrid_search

        def retrieve(question: str, recorder: Any | None) -> list[Any]:
            return hybrid_search(
                question,
                top_k=HYBRID_TOP_K,
                top_n=EVAL_HYBRID_TOP_N,
                store=store,
                index=index,
                trace=recorder,
            )

        return retrieve

    if stage == "rerank":
        if store is None or index is None or reranker is None:
            raise SystemExit("rerank 需要向量 + BM25 + 精排三者，但有组件不可用")
        from app.retrieval.rrf import hybrid_search

        def retrieve(question: str, recorder: Any | None) -> list[Any]:
            candidates = hybrid_search(
                question,
                top_k=HYBRID_TOP_K,
                top_n=EVAL_RERANK_CANDIDATES,
                store=store,
                index=index,
                trace=recorder,
            )
            before = list(candidates[:EVAL_RERANK_TOP_M])
            started = time.perf_counter()
            reranked = reranker.rerank(question, candidates, top_m=EVAL_RERANK_TOP_M)
            if recorder is not None:
                recorder.record_rerank(before, reranked, (time.perf_counter() - started) * 1000)
            return reranked

        return retrieve

    raise ValueError(f"未知 stage：{stage}（可选 {STAGES}）")


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def generate_answer(question: str, hits: Sequence[Any]) -> dict[str, Any]:
    """把 hits 组装成提示词并生成答案——三阶段共用，保证提示词完全一致。

    生成后顺手跑一次年份校验（确定性，不调模型）。评测链路走的是**裸模型**，
    不过线上那层告警；这里把校验结果记进 records，用来统计「有多少题其实会被拦下」。
    """
    from app.generation.answer import build_contexts, build_messages
    from app.generation.consistency import check_year_consistency

    contexts = build_contexts(hits)
    messages = build_messages(question, contexts)
    answer, usage = call_llm(messages)
    check = check_year_consistency(question, answer, contexts)
    return {
        "answer": answer,
        "contexts": [ctx.text for ctx in contexts],
        "context_chunk_ids": [ctx.chunk_id for ctx in contexts],
        "usage": usage,
        "messages": messages,
        "answer_check": check.to_dict(),
    }


def run_stage(
    stage: str,
    golden: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    on_record: Callable[[list[dict[str, Any]]], None] | None = None,
    trace: bool = False,
) -> list[dict[str, Any]]:
    """逐题跑检索 + 生成。

    单题失败（LLM 超时、检索异常）不会中断整批：该题记为 ``failed`` 并继续，
    已完成的记录通过 ``on_record`` 增量落盘，进程被杀也不会全丢。

    ``trace=True`` 时每题的 record 会多一个 ``trace`` 字段，记录 vector / bm25 /
    rrf / rerank / context / timing 六个阶段——用于事后定位「正确片段丢在哪一步」。
    """
    from app.generation.trace import TraceRecorder  # 延迟导入，不影响无追踪的常规跑批

    pipeline = get_pipeline()
    retrieve = make_retriever(stage, pipeline)

    records: list[dict[str, Any]] = []
    selected = golden[:limit] if limit else list(golden)
    total = len(selected)

    for position, item in enumerate(selected, start=1):
        question = str(item.get("question", "")).strip()
        base = {
            "stage": stage,
            "id": item.get("id"),
            "question": question,
            "ground_truth": item.get("ground_truth", ""),
            "reference_chunk_ids": item.get("relevant_chunk_ids", []),
        }
        recorder = TraceRecorder(question) if trace else None

        try:
            started = time.perf_counter()
            hits = retrieve(question, recorder)
            retrieval_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            payload = generate_answer(question, hits)
            generation_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - 单题失败不该废掉整批
            record = {
                **base,
                "answer": "",
                "contexts": [],
                "context_chunk_ids": [],
                "retrieved": [],
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "usage": None,
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if recorder is not None:
                recorder.mark_fallback("retrieval", f"{type(exc).__name__}: {exc}")
                record["trace"] = recorder.to_dict()
            records.append(record)
            print(f"[{position}/{total}] ✗ 失败 {question[:36]}… —— {type(exc).__name__}: {exc}")
            if on_record is not None:
                on_record(records)
            continue

        if recorder is not None:
            recorder.add_timing("retrieval_ms", retrieval_ms)
            recorder.add_timing("generation_ms", generation_ms)
            recorder.record_context(payload["contexts"])

        record = {
            **base,
            "answer": payload["answer"],
            "contexts": payload["contexts"],
            "context_chunk_ids": payload["context_chunk_ids"],
            "retrieved": [
                {
                    "chunk_id": getattr(hit, "chunk_id", ""),
                    "score": float(getattr(hit, "score", 0.0)),
                    "source": getattr(hit, "source", ""),
                    "page": int(getattr(hit, "page", 0) or 0),
                    "block_type": getattr(hit, "block_type", ""),
                }
                for hit in hits
            ],
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "usage": payload["usage"],
            # 年份校验：线上靠它弹告警，评测里用来统计「多少题其实会被拦下」
            "answer_check": payload.get("answer_check"),
            "failed": False,
        }
        if recorder is not None:
            record["trace"] = recorder.to_dict()
        records.append(record)
        check = payload.get("answer_check") or {}
        flag = "  ⚠ 年份不符" if check.get("status") == "mismatch" else ""
        print(
            f"[{position}/{total}] {question[:40]}…  检索 {retrieval_ms:.0f}ms "
            f"生成 {generation_ms:.0f}ms  答案 {len(payload['answer'])} 字{flag}"
        )
        if on_record is not None:
            on_record(records)
    return records


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Golden Set 不存在或为空：{path}（先跑 build_golden.py）")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Golden Set 格式不正确：{path}")
    return data


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """写 records。先写临时文件再替换，避免中途被杀留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run_pipeline",
        description="按 stage 跑检索+生成，落评估原料。",
    )
    parser.add_argument("--stage", required=True, choices=STAGES, help="检索策略")
    parser.add_argument("--golden", default=EVAL_GOLDEN_PATH, help="Golden Set 路径")
    parser.add_argument(
        "--output",
        help=f"输出路径，默认 {EVAL_RESULTS_DIR}/<stage>_records.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，0=全部）")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="每完成 N 条落盘一次（默认 1，进程被杀也不丢已完成的部分）",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="为每题记录各阶段追踪线索（vector / bm25 / rrf / rerank / context / timing）",
    )
    args = parser.parse_args(argv)

    golden = load_golden(Path(args.golden))
    output = Path(args.output) if args.output else Path(EVAL_RESULTS_DIR) / f"{args.stage}_records.json"

    print(f"stage={args.stage}  问题数={len(golden)}  输出={output.as_posix()}")

    # 增量落盘：每 flush_every 条写一次
    pending = {"count": 0}

    def on_record(records: list[dict[str, Any]]) -> None:
        pending["count"] += 1
        if pending["count"] % max(1, args.flush_every) == 0:
            write_records(output, records)

    records = run_stage(
        args.stage,
        golden,
        limit=args.limit or None,
        on_record=on_record,
        trace=args.trace,
    )
    write_records(output, records)

    failed = [r for r in records if r.get("failed")]
    total_retrieval = sum(r["retrieval_ms"] for r in records)
    total_generation = sum(r["generation_ms"] for r in records)
    print(
        f"完成 {len(records) - len(failed)}/{len(records)} 条："
        f"检索合计 {total_retrieval / 1000:.1f}s，生成合计 {total_generation / 1000:.1f}s "
        f"→ {output.as_posix()}"
    )

    # 年份校验汇总：这是「裸模型」的出错率，线上会被 answer_check 拦下
    checks = [r.get("answer_check") or {} for r in records if not r.get("failed")]
    mismatch = [r for r, c in zip(records, checks) if c.get("status") == "mismatch"]
    ok_count = sum(1 for c in checks if c.get("status") == "ok")
    skipped = sum(1 for c in checks if c.get("status") == "skipped")
    if checks:
        print(
            f"年份校验：通过 {ok_count} / 不符 {len(mismatch)} / 跳过 {skipped}"
            f"（共 {len(checks)} 条）"
        )
        for record in mismatch:
            detail = (record.get("answer_check") or {}).get("detail", "")
            print(f"  ⚠ {record.get('id')} {record['question'][:32]}… —— {detail}")

    if failed:
        print(f"其中 {len(failed)} 条失败（已写入 records，评分时会因空上下文得低分）：")
        for record in failed:
            print(f"  - {record.get('id')} {record['question'][:40]}… —— {record.get('error')}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
