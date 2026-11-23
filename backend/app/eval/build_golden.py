"""Golden Set 的构造与校验（评估流程阶段 0）。

这个文件只负责「把尺子做准」，不产出任何评估数字：

1. 从 PostgreSQL 的 ``chunks`` 表读取已入库的分块；
2. 按文档均匀抽样，用本地 llama.cpp（Qwen3.5-4B）从 chunk 反向生成
   「问题 + 标准答案」，生成结果标记为 ``source="llm"``、``reviewed=false``，
   需要人工复核后才应作为正式评测集；
3. 校验格式：``question / ground_truth / relevant_chunk_ids`` 必填；
4. 校验 ``relevant_chunk_ids`` 真实性：拿 id 回查数据库，确认 chunk 真实存在
   （拼错的 id 会让后面的 Recall@K 静默算错）；
5. 统计题型分布：精确型问题（含编号 / 日期 / 金额 / 百分比）占比低于 30% 时告警。

用法::

    # 只校验已有的 golden set（比如人工标完后）
    python -m app.eval.build_golden --validate-only

    # 从数据库抽样生成 12 条，再和已有条目合并、校验、写盘
    python -m app.eval.build_golden --generate 12

    # 合并一份手工标注的文件
    python -m app.eval.build_golden --from-file my_labeled.json --validate-only

输出：``data/golden_set.json``（数组，每条一个 QA 及其依据的 chunk id）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import psycopg2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402
    DSN,
    EVAL_GOLDEN_PATH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
)
from app.generation.answer import call_llm  # noqa: E402

# 精确型问题的判定：出现编号/日期/金额/百分比等硬信息
EXACT_PATTERNS = [
    re.compile(r"\d{4}\s*年"),
    re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"\d+(\.\d+)?\s*%"),
    re.compile(r"\d[\d,]*\s*(元|万元|亿元|千元|股|人|家|项)"),
    re.compile(r"[A-Z]?\d{6,}"),
    re.compile(r"第[一二三四五六七八九十\d]+"),
]

# 近似重复阈值：两问句的相似度超过它即视为同一题（LLM 换个问法就能绕开精确去重）
NEAR_DUPLICATE_RATIO = 0.85

GENERATE_SYSTEM = (
    "你是财报数据集标注员。根据给出的年报原文，出 1 个能用这段原文准确回答的问题，"
    "并给出简短的标准答案。要求：\n"
    "1. 问题要具体，尽量包含公司名或年份、以及具体指标名；\n"
    "2. 标准答案只能来自原文，保留原文的数值和单位，不要补充外部知识；\n"
    "3. 原文信息不足时，把问题聚焦到原文确实写到的内容上；\n"
    "4. **如果原文里没有任何具体数值（例如只是表头、行标签或科目名称），"
    "绝对不要出需要数字的题目**，改为问文中确实写到的定性内容；"
    "标准答案里绝不能出现原文没有的数字（包括 0）；\n"
    "5. 只输出一个 JSON 对象，形如："
    '{"question": "...", "ground_truth": "..."}，不要输出其他文字。'
)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    block_type: str
    text: str


# --------------------------------------------------------------------------- #
# 数据读取与抽样
# --------------------------------------------------------------------------- #
def load_chunks(dsn: str | None = None) -> list[Chunk]:
    """从数据库读取候选 chunk（只要正文块，过短/过长的表格块不适合出题）。"""
    conn = psycopg2.connect(dsn or DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, source, page, block_type, text FROM chunks "
                "WHERE block_type = 'text' AND char_length(text) BETWEEN 150 AND 900 "
                "ORDER BY source, page, chunk_id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [Chunk(*row) for row in rows]


def sample_chunks(chunks: Sequence[Chunk], count: int, *, seed: int = 42) -> list[Chunk]:
    """按文档轮转抽样，保证覆盖多个文档，而不是集中在一份年报里。"""
    if count <= 0 or not chunks:
        return []

    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    rng = random.Random(seed)
    for items in by_source.values():
        rng.shuffle(items)

    picked: list[Chunk] = []
    sources = sorted(by_source)
    index = 0
    while len(picked) < count and any(by_source[s] for s in sources):
        source = sources[index % len(sources)]
        if by_source[source]:
            picked.append(by_source[source].pop())
        index += 1
    return picked


# --------------------------------------------------------------------------- #
# LLM 辅助生成
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出最后一个合法 JSON 对象（兼顾思考过程、代码围栏）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)

    candidates = [cleaned]
    starts = [i for i, ch in enumerate(cleaned) if ch == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(cleaned)):
            if cleaned[end] == "{":
                depth += 1
            elif cleaned[end] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start : end + 1])
                    break

    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("question") and data.get("ground_truth"):
            return data
    return None


def generate_qa(chunk: Chunk) -> dict[str, Any] | None:
    """让本地 LLM 根据一个 chunk 反推问题与标准答案。"""
    messages = [
        {"role": "system", "content": GENERATE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"公司/文档：{Path(chunk.source).name}\n第 {chunk.page} 页\n"
                f"原文：\n{chunk.text[:900]}"
            ),
        },
    ]
    answer, _ = call_llm(messages, temperature=0.3, max_tokens=min(LLM_MAX_TOKENS, 400))
    data = _extract_json(answer)
    if not data:
        return None
    return {
        "question": str(data["question"]).strip(),
        "ground_truth": str(data["ground_truth"]).strip(),
        "relevant_chunk_ids": [chunk.chunk_id],
        "source": "llm",
        "reviewed": False,
    }


# --------------------------------------------------------------------------- #
# 校验与统计
# --------------------------------------------------------------------------- #
def is_exact_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in EXACT_PATTERNS)


def validate(items: Sequence[dict[str, Any]], known_ids: set[str]) -> tuple[list[str], list[str]]:
    """校验格式与 id 真实性，返回 ``(错误, 警告)``。"""
    errors: list[str] = []
    warnings: list[str] = []
    seen_questions: dict[str, int] = {}

    for index, item in enumerate(items, start=1):
        label = f"第 {index} 条"
        for field in ("question", "ground_truth", "relevant_chunk_ids"):
            if field not in item:
                errors.append(f"{label} 缺少字段 {field}")
        if errors and errors[-1].startswith(label):
            continue

        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        ids = item.get("relevant_chunk_ids") or []

        if not question:
            errors.append(f"{label} 的 question 为空")
        if not ground_truth:
            errors.append(f"{label} 的 ground_truth 为空")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{label} 的 relevant_chunk_ids 必须是非空数组")
        else:
            missing = [cid for cid in ids if cid not in known_ids]
            if missing:
                errors.append(f"{label} 引用了不存在的 chunk_id：{missing[:3]}")

        key = re.sub(r"\s+", "", question)
        if key in seen_questions:
            warnings.append(f"{label} 与第 {seen_questions[key]} 条问题重复：{question[:30]}…")
        else:
            seen_questions[key] = index

    return errors, warnings


def summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """统计题型分布与文档覆盖。"""
    total = len(items)
    exact = sum(1 for item in items if is_exact_question(str(item.get("question", ""))))
    by_source: dict[str, int] = {}
    for item in items:
        source = str(item.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1
    reviewed = sum(1 for item in items if item.get("reviewed"))
    return {
        "total": total,
        "exact": exact,
        "exact_ratio": (exact / total) if total else 0.0,
        "by_source": by_source,
        "reviewed": reviewed,
    }


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def write_json(path: Path, items: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(list(items), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", str(question))


def dedupe_near_duplicates(
    items: Sequence[dict[str, Any]],
    *,
    threshold: float = NEAR_DUPLICATE_RATIO,
) -> tuple[list[dict[str, Any]], int]:
    """去掉「问法只差几个字」的近似重复，保留首次出现的那条。

    LLM 从相邻 chunk 出题时很容易写出同一件事的两种问法（比如
    「……应计入哪个科目？」与「根据 XX 年报，……应计入哪个科目？」），
    完全相同的文本去重挡不住它们。
    """
    kept: list[dict[str, Any]] = []
    kept_questions: list[str] = []
    removed = 0

    for item in items:
        question = _normalize_question(item.get("question", ""))
        if not question:
            kept.append(item)
            continue
        if any(
            SequenceMatcher(None, question, other).ratio() >= threshold for other in kept_questions
        ):
            removed += 1
            continue
        kept.append(item)
        kept_questions.append(question)
    return kept, removed


def merge(existing: Sequence[dict[str, Any]], generated: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 question 去重合并，已有条目优先（人工标注不被覆盖）。"""
    merged = list(existing)
    seen = {_normalize_question(item.get("question", "")) for item in merged}
    for item in generated:
        key = _normalize_question(item.get("question", ""))
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    merged, _ = dedupe_near_duplicates(merged)
    for index, item in enumerate(merged, start=1):
        item["id"] = item.get("id") or f"gs_{index:03d}"
    return merged


def report(items: Sequence[dict[str, Any]], errors: Sequence[str], warnings: Sequence[str]) -> int:
    stats = summarize(items)
    print(f"Golden Set：{stats['total']} 条")
    print(
        f"  精确型问题 {stats['exact']} 条（{stats['exact_ratio']:.0%}），"
        f"已人工复核 {stats['reviewed']} 条"
    )
    print(f"  来源分布：{stats['by_source']}")
    if stats["exact_ratio"] < 0.3 and stats["total"] >= 5:
        print("  ⚠ 精确型问题占比低于 30%，建议补充带编号/日期/金额的题目")
    for warning in warnings:
        print(f"  ⚠ {warning}")
    if errors:
        print(f"  ✗ 校验失败 {len(errors)} 处：")
        for error in errors[:20]:
            print(f"     - {error}")
        return 1
    print("  ✓ 校验通过（字段齐全、chunk_id 均真实存在）")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.build_golden",
        description="构造/校验 Golden Set。",
    )
    parser.add_argument("--output", default=EVAL_GOLDEN_PATH, help=f"输出路径，默认 {EVAL_GOLDEN_PATH}")
    parser.add_argument("--generate", type=int, default=0, help="用 LLM 从数据库抽样生成多少条")
    parser.add_argument("--from-file", help="并入一份已有的 golden set（JSON 数组）")
    parser.add_argument("--validate-only", action="store_true", help="只校验，不生成、不写盘")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = parser.parse_args(argv)

    output = Path(args.output)
    known_ids: set[str] = set()
    try:
        known_ids = {chunk.chunk_id for chunk in load_chunks()}
    except Exception as exc:  # noqa: BLE001
        print(f"警告：无法连接数据库校验 chunk_id（{type(exc).__name__}: {exc}）", file=sys.stderr)

    items = load_json(output)
    if args.from_file:
        items = merge(items, load_json(Path(args.from_file)))

    if args.generate > 0:
        candidates = sample_chunks(load_chunks(), args.generate, seed=args.seed)
        print(f"抽样 {len(candidates)} 个 chunk，开始用 {LLM_MODEL} 生成问题…")
        generated = []
        for index, chunk in enumerate(candidates, start=1):
            try:
                item = generate_qa(chunk)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{index}/{len(candidates)}] 失败：{type(exc).__name__}: {exc}")
                continue
            if item:
                generated.append(item)
                print(f"  [{index}/{len(candidates)}] {item['question'][:50]}")
            else:
                print(f"  [{index}/{len(candidates)}] 模型输出无法解析为 JSON，跳过")
        items = merge(items, generated)

    if not items:
        print("没有可写入的条目：用 --generate 生成，或 --from-file 并入一份标注文件", file=sys.stderr)
        return 2

    errors, warnings = validate(items, known_ids)
    code = report(items, errors, warnings)

    if not args.validate_only and not errors:
        write_json(output, items)
        print(f"已写入 {output.as_posix()}")
    return code


if __name__ == "__main__":
    sys.exit(main())
