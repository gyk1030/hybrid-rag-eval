"""答案年份一致性校验（确定性，不依赖模型）。

背景：3B 模型会答非所问——把 2024 年报表格里的 1.39% 当成「2023 年末」的答案，
而正确资料（2023 年报 1.37%）当时就排在上下文第一位。这类错误比幻觉更危险：
数字是真的，只是年份配错了，看起来完全可信。

校验思路分两步，都只看确定性信息：

1. **数值归属**：把答案里的数字逐个拿到上下文里比对，找出「答案的数字来自哪一条资料」，
   再看那条资料的文件名年份是否在问题所问年份里。这是主判据。
2. **年份直述**：答案里明写了某一年（如「2024 年末」），而该年份不在问题年份里，
   且数值归属无法判断时，作为兜底判据。

之所以不直接「答案里出现别的年份就算错」：财报答案经常需要对比（「较 2022 年末上升
0.05 个百分点」），那些年份是合法引用，不能误伤。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# 「2023 年」「2023年12月31日」「2023 年度」
YEAR_PATTERN = re.compile(r"(20\d{2})\s*年")
# 「2021-2024 年」这类区间：问句里两头都算相关年份
YEAR_RANGE_PATTERN = re.compile(r"(20\d{2})\s*[-–—~～至到]\s*(20\d{2})")
# 答案里的数值：保留小数与百分号，忽略千分位逗号
NUMBER_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
# 数值后面是否紧跟「年」（用来把 2023 这类年份从数值里排除）
YEAR_SUFFIX = re.compile(r"\s*年")

# 判定为「年份」而非「数值」的范围：不参与数值归属比对
YEAR_MIN, YEAR_MAX = 2000, 2100


@dataclass(slots=True)
class YearCheck:
    """一次年份校验的结果。"""

    # ok：通过；mismatch：不通过；skipped：问题未提年份，无从校验
    status: str
    asked_years: list[int] = field(default_factory=list)
    # 答案里明写的年份
    answer_years: list[int] = field(default_factory=list)
    # 答案数值所属资料的文件年份（None 表示无法归属）
    source_year: int | None = None
    # 归属到的资料序号（上下文里的 index），便于前端高亮
    source_index: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "asked_years": self.asked_years,
            "answer_years": self.answer_years,
            "source_year": self.source_year,
            "source_index": self.source_index,
            "detail": self.detail,
        }


def extract_years(text: str) -> list[int]:
    """按出现顺序提取文本里的四位年份（去重）。

    「2021-2024 年」这种区间会把两端都算进来（正则只匹配到「2024 年」，
    所以额外扫一遍区间）。
    """
    raw = str(text or "")
    found: list[tuple[int, int]] = []

    for match in YEAR_PATTERN.finditer(raw):
        year = int(match.group(1))
        if YEAR_MIN <= year <= YEAR_MAX:
            found.append((match.start(), year))
    for match in YEAR_RANGE_PATTERN.finditer(raw):
        for group in (1, 2):
            year = int(match.group(group))
            if YEAR_MIN <= year <= YEAR_MAX:
                found.append((match.start(), year))

    seen: list[int] = []
    for _, year in sorted(found):
        if year not in seen:
            seen.append(year)
    return seen


def question_years(question: str) -> list[int]:
    """问题真正相关的年份集合：区间会被展开（「2021-2024 年」→ 2021~2024）。

    用于「允许哪些年份的资料」——区间题往往需要中间年份的数据。
    """
    years = set(extract_years(question))
    for match in YEAR_RANGE_PATTERN.finditer(str(question or "")):
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and end - start <= 10:
            years.update(range(start, end + 1))
    return sorted(years)


def extract_numbers(text: str) -> list[float]:
    """提取答案里有意义的数值：排除年份本身，保留小数。"""
    raw = str(text or "")
    values: list[float] = []
    for match in NUMBER_PATTERN.finditer(raw):
        token = match.group(1)
        value = float(token)
        # 跳过「2023 年」这类年份写法（数字后可能带空格再跟「年」）
        if token.isdigit() and YEAR_MIN <= value <= YEAR_MAX and YEAR_SUFFIX.match(raw, match.end()):
            continue
        values.append(value)
    return values


def source_year(source: str) -> int | None:
    """从文件名里取报告年度，如 ``000001_平安银行_2023年年度报告.pdf`` → 2023。"""
    name = Path(str(source or "")).name
    match = YEAR_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _numbers_in_text(text: str) -> set[float]:
    """上下文里出现的数值集合，用于「答案数字来自哪条资料」的比对。"""
    values: set[float] = set()
    for match in NUMBER_PATTERN.finditer(str(text or "")):
        try:
            values.add(float(match.group(1)))
        except ValueError:  # pragma: no cover - 正则已保证可解析
            continue
    return values


def _format_years(years: Sequence[int]) -> str:
    return "、".join(str(year) for year in years) if years else "无"


def check_year_consistency(
    question: str,
    answer: str,
    contexts: Iterable[Any],
) -> YearCheck:
    """校验答案的年份是否与问题一致。

    ``contexts`` 中的元素需要带 ``source`` / ``text`` / ``index`` 属性
    （即 :class:`app.generation.answer.RetrievedContext`）。
    """
    asked = question_years(question)
    answer_years = extract_years(answer)

    if not asked:
        return YearCheck(status="skipped", answer_years=answer_years, detail="问题未提年份，跳过校验")

    # ---- 1. 数值归属：答案的数字主要来自哪一条资料 ----
    answer_numbers = [value for value in extract_numbers(answer) if value not in (0.0,)]
    best_index: int | None = None
    best_score = 0
    best_year: int | None = None

    if answer_numbers:
        for position, context in enumerate(contexts, start=1):
            text_numbers = _numbers_in_text(getattr(context, "text", "") or "")
            if not text_numbers:
                continue
            hit = sum(1 for value in answer_numbers if value in text_numbers)
            if hit > best_score:
                best_score = hit
                best_index = int(getattr(context, "index", position) or position)
                best_year = source_year(getattr(context, "source", "") or "")

    # ---- 2. 判定 ----
    if best_score > 0 and best_year is not None:
        if best_year in asked:
            return YearCheck(
                status="ok",
                asked_years=asked,
                answer_years=answer_years,
                source_year=best_year,
                source_index=best_index,
                detail=f"答案数值来自 {best_year} 年报，与问题年份一致",
            )
        return YearCheck(
            status="mismatch",
            asked_years=asked,
            answer_years=answer_years,
            source_year=best_year,
            source_index=best_index,
            detail=(
                f"问题问的是 {_format_years(asked)} 年，"
                f"但答案的数值来自 {best_year} 年报（参考资料 {best_index}）"
            ),
        )

    # ---- 3. 兜底：答案明写的年份不在问题年份里 ----
    wrong_years = [year for year in answer_years if year not in asked]
    if wrong_years:
        return YearCheck(
            status="mismatch",
            asked_years=asked,
            answer_years=answer_years,
            detail=(
                f"问题问的是 {_format_years(asked)} 年，"
                f"但答案出现 {_format_years(wrong_years)} 年"
            ),
        )

    return YearCheck(
        status="ok",
        asked_years=asked,
        answer_years=answer_years,
        detail="未发现年份冲突",
    )
