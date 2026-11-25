"""混合检索 RAG 示例项目的 PDF -> ``chunks.jsonl`` 解析流水线。

正文由 PyMuPDF（``pymupdf``）提取，表格由 pdfplumber 提取，每个 chunk 都经过
:mod:`app.ingestion.schema` 中的 pydantic 模型校验。

命令行用法::

    python -m app.ingestion.pdf_parser --input data/corpus --output data/parsed/chunks.jsonl --errors data/parsed/errors.jsonl

单个文件的处理步骤：

1. 用 PyMuPDF 逐页读取文本行，并保留每行的 bbox；
2. 在同一批页面上用 pdfplumber 检测表格，记录表格 bbox；
3. 丢弃落在表格 bbox 内的文本行，避免表格内容混进正文块；
4. 丢弃重复出现的短行（页眉、页脚、页码）：既用指南的「出现次数 > 总页数 60%」
   频率规则，也用边距带规则处理按章节变化的页眉；
5. 合并 PDF 文本提取造成的硬换行；
6. 按约 400 字符分块、80 字符重叠，并尽量在句末断开；
7. 把每张表格转成带标题前缀的 Markdown 块，超过 10 行的表格分组切分且每组重复表头。

需要的第三方依赖：``pymupdf>=1.24``、``pdfplumber>=0.11``、``pydantic>=2``。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Sequence

import pdfplumber
import pymupdf

# 允许用脚本方式直接运行（python app/ingestion/pdf_parser.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里；
# 统一改走包内绝对导入，两种运行方式（-m 与直接路径）都能工作。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ingestion.schema import Chunk, ChunkMetadata  # noqa: E402
from app.ingestion.utils import (  # noqa: E402
    configure_logging,
    counting_key,
    doc_id_for,
    get_logger,
    normalize_inline,
    relative_source,
)

# --------------------------------------------------------------------------- #
# 可调参数（见 pdf_parser_guide.md 第 6 节）
# --------------------------------------------------------------------------- #

DEFAULT_INPUT = Path("data/corpus")
DEFAULT_OUTPUT = Path("data/parsed/chunks.jsonl")
DEFAULT_ERRORS = Path("data/parsed/errors.jsonl")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

HEADER_FOOTER_MAX_CHARS = 50
HEADER_FOOTER_PAGE_RATIO = 0.6
# 页眉页脚补充规则使用的边距带（占页面高度的比例），以及某行需要在多少页上
# 重复出现才判定为页眉页脚。
HEADER_FOOTER_MARGIN_TOP_RATIO = 0.09
HEADER_FOOTER_MARGIN_BOTTOM_RATIO = 0.91
HEADER_FOOTER_MARGIN_MIN_PAGES = 3

TABLE_SPLIT_ROWS = 10
TABLE_CAPTION_MAX_CHARS = 80
TABLE_CAPTION_MAX_GAP = 40.0
TABLE_EDGE_GATE = 5

# 跨页续表：pdfplumber 会把一张跨页表格拆成两张，下一页那张没有表头。
# 判定为续表需要同时满足「列数一致」和下面任一位置条件：
#   - 上一页表格底边贴近页底（表格没画完就翻页）
#   - 本页表格顶边贴近页顶（表格从页面顶端接着画）
TABLE_CONTINUE_BOTTOM_RATIO = 0.85
TABLE_CONTINUE_TOP_RATIO = 0.15

# 单元格是否「纯数值」：用于区分表头行与数据行。
# 允许括号负数、千分位逗号、百分号，例如 1,234 / 5.5% / (1.2%) / -。
NUMERIC_CELL_RE = re.compile(r"^[(（]?\s*[-–—]?\s*[\d,，]+(?:\.\d+)?\s*[%％]?\s*[)）]?$")
# 代码里检测年份：出现「20XX 年」说明该块内的数值可以归年。
YEAR_IN_TEXT_RE = re.compile(r"20\d{2}\s*年")
# 财税意义上的「有数值」：带百分号，或带千分位逗号的金额。
# 这样「000001」这种股票代码不会触发「无表头」警示。
FINANCIAL_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*[%％]|\d{1,3}(?:[,，]\d{3})+(?:\.\d+)?")
# 检测到的网格只有在确实是表格时才保留：至少表头加一行数据、至少两列、
# 且非空单元格占比达标。没有这道闸门，带边框的正文段落会被识别成单列表格
# （正文随之从文本流里消失），稀疏的版面残迹会变成几乎没有内容的 30 列网格。
TABLE_MIN_ROWS = 2
TABLE_MIN_COLUMNS = 2
TABLE_MIN_FILL_RATIO = 0.3

SENTENCE_TERMINATORS = "。！？；：!?;:"
TRAILING_QUOTES = "\"'”’）)】」』》>"
UNNAMED_TABLE = "未命名表格"
CONTINUED_SUFFIX = "（续）"

# 整行只有页码时，绝不可能是表格标题。
_PAGE_NUMBER_RE = re.compile(r"^第\s*\d+\s*页$|^\d+\s*/\s*\d+$|^[-–—\s]*\d+[-–—\s]*$")

# 显式标题（"表3-2 ..."、"附表 1 ..."、"Table 4 ..."）属于表格块；
# 仅仅位于表格上方的章节标题（"1.1.1 公司信息"）会用作标题，但仍留在正文流中。
_EXPLICIT_CAPTION_RE = re.compile(
    r"^(?:续\s*)?(?:附\s*)?表\s*[A-Za-z0-9一二三四五六七八九十]"
    r"|^(?:续\s*)?(?:Table|Tab\.)\s*[0-9A-Za-z]",
    re.IGNORECASE,
)


class PdfParseError(RuntimeError):
    """文件完全无法解析时抛出（会记录到 errors.jsonl）。"""


# --------------------------------------------------------------------------- #
# 轻量值对象
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Line:
    """PyMuPDF 提取出的一行文本。"""

    text: str
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class TableBlock:
    """页面上检测到的一张表格。"""

    page: int
    index: int
    caption: str
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    caption_bbox: tuple[float, float, float, float] | None = None
    # 跨页续表时记录表头来自哪一页（None 表示自带表头）
    inherited_from: int | None = None


@dataclass(slots=True)
class ParseStats:
    """一次 CLI 运行的汇总计数。"""

    files: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 文本提取
# --------------------------------------------------------------------------- #


def _extract_page_lines(page: "pymupdf.Page") -> list[Line]:
    """按阅读顺序返回 ``page`` 中非空的文本行。"""
    lines: list[Line] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # 0 表示文本块
            continue
        for line in block.get("lines", []):
            text = normalize_inline("".join(span.get("text", "") for span in line.get("spans", [])))
            if not text:
                continue
            lines.append(Line(text=text, bbox=tuple(line["bbox"])))  # type: ignore[arg-type]
    return lines


def _in_margin_band(bbox: Sequence[float], page_height: float) -> bool:
    """判断该行是否位于页面上边距或下边距带内。"""
    top = page_height * HEADER_FOOTER_MARGIN_TOP_RATIO
    bottom = page_height * HEADER_FOOTER_MARGIN_BOTTOM_RATIO
    return float(bbox[3]) <= top or float(bbox[1]) >= bottom


def _drop_headers_and_footers(
    page_lines: dict[int, list[Line]],
    page_tables: dict[int, list[TableBlock]],
    page_heights: dict[int, float],
    total_pages: int,
    logger: logging.Logger,
    source: str,
) -> tuple[dict[int, list[Line]], int, set[str]]:
    """剔除文档中重复出现的页眉页脚行。

    对短行（少于 50 字符）应用两条互补规则：

    1. 指南的频率规则——出现次数超过 ``total_pages * 0.6`` 的行视为版面噪声；
    2. 边距规则——某行在页面上边距或下边距带内、且至少在
       :data:`HEADER_FOOTER_MARGIN_MIN_PAGES` 个页面上重复出现。年报的页眉
       按章节变化（"第一章 公司简介 2023 年年度报告"），没有任何变体能达到
       60% 的页面，只靠频率规则会把页眉留在正文里。

    规则 1 要求一行至少出现两次：只出现一次无法证明它是页眉，否则单页 PDF 里
    每一行短文本都会被删掉。
    """
    blocked: dict[int, list[Sequence[float]]] = {
        page_no: [table.bbox for table in tables] for page_no, tables in page_tables.items()
    }

    counts: Counter[str] = Counter()
    margin_pages: dict[str, set[int]] = defaultdict(set)
    for page_no, lines in page_lines.items():
        height = page_heights.get(page_no, 0.0)
        for line in lines:
            if len(line.text) >= HEADER_FOOTER_MAX_CHARS:
                continue
            # 表格单元格属于表格块，不属于正文。
            if blocked.get(page_no) and _line_inside_boxes(line.bbox, blocked[page_no]):
                continue
            key = counting_key(line.text)
            counts[key] += 1
            if height and _in_margin_band(line.bbox, height):
                margin_pages[key].add(page_no)

    threshold = total_pages * HEADER_FOOTER_PAGE_RATIO
    repeated = {key for key, count in counts.items() if count > threshold and count > 1}
    margin_repeated = {
        key for key, pages in margin_pages.items() if len(pages) >= HEADER_FOOTER_MARGIN_MIN_PAGES
    }

    cleaned: dict[int, list[Line]] = {}
    removed = 0
    removed_by_margin = 0
    for page_no, lines in page_lines.items():
        height = page_heights.get(page_no, 0.0)
        kept: list[Line] = []
        for line in lines:
            if len(line.text) < HEADER_FOOTER_MAX_CHARS:
                key = counting_key(line.text)
                if key in repeated or (
                    key in margin_repeated and height and _in_margin_band(line.bbox, height)
                ):
                    if key not in repeated:
                        removed_by_margin += 1
                    removed += 1
                    continue
            kept.append(line)
        cleaned[page_no] = kept

    if removed:
        logger.info(
            "%s: 清理页眉页脚 %d 行（频率规则阈值 >%.1f 次，边距规则 ≥%d 页：%d 行）",
            source,
            removed,
            threshold,
            HEADER_FOOTER_MARGIN_MIN_PAGES,
            removed_by_margin,
        )
    return cleaned, removed, repeated | margin_repeated


def _merge_broken_lines(lines: Sequence[Line]) -> list[str]:
    """拼接相邻行，直到句读符号出现才认为一段结束。"""
    paragraphs: list[str] = []
    buffer = ""
    for line in lines:
        text = normalize_inline(line.text)
        if not text:
            continue
        buffer = _join_lines(buffer, text) if buffer else text
        if _ends_with_terminator(text):
            paragraphs.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        paragraphs.append(buffer.strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def _join_lines(left: str, right: str) -> str:
    """拼接两行：英文单词之间补空格，中文直接相连。"""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return f"{left} {right}"
    return left + right


def _ends_with_terminator(text: str) -> bool:
    """判断行尾（忽略右引号、右括号等收尾符号）是否为句读符号。"""
    stripped = text.rstrip().rstrip(TRAILING_QUOTES)
    return bool(stripped) and stripped[-1] in SENTENCE_TERMINATORS


# --------------------------------------------------------------------------- #
# 文本分块
# --------------------------------------------------------------------------- #


def _split_sentences(text: str) -> list[str]:
    """在每个句读符号之后切分 ``text``（句读符号保留在句子里）。"""
    sentences: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in SENTENCE_TERMINATORS:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """切分单句过长的文本，片段之间保留 ``overlap`` 个字符的重叠。"""
    step = max(1, size - overlap)
    pieces = [text[start : start + size] for start in range(0, len(text), step)]
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(
    paragraphs: Iterable[str],
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把句子装进 ``size`` 字符的块，并用尾部句子实现 ``overlap`` 重叠。"""
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(_split_sentences(paragraph))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        body = "".join(current).strip()
        if body:
            chunks.append(body)
        # 把尾部句子保留下来，作为下一块的重叠部分。
        tail: list[str] = []
        tail_len = 0
        for sentence in reversed(current):
            if tail_len + len(sentence) > overlap:
                break
            tail.insert(0, sentence)
            tail_len += len(sentence)
        if tail_len >= len("".join(current)):
            tail, tail_len = [], 0
        current, current_len = tail, tail_len

    for sentence in sentences:
        if len(sentence) > size:
            flush()
            chunks.extend(_hard_split(sentence, size, overlap))
            current, current_len = [], 0
            continue
        if current and current_len + len(sentence) > size:
            flush()
        current.append(sentence)
        current_len += len(sentence)

    if current:
        body = "".join(current).strip()
        if body:
            chunks.append(body)

    return _dedupe_adjacent(chunks)


def _dedupe_adjacent(chunks: Sequence[str]) -> list[str]:
    """去掉空块以及与上一块完全相同的重复块。"""
    result: list[str] = []
    for chunk in chunks:
        if not chunk or (result and result[-1] == chunk):
            continue
        result.append(chunk)
    return result


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #


def _clean_cell(value: object) -> str:
    """归一化单个单元格：换行变成空格，``|`` 转义。"""
    if value is None:
        return ""
    return normalize_inline(str(value)).replace("|", r"\|")


def _normalise_rows(rows: Sequence[Sequence[object]]) -> list[list[str]]:
    """清洗单元格，丢弃空行与末尾全空列，并把各行补齐到相同列数。"""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_clean_cell(cell) for cell in row]
        if not any(cells):
            continue
        cleaned.append(cells)
    if not cleaned:
        return []

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    while width > 1 and all(not row[width - 1] for row in cleaned):
        for row in cleaned:
            row.pop()
        width -= 1
    return cleaned


def _is_usable_table(rows: Sequence[Sequence[str]]) -> bool:
    """剔除不是表格的检测结果（带边框的正文、稀疏网格）。"""
    if len(rows) < TABLE_MIN_ROWS:
        return False
    width = max(len(row) for row in rows)
    if width < TABLE_MIN_COLUMNS:
        return False
    total = sum(len(row) for row in rows)
    if total == 0:
        return False
    filled = sum(1 for row in rows for cell in row if cell)
    return filled / total >= TABLE_MIN_FILL_RATIO


def table_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """把行渲染成 Markdown 表格（第一行作为表头）。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    if width == 0:
        return ""
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def split_table_rows(
    rows: Sequence[Sequence[str]],
    *,
    max_rows: int = TABLE_SPLIT_ROWS,
    header_count: int = 1,
) -> list[tuple[int, int, list[list[str]]]]:
    """把表格正文行按每组至多 ``max_rows`` 行切分。

    ``header_count`` 是前导表头行数（两级表头时为 2）；这些行不进正文分组，
    但每个分组渲染时都会重复表头。

    返回 ``(row_start, row_end, rows)``，行号是正文行（不含表头）的 1 基闭区间。
    """
    if not rows:
        return []
    body = [list(row) for row in rows[max(1, header_count) :]]
    if not body:
        return [(1, 0, [])]
    if len(body) <= max_rows:
        return [(1, len(body), body)]
    groups: list[tuple[int, int, list[list[str]]]] = []
    for start in range(0, len(body), max_rows):
        part = body[start : start + max_rows]
        groups.append((start + 1, start + len(part), part))
    return groups


def _is_numeric_cell(value: str) -> bool:
    """单元格是否是「纯数值」（含百分号、括号负数、千分位）。

    「2024年12月31日」这种带中文的年份不算数值——它是表头里的年度列名。
    """
    cell = normalize_inline(value).replace(" ", "")
    if not cell:
        return False
    return bool(NUMERIC_CELL_RE.match(cell))


def header_row_count(rows: Sequence[Sequence[str]]) -> int:
    """返回前导表头行数（支持两级表头）。

    财报表格常见两级表头::

        | 项 目 | 2024年12月31日 |     |     | 2023年12月31日 |     |     | 不良率增减 |
        |      | 余额 | 占比 | 不良率 | 余额 | 占比 | 不良率 |          |

    两行都不含纯数值 → 都算表头；遇到第一行含数值的即为数据行。
    返回 0 表示这张表「一上来就是数据」，即没有表头。
    """
    count = 0
    for row in rows:
        if any(_is_numeric_cell(cell) for cell in row):
            break
        count += 1
    return count


def _row_width(rows: Sequence[Sequence[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def link_continued_tables(
    page_tables: dict[int, list[TableBlock]],
    page_heights: dict[int, float],
) -> int:
    """把跨页续表接到上一页的表头上，返回接上的表格数。

    背景：``pdfplumber`` 按页识别表格，一张跨页表格会被拆成两张，
    下一页那张的起始行直接是数据（如「贴现 | 185,729 | 5.5% | …」），
    导致块内没有任何年份标记，数值无法归年——这正是「1.39% 被当成 2023 年数据」
    的根源之一。

    只在两个条件同时成立时才接表头，避免把两张无关的表格拼在一起：
    列数一致，且（上一页表格贴着页底 或 本页表格贴着页顶）。
    """
    linked = 0
    for page_no in sorted(page_tables):
        current = page_tables[page_no]
        if not current:
            continue

        first = current[0]
        if header_row_count(first.rows) > 0:
            continue  # 自带表头，不是续表
        previous = page_tables.get(page_no - 1) or []
        if not previous:
            continue

        prev_last = previous[-1]
        header_count = header_row_count(prev_last.rows)
        if header_count == 0:
            continue  # 上一页最后一张表也没有表头，无从继承

        header = prev_last.rows[:header_count]
        if _row_width(header) != _row_width(first.rows):
            continue  # 列数不同，大概率是另一张表

        prev_height = page_heights.get(page_no - 1, 0.0)
        current_height = page_heights.get(page_no, 0.0)
        prev_bottom_ratio = float(prev_last.bbox[3]) / prev_height if prev_height else 0.0
        current_top_ratio = float(first.bbox[1]) / current_height if current_height else 1.0
        if not (
            prev_bottom_ratio >= TABLE_CONTINUE_BOTTOM_RATIO
            or current_top_ratio <= TABLE_CONTINUE_TOP_RATIO
        ):
            continue

        first.rows = [list(row) for row in header] + list(first.rows)
        first.inherited_from = page_no - 1
        base = prev_last.caption
        if not base or base == UNNAMED_TABLE:
            base = first.caption or UNNAMED_TABLE
        first.caption = f"{base}{CONTINUED_SUFFIX}"
        linked += 1
    return linked


def _needs_header_warning(rows: Sequence[Sequence[str]]) -> bool:
    """是否该提示「无表头」：含财税数值但整表没有任何年份标记。

    这类块里的数值无法归年，模型只能猜，必须让它在文本里看得见。
    只认「带百分号」或「带千分位」的数字，避免把股票代码、页码误判成数值。
    """
    joined = " ".join(cell for row in rows for cell in row)
    if not FINANCIAL_NUMBER_RE.search(joined):
        return False
    return not YEAR_IN_TEXT_RE.search(joined)


def _is_page_frame(bbox: Sequence[float], page_width: float, page_height: float) -> bool:
    """覆盖整页的 bbox 是页面边框，不是表格。"""
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return width >= 0.95 * float(page_width) and height >= 0.95 * float(page_height)


def _looks_like_caption(text: str) -> bool:
    """判断一行文本是否可能是表格标题。"""
    text = normalize_inline(text)
    if not text or len(text) > TABLE_CAPTION_MAX_CHARS:
        return False
    if _PAGE_NUMBER_RE.fullmatch(text):
        return False
    return not _ends_with_terminator(text)


def _find_table_caption(
    lines: Sequence[Line],
    bbox: Sequence[float],
    *,
    page_height: float = 0.0,
    excluded: Collection[str] = (),
) -> Line | None:
    """返回紧挨表格上方的、最接近的类标题行。

    已知的版面噪声（``excluded`` 中的键，或位于边距带内的任何行）绝不会被当作
    表格标题。
    """
    top = float(bbox[1])
    best: tuple[float, Line] | None = None
    for line in lines:
        if page_height and _in_margin_band(line.bbox, page_height):
            continue
        if excluded and counting_key(line.text) in excluded:
            continue
        gap = top - float(line.bbox[3])
        if gap < -1.0 or gap > TABLE_CAPTION_MAX_GAP:
            continue
        if not _looks_like_caption(line.text):
            continue
        if best is None or gap < best[0]:
            best = (gap, line)
    return best[1] if best else None


def _extract_page_tables(page, page_no: int, page_lines: Sequence[Line]) -> list[TableBlock]:
    """在单个 pdfplumber 页面上检测表格（先用廉价的开销闸门过滤）。"""
    if len(page.edges) < TABLE_EDGE_GATE:
        return []

    page_height = float(page.height)
    tables: list[TableBlock] = []
    for found in page.find_tables():
        bbox = tuple(float(value) for value in found.bbox)
        if _is_page_frame(bbox, page.width, page.height):
            continue
        rows = _normalise_rows(found.extract() or [])
        if not _is_usable_table(rows):
            continue
        caption_line = _find_table_caption(page_lines, bbox, page_height=page_height)
        caption = normalize_inline(caption_line.text) if caption_line else UNNAMED_TABLE
        caption_bbox = None
        if caption_line is not None and _EXPLICIT_CAPTION_RE.match(caption):
            # 显式标题属于表格块，不属于正文。
            caption_bbox = caption_line.bbox
        tables.append(
            TableBlock(
                page=page_no,
                index=len(tables) + 1,
                caption=caption or UNNAMED_TABLE,
                bbox=bbox,  # type: ignore[arg-type]
                rows=rows,
                caption_bbox=caption_bbox,
            )
        )
    return tables


def _release_page(page) -> None:
    """释放 pdfplumber 的单页缓存，避免解析 300 页的年报时内存失控。"""
    try:
        page.flush_cache()
    except Exception:  # pragma: no cover - 仅作兜底
        pass


def _line_inside_boxes(
    bbox: Sequence[float],
    boxes: Sequence[Sequence[float]],
    tolerance: float = 1.0,
) -> bool:
    """``bbox`` 的中心点落在任一 ``boxes`` 内时返回 True。"""
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    for box in boxes:
        if (
            float(box[0]) - tolerance <= cx <= float(box[2]) + tolerance
            and float(box[1]) - tolerance <= cy <= float(box[3]) + tolerance
        ):
            return True
    return False


# --------------------------------------------------------------------------- #
# 组装 chunk
# --------------------------------------------------------------------------- #


def _make_chunk(
    *,
    text: str,
    source: str,
    doc_id: str,
    page: int,
    block_type: str,
    block_index: int | None = None,
    table_index: int | None = None,
    table_caption: str | None = None,
    row_range: list[int] | None = None,
    split_table: bool = False,
) -> Chunk:
    """按约定规则生成 chunk_id 并构造经过校验的 :class:`Chunk`。"""
    if block_type == "table":
        if table_index is None:
            raise ValueError("表格块必须提供 table_index")
        chunk_id = f"{doc_id}_p{page:03d}_tbl{table_index:03d}"
        if split_table:
            if row_range is None:
                raise ValueError("切分的表格块必须提供 row_range")
            # 被切成多块的表格保留稳定的父级 id，再追加每块的行偏移，
            # 这样不用 UUID 也能保证 id 唯一。
            chunk_id += f"_r{row_range[0]:03d}"
    else:
        if block_index is None:
            raise ValueError("文本块必须提供 block_index")
        chunk_id = f"{doc_id}_p{page:03d}_b{block_index:03d}"

    metadata = ChunkMetadata(
        source=source,
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        block_type=block_type,  # type: ignore[arg-type]
        char_count=len(text),
        table_caption=table_caption,
        table_index=table_index,
        row_range=row_range,
    )
    return Chunk(text=text, metadata=metadata)


def _text_chunks_for_page(
    page_no: int,
    lines: Sequence[Line],
    *,
    source: str,
    doc_id: str,
) -> list[Chunk]:
    """把一页的文本行合并成段落、分块，并组装成文本块列表。"""
    paragraphs = _merge_broken_lines(lines)
    chunks: list[Chunk] = []
    for index, text in enumerate(chunk_text(paragraphs), start=1):
        chunks.append(
            _make_chunk(
                text=text,
                source=source,
                doc_id=doc_id,
                page=page_no,
                block_type="text",
                block_index=index,
            )
        )
    return chunks


def _table_chunks_for_page(
    page_no: int,
    tables: Sequence[TableBlock],
    *,
    source: str,
    doc_id: str,
) -> list[Chunk]:
    """把一页的表格逐个转成 Markdown 块，超长表格分组并重复表头。

    三处细节直接关系到「数值能不能归年」：

    1. 表头行数不止一行（年度行 + 指标行），全部保留重复；
    2. 跨页续表已在 :func:`link_continued_tables` 里接上上一页表头，
       这里在标题后标出「表头沿用第 N 页」；
    3. 整表既没年份又含数字时，标出「无表头·数值不可归年」，
       避免模型把没有年份的数值硬安到问题所问的年份上。
    """
    chunks: list[Chunk] = []
    for table in tables:
        header_count = header_row_count(table.rows)
        if header_count > 0:
            header = [list(row) for row in table.rows[:header_count]]
        else:
            # 没有识别到表头：沿用旧行为把首行当表头，同时打上警示标记
            header = [list(table.rows[0])] if table.rows else []
        groups = split_table_rows(table.rows, header_count=max(1, header_count))
        split = len(groups) > 1
        warn_headerless = header_count == 0 and _needs_header_warning(table.rows)

        for row_start, row_end, body_rows in groups:
            caption = table.caption
            if split and row_start > 1:
                caption = f"{caption}{CONTINUED_SUFFIX}"
            notes: list[str] = []
            if table.inherited_from:
                notes.append(f"表头沿用第 {table.inherited_from} 页")
            if warn_headerless:
                notes.append("无表头·数值不可归年")
            note = f"（{'；'.join(notes)}）" if notes else ""
            markdown = table_to_markdown(header + body_rows)
            text = f"{caption}{note}\n{markdown}".strip()
            if not text:
                continue
            chunks.append(
                _make_chunk(
                    text=text,
                    source=source,
                    doc_id=doc_id,
                    page=page_no,
                    block_type="table",
                    table_index=table.index,
                    table_caption=table.caption,
                    row_range=[row_start, row_end],
                    split_table=split,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# 单文件解析
# --------------------------------------------------------------------------- #


def _read_pages(
    doc: "pymupdf.Document",
    path: Path,
    total_pages: int,
    logger: logging.Logger,
    source: str,
) -> tuple[dict[int, list[Line]], dict[int, list[TableBlock]], dict[int, float]]:
    """逐页收集文本行、表格以及页面高度。"""
    page_lines: dict[int, list[Line]] = {}
    page_tables: dict[int, list[TableBlock]] = {}
    page_heights: dict[int, float] = {}

    plumber_doc = None
    try:
        plumber_doc = pdfplumber.open(path)
    except Exception as exc:  # noqa: BLE001 - 退化为纯文本解析，但不静默放过
        logger.warning(
            "%s: pdfplumber 无法打开（%s: %s），本次仅输出文本块",
            source,
            type(exc).__name__,
            exc,
        )

    try:
        for page_no in range(1, total_pages + 1):
            fitz_page = doc.load_page(page_no - 1)
            page_heights[page_no] = float(fitz_page.rect.height)
            lines = _extract_page_lines(fitz_page)
            page_lines[page_no] = lines
            page_tables[page_no] = []
            if plumber_doc is None:
                continue
            try:
                plumber_page = plumber_doc.pages[page_no - 1]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s: 第 %d 页表格页打开失败（%s: %s）", source, page_no, type(exc).__name__, exc
                )
                continue
            try:
                page_tables[page_no] = _extract_page_tables(plumber_page, page_no, lines)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s: 第 %d 页表格提取失败（%s: %s）", source, page_no, type(exc).__name__, exc
                )
            finally:
                _release_page(plumber_page)
    finally:
        if plumber_doc is not None:
            plumber_doc.close()

    return page_lines, page_tables, page_heights


def _refine_table_captions(
    page_tables: dict[int, list[TableBlock]],
    raw_page_lines: dict[int, list[Line]],
    page_heights: dict[int, float],
    furniture_keys: Collection[str],
) -> None:
    """首个候选标题是页眉行时，重新挑选表格标题。"""
    for page_no, tables in page_tables.items():
        lines = raw_page_lines.get(page_no, [])
        height = page_heights.get(page_no, 0.0)
        for table in tables:
            if counting_key(table.caption) not in furniture_keys:
                continue
            replacement = _find_table_caption(
                lines,
                table.bbox,
                page_height=height,
                excluded=furniture_keys,
            )
            caption = normalize_inline(replacement.text) if replacement else ""
            table.caption = caption or UNNAMED_TABLE
            table.caption_bbox = (
                replacement.bbox
                if replacement is not None and _EXPLICIT_CAPTION_RE.match(table.caption)
                else None
            )


def parse_pdf(
    path: str | Path,
    *,
    source: str | None = None,
    logger: logging.Logger | None = None,
) -> list[Chunk]:
    """解析单个 PDF 并返回通过校验的 chunk 列表（文件不可读或为空时抛异常）。"""
    logger = logger or get_logger()
    path = Path(path)
    if not path.is_file():
        raise PdfParseError(f"文件不存在: {path.name}")

    doc_id = doc_id_for(path)
    source = source or relative_source(path)

    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise PdfParseError("PDF 已加密，需要密码才能读取")
        total_pages = doc.page_count
        if total_pages <= 0:
            raise PdfParseError("PDF 不包含任何页面")
        page_lines, page_tables, page_heights = _read_pages(doc, path, total_pages, logger, source)

    raw_page_lines = page_lines
    page_lines, _removed, furniture_keys = _drop_headers_and_footers(
        page_lines, page_tables, page_heights, total_pages, logger, source
    )
    _refine_table_captions(page_tables, raw_page_lines, page_heights, furniture_keys)

    # 跨页续表：把上一页的表头接到本页没有表头的表格上
    linked = link_continued_tables(page_tables, page_heights)
    if linked:
        logger.info("%s: 跨页续表接回表头 %d 张", source, linked)

    chunks: list[Chunk] = []
    for page_no in range(1, total_pages + 1):
        tables = page_tables.get(page_no, [])
        # 表格单元格和显式标题已由表格块承载，不能再混进正文块。
        blocked: list[Sequence[float]] = [table.bbox for table in tables]
        blocked.extend(table.caption_bbox for table in tables if table.caption_bbox)
        lines = [
            line
            for line in page_lines.get(page_no, [])
            if not _line_inside_boxes(line.bbox, blocked)
        ]
        chunks.extend(_text_chunks_for_page(page_no, lines, source=source, doc_id=doc_id))
        chunks.extend(_table_chunks_for_page(page_no, tables, source=source, doc_id=doc_id))

    return chunks


# --------------------------------------------------------------------------- #
# 批量解析与 CLI
# --------------------------------------------------------------------------- #


def iter_pdf_files(input_path: str | Path) -> list[Path]:
    """返回 ``input_path`` 下的所有 ``.pdf``（递归，顺序确定）。"""
    root = Path(input_path)
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(files, key=lambda p: p.as_posix())


def parse_corpus(
    input_path: str | Path,
    output_path: str | Path,
    errors_path: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> ParseStats:
    """把 ``input_path`` 下的每个 PDF 解析并写入 ``output_path``。

    解析失败的文件会追加到 ``errors_path``，批次继续处理下一个文件。
    """
    logger = logger or get_logger()
    files = iter_pdf_files(input_path)
    stats = ParseStats(files=len(files))

    output_path = Path(output_path)
    errors_path = Path(errors_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始解析：%d 个 PDF（输入 %s）", len(files), Path(input_path).as_posix())

    with output_path.open("w", encoding="utf-8", newline="\n") as output_handle, errors_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as error_handle:
        for position, path in enumerate(files, start=1):
            source = relative_source(path)
            logger.info("[%d/%d] 解析 %s", position, len(files), source)
            try:
                chunks = parse_pdf(path, source=source, logger=logger)
            except Exception as exc:  # noqa: BLE001 - 单个坏文件不能中断整个批次
                stats.failed += 1
                stats.failures.append(source)
                logger.error("%s: 解析失败（%s: %s）", source, type(exc).__name__, exc)
                record = {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                error_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            if not chunks:
                logger.warning("%s: 未产出任何 chunk（可能为扫描件或空文档）", source)

            for chunk in chunks:
                output_handle.write(chunk.to_json_line() + "\n")

            stats.succeeded += 1
            stats.chunks += len(chunks)
            for chunk in chunks:
                if chunk.metadata.block_type == "table":
                    stats.table_chunks += 1
                else:
                    stats.text_chunks += 1
            logger.info("[%d/%d] 完成 %s -> %d chunks", position, len(files), source, len(chunks))

    logger.info(
        "解析结束：成功 %d / 失败 %d / chunks %d（文本 %d，表格 %d）",
        stats.succeeded,
        stats.failed,
        stats.chunks,
        stats.text_chunks,
        stats.table_chunks,
    )
    if stats.failed:
        logger.error("失败文件 %d 个，详见 %s", stats.failed, errors_path.as_posix())
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.pdf_parser",
        description="把 PDF 解析为 chunks.jsonl（PyMuPDF 文本 + pdfplumber 表格）。",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="PDF 目录（递归），默认 data/corpus")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="chunks.jsonl 输出路径")
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS), help="errors.jsonl 输出路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = build_arg_parser().parse_args(argv)
    configure_logging(logging.INFO)
    logger = get_logger()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("输入路径不存在：%s", input_path.as_posix())
        return 2

    stats = parse_corpus(input_path, args.output, args.errors, logger=logger)
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
