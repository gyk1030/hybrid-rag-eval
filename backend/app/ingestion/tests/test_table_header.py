"""表格表头处理的单元测试：跨页续表、两级表头、无表头警示。

背景：平安银行 2024 年报第 37 页的表格是第 36 页那张表的跨页续篇，
pdfplumber 把它单独识别成一张新表，起始行直接是数据（「贴现 | 185,729 | …」），
于是 1.39% 与 1.37% 在块内无法归年，模型把 2024 年的数字当成了 2023 年的答案。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.pdf_parser import (  # noqa: E402
    TableBlock,
    _needs_header_warning,
    _table_chunks_for_page,
    header_row_count,
    link_continued_tables,
    split_table_rows,
)

# 两级表头：年度行 + 指标行
HEADER = [
    ["项 目", "2024年12月31日", "", "", "2023年12月31日", "", "", "不良率增减"],
    ["", "余额", "占比", "不良率", "余额", "占比", "不良率", ""],
]
DATA = [
    ["贴现", "185,729", "5.5%", "-", "214,799", "6.3%", "-", "-"],
    ["个人贷款（注）", "1,767,168", "52.4%", "1.39%", "1,977,719", "58.0%", "1.37%", "+0.02个百分点"],
]


def make_table(page: int, rows, bbox, caption="未命名表格", index=1) -> TableBlock:
    return TableBlock(page=page, index=index, caption=caption, bbox=bbox, rows=rows)


# --------------------------------------------------------------------------- #
# 表头识别
# --------------------------------------------------------------------------- #
def test_header_row_count_two_level():
    assert header_row_count(HEADER + DATA) == 2


def test_header_row_count_single_level():
    rows = [["项 目", "2023年", "2022年"], ["营业收入", "1,234", "1,111"]]
    assert header_row_count(rows) == 1


def test_header_row_count_returns_zero_for_data_first():
    """跨页续表的特征：一上来就是数据行。"""
    assert header_row_count(DATA) == 0


def test_header_row_count_treats_year_cell_as_header():
    """「2024年12月31日」是列名不是数值，不能被当成数据。"""
    rows = [["", "2024年12月31日", "2023年12月31日"]]
    assert header_row_count(rows) == 1


# --------------------------------------------------------------------------- #
# 无表头警示
# --------------------------------------------------------------------------- #
def test_needs_header_warning_when_numbers_without_year():
    assert _needs_header_warning([["贴现", "185,729", "5.5%", "1.39%"]])


def test_no_warning_when_year_present():
    assert not _needs_header_warning(HEADER + DATA)


def test_no_warning_when_no_numbers():
    """纯文字表格（如公司基本信息）不需要年份，不该打警示。"""
    assert not _needs_header_warning([["股票简称", "平安银行"], ["股票代码", "000001"]])


# --------------------------------------------------------------------------- #
# 跨页续表接表头
# --------------------------------------------------------------------------- #
PAGE_HEIGHT = 842.0


def test_link_continued_table_inherits_header():
    prev = make_table(36, HEADER + DATA[:1], bbox=(58.0, 690.0, 545.0, 760.0), caption="（货币单位：人民币百万元）")
    current = make_table(37, DATA, bbox=(58.0, 74.0, 545.0, 196.0))
    page_tables = {36: [prev], 37: [current]}

    linked = link_continued_tables(page_tables, {36: PAGE_HEIGHT, 37: PAGE_HEIGHT})

    assert linked == 1
    assert current.inherited_from == 36
    assert header_row_count(current.rows) == 2
    # 表头被真正拼到数据前面
    assert current.rows[0][1] == "2024年12月31日"
    assert current.rows[-1][0] == "个人贷款（注）"
    assert current.caption.endswith("（续）")


def test_link_skipped_when_column_count_differs():
    prev = make_table(36, HEADER + DATA[:1], bbox=(0, 690, 545, 760))
    current = make_table(37, [["贴现", "185,729", "5.5%"]], bbox=(0, 74, 545, 196))
    page_tables = {36: [prev], 37: [current]}

    assert link_continued_tables(page_tables, {36: PAGE_HEIGHT, 37: PAGE_HEIGHT}) == 0
    assert current.inherited_from is None


def test_link_skipped_when_table_has_header():
    prev = make_table(36, HEADER + DATA, bbox=(0, 690, 545, 760))
    current = make_table(37, HEADER + DATA, bbox=(0, 74, 545, 196))
    page_tables = {36: [prev], 37: [current]}

    assert link_continued_tables(page_tables, {36: PAGE_HEIGHT, 37: PAGE_HEIGHT}) == 0


def test_link_skipped_when_not_at_page_edges():
    """两张表都在页面中部，不是跨页关系，不能乱接。"""
    prev = make_table(36, HEADER + DATA[:1], bbox=(0, 200.0, 545, 300.0))
    current = make_table(37, DATA, bbox=(0, 400.0, 545, 500.0))
    page_tables = {36: [prev], 37: [current]}

    assert link_continued_tables(page_tables, {36: PAGE_HEIGHT, 37: PAGE_HEIGHT}) == 0


# --------------------------------------------------------------------------- #
# 分组与渲染
# --------------------------------------------------------------------------- #
def test_split_uses_header_count():
    """两级表头的第二行不应被当成正文数据，且每组都重复完整表头。"""
    rows = HEADER + [[f"行{i}", str(i), "1%", "-", str(i), "1%", "-", "-"] for i in range(1, 13)]
    groups = split_table_rows(rows, max_rows=10, header_count=2)
    assert len(groups) == 2
    assert groups[0][0] == 1 and groups[0][1] == 10
    assert groups[1][0] == 11 and groups[1][1] == 12
    # 正文首行是「行1」，不是第二行表头
    assert groups[0][2][0][0] == "行1"


def test_chunk_text_contains_inherited_note_and_header():
    table = make_table(37, DATA, bbox=(0, 74, 545, 196))
    table.rows = [list(row) for row in HEADER] + list(DATA)
    table.inherited_from = 36

    chunks = _table_chunks_for_page(37, [table], source="data/corpus/x.pdf", doc_id="a" * 16)

    assert len(chunks) == 1
    text = chunks[0].text
    assert "表头沿用第 36 页" in text
    assert "2024年12月31日" in text and "2023年12月31日" in text
    assert "1.39%" in text


def test_chunk_text_marks_headerless_table():
    table = make_table(37, [["贴现", "185,729", "5.5%", "1.39%"]], bbox=(0, 74, 545, 196))
    chunks = _table_chunks_for_page(37, [table], source="data/corpus/x.pdf", doc_id="b" * 16)

    assert len(chunks) == 1
    assert "无表头" in chunks[0].text
    assert "数值不可归年" in chunks[0].text
