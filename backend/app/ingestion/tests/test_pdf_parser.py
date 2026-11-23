""":mod:`app.ingestion.pdf_parser` 的测试。

fixture PDF 在测试时用 PyMuPDF 现场生成，仓库里不放二进制文件。它复现了解析器
必须处理的三种情况：

* 每页都有页眉页脚（页码逐页不同）；
* PDF 提取会把正文断在句子中间；
* 两张用真实表格线画出来的表格（4 行，以及 12 行以便覆盖 ``> 10 行`` 的切分分支）。

运行方式::

    python -m pytest app/ingestion/tests -q --basetemp .pytest-tmp

（只有在该解释器无法在 ``%TEMP%`` 下建目录的机器上，``--basetemp`` 才是必需的。）
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pymupdf
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # 允许直接调用 pytest
    sys.path.insert(0, str(REPO_ROOT))

from app.ingestion.pdf_parser import main as parser_main  # noqa: E402

HEADER = "某某股份有限公司 2023 年年度报告"
FOOTER = "第 {page} 页"

PAGE_ONE_LINES = [
    "本公司董事会及全体董事保证本报告内容不存在任何虚假记载、误导性陈述或者重大遗漏，",
    "并对报告内容的真实性、准确性和完整性承担个别及连带责任。",
    "报告期内，公司实现营业收入 1,234,567 万元，同比增长 12.34%，",
    "实现归属于上市公司股东的净利润 234,567 万元。",
]

LONG_PARAGRAPH = (
    "报告期内，公司持续加大研发投入，围绕主营业务开展技术攻关，全年研发费用同比增长百分之十八，"
    "新增专利授权一百二十项。公司坚持以客户为中心，持续优化产品结构，提升供应链协同效率，"
    "在行业景气度波动的背景下保持了经营业绩的稳健增长。"
)

SMALL_TABLE = [
    ["城市级别", "住宿费上限", "适用人员"],
    ["一线城市", "600", "全员"],
    ["二线城市", "400", "全员"],
    ["其他城市", "300", "普通员工"],
]

BIG_TABLE = [["项目", "本期金额", "上期金额"]] + [
    [f"会计科目{i}", f"{1000 + i}", f"{900 + i}"] for i in range(1, 13)
]

TABLE_CAPTION = "表3-2 差旅住宿费标准"
BIG_TABLE_CAPTION = "表4-1 主要会计数据"
CONTINUED_SUFFIX = "（续）"


def _new_page(doc: pymupdf.Document, page_no: int) -> "pymupdf.Page":
    """新建一页，并写上该文档统一的页眉页脚。"""
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 45), HEADER, fontname="china-s", fontsize=10)
    page.insert_text((72, 805), FOOTER.format(page=page_no), fontname="china-s", fontsize=9)
    return page


def _insert_lines(page: "pymupdf.Page", lines: list[str], *, y: float = 110.0, leading: float = 18.0) -> None:
    """从 ``y`` 开始逐行写入文本。"""
    for line in lines:
        page.insert_text((72, y), line, fontname="china-s", fontsize=11)
        y += leading


def _wrap(text: str, width: int = 38) -> list[str]:
    """把一段文字硬折成多行，行尾不添加任何句读符号。"""
    return [text[start : start + width] for start in range(0, len(text), width)]


def _draw_table(
    page: "pymupdf.Page",
    rows: list[list[str]],
    *,
    top: float,
    caption: str,
    left: float = 72.0,
    col_width: float = 150.0,
    row_height: float = 24.0,
) -> None:
    """在页面上画出带表格线的表格，并在表格上方写标题。"""
    page.insert_text((left, top - 8), caption, fontname="china-s", fontsize=10)

    n_rows = len(rows)
    n_cols = len(rows[0])
    bottom = top + n_rows * row_height
    right = left + n_cols * col_width
    for row in range(n_rows + 1):
        y = top + row * row_height
        page.draw_line((left, y), (right, y), color=(0, 0, 0), width=0.8)
    for col in range(n_cols + 1):
        x = left + col * col_width
        page.draw_line((x, top), (x, bottom), color=(0, 0, 0), width=0.8)

    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if not value:
                continue
            page.insert_text(
                (left + col_index * col_width + 4, top + row_index * row_height + row_height - 7),
                str(value),
                fontname="china-s",
                fontsize=9,
            )


def build_fixture_pdf(path: Path) -> Path:
    """生成所有测试共用的 4 页 fixture PDF。"""
    doc = pymupdf.open()
    try:
        page = _new_page(doc, 1)
        _insert_lines(page, PAGE_ONE_LINES, y=110)

        page = _new_page(doc, 2)
        _draw_table(page, SMALL_TABLE, top=140, caption=TABLE_CAPTION)

        page = _new_page(doc, 3)
        _insert_lines(page, _wrap(LONG_PARAGRAPH * 3), y=110)

        page = _new_page(doc, 4)
        _draw_table(page, BIG_TABLE, top=120, caption=BIG_TABLE_CAPTION)

        doc.save(str(path))
    finally:
        doc.close()
    return path


@pytest.fixture(scope="module")
def fixture_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("pdf_fixture")
    return build_fixture_pdf(directory / "sample_annual_report.pdf")


def _run_parser(fixture_pdf: Path, workspace: Path) -> list[dict]:
    """在一个只放 fixture 副本的语料目录上跑一遍 CLI。"""
    corpus = workspace / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    target = corpus / fixture_pdf.name
    shutil.copy2(fixture_pdf, target)

    output = workspace / "chunks.jsonl"
    errors = workspace / "errors.jsonl"
    exit_code = parser_main(
        ["--input", str(corpus), "--output", str(output), "--errors", str(errors)]
    )
    assert exit_code == 0, f"解析器退出码为 {exit_code}"
    assert errors.read_text(encoding="utf-8") == ""

    records = []
    for line in output.read_text(encoding="utf-8").splitlines():
        assert line.strip(), "chunks.jsonl 不允许出现空行"
        records.append(json.loads(line))
    return records


def test_chunks_jsonl_is_valid_and_header_footer_removed(fixture_pdf: Path, tmp_path: Path) -> None:
    """每行可 json.loads、text 非空、必填 metadata 齐全、页眉页脚已清理。"""
    records = _run_parser(fixture_pdf, tmp_path / "run")

    assert records, "解析器没有产出任何 chunk"
    for record in records:
        assert set(record) == {"text", "metadata"}
        text = record["text"]
        assert isinstance(text, str) and text.strip()

        metadata = record["metadata"]
        for key in ("source", "chunk_id", "doc_id", "page", "block_type", "char_count"):
            assert key in metadata, f"缺少 metadata 字段 {key}"
        assert metadata["block_type"] in {"text", "table"}
        assert metadata["char_count"] == len(text)
        assert isinstance(metadata["page"], int) and metadata["page"] >= 1
        assert not Path(metadata["source"]).is_absolute()
        assert metadata["chunk_id"].startswith(f"{metadata['doc_id']}_p{metadata['page']:03d}_")

        # 页眉页脚绝不能残留到 chunk 里。
        assert HEADER not in text
        assert "某股份有限公司 2023" not in text
        for page_no in range(1, 5):
            assert FOOTER.format(page=page_no) not in text

    # 第 1 页被硬折行的句子必须重新拼成一段。
    joined = "\n".join(record["text"] for record in records)
    assert "误导性陈述或者重大遗漏，并对报告内容的真实性" in joined


def test_chunk_ids_are_stable_and_unique(fixture_pdf: Path, tmp_path: Path) -> None:
    """同一 PDF 解析两次，chunk_id 序列完全一致，且没有重复 ID。"""
    first = [record["metadata"]["chunk_id"] for record in _run_parser(fixture_pdf, tmp_path / "a")]
    second = [record["metadata"]["chunk_id"] for record in _run_parser(fixture_pdf, tmp_path / "b")]

    assert first, "没有产出任何 chunk_id"
    assert first == second
    assert len(set(first)) == len(first)


def test_table_chunks_are_markdown(fixture_pdf: Path, tmp_path: Path) -> None:
    """表格块单独成块、text 为 Markdown（含 |）、不含 HTML <table>。"""
    records = _run_parser(fixture_pdf, tmp_path / "run")

    table_chunks = [r for r in records if r["metadata"]["block_type"] == "table"]
    assert table_chunks, "没有产出表格块"

    for record in table_chunks:
        text = record["text"]
        lines = text.splitlines()
        assert "|" in text
        assert "<table" not in text.lower() and "</table" not in text.lower()
        assert "<tr" not in text.lower()
        # 第一行是标题，其后是 Markdown 表格（表头 + 分隔行）
        caption = record["metadata"]["table_caption"]
        assert caption
        assert lines[0] in {caption, caption + CONTINUED_SUFFIX}, lines[0]
        assert lines[1].startswith("|") and lines[2].startswith("|---")

    # 表格单元格不得混入正文块。
    prose = "\n".join(r["text"] for r in records if r["metadata"]["block_type"] == "text")
    assert "住宿费上限" not in prose
    assert "适用人员" not in prose

    # 12 行的表格按 10 行一组切分，每组都重复表头。
    continued = [r for r in table_chunks if r["text"].startswith(BIG_TABLE_CAPTION + "（续）")]
    assert len(continued) == 1
    for record in table_chunks:
        if record["text"].startswith(BIG_TABLE_CAPTION):
            assert "| 项目 | 本期金额 | 上期金额 |" in record["text"]
    assert [r["metadata"]["row_range"] for r in table_chunks if r["text"].startswith(BIG_TABLE_CAPTION)] == [
        [1, 10],
        [11, 12],
    ]
