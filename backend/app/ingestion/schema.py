"""写入 ``data/parsed/chunks.jsonl`` 的记录所用的 Pydantic 模型。

输出文件的一行就是一个 :class:`Chunk`::

    {"text": "...", "metadata": {...}}

模型负责守住数据契约（必填字段、``chunk_id`` 格式、``char_count == len(text)``），
让不合规的块在写盘前就报错，而不是悄悄进入向量库 / BM25 索引。
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PARSER_VERSION = "pdf_parser_v1"

BlockType = Literal["text", "table"]

# doc_id 是文件 sha256 的前 16 位十六进制字符
DOC_ID_PATTERN = r"^[0-9a-f]{16}$"
# {doc_id}_p{page:03d}_b{block_index:03d}
# {doc_id}_p{page:03d}_tbl{table_index:03d}[_r{row_start:03d}]
CHUNK_ID_PATTERN = r"^[0-9a-f]{16}_p\d{3,}_(?:b\d{3,}|tbl\d{3,}(?:_r\d{3,})?)$"


class ChunkMetadata(BaseModel):
    """每个 chunk 携带的 metadata。

    文档规定的 6 个字段为必填；其余为可选字段，用于表格块溯源。
    """

    model_config = ConfigDict(extra="ignore")

    source: str = Field(min_length=1, description="相对路径，例如 data/corpus/xxx.pdf")
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN, description="确定性生成的块 ID")
    doc_id: str = Field(pattern=DOC_ID_PATTERN, description="文件 sha256 前 16 位")
    page: int = Field(ge=1, description="页码，从 1 开始")
    block_type: BlockType
    char_count: int = Field(ge=0, description="len(text)")

    table_caption: str | None = Field(default=None, description="表格标题，文本块为 None")
    table_index: int | None = Field(default=None, ge=1, description="页内表格序号")
    row_range: list[int] | None = Field(default=None, description="表格子块覆盖的行区间（1 基）")
    parser_version: str = PARSER_VERSION

    @model_validator(mode="after")
    def _validate_chunk_id(self) -> "ChunkMetadata":
        prefix = f"{self.doc_id}_p{self.page:03d}_"
        if not self.chunk_id.startswith(prefix):
            raise ValueError(
                f"chunk_id {self.chunk_id!r} 与预期前缀 {prefix!r} 不匹配"
            )
        remainder = self.chunk_id[len(prefix):]
        is_table = remainder.startswith("tbl")
        if is_table != (self.block_type == "table"):
            raise ValueError(
                f"chunk_id {self.chunk_id!r} 与 block_type={self.block_type!r} 不一致"
            )
        if not is_table and (self.table_caption or self.table_index or self.row_range):
            raise ValueError(f"文本块 {self.chunk_id!r} 不应携带表格字段")
        if is_table and self.table_index is None:
            raise ValueError(f"表格块 {self.chunk_id!r} 缺少 table_index")
        if self.row_range is not None:
            if len(self.row_range) != 2:
                raise ValueError("row_range 必须是 [start, end]")
            start, end = self.row_range
            if start < 1 or end < start - 1:
                raise ValueError(f"row_range 非法：{self.row_range!r}")
        return self


class Chunk(BaseModel):
    """一条输出记录：清洗后的正文加对应的 metadata。"""

    text: str
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Chunk":
        if not self.text.strip():
            raise ValueError(f"chunk {self.metadata.chunk_id!r} 的 text 为空白")
        if self.metadata.char_count != len(self.text):
            raise ValueError(
                f"chunk {self.metadata.chunk_id!r} 的 char_count="
                f"{self.metadata.char_count} != len(text)={len(self.text)}"
            )
        return self

    def to_json_line(self) -> str:
        """序列化为一行 UTF-8 JSONL（中文原样输出，不做转义）。"""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)
