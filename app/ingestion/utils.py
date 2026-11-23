"""PDF 解析流水线共用的小工具。

这里只依赖标准库：文件哈希、日志配置、文本归一化与相对路径渲染。
解析逻辑在 :mod:`app.ingestion.pdf_parser`，数据结构在
:mod:`app.ingestion.schema`。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

LOGGER_NAME = "app.ingestion.pdf_parser"

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"

_HASH_CHUNK_SIZE = 1 << 20
_WHITESPACE_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """返回流水线日志器（默认共用一个名称）。"""
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """配置标准 logging，默认把 INFO 级进度输出到 stderr。"""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
    )


def file_sha256(path: str | Path) -> str:
    """返回 ``path`` 的完整 ``sha256`` 十六进制摘要。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(path: str | Path) -> str:
    """返回 ``path`` 对应的 16 位文档 id（``sha256`` 前 16 位）。"""
    return file_sha256(path)[:16]


def normalize_inline(text: str | None) -> str:
    """把任意空白（含换行）折叠成一个空格并去掉首尾空白。"""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def counting_key(text: str) -> str:
    """用于识别重复页眉页脚行的归一化键。

    数字串统一替换成 ``0``，使 ``第 3 页`` 与 ``第 4 页`` 落到同一个键上；
    空白也做折叠，避免排版留白影响比对。
    """
    return _DIGITS_RE.sub("0", normalize_inline(text)).lower()


def relative_source(path: str | Path, root: str | Path | None = None) -> str:
    """把 ``path`` 渲染成相对 ``root``（默认当前工作目录）的 POSIX 路径。

    绝对路径不得出现在 ``metadata.source`` 或日志中，因此即使目标在
    ``root`` 之外，也返回 ``../`` 形式的相对路径。
    """
    target = Path(path).expanduser().resolve()
    base = Path(root or Path.cwd()).expanduser().resolve()
    try:
        relative = target.relative_to(base)
    except ValueError:
        relative = Path(os.path.relpath(target, base))
    return relative.as_posix()
