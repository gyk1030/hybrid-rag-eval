"""BM25 索引：从 PostgreSQL 读入已入库的 chunks，jieba 分词后用 BM25 建索引并支持查询。

与 :mod:`app.retrieval.vector_store` 共用同一个 PostgreSQL 库（连接串见该模块），
但这里只依赖纯 Python 的 ``jieba`` / ``rank_bm25``，**不会**加载 torch 与
FlagEmbedding——所以建索引只要几秒、几十 MB 内存，不需要显卡。

用法::

    python -m app.retrieval.bm25_index                  # 从数据库构建索引并写缓存
    python -m app.retrieval.bm25_index --rebuild        # 忽略缓存，强制重建
    python -m app.retrieval.bm25_index -q "研发投入" -k 5
    python -m app.retrieval.bm25_index -q "应收账款 坏账准备" -k 10

作为库使用::

    from app.retrieval.bm25_index import BM25Index

    index = BM25Index().build()          # 或 BM25Index().load_cache()
    for hit in index.search("研发投入", top_k=5):
        print(hit.score, hit.chunk_id, hit.text[:60])
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jieba
import numpy as np
import psycopg2
from rank_bm25 import BM25Okapi

# 允许用脚本方式直接运行（python app/retrieval/bm25_index.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里，
# `from app.config import ...` 会报 ModuleNotFoundError。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    BM25_INDEX_PATH,
    DSN,
    SEARCH_TOP_K,
)

LOGGER_NAME = "app.retrieval.bm25_index"

DEFAULT_CACHE = Path(BM25_INDEX_PATH)

# 缓存结构版本号：改分词方式或索引字段时递增，避免读到旧缓存
CACHE_VERSION = 1

# 纯标点/符号的 token（jieba 会把 "|"、"——"、"……" 也切出来），建索引前丢掉
_PUNCT_RE = re.compile(r"^[\W_]+$")

logger = logging.getLogger(LOGGER_NAME)

# jieba 默认会往 stderr 打 "Loading model cost ..." 之类的 DEBUG 日志，压到 WARNING
jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    """用 jieba 分词，并做归一化：去空白、转小写、丢弃纯标点。

    BM25 本身会用 IDF 压制高频词，所以这里不额外做停用词过滤。
    """
    tokens: list[str] = []
    for word in jieba.lcut(text):
        word = word.strip().lower()
        if not word or _PUNCT_RE.match(word):
            continue
        tokens.append(word)
    return tokens


@dataclass(slots=True)
class Bm25Hit:
    """一条 BM25 检索结果。"""

    chunk_id: str
    score: float
    text: str
    source: str
    page: int
    block_type: str

    def snippet(self, length: int = 80) -> str:
        """返回用于展示的短摘要。"""
        flat = " ".join(self.text.split())
        return flat[:length] + ("…" if len(flat) > length else "")


class BM25Index:
    """chunks 表上的 BM25 倒排索引。

    索引内容与顺序完全由 ``SELECT ... ORDER BY chunk_id`` 决定，因此同一份数据
    重复构建得到相同结果（便于和向量检索做 RRF 融合、复现评估结果）。
    """

    def __init__(self, dsn: str | None = None, cache_path: str | Path | None = None) -> None:
        self.dsn = dsn or DSN
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE

        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []
        self.tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def fetch_rows(self) -> list[tuple]:
        """从数据库读取参与索引的字段。"""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, text, source, page, block_type "
                    "FROM chunks ORDER BY chunk_id"
                )
                return cur.fetchall()
        finally:
            conn.close()

    def build(self) -> "BM25Index":
        """从数据库重建索引（分词 + 计算 IDF）。"""
        rows = self.fetch_rows()
        if not rows:
            raise RuntimeError("chunks 表为空，请先运行 vector_store.py 把 chunk 入库")

        self.chunk_ids = [row[0] for row in rows]
        self.texts = [row[1] for row in rows]
        self.metas = [{"source": row[2], "page": row[3], "block_type": row[4]} for row in rows]
        self.tokenized = [tokenize(text) for text in self.texts]
        self._bm25 = BM25Okapi(self.tokenized)

        total_tokens = sum(len(tokens) for tokens in self.tokenized)
        logger.info(
            "BM25 索引构建完成：%d 个 chunk，%d 个 token，平均 %.1f token/块，词表 %d",
            len(self.chunk_ids),
            total_tokens,
            total_tokens / len(self.tokenized),
            len(self._bm25.idf),
        )
        return self

    # ------------------------------------------------------------------ #
    # 缓存
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path | None = None) -> Path:
        """把分词结果与元数据落盘（不含 BM25 对象本身，加载时重算，避免跨版本不兼容）。"""
        target = Path(path) if path else self.cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "chunk_ids": self.chunk_ids,
            "texts": self.texts,
            "metas": self.metas,
            "tokenized": self.tokenized,
        }
        with open(target, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "索引缓存已写入 %s（%.1f MB）",
            target.as_posix(),
            target.stat().st_size / 1024**2,
        )
        return target

    def load_cache(self, path: str | Path | None = None) -> bool:
        """尝试读取缓存，成功返回 True；不存在或版本不符返回 False。"""
        source = Path(path) if path else self.cache_path
        if not source.is_file():
            return False
        with open(source, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != CACHE_VERSION:
            logger.warning("缓存版本不匹配（%s），将重建", payload.get("version"))
            return False

        self.chunk_ids = payload["chunk_ids"]
        self.texts = payload["texts"]
        self.metas = payload["metas"]
        self.tokenized = payload["tokenized"]
        self._bm25 = BM25Okapi(self.tokenized)
        logger.info("已从缓存加载 BM25 索引：%d 个 chunk", len(self.chunk_ids))
        return True

    def build_or_load(self, rebuild: bool = False) -> "BM25Index":
        """有缓存就用缓存，否则从数据库重建并写缓存。"""
        if not rebuild and self.load_cache():
            return self
        self.build()
        self.save()
        return self

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def _ensure_ready(self) -> BM25Okapi:
        if self._bm25 is None:
            raise RuntimeError("索引尚未构建，请先调用 build() 或 load_cache()")
        return self._bm25

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int = SEARCH_TOP_K, min_score: float = 0.0) -> list[Bm25Hit]:
        """返回得分最高的 ``top_k`` 条结果。

        ``min_score`` 用来丢弃弱相关结果：rank_bm25 会把高频词的负 IDF 抬成
        ``epsilon * 平均IDF`` 这样的小正值，所以「只有很常见的词命中」时分数
        依然 > 0，需要调用方按阈值裁剪（默认 0 表示保留所有正分结果）。
        """
        bm25 = self._ensure_ready()
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(scores)[::-1][: max(1, top_k)]

        hits: list[Bm25Hit] = []
        for position in order:
            score = float(scores[position])
            if score <= min_score:
                continue
            meta = self.metas[position]
            hits.append(
                Bm25Hit(
                    chunk_id=self.chunk_ids[position],
                    score=score,
                    text=self.texts[position],
                    source=meta["source"],
                    page=meta["page"],
                    block_type=meta["block_type"],
                )
            )
        return hits


def _format_hit(rank: int, hit: Bm25Hit) -> str:
    return (
        f"{rank:>2}. {hit.score:8.3f}  {hit.chunk_id}  "
        f"[{hit.block_type} p{hit.page}]  {Path(hit.source).name}\n"
        f"     {hit.snippet()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.bm25_index",
        description="从 PostgreSQL 构建 BM25 索引并查询。",
    )
    parser.add_argument("-q", "--query", help="查询语句；不传则只构建/刷新索引")
    parser.add_argument(
        "-k", "--top-k", type=int, default=SEARCH_TOP_K, help=f"返回条数，默认 {SEARCH_TOP_K}"
    )
    parser.add_argument("--rebuild", action="store_true", help="忽略缓存，强制从数据库重建")
    parser.add_argument("--index", default=str(DEFAULT_CACHE), help="索引缓存文件路径")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    index = BM25Index(cache_path=args.index)
    index.build_or_load(rebuild=args.rebuild)

    if not args.query:
        print(f"索引就绪：{index.size} 个 chunk（缓存 {index.cache_path.as_posix()}）")
        return 0

    hits = index.search(args.query, top_k=args.top_k)
    if not hits:
        print(f"没有命中：{args.query!r}")
        return 0

    print(f"查询 {args.query!r} 命中 {len(hits)} 条：")
    for rank, hit in enumerate(hits, 1):
        print(_format_hit(rank, hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
