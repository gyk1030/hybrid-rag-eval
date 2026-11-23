"""pgvector 向量库：把 chunks.jsonl 写入 PostgreSQL，并提供稠密向量检索接口。

写入侧沿用「先查已入库 chunk_id、跳过重复编码、批量提交」的策略；查询侧用
pgvector 的余弦距离算子 ``<=>``（对应表上的 HNSW 索引 ``chunks_dense_embedding_idx``）。

用法::

    python -m app.retrieval.vector_store                       # 默认读 data/parsed/chunks.jsonl
    python -m app.retrieval.vector_store -i data/parsed/chunks.jsonl
    python -m app.retrieval.vector_store -q "研发投入" -k 5
    python -m app.retrieval.vector_store -q "应收账款 坏账准备" --dsn "postgresql://..."

作为库使用::

    from app.retrieval.vector_store import VectorStore

    store = VectorStore("data/parsed/chunks.jsonl")
    for hit in store.search("研发投入", top_k=5):
        print(hit.score, hit.chunk_id, hit.snippet())
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg2
import torch
from FlagEmbedding import BGEM3FlagModel
from psycopg2.extras import execute_values

# 允许用脚本方式直接运行（python app/retrieval/vector_store.py）。
# 脚本方式下 sys.path[0] 是脚本所在目录，项目根目录不在搜索路径里，
# `from app.config import ...` 会报 ModuleNotFoundError。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    CHUNKS_PATH,
    DB_BATCH_SIZE,
    DSN,
    EMBED_USE_BF16,
    EMBED_USE_FP16,
    EMBEDDING_MODEL_PATH,
    SEARCH_TOP_K,
)

INSERT_SQL = """
INSERT INTO chunks
    (chunk_id, doc_id, source, page, block_type, text, table_caption, table_index,
     row_start, row_end, metadata, dense_embedding, sparse_embedding)
VALUES %s
ON CONFLICT (chunk_id) DO NOTHING
"""

# execute_values 的每行模板：最后两列需要显式转型，与原始单条 INSERT 保持一致
INSERT_TEMPLATE = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::sparsevec)"

# 余弦距离 <=> 越小越相似；用 1 - 距离 换成相似度，分数越大越相关
SEARCH_SQL = """
SELECT chunk_id, text, source, page, block_type,
       1 - (dense_embedding <=> %s::vector) AS score
FROM chunks
ORDER BY dense_embedding <=> %s::vector
LIMIT %s
"""

_MODEL = None


def get_model() -> BGEM3FlagModel:
    """首次编码时才加载模型，避免 import 本模块就吃掉 2 GB 内存、耽误启动。"""
    global _MODEL
    if _MODEL is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        on_gpu = device.startswith("cuda")
        # 精度默认按设备自动（GPU→bf16，CPU→fp32），可在 app/config.py 里覆盖。
        # 本机 GTX 1650 SUPER（Turing TU116，无 Tensor Core）实测前向速度：
        #   fp16 244 ms/条 → bf16 62 ms/条 → fp32 47 ms/条
        # 所以显卡上用 bf16（比 fp32 只慢一点，但权重只占一半显存）；CPU 保持 fp32。
        _MODEL = BGEM3FlagModel(
            EMBEDDING_MODEL_PATH,
            use_fp16=EMBED_USE_FP16 if EMBED_USE_FP16 is not None else False,
            use_bf16=EMBED_USE_BF16 if EMBED_USE_BF16 is not None else on_gpu,
            devices=device,
        )
    return _MODEL


def iter_chunks(path):
    """逐行读取 chunks.jsonl，不把整份文件读进内存。"""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass(slots=True)
class VectorHit:
    """一条向量检索结果（字段与 Bm25Hit 对齐，便于后续做 RRF 融合）。"""

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


class VectorStore():
    def __init__(
        self,
        copuse_path: str,
        dsn: str | None = None,
        db_batch_size: int = DB_BATCH_SIZE,
    ):
        self.copuse_path = copuse_path
        self.dsn = dsn or DSN
        self.db_batch_size = db_batch_size
        self.conn, self.cur = self.connection()

    # ------------------------------------------------------------------ #
    # 编码
    # ------------------------------------------------------------------ #
    def _encode_text(self, text: str):
        out = get_model().encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # numpy array, shape (1024,)
        dense_vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        # dict: {"4865": 0.08, ...}
        sparse_dict = out["lexical_weights"][0]
        return dense_vec, sparse_dict

    def encode(self, chunk):
        """编码一个 chunk（入库用）。"""
        return self._encode_text(chunk["text"])

    def encode_query(self, query: str):
        """编码一条查询。BGE-M3 的查询与文档共用同一套编码逻辑，无需额外指令前缀。"""
        return self._encode_text(query)

    @staticmethod
    def to_vector_literal(dense_vec) -> str:
        """numpy 向量 -> pgvector 的文本字面量，配合 ``%s::vector`` 使用。"""
        return json.dumps(np.asarray(dense_vec, dtype=np.float32).tolist())

    def to_sparsevec_literal(self, sparse_dict, dim=250002):
        if not sparse_dict:
            return None
        # pgvector 的 sparsevec 下标从 1 开始，0 是非法值，需要过滤掉
        pairs = []
        for key, value in sparse_dict.items():
            index = int(key)
            if index <= 0:
                continue
            pairs.append((index, float(value)))
        if not pairs:
            return None
        pairs.sort()
        items = ", ".join(f"{index}:{value}" for index, value in pairs)
        return f"{{{items}}}/{dim}"

    # ------------------------------------------------------------------ #
    # 数据库
    # ------------------------------------------------------------------ #
    def connection(self):
        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()
        return conn, cur

    def existing_chunk_ids(self):
        """已入库的 chunk_id，用来跳过重复编码（重跑时几乎零成本）。"""
        self.cur.execute("SELECT chunk_id FROM chunks")
        return {row[0] for row in self.cur.fetchall()}

    def build_row(self, chunk, dense_vec, sparse_dict):
        meta = chunk["metadata"]
        is_table = meta["block_type"] == "table"
        row_range = meta.get("row_range") or [None, None]
        return (
            meta["chunk_id"],
            meta["doc_id"],
            meta["source"],
            meta["page"],
            meta["block_type"],
            chunk["text"],

            meta.get("table_caption") if is_table else None,
            meta.get("table_index") if is_table else None,
            row_range[0] if is_table else None,
            row_range[1] if is_table else None,

            json.dumps(meta, ensure_ascii=False),
            self.to_vector_literal(dense_vec),
            self.to_sparsevec_literal(sparse_dict),
        )

    def flush(self, rows):
        """一批行一次性写入，只提交一次。"""
        if not rows:
            return 0
        execute_values(
            self.cur,
            INSERT_SQL,
            rows,
            template=INSERT_TEMPLATE,
            page_size=self.db_batch_size,
        )
        self.conn.commit()
        return len(rows)

    def save(self, chunk, dense_vec, sparse_dict):
        """保留原有的单条写入接口。"""
        return self.flush([self.build_row(chunk, dense_vec, sparse_dict)])

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = SEARCH_TOP_K) -> list[VectorHit]:
        """按稠密向量做余弦相似度检索，返回 ``top_k`` 条最相似的结果。

        用的是 pgvector 的余弦距离算子 ``<=>``（对应索引 ``chunks_dense_embedding_idx``，
        HNSW + vector_cosine_ops）。注意：走不走索引由优化器决定——当前 5,656 行的规模下
        它选的是「全表扫描 + top-N 排序」（约 40 ms，结果是**精确**的）；实测强制
        ``SET enable_seqscan = off`` 会走 HNSW，约 1.3 ms，但那是近似检索（ef_search=40）。
        数据量继续增长后优化器会自动切到索引。
        """
        if not query or not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        dense_vec, _ = self.encode_query(query)
        literal = self.to_vector_literal(dense_vec)

        # 用独立游标，避免和批量写入共用同一个 cursors 的状态
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, (literal, literal, top_k))
            rows = cur.fetchall()

        return [
            VectorHit(
                chunk_id=row[0],
                text=row[1],
                source=row[2],
                page=row[3],
                block_type=row[4],
                score=float(row[5]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # 入库
    # ------------------------------------------------------------------ #
    def run(self):
        existing = self.existing_chunk_ids()
        buffer = []
        seen = skipped = written = 0

        for seen, chunk in enumerate(iter_chunks(self.copuse_path), 1):
            if chunk["metadata"]["chunk_id"] in existing:
                skipped += 1
                continue

            dense_vec, sparse_dict = self.encode(chunk)
            buffer.append(self.build_row(chunk, dense_vec, sparse_dict))

            if len(buffer) >= self.db_batch_size:
                written += self.flush(buffer)
                buffer.clear()
                print(f'[{seen}] 已写入 {written} 条，跳过已存在 {skipped} 条')

        written += self.flush(buffer)
        print(f'完成：扫描 {seen} 条，新写入 {written} 条，跳过已存在 {skipped} 条')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.vector_store",
        description="把 chunks.jsonl 写入 pgvector，或对已入库向量做相似度检索。",
    )
    parser.add_argument(
        "-i", "--input", default=CHUNKS_PATH, help=f"chunks.jsonl 路径，默认 {CHUNKS_PATH}"
    )
    parser.add_argument("-q", "--query", help="给定时执行检索（不入库）")
    parser.add_argument(
        "-k", "--top-k", type=int, default=SEARCH_TOP_K, help=f"检索返回条数，默认 {SEARCH_TOP_K}"
    )
    parser.add_argument("--dsn", default=DSN, help="PostgreSQL 连接串")
    parser.add_argument("--batch-size", type=int, default=DB_BATCH_SIZE, help="每批写库条数")
    args = parser.parse_args(argv)

    if args.query is None and not Path(args.input).is_file():
        print(f"找不到输入文件：{args.input}", file=sys.stderr)
        return 2

    store = VectorStore(args.input, dsn=args.dsn, db_batch_size=args.batch_size)

    if args.query:
        hits = store.search(args.query, top_k=args.top_k)
        if not hits:
            print(f"没有命中：{args.query!r}")
            return 0
        print(f"查询 {args.query!r} 命中 {len(hits)} 条：")
        for rank, hit in enumerate(hits, 1):
            print(
                f"{rank:>2}. {hit.score:8.4f}  {hit.chunk_id}  "
                f"[{hit.block_type} p{hit.page}]  {Path(hit.source).name}"
            )
            print(f"     {hit.snippet()}")
        return 0

    store.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
