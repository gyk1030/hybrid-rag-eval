"""BGE-Reranker 精排：对 RRF 融合后的候选做交叉编码重排，取 TopM。

两阶段检索里的第二阶段：召回（向量 + BM25 → RRF）给出 TopN 候选，本模块用
BGE-Reranker 把 (query, passage) 成对送进交叉编码器打分，再按分数取 TopM。
交叉编码器比双塔（BGE-M3）精度高但慢，所以只对少量候选使用。

模型需要放在本地（默认 ``models/bge-reranker-v2-m3``），不会自动联网下载::

    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
        BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

用法::

    python -m app.retrieval.reranker -q "研发投入" -m 5              # 混合召回 → 精排
    python -m app.retrieval.reranker -q "研发投入" -m 5 -c 30        # 候选池放宽到 30

作为库使用::

    from app.retrieval.reranker import Reranker
    from app.retrieval.rrf import hybrid_search

    candidates = hybrid_search("研发投入", top_k=20, top_n=20)
    top = Reranker().rerank("研发投入", candidates, top_m=5)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from FlagEmbedding import FlagReranker

# 允许用脚本方式直接运行（python app/retrieval/reranker.py）。
# 此时 Python 会把 sys.path[0] 设为脚本所在目录（app/retrieval），项目根目录不在
# 搜索路径里，main() 中延迟导入的 `app.retrieval.rrf` 就会报 ModuleNotFoundError。
# 改用 `python -m app.retrieval.reranker` 时 __package__ 非空，这段不会执行。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import (  # noqa: E402 - 必须在 sys.path 修补之后导入
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATES,
    RERANK_MAX_LENGTH,
    RERANK_TOP_M,
    RERANK_USE_FP16,
    RERANKER_MODEL_PATH,
)

# 权重文件的存在性检查：只判断目录存在是不够的，半截下载的目录会给出很晦涩的报错
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

DOWNLOAD_HINT = (
    "请先把模型放到本地（不会自动联网下载）：\n"
    "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
    "BAAI/bge-reranker-v2-m3 --local-dir {path}"
)



@dataclass(slots=True)
class RerankedHit:
    """一条精排结果。"""

    chunk_id: str
    score: float
    text: str
    source: str
    page: int
    block_type: str
    fused_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)

    def snippet(self, length: int = 80) -> str:
        """返回用于展示的短摘要。"""
        flat = " ".join(self.text.split())
        return flat[:length] + ("…" if len(flat) > length else "")


class Reranker:
    """BGE-Reranker 的封装：加载一次、可反复打分。"""

    def __init__(
        self,
        model_path: str | Path = RERANKER_MODEL_PATH,
        *,
        device: str | None = None,
        use_fp16: bool | None = None,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = RERANK_MAX_LENGTH,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        if use_fp16 is None:
            # 实测本机（GTX 1650 SUPER）精排用 fp16 更快：512 token 下 633 vs 736 ms/条，
            # 256 token 下 427 vs 1090 ms/条；四组配置的排序结果完全一致。
            # 注意这与嵌入模型（bge-m3）的实测结论相反（那里 fp16 慢 5 倍），
            # 因为精排是交叉编码器、且按 batch=8 批处理，显存带宽才是瓶颈。
            # 默认策略可在 app/config.py 的 RERANK_USE_FP16 覆盖。
            use_fp16 = (
                RERANK_USE_FP16 if RERANK_USE_FP16 is not None else device.startswith("cuda")
            )
        self.use_fp16 = use_fp16

        self._model: FlagReranker | None = None

    @property
    def model(self) -> FlagReranker:
        """延迟加载：只有真正打分时才把模型读进显存。"""
        if self._model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"找不到精排模型目录：{self.model_path}\n"
                    + DOWNLOAD_HINT.format(path=self.model_path.as_posix())
                )
            if not any((self.model_path / name).is_file() for name in WEIGHT_FILES):
                raise FileNotFoundError(
                    f"模型目录 {self.model_path} 缺少权重文件"
                    f"（{' 或 '.join(WEIGHT_FILES)}），可能是下载未完成\n"
                    + DOWNLOAD_HINT.format(path=self.model_path.as_posix())
                )
            self._model = FlagReranker(
                str(self.model_path),
                use_fp16=self.use_fp16,
                devices=self.device,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=True,  # sigmoid 到 (0, 1)，便于跨查询比较
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """给若干候选段落打分，返回与 ``passages`` 等长的相似度列表。"""
        if not passages:
            return []
        pairs = [(query, passage or "") for passage in passages]
        raw = self.model.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        # compute_score 在只有一组 pair 时返回标量
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(score) for score in raw]

    def rerank(
        self,
        query: str,
        hits: Sequence[Any],
        top_m: int = RERANK_TOP_M,
    ) -> list[RerankedHit]:
        """对候选（``FusedHit`` / ``VectorHit`` / ``Bm25Hit``）重排并取 TopM。

        入参只需要带 ``chunk_id`` 和 ``text`` 属性；``score`` / ``ranks`` 会被保留
        下来便于对比精排前后的变化。
        """
        if not hits:
            return []
        if top_m < 1:
            raise ValueError("top_m 必须 >= 1")

        passages = [getattr(hit, "text", "") or "" for hit in hits]
        scores = self.score(query, passages)

        # 分数相同时按 chunk_id 排序，保证结果可复现
        order = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], getattr(hits[index], "chunk_id", "")),
        )[:top_m]

        results: list[RerankedHit] = []
        for index in order:
            hit = hits[index]
            results.append(
                RerankedHit(
                    chunk_id=str(getattr(hit, "chunk_id", "")),
                    score=scores[index],
                    text=getattr(hit, "text", "") or "",
                    source=getattr(hit, "source", "") or "",
                    page=int(getattr(hit, "page", 0) or 0),
                    block_type=getattr(hit, "block_type", "") or "",
                    fused_score=float(getattr(hit, "score", 0.0)),
                    ranks=dict(getattr(hit, "ranks", {}) or {}),
                )
            )
        return results


def _format_hit(rank: int, hit: RerankedHit) -> str:
    lines = [
        f"{rank:>2}. {hit.score:.4f}  {hit.chunk_id}  "
        f"[{hit.block_type} p{hit.page}]  {Path(hit.source).name if hit.source else '-'}"
    ]
    if hit.fused_score is not None:
        detail = f"     rrf={hit.fused_score:.6f}"
        if hit.ranks:
            detail += "  (" + " ".join(f"{k}={v}" for k, v in sorted(hit.ranks.items())) + ")"
        lines.append(detail)
    lines.append(f"     {hit.snippet()}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.reranker",
        description="混合召回（向量 + BM25 → RRF）后用 BGE-Reranker 精排。",
    )
    parser.add_argument("-q", "--query", required=True, help="查询语句")
    parser.add_argument(
        "-m", "--top-m", type=int, default=RERANK_TOP_M, help=f"精排后返回条数，默认 {RERANK_TOP_M}"
    )
    parser.add_argument(
        "-c",
        "--candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help=f"送入精排的候选数，默认 {RERANK_CANDIDATES}",
    )
    parser.add_argument("--model", default=RERANKER_MODEL_PATH, help="精排模型目录")
    args = parser.parse_args(argv)

    # 延迟导入：只在真正做端到端检索时才加载检索栈
    from app.retrieval.rrf import hybrid_search

    candidates = hybrid_search(args.query, top_k=args.candidates, top_n=args.candidates)
    if not candidates:
        print(f"没有命中：{args.query!r}")
        return 0

    reranker = Reranker(args.model)
    try:
        top = reranker.rerank(args.query, candidates, top_m=args.top_m)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"查询 {args.query!r}（候选 {len(candidates)} → 精排 top{args.top_m}，"
        f"设备 {reranker.device}，fp16={reranker.use_fp16}）："
    )
    for position, hit in enumerate(top, 1):
        print(_format_hit(position, hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
