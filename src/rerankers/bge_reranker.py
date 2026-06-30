"""bge-reranker-v2-m3 cross-encoder (faster, lighter alternative to Jina).

Reimplemented against the uniform ``rerank(query, list[str]) -> list[float]``
contract (ARag's original ``BGEReranker`` took a chunk-id dict). A plain
cross-encoder: pairs ``[query, doc]`` scored independently, so it is far cheaper
on CPU than the listwise Jina model — kept here as a drop-in the service can
select via ``RERANKER=bge`` when latency matters more than listwise quality.
"""
from __future__ import annotations

from typing import List

from src.rerankers.base_reranker import BaseReranker
from src.logging_config import get_logger

logger = get_logger(__name__)


class BGEReranker(BaseReranker):
    model_name = "BAAI/bge-reranker-v2-m3"

    def __init__(self, batch_size: int = 8, use_fp16: bool = False):
        from FlagEmbedding import FlagReranker

        # fp16 is a no-op (often unsupported) on CPU; default off here.
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=use_fp16, normalize=True)
        self.batch_size = batch_size
        logger.info("loaded BGE re-ranker model - BAAI/bge-reranker-v2-m3 (batch_size=%d)", batch_size)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.compute_score(pairs, batch_size=self.batch_size)
        # FlagReranker returns a float for a single pair, else a list.
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]
