"""jina-reranker-v3 reranker (the service default).

Moved verbatim (behaviour-preserving) from XbrlRag's ``src/retriever/jina_reranker.py``
so the bounded-blocking memory fix travels with the model. This is a **local**
``transformers`` model — there is no Jina API call and no ``JINA_API_KEY`` is
read; ``rerank`` runs a CPU forward pass.
"""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModel

from src.rerankers.base_reranker import BaseReranker
from src.logging_config import get_logger

logger = get_logger(__name__)

# jina-reranker-v3 is *listwise*: ``rerank()`` concatenates the query + a block of
# documents into a single sequence, so one forward pass costs O(total_tokens^2)
# memory. The model already has token-budget blocking to cap a block at
# ``tokenizer.model_max_length`` — but that ships as the Qwen3 sentinel (131072),
# which never triggers a flush, so every candidate is packed into one giant
# forward. With long chunks (SEC tables run to ~6k chars) that attention tensor
# exceeds 6 GB and segfaults the process on CPU (Windows access violation).
#
# We bound it explicitly:
#   * _MAX_DOC_TOKENS    — per-document truncation. 512 is top-k neutral vs the
#                          model default (2048) on real candidates, and packing
#                          shorter docs means fewer blocks (less listwise drift).
#   * _BLOCK_TOKEN_BUDGET — tokens per forward. The model re-blocks to honor it,
#                          keeping the largest attention tensor ~O(4096^2) (<1 GB).
_MAX_DOC_TOKENS = 512
_BLOCK_TOKEN_BUDGET = 4096

# trust_remote_code runs Python from the model repo, so the commit is pinned: a new
# upstream push can't change what executes here. Keep in step with REVISIONS in
# infra/download_models.py and with the snapshot seeded into the weights bucket
# (MODEL_SOURCE=s3 runs offline, so a commit missing from the cache fails to load).
_MODEL_REVISION = "050e171c4f75dfec5b648ed8470a2475e5a30f30"


class JinaReranker(BaseReranker):
    model_name = "jinaai/jina-reranker-v3"

    def __init__(self, dtype: str | None = None):
        # Weights ship as bf16. Without native CPU bf16 (most x86) that path is
        # emulated: measured 61.6s bf16 vs 33.5s fp32 for 8 docs on an
        # x86 desktop CPU, max score diff 0.003. Hence fp32 by default (RERANKER_DTYPE).
        if dtype is None:
            from src.config.settings import get_settings

            dtype = get_settings().reranker_dtype
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self.reranker = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3",
            revision=_MODEL_REVISION,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.reranker.eval()
        # Force the model's internal token-budget blocking to actually engage:
        # its default model_max_length sentinel disables it -> single-forward OOM.
        self.reranker._ensure_tokenizer()
        self.reranker._tokenizer.model_max_length = _BLOCK_TOKEN_BUDGET
        logger.info(
            "loaded Jina re-ranker model - jinaai/jina-reranker-v3 "
            "(dtype=%s, block budget=%d tok, per-doc cap=%d tok)",
            next(self.reranker.parameters()).dtype,
            _BLOCK_TOKEN_BUDGET,
            _MAX_DOC_TOKENS,
        )

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Score each document against *query*; returns scores aligned to input
        order. The model handles any number of candidates safely now that its
        blocking is bounded — no outer batching needed here."""
        if not documents:
            return []
        with torch.inference_mode():
            results = self.reranker.rerank(query, documents, max_doc_length=_MAX_DOC_TOKENS)
        scores = [0.0] * len(documents)
        for r in results:
            scores[r["index"]] = float(r["relevance_score"])
        return scores
