"""BAAI/bge-m3 dense+sparse embedder (the service default).

Wraps ``BGEM3FlagModel`` exactly like XbrlRag's
``src/embeddings/bge_embedding.py`` (same model, same memory-safe ``batch_size``
/ ``max_length`` defaults), but returns JSON-able dense lists + sparse
``{indices, values}`` instead of XbrlRag's dataclasses, which the client rebuilds.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.embedders.base_embedder import BaseEmbedder
from src.logging_config import get_logger

logger = get_logger(__name__)


class BGEEmbedder(BaseEmbedder):
    model_name = "BAAI/bge-m3"

    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 12, max_length: int = 8192):
        from FlagEmbedding import BGEM3FlagModel

        # batch_size/max_length capped for CPU memory safety — FlagEmbedding's
        # default batch_size=256 builds one huge forward pass and segfaults.
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(model)
        logger.info("loaded BGE-M3 embedder model=%s batch_size=%d max_length=%d",
                    model, batch_size, max_length)

    def embed(self, texts: List[str], dense: bool = True, sparse: bool = True) -> List[Dict[str, Any]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts, return_dense=dense, return_sparse=sparse,
            batch_size=self.batch_size, max_length=self.max_length,
        )
        dense_vecs = vectors["dense_vecs"] if dense else []
        lexical = vectors["lexical_weights"] if sparse else []

        out: List[Dict[str, Any]] = []
        for i in range(len(texts)):
            item: Dict[str, Any] = {}
            if dense:
                item["dense"] = [float(x) for x in dense_vecs[i]]
            if sparse:
                weights = lexical[i]
                # token-id -> weight; cast to JSON-able int/float (mirrors
                # BaseEmbeddingGenerator.dict_to_sparse_list on the XbrlRag side).
                indices = [int(k) for k in weights.keys()]
                values = [float(weights[k]) for k in weights.keys()]
                item["sparse"] = {"indices": indices, "values": values}
            out.append(item)
        return out
