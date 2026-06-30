"""Wire schemas for the inference service (embed + rerank)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Rerank
# --------------------------------------------------------------------------- #
class RerankRequest(BaseModel):
    query: str = Field(..., description="The query to score documents against.")
    documents: List[str] = Field(..., description="Candidate document texts, scored in place.")
    model: Optional[str] = Field(
        None, description="Optional reranker override ('jina' | 'bge'). Defaults to the server's loaded model."
    )


class RerankResponse(BaseModel):
    scores: List[float] = Field(..., description="One relevance score per input document, aligned to order.")
    model: str = Field(..., description="The reranker model that produced the scores.")
    count: int = Field(..., description="Number of documents scored.")


# --------------------------------------------------------------------------- #
# Embed
# --------------------------------------------------------------------------- #
class SparseVectorModel(BaseModel):
    indices: List[int] = Field(..., description="Sparse vector token-id indices.")
    values: List[float] = Field(..., description="Sparse vector weights, aligned to indices.")


class EmbeddingItem(BaseModel):
    dense: Optional[List[float]] = Field(None, description="Dense vector (None if dense not requested).")
    sparse: Optional[SparseVectorModel] = Field(None, description="Sparse vector (None if sparse not requested).")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed, results aligned to input order.")
    dense: bool = Field(True, description="Return dense vectors.")
    sparse: bool = Field(True, description="Return sparse (lexical) vectors.")
    model: Optional[str] = Field(None, description="Optional embedder override ('bge').")


class EmbedResponse(BaseModel):
    embeddings: List[EmbeddingItem] = Field(..., description="One embedding per input text, aligned to order.")
    model: str = Field(..., description="The embedder model that produced the vectors.")
    count: int = Field(..., description="Number of texts embedded.")


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    reranker: str
    embedder: str
    reranker_loaded: bool
    embedder_loaded: bool
