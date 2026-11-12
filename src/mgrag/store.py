"""Qdrant hybrid index: dense vectors (meaning) + BM25 sparse vectors (exact terms
like 'Mg(TFSI)2', 'Chevrel', 'APC') fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import uuid
from functools import lru_cache

from .config import settings

DENSE, SPARSE = "dense", "bm25"


@lru_cache
def client():
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    # Embedded mode: one process at a time. Use docker compose + QDRANT_URL for app + MCP together.
    return QdrantClient(path=str(settings.qdrant_path))


@lru_cache
def dense_model():
    from fastembed import TextEmbedding
    return TextEmbedding(settings.dense_model)


@lru_cache
def sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(settings.sparse_model)


def embed_text(chunk: dict) -> str:
    """What the embedder sees: paper title + chunk (+ figure description if any)."""
    parts = [chunk.get("title", ""), chunk["text"]]
    if chunk.get("description"):
        parts.append(chunk["description"])
    return "\n".join(p for p in parts if p)


def build_index(chunks: list[dict], recreate: bool = True, batch: int = 64) -> None:
    from qdrant_client import models

    c = client()
    dim = len(next(iter(dense_model().embed(["probe"]))))
    if recreate and c.collection_exists(settings.collection):
        c.delete_collection(settings.collection)
    if not c.collection_exists(settings.collection):
        c.create_collection(
            settings.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        texts = [embed_text(ch) for ch in part]
        dense = list(dense_model().embed(texts))
        sparse = list(sparse_model().embed(texts))
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, ch["id"])),
                vector={DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                payload=ch,
            )
            for ch, d, s in zip(part, dense, sparse, strict=True)
        ]
        c.upsert(settings.collection, points=points)
        print(f"  indexed {min(start + batch, len(chunks))}/{len(chunks)}")


def search(query: str, k: int = 30, mode: str = "hybrid", types: list[str] | None = None) -> list[dict]:
    """mode: 'hybrid' (default), 'dense' or 'sparse' — the last two exist for ablation studies."""
    from qdrant_client import models

    flt = None
    if types:
        flt = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchAny(any=types))])
    qd = next(iter(dense_model().query_embed(query))).tolist()
    qs_raw = next(iter(sparse_model().query_embed(query)))
    qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())

    if mode == "dense":
        res = client().query_points(settings.collection, query=qd, using=DENSE, limit=k,
                                    query_filter=flt, with_payload=True)
    elif mode == "sparse":
        res = client().query_points(settings.collection, query=qs, using=SPARSE, limit=k,
                                    query_filter=flt, with_payload=True)
    else:
        res = client().query_points(
            settings.collection,
            prefetch=[models.Prefetch(query=qd, using=DENSE, limit=k, filter=flt),
                      models.Prefetch(query=qs, using=SPARSE, limit=k, filter=flt)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        )
    return [dict(p.payload, retrieval_score=float(p.score)) for p in res.points]


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Cross-encoder reads (query, chunk) together → much sharper relevance than vectors.
    Scores are squashed to 0..1 with a sigmoid so one threshold works for the guardrail."""
    import numpy as np
    import torch

    if not candidates:
        return []
    pairs = [(query, c["text"] + ("\n" + c["description"] if c.get("description") else ""))
             for c in candidates]
    try:
        logits = reranker().predict(pairs, activation_fn=torch.nn.Identity())
    except TypeError:  # sentence-transformers < 3
        logits = reranker().predict(pairs, activation_fct=torch.nn.Identity())
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    for c, s in zip(candidates, scores, strict=True):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
