import logging
import time
from typing import Any, Iterable, List, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis_client import cache_get_json, cache_set_json, make_cache_key
from app.core.config import settings


logger = logging.getLogger(__name__)


async def vector_search(
    db: AsyncSession,
    query_embedding: Sequence[float],
    top_k: int = 20,
) -> List[dict]:
    # pgvector expects vector params in its textual format (e.g. "[0.1,0.2,...]"),
    # not a raw Python list when using asyncpg with raw SQL.
    query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    sql = text(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_k
        """
    )
    rows = await db.execute(
        sql, {"query_embedding": query_embedding_str, "top_k": top_k}
    )
    return [
        {
            "id": str(r.id),
            "content": r.content,
            "metadata": r.metadata,
            "score": float(r.score),
        }
        for r in rows
    ]


async def bm25_search(
    db: AsyncSession,
    query: str,
    top_k: int = 20,
) -> List[dict]:
    # Simple BM25-like ranking using PostgreSQL full-text search
    sql = text(
        """
        SELECT id, content, metadata,
               ts_rank_cd(
                   to_tsvector('english', content),
                   plainto_tsquery('english', :query)
               ) AS score
        FROM documents
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :top_k
        """
    )
    rows = await db.execute(sql, {"query": query, "top_k": top_k})
    return [
        {
            "id": str(r.id),
            "content": r.content,
            "metadata": r.metadata,
            "score": float(r.score),
        }
        for r in rows
    ]


def _merge_results(
    vector_results: Iterable[dict],
    bm25_results: Iterable[dict],
    k: int,
) -> List[dict]:
    combined: dict[str, dict] = {}
    for r in vector_results:
        combined.setdefault(r["id"], {**r, "score_vector": r["score"], "score_bm25": 0.0})
    for r in bm25_results:
        if r["id"] in combined:
            combined[r["id"]]["score_bm25"] = r["score"]
        else:
            combined[r["id"]] = {
                **r,
                "score_vector": 0.0,
                "score_bm25": r["score"],
            }
    for r in combined.values():
        r["score"] = 0.5 * r.get("score_vector", 0.0) + 0.5 * r.get("score_bm25", 0.0)
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:k]


async def hybrid_search(
    db: AsyncSession,
    query: str,
    query_embedding: Sequence[float],
    top_k: int = 20,
) -> Tuple[List[dict], dict]:
    cache_key = make_cache_key("hybrid_search", query)
    cached = await cache_get_json(cache_key)
    if cached:
        return cached["results"], cached["metrics"]

    t0 = time.perf_counter()
    vector_results = await vector_search(db, query_embedding, top_k=top_k)
    t_vector = time.perf_counter()
    if not settings.bm25_enabled:
        # Speed mode: vector-only retrieval.
        metrics = {
            "vector_search_ms": (t_vector - t0) * 1000,
            "bm25_search_ms": 0.0,
            "merge_ms": 0.0,
        }
        await cache_set_json(
            cache_key, {"results": vector_results, "metrics": metrics}, 600
        )
        return vector_results, metrics

    bm25_results = await bm25_search(db, query, top_k=top_k)
    t_bm25 = time.perf_counter()

    merged = _merge_results(vector_results, bm25_results, k=top_k)
    t_end = time.perf_counter()

    metrics = {
        "vector_search_ms": (t_vector - t0) * 1000,
        "bm25_search_ms": (t_bm25 - t_vector) * 1000,
        "merge_ms": (t_end - t_bm25) * 1000,
    }

    await cache_set_json(cache_key, {"results": merged, "metrics": metrics}, 600)
    return merged, metrics

