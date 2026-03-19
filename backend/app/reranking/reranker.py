import logging
import time
from typing import Iterable, List, Tuple

from sentence_transformers import CrossEncoder

from app.core.config import settings


logger = logging.getLogger(__name__)

_cross_encoder: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(settings.bge_reranker_model_name)
    return _cross_encoder


def rerank_passages(
    query: str,
    passages: Iterable[dict],
    top_k: int = 10,
) -> Tuple[List[dict], float]:
    start = time.perf_counter()
    reranker = get_reranker()
    passages_list = list(passages)
    if not passages_list:
        # Nothing to rerank.
        return [], 0.0

    texts = [p["content"] for p in passages_list]
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs, convert_to_numpy=True)

    enriched = []
    for passage, score in zip(passages_list, scores):
        p = dict(passage)
        p["rerank_score"] = float(score)
        enriched.append(p)
    ranked = sorted(enriched, key=lambda x: x["rerank_score"], reverse=True)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ranked[:top_k], elapsed_ms

