import logging
import textwrap
import time
from typing import Dict, List, Tuple

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.embedding.embedder import embed_texts
from app.retrieval.retriever import hybrid_search
from app.reranking.reranker import rerank_passages


logger = logging.getLogger(__name__)

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

# Once we hit Gemini quota/rate-limit, avoid repeatedly triggering long waits
# on subsequent requests. We degrade gracefully to retrieval-only behavior.
_GEMINI_GENERATE_DISABLED = False


async def generate_query_variants(question: str) -> List[str]:
    if not settings.gemini_api_key:
        return [question]
    if not settings.gemini_query_variants_enabled:
        return []
    global _GEMINI_GENERATE_DISABLED
    if _GEMINI_GENERATE_DISABLED:
        return []
    prompt = textwrap.dedent(
        f"""
        You are a system that generates alternative search queries.
        Given the user's question, produce three semantically different but relevant search queries.
        Return them as a numbered list.

        Question: {question}
        """
    )
    model = genai.GenerativeModel(settings.gemini_chat_model)
    try:
        resp = model.generate_content(prompt)
    except ResourceExhausted:
        _GEMINI_GENERATE_DISABLED = True
        # Free-tier quota/rate-limit hit: degrade gracefully.
        return [question]
    content = resp.text or ""
    variants: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        variants.append(line)
    if not variants:
        variants = [question]
    return variants[:3]


async def rewrite_query(question: str) -> str:
    if not settings.gemini_api_key:
        return question
    if not settings.gemini_query_rewrite_enabled:
        return question
    global _GEMINI_GENERATE_DISABLED
    if _GEMINI_GENERATE_DISABLED:
        return question
    prompt = textwrap.dedent(
        f"""
        Rewrite the following question to be clearer and more specific while preserving its meaning.
        Return only the rewritten question.

        Question: {question}
        """
    )
    model = genai.GenerativeModel(settings.gemini_chat_model)
    try:
        resp = model.generate_content(prompt)
    except ResourceExhausted:
        _GEMINI_GENERATE_DISABLED = True
        # Free-tier quota/rate-limit hit: degrade gracefully.
        return question
    return (resp.text or question).strip()


def compress_context(passages: List[dict], max_chars: int = 6000) -> Tuple[str, List[dict]]:
    context_pieces: List[str] = []
    used_passages: List[dict] = []
    remaining = max_chars
    for p in passages:
        chunk = p["content"].strip()
        if not chunk:
            continue
        if len(chunk) + 2 > remaining:
            break
        context_pieces.append(chunk)
        used_passages.append(p)
        remaining -= len(chunk) + 2
    return "\n\n".join(context_pieces), used_passages


async def run_rag_pipeline(
    db: AsyncSession,
    question: str,
) -> Dict:
    # Gemini is optional: retrieval can work without it.
    gemini_available = bool(settings.gemini_api_key) and (
        settings.gemini_query_rewrite_enabled
        or settings.gemini_query_variants_enabled
        or settings.gemini_answer_enabled
    )
    if not gemini_available and not settings.gemini_answer_enabled:
        # Still allow retrieval-only flow
        pass

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    rewritten = await rewrite_query(question)
    variants = await generate_query_variants(rewritten)
    timings["query_rewrite_ms"] = (time.perf_counter() - t0) * 1000

    all_results: List[dict] = []
    retrieval_metrics: Dict[str, float] = {}
    t_retrieval_start = time.perf_counter()

    # Multi-query retrieval (de-dup while keeping order)
    query_texts: List[str] = [rewritten] + list(variants)
    seen: set[str] = set()
    unique_query_texts: List[str] = []
    for qt in query_texts:
        if qt in seen:
            continue
        seen.add(qt)
        unique_query_texts.append(qt)

    for v in unique_query_texts:
        embed_t0 = time.perf_counter()
        [q_vec] = await embed_texts([v])
        embed_t1 = time.perf_counter()
        results, metrics = await hybrid_search(db, v, q_vec, top_k=settings.retrieval_top_k)
        retrieval_metrics.setdefault("embedding_ms", 0.0)
        retrieval_metrics["embedding_ms"] += (embed_t1 - embed_t0) * 1000
        for key, val in metrics.items():
            retrieval_metrics[key] = retrieval_metrics.get(key, 0.0) + float(val)
        all_results.extend(results)

    # Deduplicate by document id
    by_id: Dict[str, dict] = {}
    for r in all_results:
        if r["id"] not in by_id or r["score"] > by_id[r["id"]]["score"]:
            by_id[r["id"]] = r
    merged_results = list(by_id.values())
    timings["retrieval_total_ms"] = (time.perf_counter() - t_retrieval_start) * 1000
    timings.update(retrieval_metrics)

    # Rerank (optional). If Gemini generation is disabled due to quota/rate-limit,
    # avoid loading the large cross-encoder model and just take top results.
    if not merged_results:
        # No passages to rerank/reason over.
        return {
            "answer": "Not in document.",
            "sources": [],
            "metrics": timings,
        }

    global _GEMINI_GENERATE_DISABLED
    if (not settings.reranker_enabled) or _GEMINI_GENERATE_DISABLED:
        # Speed mode: skip cross-encoder reranking entirely.
        reranked = merged_results[: settings.reranker_top_k]
        timings["rerank_ms"] = 0.0
    else:
        reranked, rerank_ms = rerank_passages(
            rewritten, merged_results, top_k=settings.reranker_top_k
        )
        timings["rerank_ms"] = rerank_ms

    # Context compression
    context_str, used_passages = compress_context(reranked, max_chars=6000)
    timings["context_chars"] = len(context_str)

    # LLM answer (optional). If Gemini is disabled/quota-limited, fall back to extractive.
    system_prompt = (
        "You are an AI assistant that answers questions strictly based on the provided context. "
        "If the answer is not contained in the context, reply exactly with: Not in document.\n\n"
    )
    user_prompt = f"{system_prompt}Question:\n{question}\n\nContext:\n{context_str}"

    llm_t0 = time.perf_counter()
    try:
        if _GEMINI_GENERATE_DISABLED or not settings.gemini_answer_enabled or not context_str:
            return {
                # Cheap/no-token fallback: return the most relevant context snippet.
                "answer": (context_str[:1500] if context_str else "Not in document."),
                "sources": used_passages,
                "metrics": timings,
            }

        model = genai.GenerativeModel(settings.gemini_chat_model)
        completion = model.generate_content(user_prompt)
    except ResourceExhausted:
        _GEMINI_GENERATE_DISABLED = True
        # If we can't generate an answer right now, still return sources/metrics.
        return {
            "answer": (context_str[:1500] if context_str else "Not in document."),
            "sources": used_passages,
            "metrics": timings,
        }
    llm_t1 = time.perf_counter()

    answer = completion.text or ""
    timings["llm_ms"] = (llm_t1 - llm_t0) * 1000

    return {
        "answer": answer.strip(),
        "sources": used_passages,
        "metrics": timings,
    }

