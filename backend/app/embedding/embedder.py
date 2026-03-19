import hashlib
import logging
import os
from typing import Iterable, List, Sequence

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis_client import cache_get_json, cache_set_json, make_cache_key
from app.core.config import settings
from app.models.document import Document, DocumentImage


logger = logging.getLogger(__name__)

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

# Local embeddings cache (open-source, free).
_local_embedder: SentenceTransformer | None = None


def get_local_embedder() -> SentenceTransformer:
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = SentenceTransformer(
            settings.bge_embedding_model_name,
            device=settings.local_embeddings_device,
        )
    return _local_embedder


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def embed_texts(
    texts: Sequence[str],
    *,
    target_dim: int | None = None,
) -> List[List[float]]:
    provider = (settings.embeddings_provider or "gemini").lower()
    effective_dim = (
        int(target_dim)
        if target_dim is not None
        else int(settings.pgvector_text_embedding_dim)
    )

    # Local open-source embeddings (recommended to reduce cost/quota).
    if provider == "local":
        model_name = settings.bge_embedding_model_name
        cache_key = make_cache_key(
            "embeddings:local",
            model_name,
            str(effective_dim),
            _hash_text("||".join(texts)),
        )
        cached = await cache_get_json(cache_key)
        if cached:
            return cached

        embedder = get_local_embedder()
        batch_size = settings.local_embeddings_batch_size
        vectors: List[List[float]] = []
        import numpy as np
        texts_list = list(texts)
        for i in range(0, len(texts_list), batch_size):
            chunk = texts_list[i : i + batch_size]
            # sentence-transformers encode is synchronous; run in async context anyway.
            batch_vectors = embedder.encode(
                chunk,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            # Pad/truncate to pgvector schema dimension.
            orig_dim = int(batch_vectors.shape[1])
            if orig_dim < effective_dim:
                padded = np.zeros((batch_vectors.shape[0], effective_dim), dtype=batch_vectors.dtype)
                padded[:, :orig_dim] = batch_vectors
            elif orig_dim > effective_dim:
                padded = batch_vectors[:, :effective_dim]
            else:
                padded = batch_vectors
            vectors.extend(padded.tolist())

        await cache_set_json(cache_key, vectors, ttl_seconds=24 * 3600)
        return vectors

    # Gemini embeddings (paid/quota-limited).
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini not configured (GEMINI_API_KEY missing)")

    model_name = settings.gemini_embedding_model
    # google-generativeai embed_content expects fully-qualified model name.
    # Example: "models/text-embedding-004"
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    # Compatibility: some SDK/API combos don't support text-embedding-004 via
    # embedContent; gemini-embedding-001 is available and supported.
    if model_name.endswith("text-embedding-004"):
        model_name = "models/gemini-embedding-001"
    # At this point model_name is guaranteed to start with "models/".
    cache_key = make_cache_key(
        "embeddings:gemini",
        model_name,
        str(effective_dim),
        _hash_text("||".join(texts)),
    )
    cached = await cache_get_json(cache_key)
    if cached:
        return cached

    # Batch embed_content requests to drastically reduce API call count.
    # Signature supports `content` as an iterable.
    batch_size = int(os.getenv("GEMINI_EMBED_BATCH_SIZE", "32"))
    vectors: List[List[float]] = []
    texts_list = list(texts)
    for i in range(0, len(texts_list), batch_size):
        chunk = texts_list[i : i + batch_size]
        result = genai.embed_content(model=model_name, content=chunk)
        # `result["embedding"]` is a list[embedding] when content is a list.
        vectors.extend(
            [
                (v[:effective_dim] if len(v) >= effective_dim else v + [0.0] * (effective_dim - len(v)))
                for v in result["embedding"]
            ]
        )

    await cache_set_json(cache_key, vectors, ttl_seconds=24 * 3600)
    return vectors


async def embed_and_store_documents(
    db: AsyncSession, documents: Iterable[Document]
) -> None:
    docs = list(documents)
    contents = [d.content for d in docs]
    vectors = await embed_texts(contents)
    for doc, vec in zip(docs, vectors):
        doc.embedding = vec
    await db.flush()


async def embed_and_store_images(
    db: AsyncSession, images: Iterable[DocumentImage]
) -> None:
    # Placeholder: in a full implementation, call CLIP/BLIP to embed images or captions.
    # Here we only embed captions if available.
    imgs = [img for img in images if img.caption]
    if not imgs:
        return
    captions = [img.caption for img in imgs]  # type: ignore[arg-type]
    vectors = await embed_texts(captions, target_dim=settings.pgvector_image_embedding_dim)
    for img, vec in zip(imgs, vectors):
        img.embedding = vec
    await db.flush()

