import json
import logging
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis  # type: ignore[import-not-found]
except ModuleNotFoundError:
    redis = None  # type: ignore[assignment]
    logger.warning(
        "redis package not installed; caching is disabled. "
        "Install 'redis' to enable Redis-based caching."
    )

_redis_client: Optional["redis.Redis"] = None  # type: ignore[name-defined]
_redis_disabled: bool = False
_redis_warned: bool = False


def get_redis_client() -> Optional["redis.Redis"]:  # type: ignore[name-defined]
    if redis is None:
        return None
    global _redis_disabled
    if _redis_disabled:
        return None
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(  # type: ignore[attr-defined]
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
    return _redis_client


def make_cache_key(prefix: str, *parts: str) -> str:
    safe_parts = [prefix] + [p.replace(" ", "_") for p in parts]
    return ":".join(safe_parts)


async def cache_set_json(key: str, value: Any, ttl_seconds: int = 3600) -> None:
    client = get_redis_client()
    if client is None:
        return
    payload = json.dumps(value, ensure_ascii=False)
    try:
        await client.set(key, payload, ex=ttl_seconds)
    except Exception as e:  # pragma: no cover
        global _redis_disabled, _redis_warned
        _redis_disabled = True
        if not _redis_warned:
            logger.warning("Redis unreachable; disabling cache: %s", e)
            _redis_warned = True
        # If Redis isn't reachable (not started, wrong host/port, etc.)
        # we don't want to crash the whole API; just skip caching.
        return


async def cache_get_json(key: str) -> Any | None:
    client = get_redis_client()
    if client is None:
        return None
    try:
        data = await client.get(key)
    except Exception as e:  # pragma: no cover
        global _redis_disabled, _redis_warned
        _redis_disabled = True
        if not _redis_warned:
            logger.warning("Redis unreachable; disabling cache: %s", e)
            _redis_warned = True
        logger.warning("Redis get failed; treating as cache miss: %s", e)
        return None
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON from cache for key %s", key)
        return None

