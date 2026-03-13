"""
Redis caching utility for DocForge Hub.

Provides a thin async wrapper around redis.asyncio with JSON serialisation.
Falls back gracefully — if Redis is unavailable, all cache calls are no-ops
so the app continues to function without caching.

Usage:
    from api.redis_cache import get_cache, set_cache, delete_cache, flush_prefix

    data = await get_cache("departments")
    if data is None:
        data = await fetch_from_mongo()
        await set_cache("departments", data, ttl=300)
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger("api.redis_cache")

# ── Lazy client — imported only once ────────────────────────────────────────
_redis_client = None

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DEFAULT_TTL = 3600  


async def _get_client():
    """Return a connected async Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as aioredis  # pip install redis
        client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        await client.ping()
        _redis_client = client
        logger.info("✅ Redis connected at %s", REDIS_URL)
        return _redis_client
    except Exception as exc:
        logger.warning("⚠️  Redis unavailable (%s) — caching disabled", exc)
        return None


async def get_cache(key: str) -> Any | None:
    """
    Retrieve a JSON-deserialised value from Redis.
    Returns None on cache miss or Redis unavailability.
    """
    client = await _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            logger.info("❌ Cache MISS key=%s", key)
            return None
        logger.info("✅ Cache HIT  key=%s", key)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis GET error for key=%s: %s", key, exc)
        return None


async def set_cache(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """
    Serialise value to JSON and store in Redis with a TTL (seconds).
    Silently skips if Redis is unavailable.
    """
    client = await _get_client()
    if client is None:
        return
    try:
        await client.set(key, json.dumps(value), ex=ttl)
        logger.info("💾 Cache SET  key=%s  ttl=%ds", key, ttl)
    except Exception as exc:
        logger.warning("Redis SET error for key=%s: %s", key, exc)


async def delete_cache(key: str) -> None:
    """Delete a single cache key (e.g. after a write invalidates stale data)."""
    client = await _get_client()
    if client is None:
        return
    try:
        await client.delete(key)
        logger.info("🗑️  Cache DEL  key=%s", key)
    except Exception as exc:
        logger.warning("Redis DEL error for key=%s: %s", key, exc)


async def flush_prefix(prefix: str) -> int:
    """
    Delete all keys that start with `prefix`.
    Returns the number of keys deleted.
    Useful for bulk-invalidating a family of keys (e.g. "doc_types:*").
    """
    client = await _get_client()
    if client is None:
        return 0
    try:
        keys = await client.keys(f"{prefix}*")
        if keys:
            await client.delete(*keys)
            logger.info("🗑️  Cache FLUSH prefix=%s  deleted=%d", prefix, len(keys))
            return len(keys)
        return 0
    except Exception as exc:
        logger.warning("Redis FLUSH error for prefix=%s: %s", prefix, exc)
        return 0


async def close_redis() -> None:
    """Gracefully close the Redis connection (call from lifespan shutdown)."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
            logger.info("Redis connection closed.")
        except Exception:
            pass
        _redis_client = None