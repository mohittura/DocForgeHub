"""
rag/pipeline/redis_cache_rag.py

Redis caching layer for CiteRagLab.

Provides three distinct namespaces:
  1. Retrieval cache  — query_hash → chunks list   (TTL 10 min)
  2. Session context  — session_id → chat history  (TTL 24 h)
  3. Rate limiting    — Notion read counter         (100 / minute)

Requires a locally installed Redis server — no Docker needed.
Install Redis:
    Ubuntu:  sudo apt install redis-server && sudo service redis start
    macOS:   brew install redis && brew services start redis
    Windows: winget install Redis.Redis

Falls back gracefully: if Redis is unavailable all cache operations
are silent no-ops so the pipeline continues to function (uncached).
"""

import json
import hashlib
import logging
import os
from typing import Any, Optional
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.pipeline.redis_cache_rag")

REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_RETRIEVAL  = 600    # 10 minutes — retrieval results
TTL_SESSION    = 86400  # 24 hours   — chat history
RATE_LIMIT_RPM = 100    # max Notion reads per minute

_client = None   # async redis client singleton


async def _get_client():
    """
    Return a connected async Redis client, or None if Redis is unavailable.
    Attempts connection once and caches the result — avoids repeated
    connection attempts on every cache call.
    """
    global _client
    if _client is not None:
        return _client
    try:
        
        c = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        await c.ping()
        _client = c
        logger.info("✅ RAG Redis connected at %s", REDIS_URL)
        return _client
    except Exception as err:
        logger.warning(
            "⚠️  RAG Redis unavailable (%s) — all cache operations are no-ops",
            err,
        )
        return None


# ── Retrieval cache ──────────────────────────────────────────────────────────

def _retrieval_key(query: str, filters: Optional[dict]) -> str:
    """Deterministic cache key from query + filters."""
    raw = json.dumps({"q": query, "f": filters or {}}, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"rag:retrieval:{digest}"


async def get_retrieval_cache(
    query: str,
    filters: Optional[dict] = None,
) -> Optional[list]:
    """
    Return cached chunks for (query, filters) or None on miss / unavailability.
    """
    client = await _get_client()
    if not client:
        return None
    try:
        raw = await client.get(_retrieval_key(query, filters))
        if raw:
            logger.info("✅ Cache HIT  retrieval key='%s…'", _retrieval_key(query, filters)[:20])
            return json.loads(raw)
        logger.info("❌ Cache MISS retrieval key='%s…'", _retrieval_key(query, filters)[:20])
        return None
    except Exception as err:
        logger.warning("Redis retrieval GET error: %s", err)
        return None


async def set_retrieval_cache(
    query: str,
    filters: Optional[dict],
    chunks: list,
) -> None:
    """Store chunks in the retrieval cache with TTL_RETRIEVAL."""
    client = await _get_client()
    if not client:
        return
    try:
        key = _retrieval_key(query, filters)
        await client.set(key, json.dumps(chunks), ex=TTL_RETRIEVAL)
        logger.info(
            "💾 Cache SET  retrieval key='%s…'  ttl=%ds  chunks=%d",
            key[:20], TTL_RETRIEVAL, len(chunks),
        )
    except Exception as err:
        logger.warning("Redis retrieval SET error: %s", err)


# ── Session context ──────────────────────────────────────────────────────────

async def get_session_history(session_id: str) -> list[dict]:
    """
    Return the chat history list for session_id, or [] on miss.
    Each entry is {role: str, content: str}.
    """
    client = await _get_client()
    if not client:
        return []
    try:
        raw = await client.get(f"rag:session:{session_id}")
        if raw:
            history = json.loads(raw)
            logger.info(
                "✅ Cache HIT  session=%s  turns=%d",
                session_id, len(history),
            )
            return history
        logger.info("❌ Cache MISS session=%s", session_id)
        return []
    except Exception as err:
        logger.warning("Redis session GET error (session=%s): %s", session_id, err)
        return []


async def set_session_history(session_id: str, history: list[dict]) -> None:
    """Persist updated chat history for session_id with TTL_SESSION."""
    client = await _get_client()
    if not client:
        return
    try:
        await client.set(
            f"rag:session:{session_id}",
            json.dumps(history),
            ex=TTL_SESSION,
        )
        logger.info(
            "💾 Cache SET  session=%s  turns=%d  ttl=%ds",
            session_id, len(history), TTL_SESSION,
        )
    except Exception as err:
        logger.warning("Redis session SET error (session=%s): %s", session_id, err)


async def delete_session(session_id: str) -> None:
    """Delete a session's chat history from Redis."""
    client = await _get_client()
    if not client:
        return
    try:
        await client.delete(f"rag:session:{session_id}")
        logger.info("🗑️  Cache DEL  session=%s", session_id)
    except Exception as err:
        logger.warning("Redis session DEL error (session=%s): %s", session_id, err)


# ── Rate limiting ────────────────────────────────────────────────────────────

async def check_notion_rate_limit() -> bool:
    """
    Increment the Notion read counter for the current minute.
    Returns True if under RATE_LIMIT_RPM, False if the limit is exceeded.
    Falls back to True (permissive) when Redis is unavailable.
    """
    client = await _get_client()
    if not client:
        return True
    try:
        key   = "rag:notion:reads"
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, 60)
        if count > RATE_LIMIT_RPM:
            logger.warning(
                "⚠️  Notion rate limit exceeded — %d reads this minute (limit=%d)",
                count, RATE_LIMIT_RPM,
            )
            return False
        logger.info("ℹ️  Notion rate limit: %d/%d reads this minute", count, RATE_LIMIT_RPM)
        return True
    except Exception as err:
        logger.warning("Redis rate-limit check error: %s", err)
        return True


# ── Shutdown ─────────────────────────────────────────────────────────────────

async def close_rag_redis() -> None:
    """Gracefully close the Redis connection (call from lifespan shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
            logger.info("Redis RAG connection closed")
        except Exception:
            pass
        _client = None