"""
rag/retrieval/milvus_client_rag.py

Milvus vector-store client for CiteRagLab.

Uses milvus-lite (pip install milvus-lite) — runs fully in-process,
zero external services or Docker required.
Data is persisted to the local file at MILVUS_URI.

To migrate to a full Milvus server, change MILVUS_URI in .env from a
file path (e.g. "./rag_data/milvus.db") to a server address
(e.g. "http://localhost:19530") — no other code changes needed.

Collection schema  (matches DocForge Hub Library Notion database columns)
─────────────────────────────────────────────────────────────────────────
    id          INT64        primary key, auto
    embedding   FLOAT_VECTOR(1536)   text-embedding-3-small
    chunk_text  VARCHAR(4096)        extracted text content
    doc_id      VARCHAR(256)         Notion page UUID (same as page_id)
    title       VARCHAR(512)         database row Title column
    section     VARCHAR(256)         most recent heading above this chunk
    industry    VARCHAR(128)         database row Industry (select) column
    doc_type    VARCHAR(128)         database row Type (select) column
    version     VARCHAR(32)          database row Version (rich_text) column
    tags        VARCHAR(512)         database row tags (multi_select), joined as "tag1,tag2"
    page_id     VARCHAR(128)         Notion page UUID
    block_range VARCHAR(64)          e.g. "12-18" — blocks this chunk covers

Index: HNSW · COSINE · M=16 · efConstruction=200
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.retrieval.milvus_client_rag")

# ── Connection config ─────────────────────────────────────────────────────────
MILVUS_URI      = os.getenv("MILVUS_URI", "./rag_data/milvus.db")
COLLECTION_NAME = "notion_documents"
EMBEDDING_DIM   = 1536   # text-embedding-3-small output dimension

# ── Field name constants ──────────────────────────────────────────────────────
FIELD_ID          = "id"
FIELD_EMBEDDING   = "embedding"
FIELD_CHUNK_TEXT  = "chunk_text"
FIELD_DOC_ID      = "doc_id"
FIELD_TITLE       = "title"
FIELD_SECTION     = "section"
FIELD_INDUSTRY    = "industry"
FIELD_DOC_TYPE    = "doc_type"
FIELD_VERSION     = "version"
FIELD_TAGS        = "tags"          # multi_select column — stored as comma-joined string
FIELD_PAGE_ID     = "page_id"
FIELD_BLOCK_RANGE = "block_range"

_collection = None   # module-level singleton


def _tags_to_str(tags: list[str] | str | None) -> str:
    """
    Normalise the tags value coming from the ingestion pipeline.

    The Notion loader returns tags as list[str].
    We store them as a comma-joined VARCHAR so Milvus can filter with
    a simple string contains expression.
    """
    if tags is None:
        return ""
    if isinstance(tags, list):
        return ",".join(t.strip() for t in tags if t.strip())
    return str(tags).strip()


def get_collection():
    """
    Return the Milvus collection, creating it with an HNSW index if it
    does not yet exist.  milvus-lite starts the engine automatically on
    first connection — no external service needed.
    """
    global _collection

    if _collection is not None:
        logger.info("✅ Returning cached Milvus collection '%s'", COLLECTION_NAME)
        return _collection

    logger.info("🔌 Connecting to Milvus at URI='%s'", MILVUS_URI)

    # Create the data directory for file-based URIs
    if not MILVUS_URI.startswith("http"):
        data_dir = os.path.dirname(os.path.abspath(MILVUS_URI))
        os.makedirs(data_dir, exist_ok=True)
        logger.info("   📁 Milvus data directory: %s", data_dir)

    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

    connections.connect(uri=MILVUS_URI)
    logger.info("   ✅ Connected to Milvus (uri=%s)", MILVUS_URI)

    if utility.has_collection(COLLECTION_NAME):
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        logger.info(
            "   ✅ Loaded existing collection '%s' — %d entities",
            COLLECTION_NAME,
            _collection.num_entities,
        )
        return _collection

    # ── Create collection schema ──────────────────────────────────────────────
    logger.info("   📐 Creating collection '%s' with HNSW index…", COLLECTION_NAME)
    fields = [
        FieldSchema(name=FIELD_ID,          dtype=DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema(name=FIELD_EMBEDDING,   dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name=FIELD_CHUNK_TEXT,  dtype=DataType.VARCHAR,      max_length=4096),
        FieldSchema(name=FIELD_DOC_ID,      dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name=FIELD_TITLE,       dtype=DataType.VARCHAR,      max_length=512),
        FieldSchema(name=FIELD_SECTION,     dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name=FIELD_INDUSTRY,    dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name=FIELD_DOC_TYPE,    dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name=FIELD_VERSION,     dtype=DataType.VARCHAR,      max_length=32),
        FieldSchema(name=FIELD_TAGS,        dtype=DataType.VARCHAR,      max_length=512),
        FieldSchema(name=FIELD_PAGE_ID,     dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name=FIELD_BLOCK_RANGE, dtype=DataType.VARCHAR,      max_length=64),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="CiteRagLab — DocForge Hub Library document chunks",
    )
    _collection = Collection(name=COLLECTION_NAME, schema=schema)

    # ── HNSW index ────────────────────────────────────────────────────────────
    index_params = {
        "metric_type": "COSINE",
        "index_type":  "HNSW",
        "params":      {"M": 16, "efConstruction": 200},
    }
    _collection.create_index(field_name=FIELD_EMBEDDING, index_params=index_params)
    _collection.load()
    logger.info(
        "   ✅ Collection '%s' created — HNSW index (M=16, ef=200, COSINE)",
        COLLECTION_NAME,
    )
    return _collection


def insert_chunks(chunks: list[dict]) -> int:
    """
    Insert a list of chunk dicts into the Milvus collection.

    Required keys per chunk dict:
        embedding   (list[float])
        chunk_text  (str)
        doc_id      (str)
        title       (str)
        section     (str)
        industry    (str)
        doc_type    (str)
        version     (str)
        tags        (list[str] or str — the multi_select tags from Notion)
        page_id     (str)
        block_range (str)

    Returns the number of chunks inserted.
    """
    if not chunks:
        logger.warning("   ⚠️  insert_chunks: called with empty list — skipping")
        return 0

    logger.info(
        "💾 Inserting %d chunks into Milvus collection '%s'…",
        len(chunks), COLLECTION_NAME,
    )

    collection = get_collection()

    # Milvus inserts column-by-column
    data = [
        [c["embedding"]              for c in chunks],   # FIELD_EMBEDDING
        [c.get("chunk_text",  "")    for c in chunks],   # FIELD_CHUNK_TEXT
        [c.get("doc_id",      "")    for c in chunks],   # FIELD_DOC_ID
        [c.get("title",       "")    for c in chunks],   # FIELD_TITLE
        [c.get("section",     "")    for c in chunks],   # FIELD_SECTION
        [c.get("industry",    "")    for c in chunks],   # FIELD_INDUSTRY
        [c.get("doc_type",    "")    for c in chunks],   # FIELD_DOC_TYPE
        [c.get("version",     "")    for c in chunks],   # FIELD_VERSION
        [_tags_to_str(c.get("tags")) for c in chunks],   # FIELD_TAGS
        [c.get("page_id",     "")    for c in chunks],   # FIELD_PAGE_ID
        [c.get("block_range", "")    for c in chunks],   # FIELD_BLOCK_RANGE
    ]

    collection.insert(data)
    collection.flush()

    logger.info(
        "   ✅ Inserted %d chunks — collection now has %d total entities",
        len(chunks),
        collection.num_entities,
    )
    return len(chunks)


def search_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Cosine-similarity vector search against the collection.

    filters (optional dict) supported keys:
        industry  — exact match on the Industry select column
        doc_type  — exact match on the Type select column
        version   — exact match on the Version column
        tags      — substring match: returns chunks whose tags string
                    contains the given tag value

    Returns a ranked list of chunk dicts, each with:
        chunk_text, doc_id, title, section, industry,
        doc_type, version, tags, page_id, block_range, score
    """
    collection = get_collection()

    # ── Build Milvus boolean expression ──────────────────────────────────────
    expr_parts = []
    if filters:
        # Exact-match fields
        for field_name, filter_key in [
            (FIELD_INDUSTRY, "industry"),
            (FIELD_DOC_TYPE, "doc_type"),
            (FIELD_VERSION,  "version"),
        ]:
            val = filters.get(filter_key, "")
            if val and str(val).strip():
                expr_parts.append(f'{field_name} == "{str(val).strip()}"')

        # Tags — stored as comma-joined string, use LIKE for substring match
        # e.g. tags filter "HR" matches "Offer Letter Templates,HR"
        tag_filter = filters.get("tags", "")
        if tag_filter and str(tag_filter).strip():
            expr_parts.append(f'{FIELD_TAGS} like "%{str(tag_filter).strip()}%"')

    expr = " && ".join(expr_parts) if expr_parts else None

    logger.info(
        "🔍 search_chunks — top_k=%d  expr='%s'",
        top_k,
        expr or "(none)",
    )

    output_fields = [
        FIELD_CHUNK_TEXT,
        FIELD_DOC_ID,
        FIELD_TITLE,
        FIELD_SECTION,
        FIELD_INDUSTRY,
        FIELD_DOC_TYPE,
        FIELD_VERSION,
        FIELD_TAGS,
        FIELD_PAGE_ID,
        FIELD_BLOCK_RANGE,
    ]
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    results = collection.search(
        data=[query_embedding],
        anns_field=FIELD_EMBEDDING,
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields,
    )

    hits = []
    for hit in results[0]:
        entity = hit.entity
        # Convert stored comma-joined tags string back to a list for consumers
        raw_tags = entity.get(FIELD_TAGS, "") or ""
        tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]

        hits.append({
            "chunk_text":  entity.get(FIELD_CHUNK_TEXT,  ""),
            "doc_id":      entity.get(FIELD_DOC_ID,      ""),
            "title":       entity.get(FIELD_TITLE,       ""),
            "section":     entity.get(FIELD_SECTION,     ""),
            "industry":    entity.get(FIELD_INDUSTRY,    ""),
            "doc_type":    entity.get(FIELD_DOC_TYPE,    ""),
            "version":     entity.get(FIELD_VERSION,     ""),
            "tags":        tags_list,
            "page_id":     entity.get(FIELD_PAGE_ID,     ""),
            "block_range": entity.get(FIELD_BLOCK_RANGE, ""),
            "score":       round(float(hit.score), 4),
        })

    logger.info(
        "   ✅ search_chunks — %d hits returned (top score: %.4f)",
        len(hits),
        hits[0]["score"] if hits else 0.0,
    )
    return hits


def drop_collection() -> None:
    """
    Drop the entire collection (use before a full re-ingest from scratch).
    Resets the singleton so the next get_collection() call recreates it.
    """
    global _collection
    logger.warning("🗑️  Dropping collection '%s'…", COLLECTION_NAME)

    from pymilvus import connections, utility
    connections.connect(uri=MILVUS_URI)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.warning("   ✅ Collection '%s' dropped", COLLECTION_NAME)
    else:
        logger.info(
            "   ℹ️  Collection '%s' did not exist — nothing to drop",
            COLLECTION_NAME,
        )

    _collection = None