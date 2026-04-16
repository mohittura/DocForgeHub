"""
rag/retrieval/milvus_client_rag.py

Milvus vector-store client for CiteRagLab.

Uses milvus-lite (pymilvus[milvus_lite]==2.5.10) — runs fully in-process,
zero external services or Docker required.

milvus-lite limitations vs full Milvus server
──────────────────────────────────────────────
  • Supported index types : FLAT, IVF_FLAT, AUTOINDEX  (no HNSW, no sparse)
  • No SPARSE_FLOAT_VECTOR field type support
  • No hybrid_search / BM25 support

Therefore this file uses dense-only vector search with AUTOINDEX + COSINE.
The RRF fusion and BM25 sparse arm are not used with milvus-lite.
When you migrate to a full Milvus server, swap AUTOINDEX → HNSW and
add the sparse field + hybrid_search back.

Collection schema  (matches DocForge Hub Library Notion database columns)
─────────────────────────────────────────────────────────────────────────
    id          INT64              primary key, auto-generated
    embedding   FLOAT_VECTOR(3072) dense vector — text-embedding-3-large
    chunk_text  VARCHAR(8192)      extracted text content
    doc_id      VARCHAR(256)       Notion page UUID
    title       VARCHAR(512)       database Title column
    section     VARCHAR(256)       most recent heading above this chunk
    industry    VARCHAR(128)       database Industry (select) column
    doc_type    VARCHAR(128)       database Type (select) column
    version     VARCHAR(32)        database Version (rich_text) column
    tags        VARCHAR(512)       database tags (multi_select), comma-joined
    page_id     VARCHAR(128)       Notion page UUID
    block_range VARCHAR(64)        e.g. "12-18"

Index: AUTOINDEX · COSINE  (milvus-lite compatible)
"""

import os
import sys
import logging
from types import ModuleType
from typing import Optional
from dotenv import load_dotenv
from importlib.metadata import version as _meta_version, PackageNotFoundError
from pymilvus import connections
from pymilvus import (
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

# ── milvus_lite pkg_resources workaround ─────────────────────────────────────
# milvus_lite/__init__.py imports pkg_resources purely to read its version.
# We inject a minimal mock so it works without setuptools being installed.
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources  # noqa: F401 #type: ignore
    except ModuleNotFoundError:
        

        _mock_pkg = ModuleType("pkg_resources")

        class _MockDist:
            def __init__(self, v: str):
                self.version = v

        class _DistributionNotFound(Exception):
            pass

        def _get_distribution(name: str) -> _MockDist:
            try:
                return _MockDist(_meta_version(name))
            except PackageNotFoundError:
                raise _DistributionNotFound(name)

        _mock_pkg.get_distribution     = _get_distribution          # type: ignore
        _mock_pkg.DistributionNotFound = _DistributionNotFound       # type: ignore
        sys.modules["pkg_resources"]   = _mock_pkg

load_dotenv()

logger = logging.getLogger("rag.retrieval.milvus_client_rag")

# ── Connection config ─────────────────────────────────────────────────────────
MILVUS_URI      = os.getenv("MILVUS_URI", "./rag_data/milvus.db")
COLLECTION_NAME = "notion_documents"
EMBEDDING_DIM   = 3072   # text-embedding-3-large output dimension

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
FIELD_TAGS        = "tags"
FIELD_PAGE_ID     = "page_id"
FIELD_BLOCK_RANGE = "block_range"

_collection = None   # module-level singleton
_connected  = False  # track whether connections.connect() has been called


def _ensure_connected() -> None:
    """Connect to Milvus exactly once per process."""
    global _connected
    if _connected:
        return

    

    logger.info("🔌 Connecting to Milvus at URI='%s'", MILVUS_URI)

    if not MILVUS_URI.startswith("http"):
        data_dir = os.path.dirname(os.path.abspath(MILVUS_URI))
        os.makedirs(data_dir, exist_ok=True)
        logger.info("   📁 Milvus data directory: %s", data_dir)

    connections.connect(uri=MILVUS_URI)
    _connected = True
    logger.info("   ✅ Connected to Milvus (uri=%s)", MILVUS_URI)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tags_to_str(tags: list[str] | str | None) -> str:
    if tags is None:
        return ""
    if isinstance(tags, list):
        return ",".join(t.strip() for t in tags if t.strip())
    return str(tags).strip()


def _build_filter_expr(filters: Optional[dict]) -> Optional[str]:
    if not filters:
        return None
    expr_parts = []
    for field_name, filter_key in [
        (FIELD_INDUSTRY, "industry"),
        (FIELD_DOC_TYPE, "doc_type"),
        (FIELD_VERSION,  "version"),
    ]:
        val = filters.get(filter_key, "")
        if val and str(val).strip():
            expr_parts.append(f'{field_name} == "{str(val).strip()}"')
    tag_filter = filters.get("tags", "")
    if tag_filter and str(tag_filter).strip():
        expr_parts.append(f'{FIELD_TAGS} like "%{str(tag_filter).strip()}%"')
    return " && ".join(expr_parts) if expr_parts else None


def _build_output_fields() -> list[str]:
    return [
        FIELD_CHUNK_TEXT, FIELD_DOC_ID, FIELD_TITLE, FIELD_SECTION,
        FIELD_INDUSTRY,   FIELD_DOC_TYPE, FIELD_VERSION,
        FIELD_TAGS,       FIELD_PAGE_ID,  FIELD_BLOCK_RANGE,
    ]


def _hit_to_dict(hit) -> dict:
    entity    = hit.entity
    raw_tags  = entity.get(FIELD_TAGS, "") or ""
    tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
    return {
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
    }


# ── Collection lifecycle ──────────────────────────────────────────────────────

def get_collection():
    """
    Return the Milvus collection, creating it with AUTOINDEX if it does
    not yet exist.  Connection is established once per process.
    """
    global _collection

    if _collection is not None:
        return _collection

    _ensure_connected()



    if utility.has_collection(COLLECTION_NAME):
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        logger.info(
            "   ✅ Loaded existing collection '%s' — %d entities",
            COLLECTION_NAME, _collection.num_entities,
        )
        return _collection

    # ── Schema ────────────────────────────────────────────────────────────────
    logger.info("   📐 Creating collection '%s'…", COLLECTION_NAME)
    fields = [
        FieldSchema(name=FIELD_ID,          dtype=DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema(name=FIELD_EMBEDDING,   dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name=FIELD_CHUNK_TEXT,  dtype=DataType.VARCHAR,      max_length=8192),
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

    # AUTOINDEX is the only ANN index supported by milvus-lite
    _collection.create_index(
        field_name=FIELD_EMBEDDING,
        index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
    )
    logger.info("   ✅ AUTOINDEX created (COSINE)")

    _collection.load()
    logger.info("   ✅ Collection '%s' created and loaded", COLLECTION_NAME)
    return _collection


# ── Ingestion ─────────────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]) -> int:
    """
    Insert chunk dicts into Milvus.

    Required keys per chunk: embedding, chunk_text, doc_id, title,
    section, industry, doc_type, version, tags, page_id, block_range.

    Returns the number of chunks inserted.
    """
    if not chunks:
        logger.warning("   ⚠️  insert_chunks: called with empty list — skipping")
        return 0

    logger.info(
        "💾 Inserting %d chunks into collection '%s'…",
        len(chunks), COLLECTION_NAME,
    )

    collection = get_collection()

    # Column order must match schema field order (id is auto, skip it)
    # chunk_text is truncated to 8000 chars as a safety net — Milvus enforces
    # max_length=8192 but we leave headroom for multi-byte characters.
    data = [
        [c["embedding"]                          for c in chunks],   # FIELD_EMBEDDING
        [c.get("chunk_text",  "")[:8000]         for c in chunks],   # FIELD_CHUNK_TEXT
        [c.get("doc_id",      "")                for c in chunks],   # FIELD_DOC_ID
        [c.get("title",       "")                for c in chunks],   # FIELD_TITLE
        [c.get("section",     "")                for c in chunks],   # FIELD_SECTION
        [c.get("industry",    "")                for c in chunks],   # FIELD_INDUSTRY
        [c.get("doc_type",    "")                for c in chunks],   # FIELD_DOC_TYPE
        [c.get("version",     "")                for c in chunks],   # FIELD_VERSION
        [_tags_to_str(c.get("tags"))             for c in chunks],   # FIELD_TAGS
        [c.get("page_id",     "")                for c in chunks],   # FIELD_PAGE_ID
        [c.get("block_range", "")                for c in chunks],   # FIELD_BLOCK_RANGE
    ]

    collection.insert(data)
    collection.flush()

    logger.info(
        "   ✅ Inserted %d chunks — collection now has %d total entities",
        len(chunks), collection.num_entities,
    )
    return len(chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def hybrid_search_chunks(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Dense vector search using AUTOINDEX + COSINE similarity.

    milvus-lite does not support sparse/BM25 or hybrid_search, so this
    performs dense-only search. The function signature keeps `query_text`
    for API compatibility with retriever_rag.py — it is not used here.

    When migrating to a full Milvus server, this function can be upgraded
    to true hybrid search (HNSW + BM25 + RRFRanker) by adding the
    sparse_vec field back to the schema and using collection.hybrid_search().

    Returns a ranked list of chunk dicts with a 'score' field.
    """
    collection    = get_collection()
    expr          = _build_filter_expr(filters)
    output_fields = _build_output_fields()

    logger.info(
        "🔍 search_chunks — top_k=%d  expr='%s'",
        top_k, expr or "(none)",
    )

    results = collection.search(
        data=[query_embedding],
        anns_field=FIELD_EMBEDDING,
        param={"metric_type": "COSINE"},
        limit=top_k,
        expr=expr,
        output_fields=output_fields,
    )

    hits = [_hit_to_dict(hit) for hit in results[0]]

    logger.info(
        "   ✅ search_chunks — %d hits returned (top score: %.4f)",
        len(hits),
        hits[0]["score"] if hits else 0.0,
    )
    return hits


def drop_collection() -> None:
    """
    Drop the entire collection. Use before a full re-ingest from scratch.
    Resets the singleton so the next get_collection() recreates it cleanly.
    """
    global _collection
    logger.warning("🗑️  Dropping collection '%s'…", COLLECTION_NAME)

    _ensure_connected()

    

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.warning("   ✅ Collection '%s' dropped", COLLECTION_NAME)
    else:
        logger.info("   ℹ️  Collection '%s' did not exist — nothing to drop", COLLECTION_NAME)

    _collection = None