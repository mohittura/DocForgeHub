"""
api — FastAPI backend for DocForgeHub and CiteRagLab.

Submodules:
  - main: FastAPI app with document generation and QA endpoints
  - db: MongoDB async client singleton
  - helpers: Notion API utilities
  - notion_publisher: Markdown to Notion blocks conversion
  - redis_cache: Async Redis wrapper
"""