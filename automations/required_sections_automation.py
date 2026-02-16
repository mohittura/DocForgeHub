"""
Automation: Upload Notion document schemas into MongoDB collection 'required_section'.

Copy-paste order (for presentation):
    Section 1 → Imports and logging setup
    Section 2 → Helper function (clean_department_name)
    Section 3 → Main class (NotionSectionsToMongo)
    Section 4 → CLI entry point (main)

What this script does:
    1. Scans the 'notion_documents/' folder for department sub-folders
       (e.g. "1._Product_Management", "2._Engineering__Software_Development")
    2. Derives a clean department name from the folder name
       (e.g. "1._Product_Management" → "Product Management")
    3. Reads every .json file inside each folder
    4. Upserts each document into the 'required_section' collection in MongoDB

Why?
    The agent needs the document schema (sections/structure) to generate
    professional documents.  Storing them in a MongoDB collection indexed
    by department + document_name makes lookups instant — no file scanning
    at runtime.

Run:
    python -m automations.notion_sections_to_mongo
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 1: Imports and Logging Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from pymongo import MongoClient, ASCENDING, errors
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 2: Helper Function — clean_department_name & derive_document_name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_department_name(folder_name: str) -> str:
    """
    Convert a folder name like '1._Product_Management' into 'Product Management'.

    Steps:
        1. Strip the leading number + dot + underscore  (e.g. '1._')
        2. Replace remaining underscores with spaces
        3. Collapse double-spaces caused by double-underscores
    """
    # Remove leading digits, dots, underscores  (e.g. "1._" or "10._")
    without_prefix = re.sub(r"^\d+\._", "", folder_name)

    # Replace underscores with spaces
    with_spaces = without_prefix.replace("_", " ")

    # Collapse multiple spaces into one
    clean_name = re.sub(r"\s+", " ", with_spaces).strip()

    return clean_name


def derive_document_name_from_filename(filename: str) -> str:
    """
    Derive document_name from filename to match document_qas and document_schemas collections.
    
    Example:
        "Go-to-market_alignment_document.json" → "Go-to-market alignment document"
    
    Steps:
        1. Remove .json extension
        2. Replace underscores with spaces
        3. Clean up any extra whitespace
    """
    # Remove .json extension
    name_without_ext = filename.replace(".json", "")
    
    # Replace underscores with spaces
    with_spaces = name_without_ext.replace("_", " ")
    
    # Collapse multiple spaces into one and strip
    clean_name = re.sub(r"\s+", " ", with_spaces).strip()
    
    return clean_name


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 3: Main Class — NotionSectionsToMongo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NotionSectionsToMongo:
    """
    Read Notion document JSON schemas from the file system and
    store them in the 'required_section' collection in MongoDB.

    Each document in the collection looks like:
    {
        "department":    "Product Management",
        "document_name": "Go-to-market alignment document",
        "sections":      [ ... ],          # as-is from the JSON file
        "_metadata": {
            "source_folder": "1._Product_Management",
            "source_file":   "Go-to-market_alignment_document.json",
            "stored_at":     "2026-02-14T09:45:00Z"
        }
    }
    """

    COLLECTION_NAME = "required_section"

    # ── 3a: Constructor — connect to MongoDB ─────────────────────

    def __init__(
        self,
        connection_string: str | None = None,
        database_name: str = "document_automation",
    ):
        if not connection_string:
            connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError(
                "MongoDB connection string not provided.  "
                "Set the MONGODB_CONNECTION_STRING env variable or pass it directly."
            )

        self.mongo_client = MongoClient(connection_string)
        self.database = self.mongo_client[database_name]
        self.collection = self.database[self.COLLECTION_NAME]

        logger.info("✅ Connected to MongoDB — database: %s", database_name)

        self._ensure_indexes()

    # ── 3b: Index creation ───────────────────────────────────────

    def _ensure_indexes(self):
        """
        Create a compound index on (department, document_name) so queries like
            db.required_section.find({department: "...", document_name: "..."})
        are fast.
        """
        try:
            self.collection.create_index(
                [("department", ASCENDING), ("document_name", ASCENDING)],
                unique=True,
                name="idx_department_document",
            )
            self.collection.create_index(
                [("department", ASCENDING)],
                name="idx_department",
            )
            logger.info("✅ Indexes created on '%s'", self.COLLECTION_NAME)
        except errors.OperationFailure as index_error:
            logger.warning("⚠️  Index creation warning: %s", index_error)

    # ── 3c: Upload a single JSON file ────────────────────────────

    def upload_single_file(
        self,
        json_file_path: str,
        department_name: str,
        source_folder: str,
    ) -> bool:
        """
        Read one JSON schema file and upsert it into the collection.

        Returns True on success, False on failure.
        """
        file_name = os.path.basename(json_file_path)

        try:
            with open(json_file_path, "r", encoding="utf-8") as file_handle:
                schema_data = json.load(file_handle)
        except (json.JSONDecodeError, OSError) as read_error:
            logger.error("   ❌ Cannot read %s: %s", file_name, read_error)
            return False

        # FIX: Derive document_name from filename instead of JSON content
        # This ensures consistency with document_qas and document_schemas collections
        document_name = derive_document_name_from_filename(file_name)
        sections = schema_data.get("sections", [])

        mongo_document = {
            "department": department_name,
            "document_name": document_name,
            "sections": sections,
            "_metadata": {
                "source_folder": source_folder,
                "source_file": file_name,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        try:
            # Upsert: update if (department, document_name) already exists
            self.collection.update_one(
                {"department": department_name, "document_name": document_name},
                {"$set": mongo_document},
                upsert=True,
            )
            logger.info("   ✅ %s", document_name)
            return True
        except Exception as db_error:
            logger.error("   ❌ Failed to store %s: %s", document_name, db_error)
            return False

    # ── 3d: Upload all files from a directory ────────────────────

    def upload_all_from_directory(self, notion_documents_dir: str) -> Dict[str, Any]:
        """
        Scan the given directory for department folders, read each JSON
        file inside, and upload to MongoDB.

        Returns a stats dictionary with counts.
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("📂 REQUIRED_SECTION — BATCH UPLOAD")
        logger.info("=" * 70)

        base_path = Path(notion_documents_dir)
        if not base_path.exists():
            logger.error("❌ Directory not found: %s", notion_documents_dir)
            return {"total": 0, "success": 0, "failed": 0}

        # Find all department sub-folders (sorted for predictable order)
        department_folders = sorted(
            [folder for folder in base_path.iterdir() if folder.is_dir() and not folder.name.startswith(".")]
        )

        logger.info("📁 Found %d department folders\n", len(department_folders))

        total_files_count = 0
        success_count = 0
        failed_count = 0

        for folder_index, department_folder in enumerate(department_folders, start=1):
            folder_name = department_folder.name
            department_name = clean_department_name(folder_name)

            logger.info("─" * 70)
            logger.info(
                "📂 [%d/%d] %s  →  '%s'",
                folder_index,
                len(department_folders),
                folder_name,
                department_name,
            )
            logger.info("─" * 70)

            # Find all JSON files in this department folder
            json_files = sorted(department_folder.glob("*.json"))

            if not json_files:
                logger.warning("   ⚠️  No JSON files found — skipping")
                continue

            for json_file in json_files:
                total_files_count += 1
                was_successful = self.upload_single_file(
                    json_file_path=str(json_file),
                    department_name=department_name,
                    source_folder=folder_name,
                )
                if was_successful:
                    success_count += 1
                else:
                    failed_count += 1

        # ── Final summary ────────────────────────────────────────
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎉 UPLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info("📊 Total files scanned : %d", total_files_count)
        logger.info("✅ Successfully stored : %d", success_count)
        logger.info("❌ Failed              : %d", failed_count)
        logger.info(
            "🗄️  Collection '%s' now has %d documents",
            self.COLLECTION_NAME,
            self.collection.count_documents({}),
        )
        logger.info("=" * 70)

        return {
            "total": total_files_count,
            "success": success_count,
            "failed": failed_count,
        }

    # ── 3e: Cleanup ──────────────────────────────────────────────

    def close(self):
        """Close the MongoDB connection."""
        self.mongo_client.close()
        logger.info("✅ MongoDB connection closed")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 4: CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Run the automation from the command line."""
    # Resolve the notion_documents directory relative to this project
    project_root = Path(__file__).resolve().parent.parent
    notion_documents_dir = project_root / "document_and_questions" / "notion_documents"

    logger.info("📂 Source directory: %s", notion_documents_dir)

    uploader = NotionSectionsToMongo()

    try:
        uploader.upload_all_from_directory(str(notion_documents_dir))
    finally:
        uploader.close()


if __name__ == "__main__":
    main()