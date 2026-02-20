"""
Automation to process all notion_documents JSON files and push to MongoDB Atlas.
- Adds "order" field to each section/subsection title
- Sets "department" from the parent folder name
- Pushes to `required_section` collection in MongoDB Atlas
"""

import os
import json
from pymongo import MongoClient

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb+srv://mohitturabit_db_user:yqAjPzqe5HuZ91Gl@cluster0.uzrpe8f.mongodb.net/"  # e.g. mongodb+srv://user:pass@cluster.mongodb.net/
DB_NAME = "document_automation"
COLLECTION_NAME = "required_section"

# Root folder containing department subfolders (e.g. 1._Product_Management, etc.)
ROOT_DIR = "../document_and_questions/notion_documents"
# ─────────────────────────────────────────────────────────────────────────────


def get_document_type(file_name: str) -> str:
    """
    Convert filename like 'Feature_prioritization_framework.json'
    → 'Feature prioritization framework'
    """
    name = file_name.replace(".json", "")
    return " ".join(name.replace("_", " ").split())


def get_department_name(folder_name: str) -> str:
    """
    Convert folder name like '1._Product_Management' → 'Product_Management'
    Strips the leading number prefix.
    """
    parts = folder_name.split("_", 1)
    # parts[0] is the number, parts[1] is the rest
    if len(parts) == 2 and parts[0].rstrip(".").isdigit():
        return " ".join(parts[1].replace("_", " ").split())
    return " ".join(folder_name.replace("_", " ").split())


def add_order(document: dict) -> dict:
    """
    Traverse sections and subsections, adding an 'order' field
    based on their position in the list (1-indexed).
    """
    sections = document.get("sections", [])
    for sec_idx, section in enumerate(sections):
        section["order"] = sec_idx + 1
        subsections = section.get("subsections", [])
        for sub_idx, subsection in enumerate(subsections):
            subsection["order"] = sub_idx + 1
    return document


def process_and_push():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    total_inserted = 0
    total_files = 0

    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        department = get_department_name(folder_name)

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(folder_path, file_name)
            total_files += 1

            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

            # Handle files that may contain duplicated JSON (as in your example)
            # Try parsing; if it fails, try splitting and taking the first valid JSON
            try:
                document = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt to extract the first valid JSON object
                decoder = json.JSONDecoder()
                document, _ = decoder.raw_decode(raw)

            # Add order fields
            document = add_order(document)

            # Derive document_type from filename and sync document_name to it
            document_type = get_document_type(file_name)
            document["document_type"] = document_type
            document["document_name"] = document_type

            # Add department (from folder), no other metadata
            document["department"] = department

            # Remove MongoDB _id if already present to avoid conflicts on re-runs
            document.pop("_id", None)

            collection.insert_one(document)

            # insert_one adds _id in-place; remove it from local dict (optional cleanup)
            document.pop("_id", None)

            total_inserted += 1
            print(f"  ✓ [{department}] {file_name}")

    print(f"\nDone. Processed {total_files} files, inserted {total_inserted} documents.")
    client.close()


if __name__ == "__main__":
    process_and_push()