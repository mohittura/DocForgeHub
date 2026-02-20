"""
Automation to process all notion_documents JSON files and push to MongoDB Atlas.

- Fetches exact department names from GET /departments
- Fetches exact document_type & document_name strings from GET /document-types
- API is the ONLY source of truth for department, document_type, document_name
- Local filenames are used ONLY to find the right file to read — never written to DB
- Adds "order" field to each section/subsection
- Pushes to `required_section` collection in MongoDB Atlas
"""

import os
import json
import requests
from pymongo import MongoClient

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MONGO_URI  = ""
DB_NAME    = "document_automation"
COLLECTION = "required_section"
API_BASE   = "http://localhost:8000"
ROOT_DIR   = "../document_and_questions/notion_documents"
# ─────────────────────────────────────────────────────────────────────────────


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def folder_to_normalized(folder_name: str) -> str:
    parts = folder_name.split("_", 1)
    base = parts[1] if len(parts) == 2 and parts[0].rstrip(".").isdigit() else folder_name
    return normalize(base.replace("_", " "))


def filename_to_normalized(file_name: str) -> str:
    return normalize(file_name.replace(".json", "").replace("_", " "))


def best_match(raw_file: str, doc_type_lookup: dict) -> dict | None:
    """Find the best matching API document for a normalized filename."""
    # 1. Exact
    if raw_file in doc_type_lookup:
        return doc_type_lookup[raw_file]

    file_tokens = set(raw_file.split())
    best_score  = 0
    best_entry  = None

    for api_name, doc_meta in doc_type_lookup.items():
        api_tokens = set(api_name.split())

        # 2. Substring containment
        if raw_file in api_name or api_name in raw_file:
            return doc_meta

        # 3. Jaccard token overlap
        intersection = file_tokens & api_tokens
        union        = file_tokens | api_tokens
        score        = len(intersection) / len(union) if union else 0
        if score > best_score:
            best_score = score
            best_entry = doc_meta

    return best_entry if best_score >= 0.5 else None


def fetch_departments() -> list:
    resp = requests.get(f"{API_BASE}/departments")
    resp.raise_for_status()
    return resp.json()["departments"]


def fetch_document_types(department_name: str) -> list:
    resp = requests.get(f"{API_BASE}/document-types", params={"department": department_name})
    resp.raise_for_status()
    return resp.json()["document_types"]


def add_order(document: dict) -> dict:
    for sec_idx, section in enumerate(document.get("sections", [])):
        section["order"] = sec_idx + 1
        for sub_idx, subsection in enumerate(section.get("subsections", [])):
            subsection["order"] = sub_idx + 1
    return document


def process_and_push():
    client     = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]

    print("Fetching departments from API...")
    dept_lookup = {
        normalize(d["name"]): d
        for d in fetch_departments()
    }
    print(f"  Found {len(dept_lookup)} departments\n")

    total_inserted = 0
    total_skipped  = 0

    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        dept_obj = dept_lookup.get(folder_to_normalized(folder_name))
        if not dept_obj:
            print(f"⚠  No API match for folder '{folder_name}' — skipping")
            continue

        dept_name = dept_obj["name"]
        print(f"📂 [{dept_name}]")

        try:
            api_doc_types = fetch_document_types(dept_name)
        except requests.HTTPError as e:
            print(f"   ⚠  Could not fetch document types: {e} — skipping")
            continue

        # API names → metadata lookup (normalized for matching only)
        doc_type_lookup = {normalize(d["document_name"]): d for d in api_doc_types}

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".json"):
                continue

            # Match file to API entry — filename used ONLY for lookup, never saved
            doc_meta = best_match(filename_to_normalized(file_name), doc_type_lookup)
            if not doc_meta:
                print(f"   ⚠  No API match for '{file_name}' — skipping")
                total_skipped += 1
                continue

            # Parse JSON
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                raw = f.read().strip()
            try:
                document = json.loads(raw)
            except json.JSONDecodeError:
                document, _ = json.JSONDecoder().raw_decode(raw)

            # Build document — everything comes from the API
            document = add_order(document)
            document["document_type"] = doc_meta["document_type"]   # from API
            document["document_name"] = doc_meta["document_name"]   # from API
            document["department"]    = dept_name                    # from API
            document.pop("_id", None)

            collection.insert_one(document)
            document.pop("_id", None)

            total_inserted += 1
            print(f"   ✓  {file_name}  →  '{doc_meta['document_name']}'")

    print(f"\n✅ Done. Inserted: {total_inserted}  |  Skipped: {total_skipped}")
    client.close()


if __name__ == "__main__":
    process_and_push()