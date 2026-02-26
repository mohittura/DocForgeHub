"""
Notion API helper functions for the DocForge Hub API.

Provides utilities for interacting with the Notion API, including
recursive page discovery and URL construction. These functions are
extracted from main.py to keep route handlers lean.
"""

import os
from typing import List, Dict
from notion_client import Client


# ── Notion client initialisation ─────────────────────────────────
notion_api_key = os.environ.get("NOTION_API_KEY")
if not notion_api_key:
    raise ValueError("notion api key not defined")

notion_client = Client(auth=notion_api_key)


def get_page_url_from_id(page_id: str) -> str:
    """
    Construct the Notion web URL from a page ID.

    Notion URLs use dash-free IDs, so we strip dashes from the API-provided ID.
    """
    simple_page_id = page_id.replace("-", "")
    return f"https://notion.so/{simple_page_id}"


def retrieve_all_child_pages_recursive(
    block_id: str,
    all_pages: List[Dict] = None,
) -> List[Dict]:
    """
    Recursively retrieve all child pages under a given Notion block ID.

    Each discovered page is appended to `all_pages` as a dict with keys:
        - id: the Notion page ID
        - title: the page title
        - url: the constructed Notion web URL

    Handles Notion API pagination automatically (page_size=100).
    """
    if all_pages is None:
        all_pages = []

    has_more = True
    next_cursor = None

    while has_more:
        try:
            response = notion_client.blocks.children.list(
                block_id=block_id,
                start_cursor=next_cursor,
                page_size=100,
            )
            for block_item in response["results"]:
                if block_item["type"] == "child_page":
                    page_id = block_item["id"]
                    page_title = block_item["child_page"]["title"]
                    page_url = get_page_url_from_id(page_id)
                    all_pages.append({
                        "id": page_id,
                        "title": page_title,
                        "url": page_url,
                    })
                    # Recursively discover children of this child page
                    retrieve_all_child_pages_recursive(page_id, all_pages)

            next_cursor = response.get("next_cursor")
            has_more = response.get("has_more")

        except Exception as error_msg:
            print(f"Error retrieving children for block {block_id}: {error_msg}")
            break

    return all_pages