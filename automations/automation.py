import json
import os
import re
import time
from notion_client import Client
from typing import List, Dict, Any, Optional

class NotionDocumentExtractor:
    def __init__(self, api_key: str, rate_limit_delay: float = 0.35):
        self.notion = Client(auth=api_key)
        self.rate_limit_delay = rate_limit_delay
        self.api_call_count = 0
    
    def _rate_limit(self):
        """Add delay between API calls to avoid rate limits"""
        time.sleep(self.rate_limit_delay)
        self.api_call_count += 1
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove invalid characters"""
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace(' ', '_')
        filename = filename[:200]
        return filename if filename else "untitled"
    
    def get_page_title(self, page_id: str) -> str:
        """Get the title of a page"""
        try:
            self._rate_limit()
            page = self.notion.pages.retrieve(page_id=page_id)
            properties = page.get('properties', {})
            
            for prop_name, prop_data in properties.items():
                if prop_data['type'] == 'title':
                    title_parts = prop_data['title']
                    if title_parts:
                        return ''.join([t['plain_text'] for t in title_parts])
            
            if 'title' in page:
                return ''.join([t['plain_text'] for t in page['title']])
            
            return "Untitled"
        except Exception as e:
            print(f"Error getting page title: {e}")
            return "Untitled"
    
    def get_pages_by_headings(self, parent_page_id: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all pages organized by the headings they appear under.
        Returns a dictionary: {heading_name: [list of pages under that heading]}
        """
        organized_pages = {}
        current_heading = "Uncategorized"
        
        try:
            self._rate_limit()
            # Get all blocks from the parent page
            blocks = []
            has_more = True
            start_cursor = None
            
            while has_more:
                if start_cursor:
                    response = self.notion.blocks.children.list(
                        block_id=parent_page_id,
                        start_cursor=start_cursor
                    )
                else:
                    response = self.notion.blocks.children.list(block_id=parent_page_id)
                
                blocks.extend(response['results'])
                has_more = response['has_more']
                start_cursor = response.get('next_cursor')
                self._rate_limit()
            
            # Parse blocks to find headings and child pages
            for block in blocks:
                block_type = block['type']
                
                # Check if it's a heading
                if block_type in ['heading_1', 'heading_2', 'heading_3']:
                    heading_text = ''.join([t['plain_text'] for t in block[block_type]['rich_text']])
                    if heading_text:
                        current_heading = heading_text
                        if current_heading not in organized_pages:
                            organized_pages[current_heading] = []
                
                # Check if it's a child page
                elif block_type == 'child_page':
                    page_id = block['id']
                    title = block['child_page'].get('title', 'Untitled')
                    
                    if current_heading not in organized_pages:
                        organized_pages[current_heading] = []
                    
                    organized_pages[current_heading].append({
                        'id': page_id,
                        'title': title
                    })
                
                # Check if it's a link to page (in databases)
                elif block_type == 'link_to_page':
                    if block['link_to_page']['type'] == 'page_id':
                        page_id = block['link_to_page']['page_id']
                        title = self.get_page_title(page_id)
                        
                        if current_heading not in organized_pages:
                            organized_pages[current_heading] = []
                        
                        organized_pages[current_heading].append({
                            'id': page_id,
                            'title': title
                        })
            
            # If parent is a database, get all pages from it
            if not organized_pages:
                try:
                    self._rate_limit()
                    has_more = True
                    start_cursor = None
                    
                    while has_more:
                        if start_cursor:
                            results = self.notion.databases.query(
                                database_id=parent_page_id,
                                start_cursor=start_cursor
                            )
                        else:
                            results = self.notion.databases.query(database_id=parent_page_id)
                        
                        for page in results['results']:
                            page_id = page['id']
                            title = self.get_page_title(page_id)
                            
                            # Try to get category from properties
                            properties = page.get('properties', {})
                            category = "All Documents"
                            
                            for prop_name, prop_data in properties.items():
                                if prop_data['type'] == 'select' and prop_data.get('select'):
                                    category = prop_data['select']['name']
                                    break
                            
                            if category not in organized_pages:
                                organized_pages[category] = []
                            
                            organized_pages[category].append({
                                'id': page_id,
                                'title': title
                            })
                        
                        has_more = results.get('has_more', False)
                        start_cursor = results.get('next_cursor')
                        self._rate_limit()
                        
                except Exception as e:
                    print(f"Not a database: {e}")
            
        except Exception as e:
            print(f"Error organizing pages by headings: {e}")
        
        return organized_pages
    
    def get_table_columns(self, table_block_id: str) -> List[str]:
        """Extract column headers from a table"""
        try:
            self._rate_limit()
            children = self.notion.blocks.children.list(block_id=table_block_id)
            columns = []
            
            if children['results']:
                first_row = children['results'][0]
                if first_row['type'] == 'table_row':
                    cells = first_row['table_row']['cells']
                    for cell in cells:
                        col_text = ''.join([t['plain_text'] for t in cell])
                        columns.append(col_text if col_text else f"Column {len(columns) + 1}")
            
            return columns
        except Exception as e:
            print(f"Error extracting table columns: {e}")
            return []
    
    def parse_blocks_to_sections(self, blocks: List[Dict]) -> List[Dict]:
        """Parse blocks into a hierarchical section structure"""
        sections = []
        current_section = None
        current_subsection = None
        
        for block in blocks:
            block_type = block['type']
            
            if block_type == 'heading_1':
                text = ''.join([t['plain_text'] for t in block[block_type]['rich_text']])
                
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'title': text,
                    'subsections': []
                }
                current_subsection = None
            
            elif block_type in ['heading_2', 'heading_3']:
                text = ''.join([t['plain_text'] for t in block[block_type]['rich_text']])
                
                if not current_section:
                    current_section = {
                        'title': text,
                        'subsections': []
                    }
                else:
                    current_subsection = {
                        'title': text,
                        'type': 'text'
                    }
                    current_section['subsections'].append(current_subsection)
            
            elif current_section:
                if not current_subsection and not current_section['subsections']:
                    current_subsection = {
                        'title': 'Description',
                        'type': 'text'
                    }
                    current_section['subsections'].append(current_subsection)
                
                if current_section['subsections']:
                    last_subsection = current_section['subsections'][-1]
                    
                    if block_type in ['bulleted_list_item', 'numbered_list_item']:
                        last_subsection['type'] = 'list'
                    
                    elif block_type == 'table':
                        last_subsection['type'] = 'table'
                        columns = self.get_table_columns(block['id'])
                        if columns:
                            last_subsection['columns'] = columns
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def extract_page_metadata(self, page_id: str) -> Dict[str, Any]:
        """Extract metadata from page properties"""
        self._rate_limit()
        page = self.notion.pages.retrieve(page_id=page_id)
        properties = page.get('properties', {})
        
        metadata = {}
        
        for prop_name, prop_data in properties.items():
            prop_type = prop_data['type']
            key = prop_name.lower().replace(' ', '_')
            
            if prop_type == 'title':
                title_text = ''.join([t['plain_text'] for t in prop_data['title']])
                metadata['document_name'] = title_text
            elif prop_type == 'rich_text':
                text = ''.join([t['plain_text'] for t in prop_data['rich_text']])
                if text:
                    metadata[key] = text
            elif prop_type == 'select' and prop_data['select']:
                metadata[key] = prop_data['select']['name']
            elif prop_type == 'date' and prop_data['date']:
                metadata[key] = prop_data['date']['start']
            elif prop_type == 'number' and prop_data['number'] is not None:
                metadata[key] = prop_data['number']
        
        return metadata
    
    def extract_document_structure(self, page_id: str) -> Dict[str, Any]:
        """Main function to extract document structure"""
        metadata = self.extract_page_metadata(page_id)
        
        blocks = []
        has_more = True
        start_cursor = None
        
        while has_more:
            self._rate_limit()
            if start_cursor:
                response = self.notion.blocks.children.list(
                    block_id=page_id,
                    start_cursor=start_cursor
                )
            else:
                response = self.notion.blocks.children.list(block_id=page_id)
            
            blocks.extend(response['results'])
            has_more = response['has_more']
            start_cursor = response.get('next_cursor')
        
        sections = self.parse_blocks_to_sections(blocks)
        
        document = {
            **metadata,
            'sections': sections
        }
        
        return document
    
    def generate_preview(self, documents: List[Dict[str, Any]], heading: str) -> str:
        """Generate a preview summary of processed documents"""
        preview = f"\n{'='*70}\n"
        preview += f"📁 FOLDER: {heading}\n"
        preview += f"{'='*70}\n\n"
        
        for idx, doc in enumerate(documents, 1):
            doc_name = doc.get('document_name', 'Untitled')
            sections = doc.get('sections', [])
            
            preview += f"{idx}. {doc_name}\n"
            preview += f"   └─ Sections: {len(sections)}\n"
            
            if sections:
                for section in sections[:3]:  # Show first 3 sections
                    subsection_count = len(section.get('subsections', []))
                    preview += f"      • {section['title']} ({subsection_count} subsections)\n"
                
                if len(sections) > 3:
                    preview += f"      • ... and {len(sections) - 3} more sections\n"
            
            preview += "\n"
        
        return preview
    
    def process_batch_by_heading(self, parent_page_id: str, output_base_dir: str = 'notion_documents',
                                 batch_size: int = 10):
        """
        Process documents in batches, organized by headings.
        Only loads batch_size pages at a time from Notion.
        """
        os.makedirs(output_base_dir, exist_ok=True)
        
        print("🔍 Analyzing page structure...")
        organized_pages = self.get_pages_by_headings(parent_page_id)
        
        if not organized_pages:
            print("❌ No pages found!")
            return
        
        # Display organization
        print(f"\n{'='*70}")
        print("📊 DOCUMENT ORGANIZATION")
        print(f"{'='*70}\n")
        
        total_pages = 0
        for heading, pages in organized_pages.items():
            print(f"📁 {heading}: {len(pages)} documents")
            total_pages += len(pages)
        
        print(f"\n📄 Total: {total_pages} documents")
        print(f"{'='*70}\n")
        
        # Process each heading
        overall_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'folders_created': 0
        }
        
        for heading_idx, (heading, pages) in enumerate(organized_pages.items(), 1):
            print(f"\n{'#'*70}")
            print(f"📂 HEADING {heading_idx}/{len(organized_pages)}: {heading}")
            print(f"{'#'*70}\n")
            
            if not pages:
                print("⚠️  No pages under this heading, skipping...\n")
                continue
            
            # Create folder for this heading
            folder_name = self.sanitize_filename(heading)
            output_dir = os.path.join(output_base_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            overall_stats['folders_created'] += 1
            
            # Process pages in batches
            total_in_heading = len(pages)
            
            for batch_start in range(0, total_in_heading, batch_size):
                batch_end = min(batch_start + batch_size, total_in_heading)
                batch_pages = pages[batch_start:batch_end]
                
                print(f"\n{'─'*70}")
                print(f"⚙️  BATCH: Processing {batch_start + 1}-{batch_end} of {total_in_heading} in '{heading}'")
                print(f"{'─'*70}\n")
                
                processed_docs = []
                failed_pages = []
                
                # Process each page in the batch
                for idx, page_info in enumerate(batch_pages, 1):
                    page_id = page_info['id']
                    page_title = page_info['title']
                    
                    global_idx = batch_start + idx
                    print(f"[{global_idx}/{total_in_heading}] Processing: {page_title}")
                    
                    try:
                        # Extract document structure
                        document = self.extract_document_structure(page_id)
                        processed_docs.append(document)
                        
                        # Create filename
                        safe_filename = self.sanitize_filename(page_title)
                        if not safe_filename:
                            safe_filename = f"document_{global_idx}"
                        
                        output_file = os.path.join(output_dir, f"{safe_filename}.json")
                        
                        # Save to JSON file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(document, f, indent=2, ensure_ascii=False)
                        
                        print(f"   ✅ Saved: {safe_filename}.json")
                        print(f"   📊 Sections: {len(document.get('sections', []))}")
                        
                        overall_stats['total_processed'] += 1
                        
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        failed_pages.append({'title': page_title, 'error': str(e)})
                        overall_stats['total_failed'] += 1
                
                # Show preview of this batch
                if processed_docs:
                    preview = self.generate_preview(processed_docs, heading)
                    print(preview)
                
                # Show batch summary
                print(f"{'─'*70}")
                print(f"✅ Batch Complete: {len(processed_docs)} processed, {len(failed_pages)} failed")
                print(f"📁 Saved in: {output_dir}")
                print(f"🔧 API calls: {self.api_call_count}")
                print(f"{'─'*70}")
                
                # Ask to continue if more batches in this heading
                if batch_end < total_in_heading:
                    choice = input(f"\n▶️  Continue to next batch in '{heading}'? (yes/no/skip-heading): ").strip().lower()
                    
                    if choice in ['skip-heading', 'skip', 's']:
                        print(f"⏭️  Skipping remaining batches in '{heading}'...\n")
                        break
                    elif choice not in ['yes', 'y']:
                        print("\n⏸️  Paused. Exiting...")
                        self._print_final_summary(overall_stats, output_base_dir)
                        return
            
            # Ask before moving to next heading
            if heading_idx < len(organized_pages):
                next_heading = list(organized_pages.keys())[heading_idx]
                next_count = len(organized_pages[next_heading])
                
                print(f"\n{'='*70}")
                print(f"📂 Next: '{next_heading}' ({next_count} documents)")
                print(f"{'='*70}")
                
                choice = input("\n▶️  Continue to next heading? (yes/no): ").strip().lower()
                
                if choice not in ['yes', 'y']:
                    print("\n⏸️  Stopped. Exiting...")
                    self._print_final_summary(overall_stats, output_base_dir)
                    return
        
        # Final summary
        self._print_final_summary(overall_stats, output_base_dir)
    
    def _print_final_summary(self, stats: Dict, output_dir: str):
        """Print final processing summary"""
        print(f"\n{'='*70}")
        print("🎉 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Documents processed: {stats['total_processed']}")
        print(f"❌ Documents failed: {stats['total_failed']}")
        print(f"📁 Folders created: {stats['folders_created']}")
        print(f"📂 Output directory: {output_dir}")
        print(f"🔧 Total API calls: {self.api_call_count}")
        print(f"{'='*70}\n")


# Usage
if __name__ == "__main__":
    # Configuration
    NOTION_API_KEY = "your_notion_api_key_here"
    PARENT_PAGE_ID = "your_parent_page_id_here"
    OUTPUT_DIR = "notion_documents"
    
    # Initialize extractor
    extractor = NotionDocumentExtractor(
        api_key=NOTION_API_KEY,
        rate_limit_delay=0.35
    )
    
    # Process with pagination (10 pages at a time)
    extractor.process_batch_by_heading(
        parent_page_id=PARENT_PAGE_ID,
        output_base_dir=OUTPUT_DIR,
        batch_size=10
    )