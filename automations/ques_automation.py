import json
import os
import re
from typing import List, Dict, Any, TypedDict, Annotated
from notion_client import Client
import operator
from datetime import datetime
import time

# Latest LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver


class DocumentAnalysisState(TypedDict):
    """State that flows through the agent graph"""
    page_id: str
    document_name: str
    raw_content: str
    structure_analysis: Dict[str, Any]
    pattern_detection: Dict[str, Any]
    question_draft: Dict[str, Any]
    validated_questions: Dict[str, Any]
    messages: Annotated[List, operator.add]
    errors: Annotated[List[str], operator.add]
    iteration_count: int
    current_step: str


class NotionContentExtractor:
    """Extract content from Notion with smart template detection"""
    
    def __init__(self, notion_api_key: str):
        self.notion = Client(auth=notion_api_key)
    
    def get_pages_by_headings(self, parent_page_id: str) -> Dict[str, List[Dict[str, str]]]:
        """Get all pages organized by the headings they appear under"""
        organized_pages = {}
        current_heading = "Uncategorized"
        
        try:
            # Get all blocks from the parent page
            blocks = []
            has_more = True
            start_cursor = None
            
            while has_more:
                response = self.notion.blocks.children.list(
                    block_id=parent_page_id,
                    start_cursor=start_cursor
                )
                blocks.extend(response['results'])
                has_more = response['has_more']
                start_cursor = response.get('next_cursor')
            
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
            
            # If parent is a database, get all pages
            if not organized_pages:
                try:
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
                            title = self._get_page_title(page_id)
                            
                            category = "All Documents"
                            if category not in organized_pages:
                                organized_pages[category] = []
                            
                            organized_pages[category].append({
                                'id': page_id,
                                'title': title
                            })
                        
                        has_more = results.get('has_more', False)
                        start_cursor = results.get('next_cursor')
                        
                except:
                    pass
            
        except Exception as e:
            print(f"Error organizing pages: {e}")
        
        return organized_pages
    
    def _get_page_title(self, page_id: str) -> str:
        """Get title of a page"""
        try:
            page = self.notion.pages.retrieve(page_id=page_id)
            properties = page.get('properties', {})
            
            for prop_name, prop_data in properties.items():
                if prop_data['type'] == 'title':
                    return ''.join([t['plain_text'] for t in prop_data['title']])
            
            return "Untitled"
        except:
            return "Untitled"
    
    def get_full_page_content(self, page_id: str) -> Dict[str, Any]:
        """Extract comprehensive content with template detection"""
        try:
            page = self.notion.pages.retrieve(page_id=page_id)
            properties = page.get('properties', {})
            
            doc_name = "Untitled"
            for prop_name, prop_data in properties.items():
                if prop_data['type'] == 'title':
                    doc_name = ''.join([t['plain_text'] for t in prop_data['title']])
                    break
            
            blocks = self._get_all_blocks(page_id)
            structured_content = self._parse_blocks_comprehensive(blocks)
            
            return {
                'document_name': doc_name,
                'structured_content': structured_content,
                'total_blocks': len(blocks)
            }
            
        except Exception as e:
            print(f"❌ Error extracting content: {e}")
            return None
    
    def _get_all_blocks(self, page_id: str) -> List[Dict]:
        """Recursively get all blocks"""
        blocks = []
        has_more = True
        start_cursor = None
        
        while has_more:
            response = self.notion.blocks.children.list(
                block_id=page_id,
                start_cursor=start_cursor
            )
            blocks.extend(response['results'])
            has_more = response['has_more']
            start_cursor = response.get('next_cursor')
        
        for block in blocks:
            if block.get('has_children', False):
                try:
                    nested = self._get_all_blocks(block['id'])
                    block['children'] = nested
                except:
                    pass
        
        return blocks
    
    def _parse_blocks_comprehensive(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Parse blocks with intelligent template detection"""
        structure = {
            'headings': [],
            'tables': [],
            'metadata_fields': [],
            'template_sections': [],
            'full_text': ''
        }
        
        current_section = None
        full_text_parts = []
        
        for block in blocks:
            block_type = block['type']
            block_data = block.get(block_type, {})
            
            # Headings
            if block_type in ['heading_1', 'heading_2', 'heading_3']:
                text = ''.join([t['plain_text'] for t in block_data.get('rich_text', [])])
                
                if current_section:
                    structure['template_sections'].append(current_section)
                
                current_section = {
                    'title': text,
                    'level': int(block_type[-1]),
                    'content': [],
                    'is_template': self._is_template_section(text)
                }
                
                structure['headings'].append({'level': int(block_type[-1]), 'text': text})
                full_text_parts.append(f"\n{'#' * int(block_type[-1])} {text}\n")
            
            # Paragraphs
            elif block_type == 'paragraph':
                text = ''.join([t['plain_text'] for t in block_data.get('rich_text', [])])
                if text:
                    if self._is_metadata_field(text):
                        structure['metadata_fields'].append(self._parse_metadata_field(text))
                    
                    if current_section:
                        current_section['content'].append({'type': 'paragraph', 'text': text})
                    
                    full_text_parts.append(text + '\n')
            
            # Tables
            elif block_type == 'table':
                table_data = self._extract_table(block)
                structure['tables'].append(table_data)
                
                if current_section:
                    current_section['content'].append({'type': 'table', 'data': table_data})
                
                full_text_parts.append(self._table_to_text(table_data))
            
            # Lists
            elif block_type in ['bulleted_list_item', 'numbered_list_item']:
                text = ''.join([t['plain_text'] for t in block_data.get('rich_text', [])])
                
                if current_section:
                    current_section['content'].append({'type': 'list', 'text': text})
                
                full_text_parts.append(f"* {text}\n")
            
            # Process children
            if 'children' in block:
                nested_structure = self._parse_blocks_comprehensive(block['children'])
                for key in ['headings', 'tables', 'metadata_fields']:
                    if key in nested_structure:
                        structure[key].extend(nested_structure[key])
        
        if current_section:
            structure['template_sections'].append(current_section)
        
        structure['full_text'] = ''.join(full_text_parts)[:3000]  # Limit to 3000 chars
        
        return structure
    
    def _is_metadata_field(self, text: str) -> bool:
        """Check if text is metadata (Key: Value)"""
        return bool(re.match(r'^[A-Z][A-Za-z\s]+:\s*.+$', text))
    
    def _parse_metadata_field(self, text: str) -> Dict[str, str]:
        """Parse metadata into key-value"""
        parts = text.split(':', 1)
        if len(parts) == 2:
            return {
                'field': parts[0].strip(),
                'value': parts[1].strip(),
                'original': text
            }
        return {'field': 'unknown', 'value': text, 'original': text}
    
    def _is_template_section(self, heading: str) -> bool:
        """Determine if section contains variable content"""
        template_indicators = [
            'roadmap', 'timeline', 'schedule', 'plan', 'initiatives', 'features',
            'tasks', 'activities', 'risks', 'dependencies', 'team', 'roles',
            'budget', 'costs', 'metrics', 'kpis', 'goals', 'version history',
            'change log', 'revision', 'overview', 'scope', 'assumptions'
        ]
        
        heading_lower = heading.lower()
        return any(indicator in heading_lower for indicator in template_indicators)
    
    def _extract_table(self, table_block: Dict) -> Dict[str, Any]:
        """Extract table data with comprehensive analysis"""
        table_data = {
            'id': table_block['id'],
            'column_headers': [],
            'rows': [],
            'empty_cells': [],
            'column_examples': {}  # New: Store example values per column
        }
        
        try:
            children = self.notion.blocks.children.list(block_id=table_block['id'])
            
            for row_idx, row in enumerate(children['results']):
                if row['type'] == 'table_row':
                    cells = row['table_row']['cells']
                    row_data = []
                    
                    for col_idx, cell in enumerate(cells):
                        cell_text = ''.join([t['plain_text'] for t in cell])
                        row_data.append(cell_text)
                        
                        if not cell_text.strip() and row_idx > 0:
                            table_data['empty_cells'].append({'row': row_idx, 'col': col_idx})
                        
                        # Header row
                        if row_idx == 0:
                            table_data['column_headers'].append(cell_text)
                            table_data['column_examples'][cell_text] = []
                        # Data rows - collect examples
                        elif cell_text.strip() and row_idx > 0:
                            header = table_data['column_headers'][col_idx] if col_idx < len(table_data['column_headers']) else f"col_{col_idx}"
                            if header in table_data['column_examples']:
                                table_data['column_examples'][header].append(cell_text)
                    
                    table_data['rows'].append(row_data)
        except:
            pass
        
        return table_data
    
    def _table_to_text(self, table_data: Dict) -> str:
        """Convert table to text"""
        text = "\n[TABLE]\n"
        if table_data['column_headers']:
            text += " | ".join(table_data['column_headers']) + "\n"
        for row in table_data['rows'][:3]:  # Only first 3 rows
            text += " | ".join(row) + "\n"
        return text + "\n"


class GroqLangGraphQuestionGenerator:
    """Optimized LangGraph system"""
    
    def __init__(self, groq_api_keys: list, notion_api_key: str, models: list = None):
        self.groq_api_keys = groq_api_keys
        if models is None:
            self.models = ["moonshotai/kimi-k2-instruct-0905", "qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "groq/compound"]
        else:
            self.models = models
        self.model_name = self.models[0]
        self.extractor = NotionContentExtractor(notion_api_key)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.llm = None
        self.current_key_idx = 0

    def _invoke_groq(self, messages):
        """Try all models and all keys until success or all fail"""
        for model in self.models:
            for key in self.groq_api_keys:
                try:
                    self.llm = ChatGroq(
                        api_key=key,
                        model_name=model,
                        temperature=0.2,
                        max_tokens=4000
                    )
                    response = self.llm.invoke(messages)
                    # Check for status code if available
                    if hasattr(response, 'response_metadata'):
                        status = response.response_metadata.get('status_code', 200)
                        if status != 200:
                            continue
                    self.model_name = model
                    return response
                except Exception as e:
                    continue
        raise Exception("All Groq API keys and models failed or returned non-200 status.")
    
    def _build_graph(self) -> StateGraph:
        """Build optimized workflow"""
        workflow = StateGraph(DocumentAnalysisState)
        
        workflow.add_node("analyzer", self._analyze_and_detect)
        workflow.add_node("question_generator", self._generate_questions)
        workflow.add_node("simple_validator", self._simple_validate)
        
        workflow.add_edge(START, "analyzer")
        workflow.add_edge("analyzer", "question_generator")
        workflow.add_edge("question_generator", "simple_validator")
        workflow.add_edge("simple_validator", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _analyze_and_detect(self, state: DocumentAnalysisState) -> DocumentAnalysisState:
        """Combined analysis and pattern detection"""
        
        structured = json.loads(state['raw_content'])
        
        system_prompt = """Analyze this document and identify template areas where users need to provide their own data.

IMPORTANT: Look for:
1. Metadata fields (Document ID, Version, Owner, Dates) - these ALWAYS need user input
2. Tables with example data (like roadmap items, tasks) - the rows are examples to be replaced
3. Sections with company-specific content (goals, initiatives, team members)
4. Lists with placeholder items
5. Dates, quarters (Q2 2026), specific feature names - these are examples

For TABLES, analyze deeply:
- Identify each column name
- Determine the data type for each column (text, date, select, number)
- Extract example values that show what type of data is expected
- Identify if columns have limited options (like Priority: High/Medium/Low)

Return ONLY valid JSON (no markdown, no extra text):
{
  "document_type": "string",
  "purpose": "string",
  "template_sections": [
    {
      "section_name": "string",
      "why_template": "string",
      "data_needed": ["list"]
    }
  ],
  "metadata_fields": [
    {
      "field_name": "string",
      "current_value": "string (the example value in template)",
      "data_type": "text|date|number|select"
    }
  ],
  "table_templates": [
    {
      "table_purpose": "string (e.g., Product Roadmap Items)",
      "columns": [
        {
          "name": "string (column name)",
          "data_type": "text|date|number|select",
          "example_values": ["array of example values from rows"],
          "inferred_options": ["if it's a select type, what are the options"]
        }
      ],
      "example_data_present": boolean,
      "number_of_example_rows": number
    }
  ]
}"""

        user_prompt = f"""Document: {state['document_name']}

METADATA FIELDS:
{json.dumps(structured.get('metadata_fields', []), indent=2)}

SECTIONS ({len(structured.get('template_sections', []))}):
{json.dumps([{'title': s['title'], 'is_template': s.get('is_template')} for s in structured.get('template_sections', [])], indent=2)}

TABLES ({len(structured.get('tables', []))}):
{json.dumps([{
    'headers': t['column_headers'], 
    'example_values_per_column': t.get('column_examples', {}),
    'total_rows': len(t['rows'])
} for t in structured.get('tables', [])], indent=2)}

SAMPLE TEXT (first 1500 chars):
{structured.get('full_text', '')[:1500]}

Analyze template areas. Return JSON only."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            time.sleep(0.5)
            response = self._invoke_groq(messages)
            analysis = self._extract_json(response.content)
            state['structure_analysis'] = analysis
            state['pattern_detection'] = analysis
            state['messages'] = messages + [response]
            state['current_step'] = 'analyzed'
        except Exception as e:
            state['errors'] = [f"Analysis error: {str(e)[:200]}"]
            state['structure_analysis'] = {}
            state['pattern_detection'] = {}
        return state
    
    def _generate_questions(self, state: DocumentAnalysisState) -> DocumentAnalysisState:
        """Generate questions"""
        
        system_prompt = """Create questions to fill this template document.

CRITICAL RULES FOR TABLES:
- DO NOT create a single generic "table_input" question
- Instead, create INDIVIDUAL questions for EACH column in the table
- Each column gets its own question asking for a list of values
- For repeating data tables (roadmaps, tasks, team members), ask for multiple items

EXAMPLE - For a roadmap table with columns: Timeframe | Initiative | Description | Priority | Owner | Status
Generate these questions:
1. "What timeframes are covered in your roadmap?" (answer_type: "multi_select", options: ["Q1 2026", "Q2 2026", ...])
2. "What initiatives or features are you planning?" (answer_type: "long_text", description: "List each initiative, one per line")
3. "For each initiative, provide a brief description" (answer_type: "long_text")
4. "What is the priority for each initiative?" (answer_type: "multi_select", options: ["High", "Medium", "Low"])
5. "Who owns each initiative?" (answer_type: "text")
6. "What is the status of each initiative?" (answer_type: "multi_select", options: ["Planned", "In Progress", "Completed"])

ALTERNATIVE APPROACH - Or create a structured data collection:
{
  "question_id": "roadmap_items",
  "question": "What items should be included in your product roadmap?",
  "answer_type": "structured_list",
  "required": true,
  "fields": [
    {"name": "Timeframe", "type": "select", "options": ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]},
    {"name": "Initiative / Feature", "type": "text"},
    {"name": "Description", "type": "text"},
    {"name": "Priority", "type": "select", "options": ["High", "Medium", "Low"]},
    {"name": "Owner", "type": "text"},
    {"name": "Status", "type": "select", "options": ["Planned", "In Progress", "Proposed", "Completed"]}
  ],
  "description": "Add each roadmap item with all required fields"
}

For METADATA fields (Document ID, Version, etc.), create ONE question per field.

Return ONLY valid JSON:
{
  "document_name": "string",
  "document_type": "string",
  "estimated_completion_time": "10-15 minutes",
  "total_questions": 0,
  "question_categories": [
    {
      "category": "string",
      "order": 1,
      "questions": [
        {
          "question_id": "lowercase_id",
          "question": "Clear question text",
          "answer_type": "text|long_text|date|number|select|multi_select|structured_list",
          "required": true|false,
          "placeholder": "Example answer",
          "options": ["only for select/multi_select"],
          "fields": [{"name": "string", "type": "string", "options": []}],
          "description": "Why needed"
        }
      ]
    }
  ]
}"""

        analysis_summary = {
            'metadata_count': len(state['structure_analysis'].get('metadata_fields', [])),
            'template_sections': [s['section_name'] for s in state['structure_analysis'].get('template_sections', [])[:5]],
            'tables': len(state['structure_analysis'].get('table_templates', []))
        }

        user_prompt = f"""Document: {state['document_name']}
Analysis Summary: {json.dumps(analysis_summary, indent=2)}
Full Analysis: {json.dumps(state['structure_analysis'], indent=2)[:1500]}

Generate questions. JSON only."""

        messages = state.get('messages', []) + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            time.sleep(0.5)
            response = self._invoke_groq(messages)
            questions = self._extract_json(response.content)
            total = sum(len(cat.get('questions', [])) for cat in questions.get('question_categories', []))
            questions['total_questions'] = total
            state['question_draft'] = questions
            state['messages'] = messages + [response]
            state['current_step'] = 'generated'
        except Exception as e:
            state['errors'] = state.get('errors', []) + [f"Generation error: {str(e)[:200]}"]
            state['question_draft'] = {}
        return state
    
    def _simple_validate(self, state: DocumentAnalysisState) -> DocumentAnalysisState:
        """Simple validation (no LLM call)"""
        
        questions = state.get('question_draft', {})
        
        # Clean up question IDs
        for cat in questions.get('question_categories', []):
            for q in cat.get('questions', []):
                if 'question_id' in q:
                    q['question_id'] = q['question_id'].lower().replace(' ', '_').replace('-', '_')
                
                valid_types = ['text', 'long_text', 'date', 'number', 'select', 'multi_select', 
                               'boolean', 'email', 'url', 'phone', 'structured_list']
                if q.get('answer_type') not in valid_types:
                    q['answer_type'] = 'text'
                
                # Validate structured_list has fields
                if q.get('answer_type') == 'structured_list' and not q.get('fields'):
                    q['fields'] = []
        
        state['validated_questions'] = questions
        state['current_step'] = 'validated'
        
        return state
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = text
        
        json_str = json_str.strip()
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json.loads(json_str)
    
    def process_document(self, page_id: str, output_dir: str = 'generated_questions') -> Dict[str, Any]:
        """Process single document"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        content_data = self.extractor.get_full_page_content(page_id)
        
        if not content_data:
            return None
        
        initial_state = {
            'page_id': page_id,
            'document_name': content_data['document_name'],
            'raw_content': json.dumps(content_data['structured_content']),
            'structure_analysis': {},
            'pattern_detection': {},
            'question_draft': {},
            'validated_questions': {},
            'messages': [],
            'errors': [],
            'iteration_count': 0,
            'current_step': 'initialized'
        }
        
        try:
            config = {"configurable": {"thread_id": f"doc_{page_id}"}}
            final_state = self.graph.invoke(initial_state, config)
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return None
        
        questions_data = final_state.get('validated_questions', {})
        
        if not questions_data:
            return None
        
        questions_data['_metadata'] = {
            'page_id': page_id,
            'extraction_date': datetime.now().isoformat(),
            'model_used': self.model_name
        }
        
        safe_filename = re.sub(r'[<>:"/\\|?*]', '', content_data['document_name']).replace(' ', '_')
        output_file = os.path.join(output_dir, f"{safe_filename}_questions.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        
        return questions_data
    
    def process_batch_by_heading(self, parent_page_id: str, output_base_dir: str = 'generated_questions', batch_size: int = 10):
        """Process documents in batches organized by headings"""
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("🔍 ANALYZING PAGE STRUCTURE")
        print(f"{'='*70}\n")
        
        organized_pages = self.extractor.get_pages_by_headings(parent_page_id)
        
        if not organized_pages:
            print("❌ No pages found!")
            return
        
        # Display organization with numbers - UPDATED
        total_pages = 0
        heading_list = list(organized_pages.keys())
        
        for idx, (heading, pages) in enumerate(organized_pages.items(), 1):
            print(f"📁 {idx}. {heading}: {len(pages)} documents")
            total_pages += len(pages)
        
        print(f"\n📄 Total: {total_pages} documents\n")
        
        # NEW: Ask which heading to start from
        print(f"{'='*70}")
        print("SELECT STARTING HEADING")
        print(f"{'='*70}")
        print("Enter the number to start from (or 0 to process all):")
        
        start_idx = 0
        while True:
            try:
                choice = input(f"Choice (0-{len(heading_list)}): ").strip()
                start_idx = int(choice)
                
                if 0 <= start_idx <= len(heading_list):
                    if start_idx == 0:
                        print("\n✅ Starting from the beginning\n")
                        start_idx = 0
                    else:
                        print(f"\n✅ Starting from: {heading_list[start_idx - 1]}\n")
                        start_idx = start_idx - 1
                    break
                else:
                    print(f"❌ Invalid choice. Enter 0-{len(heading_list)}")
            except ValueError:
                print("❌ Invalid input. Enter a number")
        
        overall_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'folders_created': 0
        }
        
        # Process each heading starting from selected index - UPDATED
        for heading_idx, (heading, pages) in enumerate(list(organized_pages.items())[start_idx:], start_idx + 1):
            print(f"\n{'#'*70}")
            print(f"📂 HEADING {heading_idx}/{len(organized_pages)}: {heading}")
            print(f"{'#'*70}\n")
            
            if not pages:
                print("⚠️  No pages, skipping...\n")
                continue
            
            # Create folder
            folder_name = re.sub(r'[<>:"/\\|?*]', '', heading).replace(' ', '_')
            output_dir = os.path.join(output_base_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            overall_stats['folders_created'] += 1
            
            # Process in batches
            total_in_heading = len(pages)
            
            for batch_start in range(0, total_in_heading, batch_size):
                batch_end = min(batch_start + batch_size, total_in_heading)
                batch_pages = pages[batch_start:batch_end]
                
                print(f"\n{'─'*70}")
                print(f"⚙️  BATCH: Processing {batch_start + 1}-{batch_end} of {total_in_heading}")
                print(f"{'─'*70}\n")
                
                processed_docs = []
                
                for idx, page_info in enumerate(batch_pages, 1):
                    page_id = page_info['id']
                    page_title = page_info['title']
                    
                    global_idx = batch_start + idx
                    print(f"[{global_idx}/{total_in_heading}] {page_title}")
                    
                    try:
                        result = self.process_document(page_id, output_dir)
                        
                        if result:
                            processed_docs.append(result)
                            print(f"   ✅ {result.get('total_questions', 0)} questions generated")
                            overall_stats['total_processed'] += 1
                        else:
                            print(f"   ❌ Failed")
                            overall_stats['total_failed'] += 1
                            
                    except Exception as e:
                        print(f"   ❌ Error: {str(e)[:100]}")
                        overall_stats['total_failed'] += 1
                
                # Batch summary
                print(f"\n{'─'*70}")
                print(f"✅ Batch: {len(processed_docs)} processed")
                print(f"📁 Saved in: {output_dir}")
                print(f"{'─'*70}")
                time.sleep(60)
                # Ask to continue
                if batch_end < total_in_heading:
                    choice = input(f"\n▶️  Continue to next batch in '{heading}'? (yes/no/skip): ").strip().lower()
                    
                    if choice in ['skip', 's']:
                        print(f"⏭️  Skipping remaining in '{heading}'...\n")
                        break
                    elif choice not in ['yes', 'y']:
                        print("\n⏸️  Paused.")
                        self._print_final_summary(overall_stats, output_base_dir)
                        return
            
            # Ask before next heading
            if heading_idx < len(organized_pages):
                next_heading = list(organized_pages.keys())[heading_idx]
                next_count = len(organized_pages[next_heading])
                
                print(f"\n{'='*70}")
                print(f"📂 Next: '{next_heading}' ({next_count} documents)")
                print(f"{'='*70}")
                
                choice = input("\n▶️  Continue to next heading? (yes/no): ").strip().lower()
                
                if choice not in ['yes', 'y']:
                    print("\n⏸️  Stopped.")
                    self._print_final_summary(overall_stats, output_base_dir)
                    return
        
        self._print_final_summary(overall_stats, output_base_dir)
    
    def _print_final_summary(self, stats: Dict, output_dir: str):
        """Print final summary"""
        print(f"\n{'='*70}")
        print("🎉 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Processed: {stats['total_processed']}")
        print(f"❌ Failed: {stats['total_failed']}")
        print(f"📁 Folders: {stats['folders_created']}")
        print(f"📂 Output: {output_dir}")
        print(f"{'='*70}\n")


# MAIN
if __name__ == "__main__":
    import sys
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    # Load all 7 Groq API keys
    GROQ_API_KEYS = []
    for i in range(1, 8):
        key = os.getenv(f"GROQ_API_KEY{i}")
        if key:
            GROQ_API_KEYS.append(key)
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    if not GROQ_API_KEYS:
        print("❌ At least one Groq API key required!")
        sys.exit(1)
    if not NOTION_API_KEY:
        NOTION_API_KEY = input("Notion API key: ").strip()
    if not NOTION_API_KEY:
        print("❌ Notion API key required!")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("Choose processing mode:")
    print(f"{'='*70}")
    print("1. Single document")
    print("2. Batch processing (by headings)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    # Optionally allow user to specify models (comma-separated)
    models_env = os.getenv("GROQ_MODELS")
    if models_env:
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        models = ["moonshotai/kimi-k2-instruct-0905","qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "groq/compound"]

    generator = GroqLangGraphQuestionGenerator(
        groq_api_keys=GROQ_API_KEYS,
        notion_api_key=NOTION_API_KEY,
        models=models
    )
    
    if choice == "1":
        page_id = input("Enter Notion page ID: ").strip()
        
        print(f"\n{'='*70}")
        print("🤖 PROCESSING SINGLE DOCUMENT")
        print(f"{'='*70}\n")
        
        result = generator.process_document(page_id, 'generated_questions')
        
        if result:
            print(f"\n✅ SUCCESS")
            print(f"📊 {result.get('total_questions', 0)} questions in {len(result.get('question_categories', []))} categories")
            print(f"📁 Saved in: generated_questions/\n")
    
    elif choice == "2":
        parent_page_id = input("Enter parent page ID: ").strip()
        
        generator.process_batch_by_heading(
            parent_page_id=parent_page_id,
            output_base_dir='generated_questions',
            batch_size=5
        )
    
    else:
        print("❌ Invalid choice")