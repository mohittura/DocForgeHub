# clean_reorder.py
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import traceback
from collections import Counter, defaultdict


@dataclass
class SchemaSection:
    """Schema subsection"""
    title: str
    type: str
    order: int
    columns: List[str] = None
    has_questions: bool = False
    question_count: int = 0
    
    def __post_init__(self):
        self.columns = self.columns or []


@dataclass
class Question:
    """Question with metadata"""
    question_id: str
    question_text: str
    category: str
    category_order: int
    answer_type: str
    required: bool
    original_order: int
    description: str = ""
    options: List[str] = None
    fields: List[Dict] = None
    placeholder: str = ""
    matched_sections: List[Dict[str, Any]] = None
    primary_schema_order: int = 999
    match_confidence: float = 0.0
    
    def __post_init__(self):
        self.matched_sections = self.matched_sections or []
        self.options = self.options or []
        self.fields = self.fields or []


class KeywordExtractor:
    """Extract keywords dynamically"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'your', 
            'this', 'that', 'how', 'should', 'does', 'do', 'can', 'will',
            'from', 'has', 'have', 'been', 'being', 'had', 'their',
            'they', 'them', 'would', 'could', 'must', 'may', 'might',
            'who', 'which', 'when', 'where', 'why'
        }
    
    def extract_from_text(self, text: str) -> Set[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return {w for w in words if len(w) >= 3 and w not in self.stop_words and not w.isdigit()}
    
    def extract_from_schema(self, schema_data: Dict) -> Set[str]:
        keywords = set()
        for section in schema_data.get('sections', []):
            keywords.update(self.extract_from_text(section.get('title', '')))
            for subsection in section.get('subsections', []):
                keywords.update(self.extract_from_text(subsection.get('title', '')))
                for col in subsection.get('columns', []):
                    keywords.update(self.extract_from_text(col))
        return keywords
    
    def extract_from_questions(self, questions_data: Dict) -> Set[str]:
        keywords = set()
        for category in questions_data.get('question_categories', []):
            keywords.update(self.extract_from_text(category['category']))
            for q in category['questions']:
                keywords.update(self.extract_from_text(q['question']))
                keywords.update(self.extract_from_text(q.get('description', '')))
                for field in q.get('fields', []):
                    keywords.update(self.extract_from_text(field.get('name', '')))
        return keywords
    
    def get_priority_keywords(self, schema_data: Dict, questions_data: Dict) -> Set[str]:
        schema_kw = self.extract_from_schema(schema_data)
        question_kw = self.extract_from_questions(questions_data)
        return schema_kw | question_kw


class QuestionReorderer:
    """Reorder questions based on substantive schema sections"""
    
    def __init__(self, schema_path: str, questions_path: str):
        self.schema_path = schema_path
        self.questions_path = questions_path
        self.document_name = Path(questions_path).stem.replace('_questions', '')
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema_data = json.load(f)
        with open(questions_path, 'r', encoding='utf-8') as f:
            self.questions_data = json.load(f)
        
        self.kw_extractor = KeywordExtractor()
        self.priority_keywords = self.kw_extractor.get_priority_keywords(
            self.schema_data, self.questions_data
        )
        
        self.schema_sections: List[SchemaSection] = []
        self.questions: List[Question] = []
        self.substantive_sections: List[SchemaSection] = []
    
    def parse_schema(self):
        sections = []
        order = 0
        
        for section in self.schema_data.get('sections', []):
            for subsection in section.get('subsections', []):
                sections.append(SchemaSection(
                    title=subsection['title'],
                    type=subsection['type'],
                    order=order,
                    columns=subsection.get('columns', [])
                ))
                order += 1
        
        self.schema_sections = sections
    
    def parse_questions(self):
        questions = []
        order = 0
        
        for category in self.questions_data.get('question_categories', []):
            for q in category['questions']:
                questions.append(Question(
                    question_id=q['question_id'],
                    question_text=q['question'],
                    category=category['category'],
                    category_order=category['order'],
                    answer_type=q['answer_type'],
                    required=q['required'],
                    original_order=order,
                    description=q.get('description', ''),
                    options=q.get('options', []),
                    fields=q.get('fields', []),
                    placeholder=q.get('placeholder', '')
                ))
                order += 1
        
        self.questions = questions
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 in t2 or t2 in t1:
            return 0.85
        
        base_score = SequenceMatcher(None, t1, t2).ratio()
        
        words1 = set(re.findall(r'\w+', t1)) - self.kw_extractor.stop_words
        words2 = set(re.findall(r'\w+', t2)) - self.kw_extractor.stop_words
        
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            priority_overlap = (words1 & words2) & self.priority_keywords
            boost = len(priority_overlap) * 0.15
            return min((base_score * 0.35) + (overlap * 0.50) + boost, 1.0)
        
        return base_score
    
    def map_and_learn(self):
        section_matches = defaultdict(list)
        
        for question in self.questions:
            matches = []
            q_context = f"{question.question_text} {question.category} {question.description}"
            q_keywords = self.kw_extractor.extract_from_text(q_context)
            
            for section in self.schema_sections:
                s_keywords = self.kw_extractor.extract_from_text(section.title)
                score = self.calculate_similarity(q_context, section.title)
                
                if q_keywords and s_keywords:
                    common = q_keywords & s_keywords
                    priority_common = common & self.priority_keywords
                    if common:
                        score += len(common) * 0.08
                    if priority_common:
                        score += len(priority_common) * 0.12
                
                if question.answer_type == 'structured_list' and section.columns:
                    q_fields = {f['name'].lower() for f in question.fields}
                    s_cols = {c.lower() for c in section.columns}
                    if q_fields and s_cols:
                        overlap = len(q_fields & s_cols) / max(len(q_fields), len(s_cols))
                        score += overlap * 0.40
                
                if section.type == 'list' and question.answer_type in ['structured_list', 'multi_select']:
                    score += 0.08
                
                if score > 0.25:
                    matches.append({
                        'section_title': section.title,
                        'section_order': section.order,
                        'score': round(min(score, 1.0), 3)
                    })
                    section_matches[section.order].append({'question_id': question.question_id, 'score': score})
            
            matches.sort(key=lambda x: x['score'], reverse=True)
            question.matched_sections = matches[:5]
            
            if matches:
                question.primary_schema_order = matches[0]['section_order']
                question.match_confidence = matches[0]['score']
        
        # Learn substantive sections
        for section in self.schema_sections:
            if section.order in section_matches:
                section.has_questions = True
                section.question_count = len(section_matches[section.order])
        
        self.substantive_sections = [s for s in self.schema_sections if s.has_questions]
        
        # Reindex
        for idx, section in enumerate(self.substantive_sections):
            section.order = idx
    
    def needs_reordering(self) -> bool:
        """Check if questions need reordering"""
        for question in self.questions:
            if question.matched_sections:
                primary_order = question.matched_sections[0]['section_order']
                original_section = self.schema_sections[primary_order]
                
                for sub_section in self.substantive_sections:
                    if sub_section.title == original_section.title:
                        question.primary_schema_order = sub_section.order
                        break
        
        ideal_order = [q.primary_schema_order for q in self.questions]
        return not all(ideal_order[i] <= ideal_order[i + 1] for i in range(len(ideal_order) - 1))
    
    def reorder_questions(self) -> Dict[str, Any]:
        """Reorder questions by substantive sections"""
        
        # Update to substantive orders
        for question in self.questions:
            if question.matched_sections:
                primary_order = question.matched_sections[0]['section_order']
                original_section = self.schema_sections[primary_order]
                
                for sub_section in self.substantive_sections:
                    if sub_section.title == original_section.title:
                        question.primary_schema_order = sub_section.order
                        break
        
        sorted_questions = sorted(self.questions, key=lambda q: (q.primary_schema_order, q.original_order))
        
        new_categories = {}
        category_order_map = {}
        current_order = 1
        
        for question in sorted_questions:
            if question.category not in new_categories:
                new_categories[question.category] = []
                category_order_map[question.category] = current_order
                current_order += 1
            new_categories[question.category].append(question)
        
        # Build clean output
        output = {
            'document_name': self.questions_data['document_name'],
            'document_type': self.questions_data['document_type'],
            'estimated_completion_time': self.questions_data['estimated_completion_time'],
            'total_questions': len(sorted_questions),
            'question_categories': [],
            '_metadata': self.questions_data.get('_metadata', {}),
            '_validation': {
                'reordered': True,
                'reordered_by': 'substantive_sections_only',
                'total_schema_sections': len(self.schema_sections),
                'substantive_sections': len(self.substantive_sections),
                'generic_sections_excluded': len(self.schema_sections) - len(self.substantive_sections),
                'reorder_timestamp': datetime.now().isoformat(),
                'validator_version': '3.0_intelligent'
            }
        }
        
        for category, questions_list in new_categories.items():
            category_data = {
                'category': category,
                'order': category_order_map[category],
                'questions': []
            }
            
            for q in questions_list:
                q_dict = {
                    'question_id': q.question_id,
                    'question': q.question_text,
                    'answer_type': q.answer_type,
                    'required': q.required,
                    'description': q.description
                }
                
                if q.placeholder:
                    q_dict['placeholder'] = q.placeholder
                if q.options:
                    q_dict['options'] = q.options
                if q.fields:
                    q_dict['fields'] = q.fields
                
                category_data['questions'].append(q_dict)
            
            output['question_categories'].append(category_data)
        
        return output
    
    def process(self) -> Tuple[Dict, bool]:
        """Process and return reordered data or original"""
        self.parse_schema()
        self.parse_questions()
        self.map_and_learn()
        
        if self.needs_reordering():
            # Needs reordering
            reordered = self.reorder_questions()
            return reordered, True
        else:
            # No reordering needed - return original with validation flag
            original = self.questions_data.copy()
            original['_validation'] = {
                'reordered': False,
                'reordered_by': 'no_reordering_needed',
                'total_schema_sections': len(self.schema_sections),
                'substantive_sections': len(self.substantive_sections),
                'generic_sections_excluded': len(self.schema_sections) - len(self.substantive_sections),
                'reorder_timestamp': datetime.now().isoformat(),
                'validator_version': '3.0_intelligent'
            }
            return original, False


class CleanBatchReorderer:
    """Clean batch processing - only outputs reordered files"""
    
    def __init__(self, questions_dir: str, schema_dir: str, output_dir: str):
        self.questions_dir = Path(questions_dir)
        self.schema_dir = Path(schema_dir)
        self.output_dir = Path(output_dir)
    
    def discover_categories(self) -> List[str]:
        categories = []
        if self.questions_dir.exists():
            for item in self.questions_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('_'):
                    categories.append(item.name)
        return sorted(categories)
    
    def find_all_pairs(self) -> List[Tuple[str, str, str]]:
        all_pairs = []
        categories = self.discover_categories()
        
        print(f"🔍 Found {len(categories)} categories")
        
        for category in categories:
            q_dir = self.questions_dir / category
            s_dir = self.schema_dir / category
            
            if not s_dir.exists():
                continue
            
            q_files = list(q_dir.glob('*_questions.json'))
            
            for q_file in q_files:
                doc_name = q_file.stem.replace('_questions', '')
                s_file = s_dir / f"{doc_name}.json"
                
                if s_file.exists():
                    all_pairs.append((category, str(s_file), str(q_file)))
        
        print(f"📊 Found {len(all_pairs)} document pairs\n")
        return all_pairs
    
    def process_single(self, args: Tuple[str, str, str, str]) -> Dict[str, Any]:
        category, schema_path, questions_path, output_dir = args
        
        try:
            reorderer = QuestionReorderer(schema_path, questions_path)
            result, was_reordered = reorderer.process()
            
            # Save to output
            output_file = Path(output_dir) / f"{reorderer.document_name}_questions.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return {
                'category': category,
                'document': reorderer.document_name,
                'status': 'success',
                'reordered': was_reordered,
                'total_sections': result['_validation']['total_schema_sections'],
                'substantive_sections': result['_validation']['substantive_sections']
            }
            
        except Exception as e:
            return {
                'category': category,
                'document': Path(questions_path).stem.replace('_questions', ''),
                'status': 'failed',
                'error': str(e)
            }
    
    def run(self, max_workers: int = 8):
        print(f"\n{'='*80}")
        print(f"🚀 CLEAN REORDERER - Output Files Only")
        print(f"{'='*80}\n")
        
        pairs = self.find_all_pairs()
        
        if not pairs:
            print("❌ No pairs found")
            return
        
        # Prepare
        prepared = []
        for category, schema, questions in pairs:
            cat_output = self.output_dir / category
            cat_output.mkdir(parents=True, exist_ok=True)
            prepared.append((category, schema, questions, str(cat_output)))
        
        print(f"⚙️  Workers: {max_workers}")
        print(f"📁 Output: {self.output_dir}\n")
        
        stats = {
            'total': len(prepared),
            'success': 0,
            'failed': 0,
            'reordered': 0,
            'kept_original': 0
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single, args): args for args in prepared}
            
            with tqdm(total=len(prepared), desc="Processing", unit="doc") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['status'] == 'success':
                        stats['success'] += 1
                        if result['reordered']:
                            stats['reordered'] += 1
                        else:
                            stats['kept_original'] += 1
                    else:
                        stats['failed'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': stats['success'],
                        'Reordered': stats['reordered']
                    })
        
        # Summary
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE")
        print(f"{'='*80}\n")
        print(f"Total:            {stats['total']}")
        print(f"✅ Success:       {stats['success']}")
        print(f"🔄 Reordered:     {stats['reordered']}")
        print(f"📋 Kept Original: {stats['kept_original']}")
        print(f"❌ Failed:        {stats['failed']}")
        print(f"\n💾 Files saved to: {self.output_dir}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧹 CLEAN QUESTION REORDERER")
    print("="*80 + "\n")
    
    QUESTIONS_DIR = "./generated_questions"
    SCHEMA_DIR = "./notion_documents"
    OUTPUT_DIR = "./reordered_questions"
    WORKERS = 8
    
    if not Path(QUESTIONS_DIR).exists():
        QUESTIONS_DIR = input("Questions directory: ").strip()
    
    if not Path(SCHEMA_DIR).exists():
        SCHEMA_DIR = input("Schema directory: ").strip()
    
    print(f"📂 Questions: {QUESTIONS_DIR}")
    print(f"📂 Schemas:   {SCHEMA_DIR}")
    print(f"📂 Output:    {OUTPUT_DIR}")
    print(f"⚙️  Workers:   {WORKERS}\n")
    
    print("ℹ️  This will:")
    print("   - Reorder questions based on substantive sections")
    print("   - Save reordered OR original files (with _validation flag)")
    print("   - NO validation reports, ONLY question files\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        reorderer = CleanBatchReorderer(QUESTIONS_DIR, SCHEMA_DIR, OUTPUT_DIR)
        reorderer.run(max_workers=WORKERS)
        print(f"\n🎉 Done! Check {OUTPUT_DIR}")
    else:
        print("❌ Cancelled")