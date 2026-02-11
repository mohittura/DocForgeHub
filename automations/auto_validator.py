# intelligent_validator.py
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import traceback
from collections import Counter, defaultdict


@dataclass
class SchemaSection:
    """Represents a schema subsection with order tracking"""
    title: str
    type: str
    order: int
    columns: List[str] = None
    parent_section: str = None
    has_questions: bool = False  # Learned from data
    question_count: int = 0
    
    def __post_init__(self):
        self.columns = self.columns or []


@dataclass
class Question:
    """Represents a question with metadata"""
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


class IntelligentKeywordExtractor:
    """Dynamically extracts keywords without hardcoding"""
    
    def __init__(self):
        # Only exclude universal stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'your', 
            'this', 'that', 'how', 'should', 'does', 'do', 'can', 'will',
            'from', 'has', 'have', 'been', 'being', 'had', 'their',
            'they', 'them', 'would', 'could', 'must', 'may', 'might',
            'who', 'which', 'when', 'where', 'why'
        }
    
    def extract_from_text(self, text: str, min_word_length: int = 3) -> Set[str]:
        """Extract keywords from text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        keywords = set()
        for word in words:
            if (len(word) >= min_word_length and 
                word not in self.stop_words and
                not word.isdigit()):
                keywords.add(word)
        
        return keywords
    
    def extract_from_schema(self, schema_data: Dict) -> Set[str]:
        """Extract all keywords from schema"""
        keywords = set()
        
        for section in schema_data.get('sections', []):
            keywords.update(self.extract_from_text(section.get('title', '')))
            
            for subsection in section.get('subsections', []):
                keywords.update(self.extract_from_text(subsection.get('title', '')))
                
                for col in subsection.get('columns', []):
                    keywords.update(self.extract_from_text(col))
        
        return keywords
    
    def extract_from_questions(self, questions_data: Dict) -> Set[str]:
        """Extract all keywords from questions"""
        keywords = set()
        
        for category in questions_data.get('question_categories', []):
            keywords.update(self.extract_from_text(category['category']))
            
            for q in category['questions']:
                keywords.update(self.extract_from_text(q['question']))
                keywords.update(self.extract_from_text(q.get('description', '')))
                
                for field in q.get('fields', []):
                    keywords.update(self.extract_from_text(field.get('name', '')))
        
        return keywords
    
    def get_shared_keywords(self, schema_data: Dict, questions_data: Dict, top_n: int = 100) -> Set[str]:
        """Get keywords that appear in BOTH schema and questions - these are important"""
        schema_keywords = self.extract_from_schema(schema_data)
        question_keywords = self.extract_from_questions(questions_data)
        
        # Shared keywords indicate substantive content
        shared = schema_keywords & question_keywords
        
        # Also get most frequent keywords
        all_text = []
        
        for section in schema_data.get('sections', []):
            for subsection in section.get('subsections', []):
                all_text.append(subsection.get('title', ''))
        
        for category in questions_data.get('question_categories', []):
            all_text.append(category['category'])
            for q in category['questions']:
                all_text.append(q['question'])
                all_text.append(q.get('description', ''))
        
        word_freq = Counter()
        for text in all_text:
            words = self.extract_from_text(text)
            word_freq.update(words)
        
        # Combine shared keywords with high-frequency keywords
        top_keywords = {word for word, _ in word_freq.most_common(top_n)}
        priority_keywords = shared | top_keywords
        
        return priority_keywords


class IntelligentValidator:
    """Learns which sections are substantive vs generic from the data"""
    
    def __init__(self, schema_path: str, questions_path: str, 
                 min_match_confidence: float = 0.25, verbose: bool = False):
        self.schema_path = schema_path
        self.questions_path = questions_path
        self.min_match_confidence = min_match_confidence
        self.verbose = verbose
        self.document_name = Path(questions_path).stem.replace('_questions', '')
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema_data = json.load(f)
            with open(questions_path, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading files: {str(e)}")
        
        self.keyword_extractor = IntelligentKeywordExtractor()
        self.priority_keywords = self.keyword_extractor.get_shared_keywords(
            self.schema_data, 
            self.questions_data
        )
        
        self.schema_sections: List[SchemaSection] = []
        self.questions: List[Question] = []
        self.substantive_sections: List[SchemaSection] = []
        
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def parse_schema(self) -> List[SchemaSection]:
        """Parse schema into flat ordered list"""
        sections = []
        order = 0
        
        for section in self.schema_data.get('sections', []):
            parent_title = section.get('title', 'Root')
            
            for subsection in section.get('subsections', []):
                sections.append(SchemaSection(
                    title=subsection['title'],
                    type=subsection['type'],
                    order=order,
                    columns=subsection.get('columns', []),
                    parent_section=parent_title
                ))
                order += 1
        
        self.schema_sections = sections
        return sections
    
    def parse_questions(self) -> List[Question]:
        """Parse questions into flat ordered list"""
        questions = []
        global_order = 0
        
        for category in self.questions_data.get('question_categories', []):
            cat_name = category['category']
            cat_order = category['order']
            
            for q in category['questions']:
                questions.append(Question(
                    question_id=q['question_id'],
                    question_text=q['question'],
                    category=cat_name,
                    category_order=cat_order,
                    answer_type=q['answer_type'],
                    required=q['required'],
                    original_order=global_order,
                    description=q.get('description', ''),
                    options=q.get('options', []),
                    fields=q.get('fields', []),
                    placeholder=q.get('placeholder', '')
                ))
                global_order += 1
        
        self.questions = questions
        return questions
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score (0-1)"""
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Exact substring match
        if t1 in t2 or t2 in t1:
            return 0.85
        
        # Base similarity
        base_score = SequenceMatcher(None, t1, t2).ratio()
        
        # Keyword overlap
        words1 = set(re.findall(r'\w+', t1))
        words2 = set(re.findall(r'\w+', t2))
        
        words1 -= self.keyword_extractor.stop_words
        words2 -= self.keyword_extractor.stop_words
        
        if words1 and words2:
            keyword_overlap = len(words1 & words2) / max(len(words1), len(words2))
            
            # Priority keyword boost (keywords in both schema and questions)
            priority_overlap = (words1 & words2) & self.priority_keywords
            priority_boost = len(priority_overlap) * 0.15
            
            return min((base_score * 0.35) + (keyword_overlap * 0.50) + priority_boost, 1.0)
        
        return base_score
    
    def map_questions_to_schema(self) -> None:
        """Map questions to schema sections and learn which sections are substantive"""
        
        # Track which sections get matched
        section_matches = defaultdict(list)
        
        for question in self.questions:
            matches = []
            
            # Combine question context
            q_context = f"{question.question_text} {question.category} {question.description}"
            q_keywords = self.keyword_extractor.extract_from_text(q_context)
            
            for section in self.schema_sections:
                s_keywords = self.keyword_extractor.extract_from_text(section.title)
                
                # Base similarity
                score = self.calculate_similarity(q_context, section.title)
                
                # Keyword overlap bonus
                if q_keywords and s_keywords:
                    common = q_keywords & s_keywords
                    priority_common = common & self.priority_keywords
                    
                    if common:
                        score += len(common) * 0.08
                    if priority_common:
                        score += len(priority_common) * 0.12
                
                # Structured data bonus (exact field matching)
                if question.answer_type == 'structured_list' and section.columns:
                    q_field_names = {f['name'].lower() for f in question.fields}
                    s_column_names = {c.lower() for c in section.columns}
                    
                    if q_field_names and s_column_names:
                        field_overlap = len(q_field_names & s_column_names) / max(len(q_field_names), len(s_column_names))
                        score += field_overlap * 0.40  # Strong signal
                
                # Type matching bonus
                if section.type == 'list' and question.answer_type in ['structured_list', 'multi_select']:
                    score += 0.08
                
                # Only keep meaningful matches
                if score > self.min_match_confidence:
                    matches.append({
                        'section_title': section.title,
                        'section_order': section.order,
                        'section_type': section.type,
                        'score': round(min(score, 1.0), 3)
                    })
                    
                    # Track for section learning
                    section_matches[section.order].append({
                        'question_id': question.question_id,
                        'score': score
                    })
            
            # Sort and keep top matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            question.matched_sections = matches[:5]
            
            # Assign primary order
            if matches:
                question.primary_schema_order = matches[0]['section_order']
                question.match_confidence = matches[0]['score']
        
        # Learn which sections are substantive (have questions)
        for section in self.schema_sections:
            if section.order in section_matches:
                section.has_questions = True
                section.question_count = len(section_matches[section.order])
        
        # Filter to only substantive sections for reordering
        self.substantive_sections = [s for s in self.schema_sections if s.has_questions]
        
        # Reindex substantive sections
        for idx, section in enumerate(self.substantive_sections):
            section.order = idx
        
        self.log(f"\n📊 Section Analysis:")
        self.log(f"   Total sections: {len(self.schema_sections)}")
        self.log(f"   Substantive (with questions): {len(self.substantive_sections)}")
        self.log(f"   Generic (no questions): {len(self.schema_sections) - len(self.substantive_sections)}")
    
    def reorder_questions_by_substantive_sections(self) -> Dict[str, Any]:
        """Reorder questions based ONLY on substantive sections (those with questions)"""
        
        # Update question orders to match substantive sections only
        for question in self.questions:
            if question.matched_sections:
                # Find if primary match is in substantive sections
                primary_section_order = question.matched_sections[0]['section_order']
                
                # Find this section in substantive list
                for sub_section in self.substantive_sections:
                    # Match by title (since original order changed)
                    original_section = self.schema_sections[primary_section_order]
                    if sub_section.title == original_section.title:
                        question.primary_schema_order = sub_section.order
                        break
        
        # Sort questions by substantive section order
        sorted_questions = sorted(
            self.questions, 
            key=lambda q: (q.primary_schema_order, q.original_order)
        )
        
        # Reorganize into categories
        new_categories = {}
        category_order_map = {}
        current_order = 1
        
        for question in sorted_questions:
            if question.category not in new_categories:
                new_categories[question.category] = []
                category_order_map[question.category] = current_order
                current_order += 1
            new_categories[question.category].append(question)
        
        # Build output structure
        reordered_data = {
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
                question_dict = {
                    'question_id': q.question_id,
                    'question': q.question_text,
                    'answer_type': q.answer_type,
                    'required': q.required,
                    'description': q.description
                }
                
                if q.placeholder:
                    question_dict['placeholder'] = q.placeholder
                if q.options:
                    question_dict['options'] = q.options
                if q.fields:
                    question_dict['fields'] = q.fields
                
                # Schema mapping metadata
                if q.matched_sections:
                    question_dict['_schema_mapping'] = {
                        'primary_match': q.matched_sections[0]['section_title'],
                        'confidence_score': q.matched_sections[0]['score'],
                        'is_substantive_section': any(
                            s.title == q.matched_sections[0]['section_title'] 
                            for s in self.substantive_sections
                        )
                    }
                
                category_data['questions'].append(question_dict)
            
            reordered_data['question_categories'].append(category_data)
        
        return reordered_data
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        
        # Coverage on substantive sections only
        substantive_coverage = {}
        for section in self.substantive_sections:
            substantive_coverage[section.title] = {
                'order': section.order,
                'type': section.type,
                'question_count': section.question_count
            }
        
        # Generic sections (no questions)
        generic_sections = [
            {
                'title': s.title,
                'type': s.type,
                'original_order': s.order
            }
            for s in self.schema_sections if not s.has_questions
        ]
        
        # Order analysis
        ideal_order = [q.primary_schema_order for q in self.questions]
        is_sorted = all(ideal_order[i] <= ideal_order[i + 1] for i in range(len(ideal_order) - 1))
        
        inversions = sum(
            1 for i in range(len(ideal_order))
            for j in range(i + 1, len(ideal_order))
            if ideal_order[i] > ideal_order[j]
        )
        
        # Match quality
        high_confidence = sum(1 for q in self.questions if q.match_confidence > 0.5)
        medium_confidence = sum(1 for q in self.questions if 0.3 <= q.match_confidence <= 0.5)
        low_confidence = sum(1 for q in self.questions if q.match_confidence < 0.3)
        
        return {
            'document_name': self.document_name,
            'validation_summary': {
                'total_questions': len(self.questions),
                'total_schema_sections': len(self.schema_sections),
                'substantive_sections': len(self.substantive_sections),
                'generic_sections': len(generic_sections),
                'is_properly_ordered': is_sorted,
                'requires_reordering': not is_sorted
            },
            'section_analysis': {
                'substantive_sections': substantive_coverage,
                'generic_sections': generic_sections,
                'learning_summary': f"Identified {len(self.substantive_sections)} substantive sections with questions, "
                                  f"{len(generic_sections)} generic sections without questions"
            },
            'match_quality': {
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence
            },
            'order_analysis': {
                'is_properly_ordered': is_sorted,
                'inversion_count': inversions,
                'needs_reordering': not is_sorted
            }
        }
    
    def run_validation(self, output_dir: str = None) -> Tuple[Dict, Optional[Dict]]:
        """Run complete intelligent validation"""
        try:
            self.parse_schema()
            self.parse_questions()
            self.map_questions_to_schema()  # This learns substantive vs generic
            
            report = self.generate_report()
            reordered = None
            
            # Always reorder based on substantive sections
            reordered = self.reorder_questions_by_substantive_sections()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                report_path = os.path.join(output_dir, f"{self.document_name}_validation_report.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                if reordered:
                    reordered_path = os.path.join(output_dir, f"{self.document_name}_questions_reordered.json")
                    with open(reordered_path, 'w', encoding='utf-8') as f:
                        json.dump(reordered, f, indent=2, ensure_ascii=False)
            
            return report, reordered
            
        except Exception as e:
            self.log(f"Error in validation: {str(e)}")
            raise


class AutoDiscoveryValidator:
    """Automatically discovers and validates all documents"""
    
    def __init__(self, base_questions_dir: str, base_schema_dir: str, output_base_dir: str):
        self.base_questions_dir = Path(base_questions_dir)
        self.base_schema_dir = Path(base_schema_dir)
        self.output_base_dir = Path(output_base_dir)
        
    def discover_all_categories(self) -> List[str]:
        """Auto-discover all category folders"""
        categories = []
        
        if self.base_questions_dir.exists():
            for item in self.base_questions_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('_'):
                    categories.append(item.name)
        
        categories.sort()
        return categories
    
    def find_all_document_pairs(self) -> List[Tuple[str, str, str]]:
        """Find ALL matching schema-question pairs across ALL categories"""
        all_pairs = []
        
        categories = self.discover_all_categories()
        
        print(f"🔍 Discovered {len(categories)} categories:")
        for cat in categories:
            print(f"   - {cat}")
        print()
        
        for category in categories:
            questions_dir = self.base_questions_dir / category
            schema_dir = self.base_schema_dir / category
            
            if not schema_dir.exists():
                print(f"⚠️  No matching schema directory for {category}")
                continue
            
            question_files = list(questions_dir.glob('*_questions.json'))
            
            category_pairs = []
            for q_file in question_files:
                doc_name = q_file.stem.replace('_questions', '')
                schema_file = schema_dir / f"{doc_name}.json"
                
                if schema_file.exists():
                    category_pairs.append((category, str(schema_file), str(q_file)))
                else:
                    print(f"⚠️  No schema for: {category}/{doc_name}")
            
            all_pairs.extend(category_pairs)
            print(f"   {category}: {len(category_pairs)} pairs")
        
        print(f"\n📊 Total: {len(all_pairs)} document pairs\n")
        return all_pairs
    
    def validate_single_document(self, args: Tuple[str, str, str, str]) -> Dict[str, Any]:
        """Validate a single document"""
        category, schema_path, questions_path, output_dir = args
        
        try:
            validator = IntelligentValidator(schema_path, questions_path, verbose=False)
            report, reordered = validator.run_validation(output_dir)
            
            return {
                'category': category,
                'document': validator.document_name,
                'status': 'success',
                'total_questions': report['validation_summary']['total_questions'],
                'total_sections': report['validation_summary']['total_schema_sections'],
                'substantive_sections': report['validation_summary']['substantive_sections'],
                'generic_sections': report['validation_summary']['generic_sections'],
                'needs_reordering': report['validation_summary']['requires_reordering']
            }
            
        except Exception as e:
            return {
                'category': category,
                'document': Path(questions_path).stem.replace('_questions', ''),
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_full_validation(self, max_workers: int = 8) -> Dict[str, Any]:
        """Run validation on ALL discovered documents"""
        
        print(f"\n{'='*80}")
        print(f"🚀 INTELLIGENT AUTO-DISCOVERY VALIDATION")
        print(f"{'='*80}\n")
        
        all_pairs = self.find_all_document_pairs()
        
        if not all_pairs:
            print("❌ No document pairs found!")
            return {}
        
        # Prepare output directories
        prepared_pairs = []
        for category, schema_path, questions_path in all_pairs:
            category_output = self.output_base_dir / category
            category_output.mkdir(parents=True, exist_ok=True)
            prepared_pairs.append((category, schema_path, questions_path, str(category_output)))
        
        print(f"⚙️  Using {max_workers} parallel workers")
        print(f"📁 Output directory: {self.output_base_dir}\n")
        
        results = {
            'total_documents': len(prepared_pairs),
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat(),
            'results': [],
            'category_stats': defaultdict(lambda: {
                'total': 0, 
                'success': 0, 
                'failed': 0,
                'avg_substantive_sections': 0,
                'avg_generic_sections': 0,
                'needs_reorder': 0
            })
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.validate_single_document, args): args 
                for args in prepared_pairs
            }
            
            with tqdm(total=len(prepared_pairs), desc="Validating", unit="doc") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results['results'].append(result)
                    
                    category = result['category']
                    results['category_stats'][category]['total'] += 1
                    
                    if result['status'] == 'success':
                        results['successful'] += 1
                        results['category_stats'][category]['success'] += 1
                        results['category_stats'][category]['avg_substantive_sections'] += result['substantive_sections']
                        results['category_stats'][category]['avg_generic_sections'] += result['generic_sections']
                        if result['needs_reordering']:
                            results['category_stats'][category]['needs_reorder'] += 1
                    else:
                        results['failed'] += 1
                        results['category_stats'][category]['failed'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results['successful'],
                        'Failed': results['failed']
                    })
        
        # Calculate averages
        for category, stats in results['category_stats'].items():
            if stats['success'] > 0:
                stats['avg_substantive_sections'] = round(stats['avg_substantive_sections'] / stats['success'], 1)
                stats['avg_generic_sections'] = round(stats['avg_generic_sections'] / stats['success'], 1)
        
        results['end_time'] = datetime.now().isoformat()
        results['category_stats'] = dict(results['category_stats'])
        
        self._save_results(results)
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save all validation results"""
        
        summary_path = self.output_base_dir / '_VALIDATION_SUMMARY.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        for category in results['category_stats'].keys():
            category_results = [r for r in results['results'] if r['category'] == category]
            
            if category_results:
                category_summary = {
                    'category': category,
                    'statistics': results['category_stats'][category],
                    'documents': category_results
                }
                
                category_path = self.output_base_dir / category / '_category_summary.json'
                with open(category_path, 'w', encoding='utf-8') as f:
                    json.dump(category_summary, f, indent=2, ensure_ascii=False)
        
        failed = [r for r in results['results'] if r['status'] == 'failed']
        if failed:
            failed_path = self.output_base_dir / '_FAILED_DOCUMENTS.json'
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(failed, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self, results: Dict):
        """Print comprehensive summary"""
        print(f"\n{'='*80}")
        print(f"✅ INTELLIGENT VALIDATION COMPLETE")
        print(f"{'='*80}\n")
        
        total = results['total_documents']
        success = results['successful']
        failed = results['failed']
        success_rate = (success / total * 100) if total > 0 else 0
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Documents:  {total}")
        print(f"   ✅ Successful:    {success} ({success_rate:.1f}%)")
        print(f"   ❌ Failed:        {failed}")
        
        print(f"\n📁 Category Breakdown:")
        print(f"{'Category':<45} {'Total':<7} {'Success':<8} {'Subst':<7} {'Generic':<8}")
        print(f"{'-'*80}")
        
        for category, stats in sorted(results['category_stats'].items()):
            print(f"{category:<45} {stats['total']:<7} {stats['success']:<8} "
                  f"{stats['avg_substantive_sections']:<7} {stats['avg_generic_sections']:<8}")
        
        successful_results = [r for r in results['results'] if r['status'] == 'success']
        if successful_results:
            total_subst = sum(r['substantive_sections'] for r in successful_results)
            total_gen = sum(r['generic_sections'] for r in successful_results)
            avg_subst = total_subst / len(successful_results)
            avg_gen = total_gen / len(successful_results)
            
            print(f"\n📈 Section Learning Insights:")
            print(f"   Avg Substantive Sections: {avg_subst:.1f} (sections WITH questions)")
            print(f"   Avg Generic Sections:     {avg_gen:.1f} (sections WITHOUT questions - LLM generatable)")
            print(f"   Total Sections Analyzed:  {total_subst + total_gen}")
        
        failed_results = [r for r in results['results'] if r['status'] == 'failed']
        if failed_results:
            print(f"\n❌ Failed Documents ({len(failed_results)}):")
            for r in failed_results[:5]:
                print(f"   {r['category']}/{r['document']}: {r.get('error', 'Unknown')[:80]}")
        
        print(f"\n💾 Results saved to: {self.output_base_dir}")


# Simple runner
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧠 INTELLIGENT VALIDATOR - Learns Generic vs Substantive Sections")
    print("="*80 + "\n")
    
    BASE_QUESTIONS_DIR = "./generated_questions"
    BASE_SCHEMA_DIR = "./notion_documents"
    OUTPUT_DIR = "./validation_results"
    MAX_WORKERS = 8
    
    if not Path(BASE_QUESTIONS_DIR).exists():
        BASE_QUESTIONS_DIR = input("Enter questions directory: ").strip()
    
    if not Path(BASE_SCHEMA_DIR).exists():
        BASE_SCHEMA_DIR = input("Enter schema directory: ").strip()
    
    print(f"Configuration:")
    print(f"   Questions: {BASE_QUESTIONS_DIR}")
    print(f"   Schemas:   {BASE_SCHEMA_DIR}")
    print(f"   Output:    {OUTPUT_DIR}")
    print(f"   Workers:   {MAX_WORKERS}\n")
    
    print("ℹ️  This validator will:")
    print("   - Learn which sections are substantive (have questions)")
    print("   - Identify generic sections (no questions - LLM generatable)")
    print("   - Reorder questions based ONLY on substantive sections\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        validator = AutoDiscoveryValidator(
            base_questions_dir=BASE_QUESTIONS_DIR,
            base_schema_dir=BASE_SCHEMA_DIR,
            output_base_dir=OUTPUT_DIR
        )
        
        results = validator.run_full_validation(max_workers=MAX_WORKERS)
        print(f"\n🎉 Done! Check {OUTPUT_DIR} for results.")
    else:
        print("❌ Cancelled")