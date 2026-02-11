import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List


class QuestionAnswerProcessor:
    """Add answer fields to questions and reorganize by topics"""
    
    def __init__(self, input_dir: str = 'generated_questions', output_dir: str = 'final_filtered_QAs'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'topics': {}
        }
    
    def add_answer_fields(self, questions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add empty answer field after each question"""
        
        # Process each category
        for category in questions_data.get('question_categories', []):
            for question in category.get('questions', []):
                # Add answer field right after question
                if 'answer' not in question:
                    question['answer'] = ""
                
                # For structured_list type, add answers array for multiple entries
                if question.get('answer_type') == 'structured_list':
                    if 'answers' not in question:
                        question['answers'] = []
        
        return questions_data
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single JSON file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add answer fields
            updated_data = self.add_answer_fields(data)
            
            return updated_data
            
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(file_path)}: {e}")
            return None
    
    def organize_by_topics(self):
        """Main function to process all files and organize by topics"""
        
        print(f"\n{'='*70}")
        print("📋 QUESTION-ANSWER FIELD PROCESSOR")
        print(f"{'='*70}\n")
        
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📂 Output directory: {self.output_dir}\n")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find all topic folders
        topic_folders = []
        if os.path.exists(self.input_dir):
            for item in os.listdir(self.input_dir):
                item_path = os.path.join(self.input_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    topic_folders.append(item)
        
        if not topic_folders:
            print("❌ No topic folders found!")
            return
        
        print(f"Found {len(topic_folders)} topic folders:\n")
        for idx, folder in enumerate(topic_folders, 1):
            print(f"  {idx}. {folder}")
        
        print(f"\n{'─'*70}\n")
        
        # Process each topic folder
        for topic_idx, topic_folder in enumerate(topic_folders, 1):
            print(f"\n{'#'*70}")
            print(f"📁 TOPIC {topic_idx}/{len(topic_folders)}: {topic_folder}")
            print(f"{'#'*70}\n")
            
            input_topic_path = os.path.join(self.input_dir, topic_folder)
            output_topic_path = os.path.join(self.output_dir, topic_folder)
            
            # Create output topic folder
            os.makedirs(output_topic_path, exist_ok=True)
            
            # Get all JSON files in this topic
            json_files = [f for f in os.listdir(input_topic_path) if f.endswith('.json') and not f.startswith('.')]
            
            print(f"Found {len(json_files)} documents in this topic\n")
            
            processed_count = 0
            failed_count = 0
            
            # Process each file
            for file_idx, filename in enumerate(json_files, 1):
                input_file_path = os.path.join(input_topic_path, filename)
                output_file_path = os.path.join(output_topic_path, filename)
                
                print(f"[{file_idx}/{len(json_files)}] Processing: {filename}")
                
                # Process the file
                updated_data = self.process_file(input_file_path)
                
                if updated_data:
                    # Save to output directory
                    try:
                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            json.dump(updated_data, f, indent=2, ensure_ascii=False)
                        
                        question_count = sum(len(cat.get('questions', [])) for cat in updated_data.get('question_categories', []))
                        print(f"   ✅ Added answer fields to {question_count} questions")
                        
                        processed_count += 1
                        self.stats['processed'] += 1
                        
                    except Exception as e:
                        print(f"   ❌ Failed to save: {e}")
                        failed_count += 1
                        self.stats['failed'] += 1
                else:
                    failed_count += 1
                    self.stats['failed'] += 1
            
            # Topic summary
            self.stats['topics'][topic_folder] = {
                'processed': processed_count,
                'failed': failed_count,
                'total': len(json_files)
            }
            
            print(f"\n{'─'*70}")
            print(f"Topic Summary: {processed_count} processed, {failed_count} failed")
            print(f"{'─'*70}")
            
            self.stats['total_files'] += len(json_files)
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print comprehensive final summary"""
        
        print(f"\n{'='*70}")
        print("🎉 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"📊 Total files: {self.stats['total_files']}")
        print(f"✅ Processed: {self.stats['processed']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📁 Output directory: {self.output_dir}")
        
        print(f"\n{'─'*70}")
        print("📂 BY TOPIC:")
        print(f"{'─'*70}")
        
        for topic, counts in self.stats['topics'].items():
            status = "✅" if counts['failed'] == 0 else "⚠️"
            print(f"{status} {topic}: {counts['processed']}/{counts['total']} documents")
        
        print(f"{'='*70}\n")
        
        if self.stats['processed'] > 0:
            print(f"✨ All documents are now ready with answer fields!")
            print(f"📍 Location: {os.path.abspath(self.output_dir)}\n")


class BatchQuestionAnswerProcessor:
    """Process questions with batch confirmation"""
    
    def __init__(self, input_dir: str = 'generated_questions', output_dir: str = 'final_filtered_QAs'):
        self.processor = QuestionAnswerProcessor(input_dir, output_dir)
    
    def preview_changes(self, sample_file: str):
        """Show a preview of what changes will be made"""
        
        print(f"\n{'='*70}")
        print("📋 PREVIEW OF CHANGES")
        print(f"{'='*70}\n")
        
        if not os.path.exists(sample_file):
            print("❌ Sample file not found for preview")
            return
        
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("BEFORE (original):")
            print("─" * 70)
            if data.get('question_categories'):
                sample_q = data['question_categories'][0]['questions'][0]
                print(json.dumps({k: sample_q[k] for k in list(sample_q.keys())[:4]}, indent=2))
            
            print("\n")
            
            # Add answer field
            updated_data = self.processor.add_answer_fields(data)
            
            print("AFTER (with answer field):")
            print("─" * 70)
            if updated_data.get('question_categories'):
                sample_q = updated_data['question_categories'][0]['questions'][0]
                print(json.dumps({k: sample_q[k] for k in list(sample_q.keys())[:5]}, indent=2))
            
            print(f"\n{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Error showing preview: {e}")
    
    def run_with_confirmation(self):
        """Run with user confirmation"""
        
        print(f"\n{'='*70}")
        print("🤖 BATCH QUESTION-ANSWER PROCESSOR")
        print(f"{'='*70}\n")
        
        # Find a sample file for preview
        sample_file = None
        if os.path.exists(self.processor.input_dir):
            for root, dirs, files in os.walk(self.processor.input_dir):
                for file in files:
                    if file.endswith('.json') and not file.startswith('.'):
                        sample_file = os.path.join(root, file)
                        break
                if sample_file:
                    break
        
        if sample_file:
            self.preview_changes(sample_file)
            
            confirm = input("Continue with processing all files? (yes/no): ").strip().lower()
            
            if confirm not in ['yes', 'y']:
                print("\n❌ Processing cancelled by user")
                return
        
        print("\n🔄 Starting batch processing...\n")
        self.processor.organize_by_topics()


def main():
    """Main entry point"""
    import sys
    
    print(f"\n{'='*70}")
    print("QUESTION-ANSWER FIELD AUTOMATION")
    print(f"{'='*70}")
    print("\nThis tool will:")
    print("  1. Read all question files from 'generated_questions/' folder")
    print("  2. Add empty 'answer' field after each question")
    print("  3. Organize by topics in 'final_filtered_QAs/' folder")
    print("  4. Maintain the same folder structure\n")
    
    # Check if input directory exists
    input_dir = 'generated_questions'
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' not found!")
        print(f"   Please make sure you have generated questions first.\n")
        sys.exit(1)
    
    # Count files
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        total_files += len([f for f in files if f.endswith('.json') and not f.startswith('.')])
    
    if total_files == 0:
        print(f"❌ No JSON files found in '{input_dir}'!")
        sys.exit(1)
    
    print(f"📊 Found {total_files} question files to process\n")
    print(f"{'─'*70}\n")
    
    # Ask for confirmation
    confirm = input("Start processing? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n❌ Processing cancelled")
        sys.exit(0)
    
    # Run processor
    batch_processor = BatchQuestionAnswerProcessor(
        input_dir='generated_questions',
        output_dir='final_filtered_QAs'
    )
    
    batch_processor.processor.organize_by_topics()


if __name__ == "__main__":
    main()