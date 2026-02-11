import json
import os
from typing import Dict, Any, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING, errors
from pathlib import Path


class DepartmentBasedMongoDBIntegration:
    """
    Store document schemas and QAs in MongoDB organized by department
    
    Collections:
    1. document_schemas - Full JSON schemas (as-is), with department field
    2. document_qas - Optimized Q&A storage with department field
    
    Department-based organization allows single-query filtering
    """
    
    def __init__(self, 
                 connection_string: str = "mongodb+srv://mohitturabit_db_user:yqAjPzqe5HuZ91Gl@cluster0.uzrpe8f.mongodb.net/",
                 database_name: str = "document_automation"):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.schemas_collection = self.db['document_schemas']
        self.qas_collection = self.db['document_qas']
        
        # Create indexes for better performance
        self._create_indexes()
        
        print(f"✅ Connected to MongoDB: {database_name}")
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Schema collection indexes
            self.schemas_collection.create_index([("department", ASCENDING)])
            self.schemas_collection.create_index([("document_type", ASCENDING)])
            self.schemas_collection.create_index([
                ("department", ASCENDING),
                ("document_name", ASCENDING)
            ])
            self.schemas_collection.create_index([("_metadata.page_id", ASCENDING)], unique=True)
            
            # QA collection indexes
            self.qas_collection.create_index([("department", ASCENDING)])
            self.qas_collection.create_index([("document_type", ASCENDING)])
            self.qas_collection.create_index([
                ("department", ASCENDING),
                ("document_type", ASCENDING)
            ])
            self.qas_collection.create_index([("schema_id", ASCENDING)])
            self.qas_collection.create_index([
                ("department", ASCENDING),
                ("category", ASCENDING),
                ("order", ASCENDING)
            ])
            
            print("✅ Department-based indexes created successfully")
        except errors.OperationFailure as e:
            print(f"⚠️  Index creation warning: {e}")
    
    def _extract_department_from_path(self, folder_name: str) -> Dict[str, str]:
        """
        Extract department info from folder name
        
        Examples:
            "1._Product_Management" -> {"code": "1", "name": "Product Management", "slug": "product_management"}
            "2._Engineering_Software_Development" -> {"code": "2", "name": "Engineering Software Development", "slug": "engineering_software_development"}
        
        Args:
            folder_name: Folder name like "1._Product_Management"
            
        Returns:
            Dictionary with department info
        """
        # Remove leading/trailing underscores and split
        parts = folder_name.split('._', 1)
        
        if len(parts) == 2:
            code = parts[0]
            name = parts[1].replace('_', ' ')
        else:
            # Fallback if format is different
            code = "0"
            name = folder_name.replace('_', ' ')
        
        slug = name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        
        return {
            'code': code,
            'name': name,
            'slug': slug
        }
    
    def store_full_schema(self, schema_data: Dict[str, Any], department: Dict[str, str]) -> str:
        """
        Store the complete document schema as-is with department info
        
        Args:
            schema_data: Full JSON schema
            department: Department information
            
        Returns:
            inserted_id: MongoDB document ID
        """
        try:
            # Add department and storage metadata
            schema_data['department'] = department
            schema_data['_storage_metadata'] = {
                'stored_at': datetime.utcnow(),
                'version': '1.0'
            }
            
            # Upsert based on page_id and department
            page_id = schema_data.get('_metadata', {}).get('page_id')
            
            if page_id:
                result = self.schemas_collection.update_one(
                    {
                        '_metadata.page_id': page_id,
                        'department.slug': department['slug']
                    },
                    {'$set': schema_data},
                    upsert=True
                )
                
                if result.upserted_id:
                    return str(result.upserted_id)
                else:
                    # Get existing document ID
                    doc = self.schemas_collection.find_one({
                        '_metadata.page_id': page_id,
                        'department.slug': department['slug']
                    })
                    return str(doc['_id']) if doc else None
            else:
                result = self.schemas_collection.insert_one(schema_data)
                return str(result.inserted_id)
                
        except Exception as e:
            print(f"❌ Error storing schema: {e}")
            return None
    
    def extract_optimized_qas(self, schema_data: Dict[str, Any], schema_id: str, department: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Extract Q&As in optimized format with department info
        
        Stores only essential fields:
        - department (for filtering by department)
        - document_type
        - document_name
        - category
        - category_order
        - question_order
        - question_id
        - question
        - answer_type
        - required
        - options (if applicable)
        - fields (if structured_list)
        - answer (empty initially)
        - answers (empty array for structured_list)
        
        Args:
            schema_data: Full schema
            schema_id: Reference to full schema document
            department: Department information
            
        Returns:
            List of optimized Q&A documents
        """
        optimized_qas = []
        
        document_type = schema_data.get('document_type', 'Unknown')
        document_name = schema_data.get('document_name', 'Unknown')
        
        for category in schema_data.get('question_categories', []):
            category_name = category.get('category', 'Uncategorized')
            category_order = category.get('order', 0)
            
            for idx, question in enumerate(category.get('questions', []), 1):
                # Extract only essential fields
                qa_doc = {
                    'schema_id': schema_id,
                    'department': department,  # Department info for filtering
                    'document_type': document_type,
                    'document_name': document_name,
                    'category': category_name,
                    'category_order': category_order,
                    'question_order': idx,
                    'question_id': question.get('question_id', f'q_{idx}'),
                    'question': question.get('question', ''),
                    'answer_type': question.get('answer_type', 'text'),
                    'required': question.get('required', False),
                    'answer': question.get('answer', ''),
                    '_runtime_metadata': {
                        'created_at': datetime.utcnow(),
                        'last_updated': datetime.utcnow(),
                        'answered': False
                    }
                }
                
                # Add optional fields only if present
                if 'placeholder' in question:
                    qa_doc['placeholder'] = question['placeholder']
                
                if 'description' in question:
                    qa_doc['description'] = question['description']
                
                # For select/multi_select types
                if question.get('answer_type') in ['select', 'multi_select']:
                    qa_doc['options'] = question.get('options', [])
                
                # For structured_list type
                if question.get('answer_type') == 'structured_list':
                    qa_doc['fields'] = question.get('fields', [])
                    qa_doc['answers'] = question.get('answers', [])
                
                optimized_qas.append(qa_doc)
        
        return optimized_qas
    
    def store_qas(self, qas: List[Dict[str, Any]]) -> int:
        """
        Store optimized Q&As in database
        
        Args:
            qas: List of optimized Q&A documents
            
        Returns:
            Number of documents inserted
        """
        try:
            if qas:
                # Clear existing QAs for this schema_id to avoid duplicates
                if qas[0].get('schema_id'):
                    self.qas_collection.delete_many({'schema_id': qas[0]['schema_id']})
                
                result = self.qas_collection.insert_many(qas)
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            print(f"❌ Error storing Q&As: {e}")
            return 0
    
    def process_single_file(self, file_path: str, department: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single JSON file and store in MongoDB
        
        Args:
            file_path: Path to the JSON file
            department: Department information
            
        Returns:
            Processing statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Store full schema with department
            schema_id = self.store_full_schema(schema_data, department)
            
            if not schema_id:
                return {'success': False, 'error': 'Failed to store schema'}
            
            # Extract and store optimized Q&As with department
            qas = self.extract_optimized_qas(schema_data, schema_id, department)
            qa_count = self.store_qas(qas)
            
            return {
                'success': True,
                'schema_id': schema_id,
                'qa_count': qa_count,
                'document_type': schema_data.get('document_type', 'Unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_directory(self, base_dir: str = 'final_filtered_QAs'):
        """
        Process all JSON files in directory structure organized by departments
        
        Args:
            base_dir: Base directory containing department folders
        """
        print(f"\n{'='*70}")
        print("🗄️  DEPARTMENT-BASED MONGODB UPLOAD")
        print(f"{'='*70}\n")
        
        stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_qas': 0,
            'departments': {}
        }
        
        # Find all department folders
        if not os.path.exists(base_dir):
            print(f"❌ Directory not found: {base_dir}")
            return stats
        
        department_folders = [f for f in os.listdir(base_dir) 
                             if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
        
        # Sort folders by their numeric prefix
        department_folders.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        
        print(f"📁 Found {len(department_folders)} departments\n")
        
        # Process each department
        for dept_idx, dept_folder in enumerate(department_folders, 1):
            department = self._extract_department_from_path(dept_folder)
            
            print(f"\n{'#'*70}")
            print(f"🏢 DEPARTMENT {dept_idx}/{len(department_folders)}: {department['name']}")
            print(f"   Code: {department['code']} | Slug: {department['slug']}")
            print(f"{'#'*70}\n")
            
            dept_path = os.path.join(base_dir, dept_folder)
            json_files = [f for f in os.listdir(dept_path) 
                         if f.endswith('.json') and not f.startswith('.')]
            
            dept_stats = {'successful': 0, 'failed': 0, 'qas': 0, 'documents': []}
            
            for file_idx, filename in enumerate(json_files, 1):
                file_path = os.path.join(dept_path, filename)
                
                print(f"[{file_idx}/{len(json_files)}] {filename}")
                
                result = self.process_single_file(file_path, department)
                
                if result['success']:
                    print(f"   ✅ Schema stored | {result['qa_count']} Q&As stored | Dept: {department['slug']}")
                    stats['successful'] += 1
                    dept_stats['successful'] += 1
                    stats['total_qas'] += result['qa_count']
                    dept_stats['qas'] += result['qa_count']
                    dept_stats['documents'].append(result['document_type'])
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    stats['failed'] += 1
                    dept_stats['failed'] += 1
                
                stats['total_files'] += 1
            
            stats['departments'][department['name']] = dept_stats
            
            print(f"\n{'─'*70}")
            print(f"Department Summary: {dept_stats['successful']} docs, {dept_stats['qas']} Q&As")
            print(f"{'─'*70}")
        
        # Final summary
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict[str, Any]):
        """Print comprehensive summary"""
        
        print(f"\n{'='*70}")
        print("🎉 UPLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total files: {stats['total_files']}")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"📝 Total Q&As stored: {stats['total_qas']}")
        
        print(f"\n{'─'*70}")
        print("🏢 BY DEPARTMENT:")
        print(f"{'─'*70}")
        
        for dept, counts in stats['departments'].items():
            status = "✅" if counts['failed'] == 0 else "⚠️"
            print(f"{status} {dept}")
            print(f"    └─ {counts['successful']} documents | {counts['qas']} questions")
        
        print(f"\n{'─'*70}")
        print("🗄️  MONGODB COLLECTIONS:")
        print(f"{'─'*70}")
        print(f"📚 document_schemas: {self.schemas_collection.count_documents({})} documents")
        print(f"❓ document_qas: {self.qas_collection.count_documents({})} questions")
        
        # Show department breakdown
        dept_counts = self.schemas_collection.aggregate([
            {
                '$group': {
                    '_id': '$department.name',
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'_id': 1}}
        ])
        
        print(f"\n{'─'*70}")
        print("📊 SCHEMAS BY DEPARTMENT:")
        print(f"{'─'*70}")
        for dept in dept_counts:
            print(f"   {dept['_id']}: {dept['count']} documents")
        
        print(f"{'='*70}\n")
    
    def get_departments(self) -> List[Dict[str, str]]:
        """Get all unique departments"""
        departments = self.qas_collection.aggregate([
            {
                '$group': {
                    '_id': '$department',
                    'doc_count': {'$sum': 1}
                }
            },
            {'$sort': {'_id.code': 1}}
        ])
        
        return [dept['_id'] for dept in departments]
    
    def get_document_types_by_department(self, department_slug: str) -> List[str]:
        """Get all document types for a specific department"""
        return self.qas_collection.distinct('document_type', {'department.slug': department_slug})
    
    def get_qas_by_department_and_type(self, department_slug: str, document_type: str) -> List[Dict[str, Any]]:
        """
        Get all Q&As for a specific department and document type (single query!)
        
        Args:
            department_slug: Department slug (e.g., "product_management")
            document_type: Type of document
            
        Returns:
            List of Q&As sorted by category and order
        """
        qas = list(self.qas_collection.find({
            'department.slug': department_slug,
            'document_type': document_type
        }).sort([
            ('category_order', ASCENDING),
            ('question_order', ASCENDING)
        ]))
        
        # Remove MongoDB _id for cleaner output
        for qa in qas:
            qa['_id'] = str(qa['_id'])
        
        return qas
    
    def get_all_qas_by_department(self, department_slug: str) -> List[Dict[str, Any]]:
        """
        Get ALL Q&As for a department (all document types)
        
        Args:
            department_slug: Department slug
            
        Returns:
            List of all Q&As in department
        """
        qas = list(self.qas_collection.find({
            'department.slug': department_slug
        }).sort([
            ('document_type', ASCENDING),
            ('category_order', ASCENDING),
            ('question_order', ASCENDING)
        ]))
        
        for qa in qas:
            qa['_id'] = str(qa['_id'])
        
        return qas
    
    def update_answer(self, qa_id: str, answer: Any):
        """
        Update answer for a specific question
        
        Args:
            qa_id: MongoDB document ID
            answer: Answer value (string or list for structured_list)
        """
        from bson import ObjectId
        
        try:
            self.qas_collection.update_one(
                {'_id': ObjectId(qa_id)},
                {
                    '$set': {
                        'answer': answer,
                        '_runtime_metadata.last_updated': datetime.utcnow(),
                        '_runtime_metadata.answered': True
                    }
                }
            )
            return True
        except Exception as e:
            print(f"❌ Error updating answer: {e}")
            return False
    
    def get_schema_by_department_and_type(self, department_slug: str, document_type: str) -> Dict[str, Any]:
        """Get full schema for a department and document type"""
        schema = self.schemas_collection.find_one({
            'department.slug': department_slug,
            'document_type': document_type
        })
        if schema:
            schema['_id'] = str(schema['_id'])
        return schema
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("✅ MongoDB connection closed")


def main():
    """Main entry point for batch upload"""
    import sys
    
    print(f"\n{'='*70}")
    print("DEPARTMENT-BASED MONGODB INTEGRATION")
    print(f"{'='*70}\n")
    
    # Configuration
    print("📋 Configuration:")
    print("─" * 70)
    
    # Get MongoDB connection string
    default_conn = "mongodb+srv://mohitturabit_db_user:yqAjPzqe5HuZ91Gl@cluster0.uzrpe8f.mongodb.net/"
    conn_string = os.getenv('MONGODB_CONNECTION_STRING', default_conn)
    
    print(f"MongoDB URI: {conn_string}")
    
    if conn_string == default_conn:
        print("💡 Using default local MongoDB")
        print("   To use remote MongoDB, set MONGODB_CONNECTION_STRING env variable")
    
    db_name = os.getenv('MONGODB_DATABASE', 'document_automation')
    print(f"Database: {db_name}")
    
    input_dir = os.getenv('INPUT_DIR', 'final_filtered_QAs')
    print(f"Input directory: {input_dir}")
    
    print("─" * 70)
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"\n❌ Directory not found: {input_dir}")
        print("   Please run add_answer_fields.py first to create the directory")
        sys.exit(1)
    
    # Count files
    total_files = 0
    dept_count = 0
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            dept_count += 1
            json_files = [f for f in os.listdir(item_path) if f.endswith('.json') and not f.startswith('.')]
            total_files += len(json_files)
    
    if total_files == 0:
        print(f"\n❌ No JSON files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n📊 Found {dept_count} departments with {total_files} total documents")
    print(f"\n{'─'*70}\n")
    
    # Confirm
    confirm = input("Continue with upload to MongoDB? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n❌ Upload cancelled")
        sys.exit(0)
    
    # Initialize MongoDB integration
    try:
        mongo = DepartmentBasedMongoDBIntegration(
            connection_string=conn_string,
            database_name=db_name
        )
        
        # Process all files
        stats = mongo.process_directory(input_dir)
        
        # Close connection
        mongo.close()
        
        print(f"\n✨ Upload complete!")
        print(f"📊 {stats['successful']}/{stats['total_files']} files uploaded successfully")
        print(f"📝 {stats['total_qas']} questions ready for Streamlit UI")
        print(f"🏢 {len(stats['departments'])} departments organized\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure MongoDB is running and accessible")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()