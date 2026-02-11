# run_clean_reorder.py
from clean_reorder import CleanBatchReorderer

reorderer = CleanBatchReorderer(
    questions_dir="./generated_questions",
    schema_dir="./notion_documents",
    output_dir="./reordered_questions"
)

reorderer.run(max_workers=8)
# ```

# ## What This Does:

# ### ✅ **Clean Output Structure**
# ```
# reordered_questions/
# ├── 1._Product_Management/
# │   ├── Feature_specification_documents_questions.json
# │   ├── Product_roadmap_questions.json
# │   └── ...
# ├── 2._Engineering__Software_Development/
# │   └── ...
# └── ...