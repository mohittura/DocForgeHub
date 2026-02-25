started the project
added all the automations
added streamlit ui
added the schemas and questions
added fastapi endpoints for getting the departments, documents, and questions from mongo db
added fastapi endpoint to receive the generated documents url
ui added fastapi helper functions to get and populate the departments, document_types, generated page history and questions in the ui
added some backend agent schema
added fastapi endpoint to get the required sections from mongo
added helper functions in agent to check table only schema as well as added a helper function to return the columns from the tabular schema
added a fill schema gaps graph node which induces relevant content to the text with areas which have low to no questions
added a fastapi post endpoint "generate" to send the request to the LLM for document generation
added build prompt node, made a generate document node, made a helper function validate document structure which validates that the generated document follows the schema structure and returns a list of error messages if its not valid in the agent backend schema
added quality gate node which checks every possibility of bad generations and will fail if bad generation is caught
added fix document node which will receive the original document, the lis tof issues and suggestions and will ask the llm to produce a corrected version
added routing logic, build the graph, made a async function to run the agent

added in ui to enhance streamlit UI with new input types, generate flow, and quality feedback
Bug fixes:

Fix get_notionpage_urls_from_fastapi to use .get("pages", []) with default to prevent NoneType crash
Guard st.session_state.history init with pages or [] to handle None response
Fix Document selectbox fallback text to "(select a department first)" to align with valid_document guard condition

New features:

Add document_name_lookup dict built from doc_types to correctly resolve document_name for /generate payload
Add select and multi_select answer type rendering (st.selectbox, st.multiselect) in the questions panel
Wire up "Generate Document" button to actually call call_generate_endpoint with a loading spinner
Build and send full Q&A payload (question, answer, category, answer_type) to the agent on generate
Display post-generation feedback: success/warning status banner with retry count
Add quality scores expander with st.metric display per criterion
Add quality issues expander listing flagged problems
Add suggestions for improvement expander
Add "📖 Preview rendered document" expander below the Markdown editor for live preview
Improve Publish button to validate content exists before triggering balloons

Refactors / polish:

Standardize Generate button label to "Generate Document" with use_container_width=True and disabled state guard
Standardize Publish button label to "Publish"

added some things in the prompts.py : feat(agent)-notFinal: made some prompts to test the agent which includes system prompts, table only prompts, Schema gap filler prompts, quality review prompts, and made builder functions for the same, these functions will build the final prompt based on the inputs received


added progressive generation in the app, made relavant endpoints (generate-section), made ui changes and added pagination to display the questions, as well as added progessive generation toggle button, made changes in the agent schema to handle the same.
some bug fixes
refactor: renamed cryptic variables to meaningful ones, added helper function _is_answered to resolve the last answer button lock issue