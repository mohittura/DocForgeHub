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