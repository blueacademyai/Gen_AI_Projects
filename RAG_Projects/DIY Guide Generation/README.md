# RAG Support System Implementation

This repository contains a Retrieval-Augmented Generation (RAG) system for support tickets. The system uses vector embeddings to find similar support tickets and leverages a large language model to generate contextual responses based on retrieved documents.

## Overview

The RAG Support System consists of three main components:

1. **Document Loader (`document_loader.py`)**: Loads and processes support ticket data from JSON and XML files.
2. **Vector Store (`vector_store.py`)**: Manages the vector embeddings database for efficient similarity search.
3. **RAG Chain (`rag_chain.py`)**: Combines retrieval and generation to provide contextual responses.

## Requirements

- Python 3.8+
- LangChain
- OpenAI API key (for embeddings and LLM)
- ChromaDB
- Required Python packages (install via `pip install -r requirements.txt`):
  ```
  langchain
  langchain_openai
  python-dotenv
  chromadb
  uuid
  ```

## Environment Setup

1. Clone the repository
2. Install the required packages
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Component Details

### 1. Document Loader (`document_loader.py`)

The Document Loader is responsible for processing support ticket files from JSON and XML formats into a standardized document format.

#### Key Features:

- Loads support tickets from both JSON and XML files
- Organizes tickets by support type (technical, product, customer)
- Creates embeddings content from subject and body fields only
- Extracts comprehensive metadata for each ticket

#### Content and Metadata Structure:

**Page Content (for embedding):**
- The page content should ONLY include the concatenation of:
  ```
  {subject}\n{body}
  ```
- This focused approach ensures embeddings are created only for the essential content, improving relevance in retrieval.

**Metadata (stored but not embedded):**
- The metadata should include the following fields:
  ```python
  {
      'ticket_id': str,             # Ticket identifier
      'support_type': str,          # Type of support (technical, product, customer)
      'type': str,                  # Type field from the original data
      'queue': str,                 # Queue information
      'priority': str,              # Priority level
      'language': str,              # Content language (defaults to 'en')
      'tags': List[str],            # List of relevant tags (tag_1 through tag_8, excluding NaN)
      'subject': str,               # Original subject field
      'body': str,                  # Original body field
      'answer': str,                # Original answer/resolution field
      'source': str                 # Source format identifier ('json' or 'xml')
  }
  ```
- All metadata is stored for filtering and retrieval but is not included in the embedding vectors.

#### Input File Requirements:

**JSON Format**:
- Must contain the following fields:
  - `subject`: The ticket subject line
  - `body`: The detailed description of the issue
  - `answer`: The solution or response provided
  - `type`: The type of ticket (e.g., Request, Incident)
  - `queue`: The department or queue assigned to the ticket
  - `priority`: The priority level (e.g., high, medium, low)
  - `Ticket ID`: A unique identifier for the ticket
- May contain tag fields (`tag_1` through `tag_8`) for categorization
- Example JSON structure:
```json
{
    "subject": "Customer Support Inquiry",
    "body": "Seeking information on digital strategies...",
    "answer": "We offer a variety of digital strategies and services...",
    "type": "Request",
    "queue": "Customer Service",
    "priority": "medium",
    "language": "en",
    "tag_1": "Feedback",
    "tag_2": "Sales",
    "tag_3": "IT",
    "tag_4": "Tech Support",
    "tag_5": "NaN",
    "tag_6": "NaN",
    "tag_7": "NaN",
    "tag_8": "NaN",
    "Ticket ID": "123abc"
}
```

**XML Format**:
- Must have `<Ticket>` elements containing the following child elements:
  - `<subject>`: The ticket subject line
  - `<body>`: The detailed description of the issue
  - `<answer>`: The solution or response provided
  - `<type>`: The type of ticket
  - `<queue>`: The department or queue assigned to the ticket
  - `<priority>`: The priority level
  - `<TicketID>`: A unique identifier for the ticket
- May contain tag fields (`<tag_1>` through `<tag_8>`) for categorization
- Example XML structure:
```xml
<SupportTickets>
  <Ticket>
    <subject>Browser Login Issue</subject>
    <body>User cannot login using Chrome browser</body>
    <answer>Clear browser cache and cookies</answer>
    <type>Technical</type>
    <queue>Tech Support</queue>
    <priority>high</priority>
    <language>en</language>
    <tag_1>browser</tag_1>
    <tag_2>login</tag_2>
    <tag_3>chrome</tag_3>
    <TicketID>12345</TicketID>
  </Ticket>
</SupportTickets>
```

### 2. Vector Store (`vector_store.py`)

The Vector Store manages the embeddings database using ChromaDB for efficient similarity search.

#### Key Features:

- Creates separate collections for each support type
- Handles metadata processing for ChromaDB compatibility
- Provides similarity search functionality with filtering options
- Properly manages conversion between native and ChromaDB data types

#### Metadata Processing:

For ChromaDB compatibility, the vector store performs these key metadata transformations:
1. **List to String Conversion**: 
   - All list values (like tags) are converted to comma-separated strings
   - Example: `['browser', 'login']` → `'browser,login'`

2. **None Value Handling**: 
   - All None values are converted to empty strings
   - Example: `None` → `''`

3. **Type Validation**:
   - Ensures all metadata values are ChromaDB-compatible primitive types (string, int, float, bool)
   - Non-primitive types are stringified

4. **Reverse Transformation**:
   - When retrieving documents, comma-separated strings are converted back to lists
   - Example: `'browser,login'` → `['browser', 'login']`

#### Important Requirements:

- Empty queries are rejected with an empty result list (not an exception)
- Query handles support type filtering correctly
- Metadata is properly processed for ChromaDB compatibility

### 3. RAG Chain (`rag_chain.py`)

The RAG Chain combines the vector store retrieval with LLM-based generation to provide contextual responses to support queries.

#### Key Features:

- Uses ChatOpenAI for response generation
- Implements proper error handling for query validation
- Formats retrieved documents into structured context for the LLM
- Uses a prompt template designed for support specialist responses

#### Context Formatting:

When preparing context for the LLM, the RAG chain formats each retrieved document with this EXACT structure:
```
Ticket {i}:
Support Type: {support_type}
Tags: {tags}
Answer: {answer}
Content: {content}
```

This structured format helps the LLM understand the relevant support ticket information and generate appropriate responses. If no documents are found, the context will be:
```
"No relevant support tickets found."
```
#### Query Validation:

The RAG chain implements strict input validation:

1. **Empty Query Handling**:
   ```python
   if not query:
       raise ValueError("Query cannot be empty")
   if query.strip() == "":
       raise ValueError("Query cannot be empty")
   ```

2. **Short Query Handling**:
   ```python
   if len(query.strip()) < 10:
       raise ValueError("Query too short. Please provide more details.")
   ```

These validations ensure that queries are substantive enough for meaningful retrieval and response generation. The error messages MUST be exactly as shown above for test compatibility.

## Data Flow

1. **Document Loading**:
   - Support ticket files are loaded and processed into Document objects
   - Each document has page_content (subject + body) and metadata fields

2. **Vector Store Creation**:
   - Documents are organized by support type
   - Embeddings are created for each document's content
   - Documents and metadata are stored in ChromaDB

3. **Query Processing**:
   - User query is validated
   - Query is converted to embedding and similar documents are retrieved
   - Retrieved documents are formatted into structured context
   - LLM generates a response based on the context and query

## Common Issues and Solutions

### Missing API Key
If you encounter an error about a missing API key, ensure your `.env` file is properly set up with OPENAI_API_KEY.

### File Not Found Errors
Ensure your data directory structure matches the expected paths for your support ticket files.

### Empty Results
If queries are returning empty results:
- Check that your query meets the minimum length requirement (10 characters)
- Verify that your vector store has been created and populated correctly
- Check if you're filtering by a support type that doesn't exist

### LLM Response Errors
If the LLM fails to generate a response:
- Check your OpenAI API key validity and quota
- Ensure your prompt template is correctly formatted
- Verify that context is correctly formatted for the LLM
