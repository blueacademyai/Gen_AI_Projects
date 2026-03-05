# RAG Support System Implementation Guide

This repository contains a Retrieval-Augmented Generation (RAG) system for support tickets. The system uses vector embeddings to find similar support tickets and leverages a large language model to generate contextual responses based on retrieved documents.

## Implementation Requirements

### 1. Document Loader (`document_loader.py`)

This module is responsible for loading and processing support ticket documents from various sources.

#### Key Requirements:

- Implement `SupportDocumentLoader` class with methods to load JSON and XML files
- Process support tickets into a standardized document format with proper metadata
- Generate unique IDs for each support ticket
- Handle various file formats and data structures

#### JSON Handling:

1. **Implement `get_json_content()` Function**:
   - Format JSON data into a standardized content string with this exact format:
     ```
     Subject: {data.get('subject', '')}
     Description: {data.get('body', '')}
     Resolution: {data.get('answer', '')}
     Type: {data.get('type', '')}
     Queue: {data.get('queue', '')}
     Priority: {data.get('priority', '')}
     ```

2. **Implement `get_json_metadata()` Function**:
   - Extract and format metadata with these exact fields:
     ```python
     {
         'ticket_id': str,             # A unique ID (format: "{support_type}_{original_id}")
         'original_ticket_id': str,    # Original ticket ID from the JSON ("Ticket ID" field)
         'support_type': str,          # Type of support (technical, product, customer)
         'type': str,                  # Type field from the original data
         'queue': str,                 # Queue information
         'priority': str,              # Priority level
         'language': str,              # Content language
         'tags': List[str],            # List of relevant tags from tag_1 through tag_8
         'source': 'json',             # Source format identifier
         # Include original content fields for reference
         'subject': str,
         'body': str,
         'answer': str
     }
     ```
   - Filter out 'NaN' or 'nan' values from tags

3. **JSONLoader Implementation**:
   - When using LangChain's `JSONLoader`, you MUST use a proper function to pass the support_type:
     ```python
     # Create a function that captures support_type in its closure
     def metadata_transform(record, metadata=None):
         return self.get_json_metadata(record, support_type)
         
     # Configure and use JSONLoader with this function
     loader = JSONLoader(
         file_path=str(file_path),
         jq_schema='.[]',
         content_key=None,
         text_content=False,
         metadata_func=metadata_transform
     )
     ```

#### XML Handling:

1. **Implement `load_xml_tickets()` Function**:
   - Parse XML files using `xml.etree.ElementTree`
   - Extract ticket elements using `root.findall('.//Ticket')`
   - Format content string with this exact format:
     ```
     Subject: {ticket_elem.findtext('subject')}
     Description: {ticket_elem.findtext('body')}
     Resolution: {ticket_elem.findtext('answer')}
     Type: {ticket_elem.findtext('type')}
     Queue: {ticket_elem.findtext('queue')}
     Priority: {ticket_elem.findtext('priority')}
     ```
   - Create metadata with these exact fields:
     ```python
     {
         'ticket_id': unique_id,           # Format: "{support_type}_xml_{original_id}"
         'original_ticket_id': str,        # Original ticket ID from the XML (TicketID field)
         'support_type': support_type,     # Type of support (technical, product, customer)
         'type': str,                      # Type field
         'queue': str,                     # Queue information
         'priority': str,                  # Priority level
         'language': str,                  # Content language
         'tags': List[str],                # List of relevant tags
         'source': 'xml'                   # Source format identifier
     }
     ```
   - Filter out 'nan' values from tags

#### Error Handling Requirements:

- Check if data path exists at initialization; raise `FileNotFoundError` if not
- Skip files that don't exist with appropriate warning logs
- Handle exceptions during file loading with error logs
- Validate unique IDs across all documents; raise `ValueError` for duplicates

### 2. Vector Store (`vector_store.py`)

This module manages the vector embeddings for support tickets and provides similarity search functionality.

#### Key Requirements:

- Implement `SupportVectorStore` class with ChromaDB integration
- Create methods for storing, retrieving, and querying embeddings
- Process metadata for ChromaDB compatibility
- Handle various edge cases and input validation

#### Input Validation Requirements:

- **Empty Queries**: You MUST check for empty queries (null or whitespace-only) in `query_similar()` and return an empty list with a warning log
- **Short Queries**: You MUST check that queries are at least 10 characters in `get_relevant_documents()` and reject shorter queries with `ValueError("Query too short. Please provide more details.")`
- **Non-existent Support Types**: Handle gracefully by returning an empty list with a warning log

#### Metadata Processing:

1. **Implement `_prepare_metadata()` Function**:
   - Convert lists to comma-separated strings
   - Convert None values to empty strings
   - Ensure all metadata values have ChromaDB-compatible types

2. **Implement `_process_metadata_for_return()` Function**:
   - Convert comma-separated tag strings back to proper lists
   - Process any other fields that need type conversion

#### Collection Management:

- Create separate collections for each support type
- Handle the case where no documents exist for a support type
- Properly initialize embeddings model for vector generation

### 3. RAG Chain (`rag_chain.py`)

This module combines vector retrieval with language model generation to provide contextual responses.

#### Key Requirements:

- Implement `SupportRAGChain` class with vector store integration
- Create methods for retrieving relevant documents and generating responses
- Format context from retrieved documents
- Implement comprehensive error handling

#### Query Validation Requirements:

The following error handling in `query()` method is MANDATORY with EXACT error messages:

1. **Empty Query Check**:
   ```python
   if not query:
       raise ValueError("Query cannot be empty")
   if query.strip() == "":
       raise ValueError("Query cannot be empty")
   ```

2. **Short Query Check**:
   ```python
   if len(query.strip()) < 10:
       raise ValueError("Query too short. Please provide more details.")
   ```

#### Context Formatting Requirements:

When preparing context in `_prepare_context()`, you MUST use the following EXACT format:

```
Ticket {i}:
Support Type: {doc['metadata'].get('support_type', 'Unknown')}
Tags: {', '.join(doc['metadata'].get('tags', []))}
Content: {doc['content']}
```

This format is required for tests to pass. Do not modify or add additional fields.

#### No Documents Case:

When no relevant documents are found, return the exact message:
```
"No relevant support tickets found."
```

## Edge Cases to Handle

### Document Loader Edge Cases:

1. **Missing Files**:
   - Skip missing files with appropriate warning logs
   - Continue processing other files

2. **Malformed JSON/XML**:
   - Catch exceptions during parsing
   - Log errors and continue with other files

3. **Empty Files or Collections**:
   - Handle gracefully with appropriate logs
   - Return empty collections when appropriate

4. **Duplicate IDs**:
   - Check for duplicate ticket IDs across all documents
   - Raise `ValueError` with message "Duplicate ticket ID found: {ticket_id}"

### Vector Store Edge Cases:

1. **Empty Query Handling**:
   - `query_similar()` MUST check for empty or whitespace-only queries
   - Return an empty list with appropriate warning log
   - Do NOT raise an exception for empty queries at this level

2. **Non-existent Support Type**:
   - Handle gracefully by returning an empty list
   - Include warning log: "Support type '{support_type}' not found"

3. **No Documents in Collection**:
   - Handle gracefully, returning empty results

### RAG Chain Edge Cases:

1. **Empty Query Handling**:
   - `query()` MUST reject empty queries with `ValueError("Query cannot be empty")`
   - Handle both null and whitespace-only queries

2. **Short Query Handling**:
   - `query()` MUST reject queries shorter than 10 characters with `ValueError("Query too short. Please provide more details.")`

3. **No Relevant Documents**:
   - Return the message "No relevant support tickets found." in the context

4. **LLM Generation Errors**:
   - Catch and log any errors during LLM response generation
   - Propagate errors with appropriate context

## File Format Examples

### JSON Structure Example

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

### XML Structure Example

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
    <!-- Additional tag_N elements may be present -->
    <TicketID>12345</TicketID>
  </Ticket>
  <!-- Additional Ticket elements -->
</SupportTickets>
```