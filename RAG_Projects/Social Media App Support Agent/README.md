# Social Media App Support Agent

This repository contains a Retrieval-Augmented Generation (RAG) system for X (formerly Twitter) support agent. The system uses vector embeddings to find relevant documentation and leverages a large language model to generate contextual responses to user queries.

## Overview

The Social Media Support Agent consists of three main components:

1. **Document Loader (`document_loader.py`)**: Loads and processes support documentation from text files.
2. **Vector Store (`vector_store.py`)**: Manages the FAISS vector embeddings database for efficient similarity search.
3. **RAG Chain (`rag_chain.py`)**: Combines retrieval and generation to provide contextual responses using LangChain Expression Language (LCEL).


## Project Structure

```
.
├── data/               # Documentation files
├── faiss_index/        # FAISS vector database storage
├── src/                # Source code
│   ├── __init__.py
│   ├── document_loader.py
│   ├── rag_chain.py
│   └── vector_store.py
├── tests/              # Test files
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_document_loader.py
│   ├── test_integration.py
│   ├── test_rag.py
│   └── test_vector_store.py
├── .env                # Environment variables
├── app.py              # Main Streamlit application
├── README.md           # This file
└── requirements.txt    # Dependencies
```

## Requirements

- Python 3.10+
- LangChain and related packages
- OpenAI API key (for embeddings and LLM)
- FAISS for vector storage
- Required Python packages (install via `pip install -r requirements.txt`):
```
langchain==0.3.22
langchain-core==0.3.49
langchain-community==0.3.20
langchain-text-splitters==0.3.7
faiss-cpu==1.10.0
openai==1.70.0
langchain-openai==0.3.11
python-dotenv==1.0.1
pytest==8.3.4
pytest-cov==6.0.0
streamlit==1.36.0
```

## Environment Setup

1. Clone the repository
2. Install the required packages with `pip install -r requirements.txt`
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MODEL_NAME=gpt-4o
   TEMPERATURE=0.0
   MAX_DOCUMENTS=4
   ```

## Component Details

### 1. Document Loader (`document_loader.py`)

The `SocialMediaDocumentLoader` class is responsible for processing X (Twitter) documentation files and preparing them for vectorization.

#### Key Features:

- Loads documentation from text files using glob patterns
- Cleans and preprocesses text content (removes non-breaking spaces, standardizes whitespace)
- Splits documents into optimally sized chunks using recursive character splitting
- Filters out chunks that are too small to be meaningful
- Removes common footer text that appears in documentation
- Preserves source filename in document metadata

### 2. Vector Store (`vector_store.py`)

The `SocialMediaVectorStore` class manages the FAISS embeddings database for efficient similarity search.

#### Key Features:

- Creates FAISS vector stores from document collections
- Uses OpenAI's embedding model for vector creation (defaults to OpenAIEmbeddings)
- Supports saving and loading vector stores to/from disk
- Provides methods for similarity search with scores
- Offers standalone embedding generation for arbitrary text
- Handles errors gracefully during vector store operations

### 3. RAG Chain (`rag_chain.py`)

The `SocialMediaRAGChain` class combines the vector store retrieval with LLM-based generation to provide contextual responses to user queries using LangChain Expression Language (LCEL).

#### Key Features:

- Implements a streamlined RAG pipeline using LCEL components
- Filters retrieved documents based on a similarity threshold
- Uses ChatOpenAI with GPT-4o model by default
- Formats retrieved documents into structured context for the LLM
- Provides source citations and document metadata
- Enables direct document retrieval for debugging purposes
- Implements robust error handling

## Usage

### Running the Support Agent

The main application is implemented as a Streamlit web app in `app.py`. To run it:

```bash
streamlit run app.py
```

This will start a web interface where users can:
- Ask questions about X (Twitter) features and policies
- Receive helpful, context-aware responses
- View the source documents used to generate the response
- Maintain a conversation history

### Application Flow

1. **Initialization**:
   - The app checks for an existing FAISS index
   - If no index exists, it creates one by processing documentation in the `data/` directory
   - A RAG chain is initialized with the vector store

2. **User Interaction**:
   - Users enter questions in the chat interface
   - The system retrieves relevant documentation based on vector similarity
   - Documents that meet the similarity threshold are used as context
   - The LLM generates a response based on the retrieved context
   - Both the response and source documents are displayed to the user

## Testing

The repository includes comprehensive test coverage:

- Unit tests for each component
- Integration tests for component interactions
- Mock implementations for external dependencies

To run tests:

```bash
pytest
```

For coverage information:

```bash
pytest --cov=src
```

## Contributing

To contribute to this project:

1. Add documentation files to the `data/` directory in text format
2. Run the application to rebuild the vector store
3. Test your changes with relevant queries
4. Submit a pull request with your improvements

## Troubleshooting

### Missing API Key
If you encounter an error about a missing API key, ensure your `.env` file is properly set up with OPENAI_API_KEY.

### File Not Found Errors
Ensure your data directory structure matches the expected paths for documentation files.

### Empty Results
If queries return "I'm sorry, I don't have enough information to answer this question confidently":
- Verify that your vector store has been created and populated correctly
- Check if the documentation contains information related to your query
- Consider adjusting the similarity threshold in the RAG chain (default is 0.7)

### LLM Response Errors
If the LLM fails to generate a response:
- Check your OpenAI API key validity and quota
- Ensure your network connectivity is stable
- Verify that context is correctly formatted for the LLM