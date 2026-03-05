# Academic Research Paper Finder

A semantic search system for finding relevant academic research papers using vector embeddings and similarity search. This system helps researchers discover papers related to their research interests based on semantic meaning rather than just keyword matching.

## Overview

Academic researchers often struggle to find relevant papers due to the limitations of keyword-based search. The Academic Research Paper Finder addresses this challenge by providing a semantic search solution that understands the context of research topics and can find conceptually similar papers even when they don't share the exact same keywords.

## Components

The system consists of three main components:

### 1. Document Loader (`document_loader.py`)

The document loader is responsible for loading and processing research papers from CSV files.

#### Key Features:
- Loads research papers from CSV using LangChain's `CSVLoader`
- Extracts important metadata (title, abstract, authors, year, etc.)
- Converts papers into a standardized document format for embedding generation

### 2. Vector Store (`vector_store.py`)

The vector store manages the embeddings for research papers using FAISS (Facebook AI Similarity Search) for efficient similarity search.

#### Key Features:
- Uses OpenAI's text-embedding-3-small model for generating high-quality embeddings
- Stores and indexes embeddings using FAISS for fast retrieval
- Provides similarity search functionality with optional recency ranking
- Persists the vector store to disk for reuse

### 3. Retriever (`retriever.py`)

The retriever provides an interface for querying the vector store and formatting the results for display.

#### Key Features:
- Validates and processes user queries
- Searches the vector store for semantically similar papers
- Formats results with relevant metadata (title, authors, year, venue, etc.)
- Supports recency-based ranking to prioritize recent publications

## Combined Score Calculation

When the `use_recency` parameter is set to `True`, the system combines semantic similarity with publication recency to rank results. This helps researchers find papers that are both relevant and recent. The calculation works as follows:

1. **Semantic Similarity Score**: The base similarity score derived from the cosine similarity between the query and document embeddings (range: 0-1, where 1 is most similar).

2. **Recency Score**: A normalized score based on the publication year:
   ```
   recency_score = (paper_year - (current_year - 30)) / 30
   ```
   This creates a score from 0-1, where papers from 30 years ago get a score of 0, and current papers get a score of 1.

3. **Combined Score**: A weighted average of the two scores:
   ```
   combined_score = 0.7 * similarity_score + 0.3 * recency_score
   ```
   This weighting gives more importance (70%) to semantic relevance while still considering recency (30%).

4. **Result Ranking**: Results are then sorted by the combined score in descending order.

This approach ensures that recent papers on the topic are ranked higher than older papers with the same semantic similarity.

## Implementation Requirements

### Document Loader Requirements

The `ResearchPaperLoader` class must:
- Accept a CSV file path at initialization
- Check if the file exists, raising `FileNotFoundError` if not
- Use LangChain's `CSVLoader` to load documents with proper source and metadata columns
- Return a list of LangChain `Document` objects with properly formatted metadata

### Vector Store Requirements

The `ResearchVectorStore` class must:
- Initialize with a path for storing the vector store
- Generate embeddings using OpenAI or a Hugging Face model
- Create and manage a FAISS index for efficient similarity search
- Save and load the vector store to/from disk
- Provide similarity search with optional recency boosting
- Handle edge cases like empty queries gracefully

### Retriever Requirements

The `ResearchPaperRetriever` class must:
- Validate queries (reject empty or too-short queries)
- Format search results with all relevant paper metadata
- Provide a clean interface for retrieving papers
- Support recency-based retrieval to prioritize newer papers

## File Format

The system expects a CSV file with the following columns:
- `title`: Paper title (used as document content)
- `abstract`: Paper abstract
- `authors`: Paper authors
- `year`: Publication year
- `venue`: Publication venue (journal/conference)
- `n_citation`: Number of citations
- `references`: Paper references
- `id`: Unique paper identifier

## Edge Cases Handling

### Document Loader Edge Cases:
- **Missing File**: Raises `FileNotFoundError` with appropriate message
- **Malformed CSV**: Logs error and raises exception
- **Empty CSV**: Handles gracefully, returns empty document list

### Vector Store Edge Cases:
- **Empty Query**: Returns empty list with warning log
- **No Index Available**: Warns user and returns empty list
- **Embedding Generation Errors**: Catches and logs errors

### Retriever Edge Cases:
- **Empty Query**: Raises `ValueError("Query cannot be empty")`
- **Too Short Query**: Raises `ValueError("Query too short. Please provide a more specific query.")`

- **No Results Found**: Returns empty list, UI shows appropriate message
