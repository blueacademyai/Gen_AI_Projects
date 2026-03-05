# Proactive Service RAG Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) pipeline for proactive service communication. The system uses vector embeddings to estimate resolution times for new tickets based on historical data and helps provide consistent service communication.

## Overview

The Proactive Service RAG Pipeline consists of three main components:

1. **Document Loader (`document_loader.py`)**: Loads and processes ticket data from CSV files, handling summarization for long descriptions.
2. **Vector Store (`vector_store.py`)**: Manages the FAISS vector embeddings database for efficient similarity search.
3. **RAG Processor (`rag_chain.py`)**: Combines retrieval and generation to estimate resolution times and create new tickets.

## Environment Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Ensure necessary directories exist:
   ```
   mkdir -p data faiss_index
   ```

## Component Details

### 1. Document Loader (`document_loader.py`)

The Document Loader is responsible for loading ticket data from CSV files and processing it into a standardized document format. It includes functionality for summarizing long descriptions to improve embedding quality.

#### Key Classes:

1. **DocumentLoader**: Loads and processes individual CSV datasets
   - Handles loading, summarization, and document creation
   - Uses OpenAI's GPT-4o for description summarization
   - Creates Document objects with standardized metadata

2. **CombinedDocumentLoader**: Manages both active and historic datasets
   - Creates separate DocumentLoader instances for each dataset
   - Provides methods to load active, historic, or all documents

#### Document Structure:

**Page Content:**
- Contains the ticket description (summarized if over 1500 characters)

**Metadata:**
- The metadata includes the following fields:
  ```python
  {
      'ticket_id': str,                    # Ticket identifier
      'location_id': str,                  # Location identifier
      'estimated_resolution_time': str     # Estimated resolution time in hours
  }
  ```

#### Input CSV File Structure:

The system requires four CSV files:

**Active Metadata (`metadata_active_ticket_detail.csv`)**
- Schema:
  - `TicketID`: Integer - Unique identifier for each ticket
  - `customerID`: Float - Customer identifier
  - `locationID`: Integer - Location identifier
  - `type`: String - Ticket type (e.g., "complaint")
  - `clusterID`: Integer - Cluster identifier for categorization
  - `estimated_resolution_time`: Integer - Estimated resolution time in hours

**Historic Metadata (`metadata_historic_ticket_detail.csv`)**
- Schema:
  - Similar to active metadata but may include additional fields like `actual_resolution_time`

**Active Descriptions (`source_active_ticket_description.csv`)**
- Schema:
  - `TicketID`: Float - Unique identifier matching the metadata file
  - `description`: String - Detailed description of the issue

**Historic Descriptions (`source_historic_ticket_description.csv`)**
- Schema:
  - `TicketID`: Integer - Unique identifier matching the metadata file
  - `description`: String - Detailed description of the issue

### 2. Vector Store (`vector_store.py`)

The Vector Store Manager (`VectorStoreManager` class) handles FAISS vector databases for efficient similarity search of ticket data.

#### Key Features:

- Creates and manages FAISS vector stores
- Provides methods for creating, loading, and updating vector stores
- Implements efficient methods for appending new tickets to existing stores
- Handles serialization and deserialization of vector stores

#### Important Methods:

1. **create_vectorstore(documents)**:
   - Creates a new FAISS vector store from a list of documents
   - Validates that the documents list is not empty
   - Saves the vector store to the specified path

2. **append_vectorstore(new_data)**:
   - Adds a new ticket to an existing vector store
   - Creates a Document object from the ticket data
   - Updates and saves the modified vector store

3. **get_vectorstore()**:
   - Loads an existing vector store if it exists
   - Returns the loaded vector store for similarity operations
   - Implements lazy loading pattern

### 3. RAG Processor (`rag_chain.py`)

The RAG Processor (`RAGProcessor` class) uses the vector stores to find similar tickets and estimate resolution times for new tickets based on historical data.

#### Key Features:

- Estimates resolution time based on similar tickets
- Performs similarity search with location-based filtering
- Creates new ticket records with appropriate metadata
- Updates the active tickets CSV file with new tickets
- Handles fallback to historical data when no active matches are found

#### Key Methods:

1. **perform_similarity_search(description, vectorstore, location_id)**:
   - Searches for similar tickets in the specified vector store
   - Filters results by location ID
   - Implements three search approaches with different thresholds
   - Returns matching documents with similarity scores

2. **get_estimated_resolution_time(description, location_id)**:
   - Tries to find similar tickets in the active database first
   - Falls back to historical data if no active matches are found
   - Calculates estimated resolution time based on similar tickets
   - Creates a new ticket with the estimated time
   - Returns ticket type and ticket data

3. **create_new_ticket(ticket_id, customer_id, location_id, description, estimated_time, clusterID)**:
   - Creates a standardized ticket record
   - Summarizes description if needed
   - Returns the new ticket dictionary

4. **append_to_csv_file(new_data)**:
   - Appends a new ticket to the active CSV files
   - Updates both metadata and description CSV files
   - Handles the separation of data between the two files

## Web Application (`app.py`)

The Streamlit web application provides a user-friendly interface for interacting with the RAG system.

#### Key Features:

- Simple web interface for submitting new tickets
- Location ID selection from available options
- Text area for entering ticket descriptions
- Display of estimated resolution time
- Option to download ticket details as CSV

#### Application Flow:

1. System initialization on startup
   - Loads environment variables
   - Initializes embedding model
   - Loads documents and vector stores

2. User submits a new ticket with:
   - Location ID (dropdown selection)
   - Description (text area)

3. System processes the ticket:
   - Estimates resolution time
   - Updates CSV files and vector store
   - Displays results to user

## Test Queries

To verify the RAG pipeline is functioning correctly, you can use the following test cases with their expected resolution times:

### Test Case 1: Network Infrastructure Upgrade

**Input:**
```
We had a big scheduled maintenance event planned for the northern region to upgrade our aging network infrastructure, and let me tell you, it was quite the ordeal. The goal was to replace outdated equipment and boost connectivity for thousands of homes and businesses that had been struggling with slow speeds and frequent outages for ages. We spent weeks preparing—coordinating crews, scheduling downtime, and sending out notices to customers so they'd know what to expect. The actual work involved digging up old cables, installing new fiber optic lines, and updating software across multiple hubs. It started early in the morning, and we hoped to wrap it up quickly, but things got messy—some areas lost service entirely, and others saw their speeds drop even lower than before. People weren't happy, and I get it; we'd promised an upgrade, not a downgrade. Still, it's done now, and the network should be stronger for it.
```

**Expected Result:**
```
Estimated Resolution Time: 16 hours
```

### Test Case 2: Blogger Internet Outage

**Input:**
```
The internet stopped last night, and it's thrown everything off. I'm a blogger, and at 2:15 PM, I couldn't post my scheduled piece—missed my window, and my readers are dropping off. Restarting the router did nothing; I'm dead in the water. This came out of nowhere, and I'm so frustrated—I need this back to keep my audience!
```

**Expected Result:**
```
Estimated Resolution Time: 30 hours
```

### Test Case 3: Plumber Phone Signal Issue

**Input:**
```
No signal on my phone all day, and it's crushing my work. I'm a plumber with my own business, and at 3:30 PM, I couldn't call clients—missed two jobs because they couldn't reach me either. I've tried everything—restarting, moving around—nothing. This outage is costing me big time!
```

**Expected Result:**
```
Estimated Resolution Time: 31 hours
```

### Test Case 4: Competitive Gamer Latency Issue

**Input:**
```
I'm a competitive esports team member, and today's extreme internet latency has completely destroyed our tournament chances. We were competing for a $5,000 prize in a match that could have been my breakthrough moment, but the network performance was catastrophic. My ping spiked to 500ms at the crucial moment, rendering me unable to play effectively. My character froze mid-fight, bullets passed through walls, and I watched helplessly as our team was eliminated in the first round. My team captain's message - 'you cost us everything' - confirmed the devastating impact. I've tried every possible technical solution: switching to Ethernet, rebooting the router multiple times, resetting my PC, and even asking my brother to stop using the internet. Nothing has resolved the persistent connection issues. Ping tests show constant instability, fluctuating between 300-600ms, making competitive gaming impossible. This was more than just a game; it was my potential pathway to a professional gaming career, an opportunity to prove myself and potentially move out of my parents' home. My team's trust is shattered, and the competitive community's response has been brutal. My neighbor, also a tech enthusiast, confirmed similar connection problems, suggesting a broader network issue. However, the service provider has been completely unresponsive - no outage alerts, no support communication. I've exhausted every troubleshooting method, from cable checking to device resets, but the connection remains unplayable. The emotional and professional toll is immense - this was supposed to be our breakthrough tournament, and now it feels like my dreams are collapsing due to technical failures beyond my control.
```

**Expected Result:**
```
Estimated Resolution Time: 25 hours
```

### Test Case 5: Retiree Landline Outage

**Input:**
```
My phone lines have been dead since yesterday, and it's got me really worried. I'm retired, and I depend on my landline to call my kids and grandkids—they check in every evening to make sure I'm okay. Last night, they couldn't reach me, and I hate thinking they're panicking. I've checked the phone, the cords—everything's plugged in right. This outage is isolating me, and I just want it back up so I can feel connected again.
```

**Expected Result:**
```
Estimated Resolution Time: 38 hours
```

## Testing

The project includes comprehensive test suites for all components:

1. **Unit Tests**:
   - `test_document_loader.py`: Tests for DocumentLoader and CombinedDocumentLoader
   - `test_vector_store.py`: Tests for VectorStoreManager
   - `test_rag.py`: Tests for RAGProcessor

2. **Integration Tests**:
   - `test_integration.py`: End-to-end tests for the complete pipeline

3. **Test Fixtures**:
   - `conftest.py`: Common test fixtures and mock objects

## Error Handling

The system implements robust error handling throughout all components:

- **DocumentLoader**: Handles file not found errors, data validation, and summarization failures
- **VectorStoreManager**: Handles initialization errors, invalid data, and vector store operations
- **RAGProcessor**: Handles query validation, similarity search failures, and data processing errors
- **Web Application**: Provides clear error messages and recovery options

## Data Flow

1. **Document Loading**:
   - CSV files with ticket data are loaded and processed
   - Long descriptions are summarized using OpenAI's model
   - Documents are created with standardized format

2. **Vector Store Operations**:
   - Documents are converted to embeddings using OpenAI's embedding model
   - Embeddings are stored in FAISS indexes for efficient retrieval
   - New tickets are appended to both CSV files and vector stores

3. **Ticket Processing**:
   - User enters location ID and ticket description via the Streamlit interface
   - System searches for similar tickets filtered by location
   - Estimated resolution time is calculated based on similar tickets
   - New ticket is created and added to the database

## Edge Cases and Resolution

1. **No Matching Tickets**:
   - System tries active tickets first, then historical tickets
   - If no matches are found, creates an "invalid query" ticket
   - Provides feedback to user about query quality

2. **Invalid Queries**:
   - System validates minimum query length (20 characters)
   - Rejects empty or too short queries with appropriate errors
   - Returns error tickets with is_valid=False flag

3. **Missing or Invalid Files**:
   - System checks for existence of required files
   - Provides clear error messages for missing files
   - Initializes new vector stores if needed


## Running the Application

To start the Streamlit web application:

```
streamlit run app.py
```