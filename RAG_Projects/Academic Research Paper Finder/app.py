import logging
import sys
from pathlib import Path
import os

import streamlit as st

from src.document_loader import ResearchPaperLoader
from src.vector_store import ResearchVectorStore
from src.retriever import ResearchPaperRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_paper_finder")

# Constants
VECTOR_STORE_DIR = "faiss_store"
DATA_PATH = "data/dataset.csv"

# Initialize Streamlit state placeholders
status_placeholder = st.empty()
progress_bar = st.progress(0)

def log_error(e: Exception) -> str:
    """
    Log an error and return formatted error message.
    
    Args:
        e (Exception): The exception to log
        
    Returns:
        str: Formatted error message for display
    """
    logger.error(e, exc_info=True)
    return f"❌ Error: {str(e)}"

def get_documents():
    """
    Load research papers from the data file.
    
    Returns:
        List[Document]: List of processed documents
    """
    try:
        status_placeholder.info("📚 Loading research papers...")
        loader = ResearchPaperLoader(DATA_PATH)
        documents = loader.create_documents()
        status_placeholder.success(f"✅ {len(documents)} research papers loaded successfully!")
        return documents
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def create_new_vector_store():
    """
    Create a new FAISS vector store from scratch.
    
    Returns:
        FAISSResearchVectorStore or None: New vector store instance or None if creation fails
    """
    try:
        vector_store = ResearchVectorStore(store_path=VECTOR_STORE_DIR)
        
        status_placeholder.info("⚙️ Creating new vector store...")
        progress_bar.progress(40)
        
        # Create documents
        documents = get_documents()
        if not documents:
            return None
        progress_bar.progress(60)
        
        # Create embeddings and vector store
        status_placeholder.info("🔨 Generating embeddings and building FAISS index...")
        vector_store.create_vector_store(documents)
        progress_bar.progress(80)
        
        # Save vector store
        status_placeholder.info("💾 Saving vector store...")
        vector_store.save()
        progress_bar.progress(100)
        
        status_placeholder.success("✅ FAISS vector store created and saved successfully!")
        return vector_store
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def load_existing_vector_store():
    """
    Load an existing FAISS vector store from disk.
    
    Returns:
        FAISSResearchVectorStore or None: Loaded vector store instance or None if loading fails
    """
    try:
        status_placeholder.info("🔄 Loading existing FAISS vector store...")
        progress_bar.progress(30)
        vector_store = ResearchVectorStore.load(VECTOR_STORE_DIR)
        progress_bar.progress(100)
        status_placeholder.success("✅ FAISS vector store loaded successfully!")
        return vector_store
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def initialize_retrieval_system():
    """
    Initialize the retrieval system by loading or creating FAISS vector store.
    
    Returns:
        FAISSResearchPaperRetriever or None: Initialized retriever or None if initialization fails
    """
    try:
        # Check if vector store directory exists and contains required files
        faiss_index_path = Path(VECTOR_STORE_DIR) / "faiss_index.bin"
        metadata_path = Path(VECTOR_STORE_DIR) / "metadata.pkl"
        documents_path = Path(VECTOR_STORE_DIR) / "documents.pkl"
        
        if all(p.exists() for p in [faiss_index_path, metadata_path, documents_path]):
            # Try to load existing vector store
            vector_store = load_existing_vector_store()
        else:
            # Create new vector store if required files don't exist
            vector_store = create_new_vector_store()
            
        if not vector_store:
            return None
        
        # Initialize retriever
        status_placeholder.info("🤖 Initializing paper retriever...")
        retriever = ResearchPaperRetriever(vector_store)
        
        status_placeholder.empty()
        return retriever
        
    except Exception as e:
        error_msg = log_error(e)
        status_placeholder.error(error_msg)
        return None
    

def render_search_results(query, nod, use_recency=False):
    """
    Render search results for a query.
    
    Args:
        query (str): User's search query
        use_recency (bool): Whether to prioritize recent papers
    """
    try:
        #Show spinner for paper retrieval
        with st.spinner("🔍 Searching for relevant papers..."):
            if use_recency:
                results = st.session_state.retriever.retrieve_papers_with_recency(query, k=nod)
                # Sort by year (descending) when prioritizing recency
                # results = sorted(results, key=lambda x: x['year'], reverse=True)
            else:
                results = st.session_state.retriever.retrieve_papers(query, k=nod)
        
        # print(results)
        # Display results
        if not results:
            st.info("No matching papers found. Try a different query.")
            return
            
        st.subheader(f"Found {len(results)} relevant papers")
        
        for i, paper in enumerate(results, 1):
            with st.expander(f"{i}. {paper['title']} ({paper['year']})"):
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Published in:** {paper['venue']}")
                st.write(f"**Citations:** {paper['citations']}")
                st.write(f"**Abstract:** {paper['abstract']}")
                st.write(f"**Similarity Score:** {paper['similarity_score']:.4f}")
        
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    """Main application function."""
    # Clear progress indicators
    progress_bar.empty()
    
    # Set up the main page
    st.title("🔍 Academic Research Paper Finder (FAISS)")
    st.write("""
    Find relevant academic papers using FAISS for efficient semantic search. Enter your research topic 
    or question to discover papers related to your area of interest.
    """)
    
    # Initialize session state
    if "retriever" not in st.session_state:
        st.session_state.retriever = initialize_retrieval_system()
    
    
    # Search interface
    query = st.text_input(
        "Enter your research topic or question:",
        placeholder="e.g., 'Transformer models in NLP' or 'Quantum computing for cryptography'"
    )

    nod = st.text_input("Number of documents",
                        placeholder="e.g. 5")
    
    # Recency toggle with help text
    use_recency = st.checkbox(
        "Prioritize recent papers", 
        value=False, 
        help="When checked, results will be sorted in descending order by publication year."
    )


    search_button = st.button("Search")

    
    # Run search when button is clicked
    if search_button and query:
        render_search_results(query, int(nod), use_recency)

if __name__ == "__main__":
    main()