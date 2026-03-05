"""
Social Media App Support - Streamlit Frontend

This module provides a Streamlit-based web interface for the RAG-powered
customer support chatbot for Social Media App.

The application allows users to:
- Ask questions about X's platform features and policies
- View answers with source attribution
- Manage conversation history
- Access support documentation through a conversational interface
"""

import os
import logging
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Import the custom modules
from src.document_loader import SocialMediaDocumentLoader
from src.vector_store import SocialMediaVectorStore
from src.rag_chain import SocialMediaRAGChain  # Updated import to use RAGChain instead of Agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

# Constants
DATA_DIR = Path("data")
FAISS_INDEX_DIR = Path("faiss_index")

def initialize_rag_system():
    """
    Initialize the RAG system by loading or creating the vector store.
    
    This function:
    1. Creates a SocialMediaVectorStore instance
    2. Attempts to load an existing vector store from disk
    3. If no vector store exists:
       - Creates a SocialMediaDocumentLoader to load and process documentation
       - Loads and processes documents from the data directory
       - Creates a new vector store with the documents
       - Saves the vector store to disk
    4. Creates and returns a SocialMediaRAGChain with the vector store
    
    Returns:
        SocialMediaRAGChain: Initialized RAG chain ready to answer questions
    """
    # Create the vector store
    vector_store = SocialMediaVectorStore(
        index_path=str(FAISS_INDEX_DIR),
        index_name="support_docs"
    )
    
    # Try to load the existing vector store
    loaded_vectorstore = vector_store.load_vectorstore()
    
    # If no vector store exists, create one
    if loaded_vectorstore is None:
        logger.info("No existing vector store found. Creating a new one...")
        
        # Load and process the documents
        document_loader = SocialMediaDocumentLoader(
            data_dir=str(DATA_DIR),
            chunk_size=1000,
            chunk_overlap=200
        )
        
        documents = document_loader.load_and_process()
        
        if not documents:
            logger.error(f"No documents found in {DATA_DIR}")
            st.error(f"No documents found in {DATA_DIR}. Please add documentation files.")
            st.stop()
        
        # Create the vector store
        vector_store.create_vectorstore(documents)
        vector_store.save_vectorstore()
    
    # Create the RAG chain instead of the agent
    rag_chain = SocialMediaRAGChain(
        vectorstore=vector_store.vectorstore,
        llm=None,  # Will use default ChatOpenAI with gpt-4o
        temperature=0.0,
        k=4,
        return_source_documents=True,
        similarity_threshold=0.7
    )
    
    return rag_chain

def format_sources(sources):
    """
    Format the source document names for display in the UI.
    
    Args:
        sources (List[str]): List of source document names
        
    Returns:
        str: Formatted source string for display in the UI
    """
    if not sources:
        return "No sources found"
    
    # Format the sources
    sources_text = "Sources:\n"
    for i, source in enumerate(sources, 1):
        # Clean up the source filename
        source_name = source.replace(".html.txt", "").replace("-", " ").title()
        sources_text += f"{i}. {source_name}\n"
    
    return sources_text

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Social Media App Support Agent",
        layout="wide"
    )
    
    # App title and description
    st.title("Social Media App Support Agent")
    st.markdown("This app uses X(formerly known as Twitter) data. Ask questions related to X platform.")
    
    # Initialize session state
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing support system..."):
            st.session_state.rag_chain = initialize_rag_system()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                if (message['content']!= "I'm sorry, but I don't understand your question. Could you please provide more details or clarify your query?") or (message['content']!= "Could you please provide more details or specify what you need help with? This will help me give you the most accurate and helpful response."):
                    with st.expander("View Sources"):
                        st.markdown(message["sources"])
                else:
                    pass
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query the RAG chain (instead of the agent)
                response = st.session_state.rag_chain.query(prompt)
                
                # Display the answer
                st.markdown(response["answer"])
                
                # Format and display sources only if there are valid sources
                sources_text = "No sources found"
                if "sources" in response and response["sources"]:
                    sources_text = format_sources(response["sources"])
                    with st.expander("View Sources"):
                        st.markdown(sources_text)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": sources_text
                })
    
    # Sidebar with options
    st.sidebar.title("Options")
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
if __name__ == "__main__":
    main()