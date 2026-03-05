import os
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import streamlit as st

from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proactive_service_rag")

# Constants for file paths
ACTIVE_METADATA_PATH = 'data/metadata_active_ticket_detail.csv'
ACTIVE_DESCRIPTION_PATH = 'data/source_active_ticket_description.csv'
HISTORY_METADATA_PATH = 'data/metadata_historic_ticket_detail.csv'
HISTORY_DESCRIPTION_PATH = 'data/source_historic_ticket_description.csv'

# Vector store paths
ACTIVE_VCT_PATH = 'faiss_index/active_faiss_index'
HISTORY_VCT_PATH = 'faiss_index/historic_faiss_index'
EMBEDDING_MODEL = "text-embedding-3-large"

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

def load_environment():
    """
    Load environment variables.
    
    Returns:
        str: OpenAI API key or None if loading fails
    """
    try:
        status_placeholder.info("🔑 Loading environment variables...")
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        dotenv_loaded = load_dotenv(dotenv_path)
        
        if not dotenv_loaded:
            status_placeholder.error(f"Failed to load .env file from {dotenv_path}. Please ensure it exists.")
            return None
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            status_placeholder.error("OPENAI_API_KEY not found in environment variables.")
            return None
            
        status_placeholder.success("✅ Environment variables loaded successfully!")
        return api_key
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def initialize_embeddings(api_key):
    """
    Initialize the embedding model.
    
    Args:
        api_key (str): OpenAI API key
        
    Returns:
        OpenAIEmbeddings: Embedding model instance or None if initialization fails
    """
    try:
        status_placeholder.info("🧠 Initializing embedding model...")
        emb_model = OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
        status_placeholder.success("✅ Embedding model initialized successfully!")
        return emb_model
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def load_documents(api_key):
    """
    Load documents from CSV files.
    
    Args:
        api_key (str): OpenAI API key
        
    Returns:
        tuple: Tuple containing (active_loader, active_data, active_documents, history_documents) or None if loading fails
    """
    try:
        status_placeholder.info("📚 Loading documents...")
        progress_bar.progress(20)
        
        # Initialize loaders with both metadata and description files
        active_loader = DocumentLoader(
            metadata_path=ACTIVE_METADATA_PATH,
            description_path=ACTIVE_DESCRIPTION_PATH,
            openai_api_key=api_key
        )
        
        history_loader = DocumentLoader(
            metadata_path=HISTORY_METADATA_PATH,
            description_path=HISTORY_DESCRIPTION_PATH,
            openai_api_key=api_key
        )
        
        progress_bar.progress(40)
        
        # Load the merged CSV data and documents
        active_data = active_loader.load_csv_data()
        active_documents = active_loader.load_documents()
        history_documents = history_loader.load_documents()
        
        progress_bar.progress(60)
        status_placeholder.success("✅ Documents loaded successfully!")
        
        return active_loader, active_data, active_documents, history_documents
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def initialize_vector_stores(emb_model, active_documents, history_documents):
    """
    Initialize and load vector stores.
    
    Args:
        emb_model: Embedding model
        active_documents: Active documents
        history_documents: History documents
        
    Returns:
        tuple: Tuple containing (active_vsm, history_vsm, active_vectorstore, history_vectorstore) or None if initialization fails
    """
    try:
        status_placeholder.info("🔢 Initializing vector stores...")
        progress_bar.progress(70)
        
        active_vsm = VectorStoreManager(ACTIVE_VCT_PATH, emb_model)
        history_vsm = VectorStoreManager(HISTORY_VCT_PATH, emb_model)
        
        # Create vector stores if they don't exist
        if not Path(ACTIVE_VCT_PATH).exists():
            status_placeholder.info("📊 Creating active vector store...")
            active_vsm.create_vectorstore(active_documents)
            
        if not Path(HISTORY_VCT_PATH).exists():
            status_placeholder.info("📊 Creating historic vector store...")
            history_vsm.create_vectorstore(history_documents)
        
        progress_bar.progress(90)
        
        # Get vector stores
        active_vectorstore = active_vsm.get_vectorstore()
        history_vectorstore = history_vsm.get_vectorstore()
        
        progress_bar.progress(100)
        status_placeholder.success("✅ Vector stores initialized successfully!")
        
        return active_vsm, history_vsm, active_vectorstore, history_vectorstore
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def initialize_rag_system():
    """
    Initialize the RAG system components.
    
    Returns:
        tuple: Tuple containing all necessary components for the RAG system or None if initialization fails
    """
    # Load environment variables
    api_key = load_environment()
    if not api_key:
        return None
    
    # Initialize embedding model
    emb_model = initialize_embeddings(api_key)
    if not emb_model:
        return None
        
    # Load documents
    doc_results = load_documents(api_key)
    if not doc_results:
        return None
        
    active_loader, active_data, active_documents, history_documents = doc_results
    
    # Initialize vector stores
    vector_results = initialize_vector_stores(emb_model, active_documents, history_documents)
        
    if not vector_results:
        return None
        
    active_vsm, history_vsm, active_vectorstore, history_vectorstore = vector_results
    
    # Initialize RAG processor with the new file paths
    try:
        status_placeholder.info("🤖 Initializing RAG processor...")
        # Pass both metadata and description paths to RAGProcessor
        rag_processor = RAGProcessor(
            metadata_path=ACTIVE_METADATA_PATH,
            description_path=ACTIVE_DESCRIPTION_PATH,
            active_vectorstore=active_vectorstore, 
            history_vectorstore=history_vectorstore, 
            openai_api_key=api_key
        )
        status_placeholder.success("✅ RAG processor initialized successfully!")
        status_placeholder.empty()
        
        return active_data, active_vsm, rag_processor
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def display_system_status():
    """
    Display the current system status and any required setup steps.
    
    Returns:
        bool: True if system is ready, False otherwise
    """
    if not st.session_state.system_components:
        st.error("⚠️ System initialization failed")
        st.info("""
        Please ensure:
        1. The .env file exists with a valid OPENAI_API_KEY
        2. The data directory contains valid CSV files:
           - metadata_active_ticket_detail.csv
           - source_active_ticket_description.csv
           - metadata_historic_ticket_detail.csv
           - source_historic_ticket_description.csv
        3. All required packages are installed
        
        Check the logs for detailed error information.
        """)
        return False
    return True

def process_ticket_request(location_id, description):
    """
    Process a ticket request and display results.
    
    Args:
        location_id (int): Location ID
        description (str): Ticket description
    """
    active_data, active_vsm, rag_processor = st.session_state.system_components
    
    try:
        with st.spinner("🔄 Processing ticket..."):
            ticket_type, new_ticket = rag_processor.get_estimated_resolution_time(description, location_id)
            
            # Update both metadata and description files
            rag_processor.append_to_csv_file(new_ticket)
            
            # Update the vector store with the new ticket
            active_vsm.append_vectorstore(new_ticket)
            
        st.success("✅ Ticket processed successfully!")
        
        # Format the response as requested
        response_format = f"""
        TicketID: {new_ticket['TicketID']}
        LocationID: {new_ticket['locationID']}
        Issue Description: {description}
        Estimated_resolution_time: {new_ticket['estimated_resolution_time']} hours
        """
        
        # Display the formatted response with custom styling
        st.markdown("""
        <style>
        .ticket-info-area {
            background-color: white !important;
            color: black !important;
            border: 2px solid #28a745;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            white-space: pre;
            height: 150px;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ticket-info-area">
            TicketID: {new_ticket['TicketID']}
            LocationID: {new_ticket['locationID']}
            Issue Description: {description}
            Estimated_resolution_time: {new_ticket['estimated_resolution_time']} hours
        </div>
        """, unsafe_allow_html=True)
        
        # Create a download button for the response
        # Escape double quotes in description properly for CSV
        safe_description = description.replace('"', '""')
        csv_content = f'TicketID,LocationID,Issue Description,Estimated_resolution_time\n{new_ticket["TicketID"]},{new_ticket["locationID"]},"{safe_description}",{new_ticket["estimated_resolution_time"]} hours'
        
        st.download_button(
            label="Download Ticket Information",
            data=csv_content,
            file_name=f"ticket_{new_ticket['TicketID']}_details.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing ticket: {str(e)}")

def main():
    """Main application function."""
    st.title("Ticket Resolution Time Estimator ")
    # st.header("Estimate Resolution Time for New Tickets")
    
    # Initialize session state for system components
    if "system_components" not in st.session_state:
        st.session_state.system_components = initialize_rag_system()
    
    # Check system status
    if not display_system_status():
        return
    
    # Clear progress indicators once system is initialized
    progress_bar.empty()
    

    # Apply custom CSS for styling
    st.markdown("""
    <style>
        /* Style for our custom dropdown container */
        .custom-dropdown {
            margin-bottom: 15px;
        }
        
        /* Style for the label */
        .dropdown-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        /* Don't hide the selectbox completely, but we'll change its display later */
        div[data-testid="stSelectbox"] {
            margin-top: -10px;  /* Negative margin to compact the layout */
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dropdown-label'>Select Location ID:</div>", unsafe_allow_html=True)

    # Use the actual Streamlit selectbox but with minimal styling
    # This ensures the selection is properly captured
    with st.form("ticket_form"):
        location_options = list(range(1, 40))
        location_id = st.selectbox(
            "", 
            options=location_options,
            index=None,  # No default selection
            placeholder="Choose an option"
        )
        
        description = st.text_area("Enter the Query/Description:")
        submit_button = st.form_submit_button("Submit")

    
    # Process form submission
    if submit_button:
        if location_id and description:
            print(f"-----------: {location_id}")
            process_ticket_request(location_id, description)
        else:
            st.error("Please provide both Location ID and Description.")

if __name__ == "__main__":
    main()