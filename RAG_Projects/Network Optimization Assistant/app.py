import streamlit as st
import pandas as pd
import os
from src.document_loader import Guide_DocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChain

def get_documents():
    """
    Load all documents from the document loader.
    
    This function initializes the document loader and loads both incident 
    and technical documents in a single operation.
    
    Returns:
        tuple: (incident_docs, tech_docs) - Lists of document objects ready for indexing
    """
    try:
        loader = Guide_DocumentLoader()
        incident_docs, tech_docs = loader.load_all_documents()
        return incident_docs, tech_docs
    except Exception as e:
        raise e

def initialize_retrieval_system(query, product_id, status_placeholder):
    """
    Initialize the hybrid retrieval system and retrieve documents for the query.
    
    This function loads documents, creates or loads vector stores, and retrieves 
    relevant documents for the given query and product ID. It uses a hybrid search 
    approach combining semantic and keyword matching for comprehensive results.
    
    The function updates the status placeholder to keep the user informed of progress
    through the document retrieval process.
    
    Args:
        query (str): User's technical question
        product_id (str): Product ID to filter results
        status_placeholder: Streamlit element to display status updates
        
    Returns:
        tuple: (incident_results, tech_results) - Retrieved document content or errors
    """
    try:
        status_placeholder.text("Loading documents...")
        incident_docs, tech_docs = get_documents()
        
        status_placeholder.text("Initializing hybrid vector store...")
        vector_manager = VectorStoreManager(
            incident_docs, 
            tech_docs,
            top_k=5,
            incident_faiss_path="index/incident_faiss", 
            tech_faiss_path="index/tech_faiss",
            incident_bm25_path="index/incident_bm25/incident_bm25.pkl", 
            tech_bm25_path="index/tech_bm25/tech_bm25.pkl",
        )
        
        vector_manager._initialize_vector_stores()

        status_placeholder.text("Retrieving incident records...")
        incident_results = vector_manager.retrieve_documents(
            query, 
            product_id, 
            store_type="incident", 
            metadata_col="SolutionDetails"
        )
    
        status_placeholder.text("Retrieving technical documentation...")
        tech_results = vector_manager.retrieve_documents(
            query, 
            product_id, 
            store_type="tech", 
            metadata_col="SolutionSteps"
        )

        return incident_results, tech_results
    except Exception as e:
        return e, e

def call_rag_chain(query, product_id, status_placeholder):
    """
    Call the RAG chain with the retrieved documents to generate a solution.
    
    This function coordinates the retrieval and generation process, handling
    error cases and formatting the final response for display. It uses the 
    gpt-4o model to generate comprehensive solutions based on retrieved documents.
    
    Args:
        query (str): User's technical question
        product_id (str): Product ID to filter results
        status_placeholder: Streamlit element to display status updates
        
    Returns:
        str: Formatted solution guide with step-by-step instructions or error message
    """
    incident_results, tech_results = initialize_retrieval_system(query, product_id, status_placeholder)
    
    if isinstance(incident_results, Exception) or isinstance(tech_results, Exception):
        raise incident_results if isinstance(incident_results, Exception) else tech_results
    
    status_placeholder.text("Generating solution guide...")
    rag_chain = RAGChain(template_name="tech_incident_template" ,model="gpt-4o")
    response = rag_chain.run(query, tech_results, incident_results)
    status_placeholder.text("Response generated successfully!")
    return response

def list_available_products():
    """
    List all available products in the dataset.
    
    This function analyzes the metadata files to identify which products
    exist in which datasets, helping users understand what data is available.
    
    Returns:
        tuple: (all_products, tech_only, incident_only, both) - Sets of product IDs
    """
    try:
        if os.path.exists('data/metadata_tech_records.csv'):
            tech_df = pd.read_csv('data/metadata_tech_records.csv')
            tech_products = set(tech_df['ProductID'])
            
            if os.path.exists('data/metadata_incident_records.csv'):
                incident_df = pd.read_csv('data/metadata_incident_records.csv')
                incident_products = set(incident_df['ProductID'])
                
                all_products = tech_products.union(incident_products)
                tech_only = tech_products - incident_products
                incident_only = incident_products - tech_products
                both = tech_products.intersection(incident_products)
                
                return all_products, tech_only, incident_only, both
            else:
                return tech_products, tech_products, set(), set()
        else:
            return set(), set(), set(), set()
    except Exception as e:
        return set(), set(), set(), set()

def main():
    """
    Main function for the Streamlit application that provides a technical support interface using a RAG system to answer user queries about specific products.
    """
    st.title("Network Optimization Assistant")
    st.subheader("Get step-by-step guides for network technical issues")

    query = st.text_area(
        "Enter your technical question:",
        placeholder="e.g., Network administrators unable to access equipment management interfaces.",
        height=100
    )
    
    try:
        if os.path.exists('data/metadata_tech_records.csv'):
            product_df = pd.read_csv('data/metadata_tech_records.csv')
            product_options = []
            for _, row in product_df.iterrows():
                product_id = row['ProductID']
                product_info = row.get('ProductInformation', '') 
                product_options.append(f"{product_id} - {product_info}")
        else:
            st.warning("Product metadata file not found. Using default product IDs.")
            product_options = [f"P{i}" for i in range(1, 51)]
    except Exception as e:
        product_options = [f"P{i}" for i in range(1, 51)]
    
    selected_product = st.selectbox(
        "Select Product ID:",
        options=product_options,
    )
    
    product_id = selected_product.split(' - ')[0] if ' - ' in selected_product else selected_product

    status_placeholder = st.empty()

    if st.button("Generate Solution"):
        if query.strip() and product_id:
            with st.spinner("Processing your request..."):
                try:
                    rag_response = call_rag_chain(query, product_id, status_placeholder)

                    st.subheader("Generated Solution")
                    st.markdown(rag_response)
                    
                    
                    st.download_button(
                        label="Download Solution",
                        data=rag_response,
                        file_name=f"solution_{product_id}.txt",
                        mime="text/plain"
                    )
                    print(rag_response)
                except Exception as e:
                    status_placeholder.text("Error occurred!")
                    st.error(f"An error occurred: {str(e)}")    
        else:
            status_placeholder.text("Please enter both a query and product ID.")
            st.warning("Please enter both a query and product ID.")

if __name__ == "__main__":
    main()