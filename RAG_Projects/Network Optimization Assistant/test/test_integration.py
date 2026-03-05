# tests/test_integration.py
import pytest
import pandas as pd
from unittest.mock import patch
from src.document_loader import Guide_DocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChain

@pytest.fixture
def create_temp_csv_files(tmp_path):
    """Create temporary CSV files with sample data for testing."""
    
    tech_src_data = pd.DataFrame({
        "step_description": ["Restart the router to resolve connectivity issues."]
    })
    tech_src_path = tmp_path / "src_tech_records.csv"
    tech_src_data.to_csv(tech_src_path, index=False)

    tech_meta_data = pd.DataFrame({
        "ProductID": ["PROD123"],
        "ProductInformation": ["Router Model X"],
        "SolutionSteps": ["Step-by-step guide"],
        "TechnicalTags": ["networking"],
        "DocumentType": ["manual"]
    })
    tech_meta_path = tmp_path / "metadata_tech_records.csv"
    tech_meta_data.to_csv(tech_meta_path, index=False)

    incident_src_data = pd.DataFrame({
        "ProblemDescription": ["User reported network dropout on Router Model X."]
    })
    incident_src_path = tmp_path / "src_incident_records.csv"
    incident_src_data.to_csv(incident_src_path, index=False)

    incident_meta_data = pd.DataFrame({
        "CustomerID": ["CUST001"],
        "ProductID": ["PROD123"],
        "ProductInformation": ["Router Model X"],
        "SolutionDetails": ["Restarted router"],
        "Status": ["Resolved"],
        "Tags": ["networking"],
        "Timestamp": ["2025-01-01"],
        "DocID": ["INC001"]
    })
    incident_meta_path = tmp_path / "metadata_incident_records.csv"
    incident_meta_data.to_csv(incident_meta_path, index=False)

    return {
        "tech_src_path": str(tech_src_path),
        "tech_meta_path": str(tech_meta_path),
        "incident_src_path": str(incident_src_path),
        "incident_meta_path": str(incident_meta_path)
    }

def test_document_loader_integration(create_temp_csv_files):
    """
    Test that the document loader can load both incident and technical documents.
    This test will FAIL if the implementation is commented out.
    """
    paths = create_temp_csv_files
    
    loader = Guide_DocumentLoader(
        tech_src_path=paths["tech_src_path"],
        tech_meta_path=paths["tech_meta_path"],
        incident_src_path=paths["incident_src_path"],
        incident_meta_path=paths["incident_meta_path"]
    )
    
    result = loader.load_all_documents()
    
    assert result is not None, "load_all_documents() should not return None"
    assert isinstance(result, tuple), "load_all_documents() should return a tuple"
    assert len(result) == 2, "load_all_documents() should return (incident_docs, tech_docs)"
    
    incident_docs, tech_docs = result
    
    assert len(incident_docs) > 0, "Should load at least one incident document"
    assert len(tech_docs) > 0, "Should load at least one technical document"
    
    assert hasattr(incident_docs[0], 'page_content'), "Documents should have page_content"
    assert hasattr(incident_docs[0], 'metadata'), "Documents should have metadata"
    assert hasattr(tech_docs[0], 'page_content'), "Documents should have page_content"
    assert hasattr(tech_docs[0], 'metadata'), "Documents should have metadata"

@patch('src.vector_store.OpenAIEmbeddings')
@patch('src.vector_store.FAISS')
@patch('src.vector_store.BM25Retriever')
@patch('langchain_openai.ChatOpenAI')
def test_full_rag_pipeline(mock_chat_openai, mock_bm25, mock_faiss, mock_embeddings, create_temp_csv_files, tmp_path):
    """
    Test the complete RAG pipeline: load documents -> create vector store -> retrieve -> generate response.
    This test will FAIL if any part of the implementation is commented out.
    """
    paths = create_temp_csv_files
    
    loader = Guide_DocumentLoader(
        tech_src_path=paths["tech_src_path"],
        tech_meta_path=paths["tech_meta_path"],
        incident_src_path=paths["incident_src_path"],
        incident_meta_path=paths["incident_meta_path"]
    )
    
    incident_docs, tech_docs = loader.load_all_documents()
    
    mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
    mock_faiss.from_documents.return_value.as_retriever.return_value.get_relevant_documents.return_value = tech_docs
    mock_bm25.from_documents.return_value.get_relevant_documents.return_value = incident_docs
    
    vector_store = VectorStoreManager(
        incident_docs=incident_docs,
        tech_docs=tech_docs,
        top_k=3,
        incident_faiss_path=str(tmp_path / "incident_faiss"),
        tech_faiss_path=str(tmp_path / "tech_faiss"),
        incident_bm25_path=str(tmp_path / "incident_bm25.pkl"),
        tech_bm25_path=str(tmp_path / "tech_bm25.pkl")
    )
    
    query = "How to troubleshoot network connectivity issues?"
    product_id = "PROD123"
    
    tech_results = vector_store.retrieve_documents(query, product_id, store_type="tech")
    incident_results = vector_store.retrieve_documents(query, product_id, store_type="incident")
    
    assert tech_results is not None, "Tech retrieval should return results"
    assert incident_results is not None, "Incident retrieval should return results"
    
    mock_chat_openai.return_value.invoke.return_value.content = "Solution: Restart the router"
    
    rag_chain = RAGChain(template_name="tech_incident_template", model="gpt-4o")
    response = rag_chain.run(query, tech_results, incident_results)
    
    assert isinstance(response, str), "RAG chain should return a string response"
    assert len(response) > 0, "Response should not be empty"