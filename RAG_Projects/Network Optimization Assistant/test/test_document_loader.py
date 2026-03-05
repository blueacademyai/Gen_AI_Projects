import pytest
import pandas as pd
import os
from src.document_loader import Guide_DocumentLoader
from langchain_core.documents import Document

@pytest.fixture
def setup_csv_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tech_src_data = pd.DataFrame({
        'step_description': ['Step 1: Configure system', 'Step 2: Restart server']
    })
    tech_meta_data = pd.DataFrame({
        'ProductID': ['P1', 'P2'],
        'ProductInformation': ['Info1', 'Info2'],
        'SolutionSteps': ['Steps1', 'Steps2'],
        'TechnicalTags': ['Tag1', 'Tag2'],
        'DocumentType': ['Manual', 'Guide']
    })
    
    incident_src_data = pd.DataFrame({
        'ProblemDescription': ['Issue with login', 'Server crash']
    })
    incident_meta_data = pd.DataFrame({
        'CustomerID': ['C1', 'C2'],
        'ProductID': ['P1', 'P2'],
        'ProductInformation': ['Info1', 'Info2'],
        'SolutionDetails': ['Solution1', 'Solution2'],
        'Status': ['Open', 'Closed'],
        'Tags': ['Tag1', 'Tag2'],
        'Timestamp': ['2023-01-01', '2023-01-02'],
        'DocID': ['D1', 'D2']
    })

    # Define file paths
    tech_src_path = data_dir / "src_tech_records.csv"
    tech_meta_path = data_dir / "metadata_tech_records.csv"
    incident_src_path = data_dir / "src_incident_records.csv"
    incident_meta_path = data_dir / "metadata_incident_records.csv"

    # Save the dataframes to CSV files
    tech_src_data.to_csv(tech_src_path, index=False)
    tech_meta_data.to_csv(tech_meta_path, index=False)
    incident_src_data.to_csv(incident_src_path, index=False)
    incident_meta_data.to_csv(incident_meta_path, index=False)

    return {
        'tech_src_path': str(tech_src_path),
        'tech_meta_path': str(tech_meta_path),
        'incident_src_path': str(incident_src_path),
        'incident_meta_path': str(incident_meta_path),
        'data_dir': data_dir
    }

@pytest.fixture
def setup_mismatched_csv_files(tmp_path):
    """Create CSV files with mismatched row counts for testing validation logic"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tech_src_data = pd.DataFrame({
        'step_description': ['Step 1: Configure system', 'Step 2: Restart server', 'Step 3: Verify setup']
    })
    tech_meta_data = pd.DataFrame({
        'ProductID': ['P1', 'P2'],
        'ProductInformation': ['Info1', 'Info2'],
        'SolutionSteps': ['Steps1', 'Steps2'],
        'TechnicalTags': ['Tag1', 'Tag2'],
        'DocumentType': ['Manual', 'Guide']
    })
    
    incident_src_data = pd.DataFrame({
        'ProblemDescription': ['Issue with login', 'Server crash']
    })
    incident_meta_data = pd.DataFrame({
        'CustomerID': ['C1', 'C2', 'C3'],
        'ProductID': ['P1', 'P2', 'P3'],
        'ProductInformation': ['Info1', 'Info2', 'Info3'],
        'SolutionDetails': ['Solution1', 'Solution2', 'Solution3'],
        'Status': ['Open', 'Closed', 'Pending'],
        'Tags': ['Tag1', 'Tag2', 'Tag3'],
        'Timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'DocID': ['D1', 'D2', 'D3']
    })

    tech_src_path = data_dir / "src_tech_records.csv"
    tech_meta_path = data_dir / "metadata_tech_records.csv"
    incident_src_path = data_dir / "src_incident_records.csv"
    incident_meta_path = data_dir / "metadata_incident_records.csv"

    tech_src_data.to_csv(tech_src_path, index=False)
    tech_meta_data.to_csv(tech_meta_path, index=False)
    incident_src_data.to_csv(incident_src_path, index=False)
    incident_meta_data.to_csv(incident_meta_path, index=False)

    return {
        'tech_src_path': str(tech_src_path),
        'tech_meta_path': str(tech_meta_path),
        'incident_src_path': str(incident_src_path),
        'incident_meta_path': str(incident_meta_path)
    }

def test_document_loader_init(setup_csv_files):
    loader = Guide_DocumentLoader(
        tech_src_path=setup_csv_files['tech_src_path'],
        tech_meta_path=setup_csv_files['tech_meta_path'],
        incident_src_path=setup_csv_files['incident_src_path'],
        incident_meta_path=setup_csv_files['incident_meta_path']
    )

    assert loader.tech_src_path == setup_csv_files['tech_src_path']
    assert loader.tech_meta_path == setup_csv_files['tech_meta_path']
    assert loader.incident_src_path == setup_csv_files['incident_src_path']
    assert loader.incident_meta_path == setup_csv_files['incident_meta_path']

def test_load_incident_documents(setup_csv_files):
    loader = Guide_DocumentLoader(
        tech_src_path=setup_csv_files['tech_src_path'],
        tech_meta_path=setup_csv_files['tech_meta_path'],
        incident_src_path=setup_csv_files['incident_src_path'],
        incident_meta_path=setup_csv_files['incident_meta_path']
    )

    documents = loader.load_incident_documents()

    assert len(documents) == 2, "Should load exactly 2 incident documents"

    assert isinstance(documents[0], Document), "Loaded object should be a Document"
    assert documents[0].metadata['CustomerID'] == "C1", "Metadata should contain correct CustomerID"
    assert documents[0].metadata['Status'] == "Open", "Metadata should contain correct Status"

def test_load_incident_documents_row_mismatch(setup_mismatched_csv_files):
    """Test that load_incident_documents properly detects and raises ValueError for row count mismatch"""
    loader = Guide_DocumentLoader(
        tech_src_path=setup_mismatched_csv_files['tech_src_path'],
        tech_meta_path=setup_mismatched_csv_files['tech_meta_path'],
        incident_src_path=setup_mismatched_csv_files['incident_src_path'],
        incident_meta_path=setup_mismatched_csv_files['incident_meta_path']
    )

    with pytest.raises(ValueError) as exc_info:
        loader.load_incident_documents()
    
    error_message = str(exc_info.value)
    assert "Row count mismatch" in error_message, "Error should mention row count mismatch"
    assert "2 in source vs 3 in metadata" in error_message, "Error should show the actual counts"

def test_load_tech_documents(setup_csv_files):
    loader = Guide_DocumentLoader(
        tech_src_path=setup_csv_files['tech_src_path'],
        tech_meta_path=setup_csv_files['tech_meta_path'],
        incident_src_path=setup_csv_files['incident_src_path'],
        incident_meta_path=setup_csv_files['incident_meta_path']
    )

    documents = loader.load_tech_documents()

    assert len(documents) == 2, "Should load exactly 2 tech documents"

    assert isinstance(documents[0], Document), "Loaded object should be a Document"
    assert documents[0].metadata['ProductID'] == "P1", "Metadata should contain correct ProductID"
    assert documents[0].metadata['DocumentType'] == "Manual", "Metadata should contain correct DocumentType"

def test_load_tech_documents_row_mismatch(setup_mismatched_csv_files):
    """Test that load_tech_documents properly detects and raises ValueError for row count mismatch"""
    
    loader = Guide_DocumentLoader(
        tech_src_path=setup_mismatched_csv_files['tech_src_path'],
        tech_meta_path=setup_mismatched_csv_files['tech_meta_path'],
        incident_src_path=setup_mismatched_csv_files['incident_src_path'],
        incident_meta_path=setup_mismatched_csv_files['incident_meta_path']
    )

    with pytest.raises(ValueError) as exc_info:
        loader.load_tech_documents()
    
    error_message = str(exc_info.value)
    assert "Row count mismatch" in error_message, "Error should mention row count mismatch"
    assert "3 in source vs 2 in metadata" in error_message, "Error should show the actual counts"

def test_load_all_documents(setup_csv_files):
    loader = Guide_DocumentLoader(
        tech_src_path=setup_csv_files['tech_src_path'],
        tech_meta_path=setup_csv_files['tech_meta_path'],
        incident_src_path=setup_csv_files['incident_src_path'],
        incident_meta_path=setup_csv_files['incident_meta_path']
    )

    incident_docs, tech_docs = loader.load_all_documents()

    assert len(incident_docs) == 2, "Should load exactly 2 incident documents"
    assert len(tech_docs) == 2, "Should load exactly 2 tech documents"

def test_load_all_documents_with_row_mismatch(setup_mismatched_csv_files):
    """Test that load_all_documents properly propagates ValueError from either method"""
    loader = Guide_DocumentLoader(
        tech_src_path=setup_mismatched_csv_files['tech_src_path'],
        tech_meta_path=setup_mismatched_csv_files['tech_meta_path'],
        incident_src_path=setup_mismatched_csv_files['incident_src_path'],
        incident_meta_path=setup_mismatched_csv_files['incident_meta_path']
    )

    with pytest.raises(ValueError) as exc_info:
        loader.load_all_documents()
    
    error_message = str(exc_info.value)
    assert "Row count mismatch" in error_message, "Error should mention row count mismatch"
    assert ("2 in source vs 3 in metadata" in error_message), "Error should show the actual counts"