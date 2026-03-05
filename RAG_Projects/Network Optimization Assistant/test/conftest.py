import pytest
from unittest.mock import MagicMock, patch
import os
import pandas as pd
from langchain_core.documents import Document


@pytest.fixture
def mock_incident_docs():
    """Fixture providing sample incident documents for testing"""
    return [
        Document(
            page_content="Network connectivity issue with Router XYZ",
            metadata={
                "CustomerID": "C001",
                "ProductID": "P001",
                "ProductInfo": "Router XYZ",
                "SolutionDetails": "Reset the router and update firmware",
                "Status": "Resolved",
                "Tags": "connectivity, router",
                "Timestamp": "2023-01-01",
                "DocID": "ID001"
            }
        ),
        Document(
            page_content="Slow connection on Switch ABC",
            metadata={
                "CustomerID": "C002",
                "ProductID": "P002",
                "ProductInfo": "Switch ABC",
                "SolutionDetails": "Update firmware and configure QoS",
                "Status": "Resolved",
                "Tags": "performance, switch",
                "Timestamp": "2023-01-02",
                "DocID": "ID002"
            }
        )
    ]


@pytest.fixture
def mock_tech_docs():
    """Fixture providing sample technical documents for testing"""
    return [
        Document(
            page_content="Router XYZ troubleshooting steps",
            metadata={
                "ProductID": "P001",
                "ProductInformation": "Router XYZ",
                "SolutionSteps": "1. Check power. 2. Reset. 3. Update firmware.",
                "TechnicalTags": "router, connectivity, firmware",
                "DocumentType": "Manual"
            }
        ),
        Document(
            page_content="Switch ABC configuration guide",
            metadata={
                "ProductID": "P002",
                "ProductInformation": "Switch ABC",
                "SolutionSteps": "1. Access config panel. 2. Set QoS. 3. Update firmware.",
                "TechnicalTags": "switch, configuration, performance",
                "DocumentType": "Guide"
            }
        )
    ]


@pytest.fixture
def mock_incident_csv_data():
    """Fixture providing sample incident CSV data"""
    return pd.DataFrame({
        'ProblemDescription': ['Network connectivity issue', 'Slow connection'],
        'CustomerID': ['C001', 'C002'],
        'ProductID': ['P001', 'P002'],
        'ProductInfo': ['Router XYZ', 'Switch ABC'],
        'SolutionDetails': ['Reset the router', 'Update firmware'],
        'Status': ['Resolved', 'Resolved'],
        'Tags': ['connectivity, router', 'performance, switch'],
        'Timestamp': ['2023-01-01', '2023-01-02'],
        'DocID': ['ID001', 'ID002']
    })


@pytest.fixture
def mock_tech_csv_data():
    """Fixture providing sample tech document CSV data"""
    return pd.DataFrame({
        'step_description': ['Router XYZ troubleshooting steps', 'Switch ABC configuration guide'],
        'ProductID': ['P001', 'P002'],
        'ProductInformation': ['Router XYZ', 'Switch ABC'],
        'SolutionSteps': ['Step 1, Step 2', 'Step 1, Step 2'],
        'TechnicalTags': ['router, connectivity', 'switch, configuration'],
        'DocumentType': ['Manual', 'Guide']
    })


@pytest.fixture
def mock_faiss_vectorstore():
    """Fixture providing a mocked FAISS vectorstore"""
    vectorstore = MagicMock()
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = [
        Document(
            page_content="Network troubleshooting guide",
            metadata={"ProductID": "P001"}
        )
    ]
    vectorstore.as_retriever.return_value = retriever
    return vectorstore


@pytest.fixture
def mock_bm25_retriever():
    """Fixture providing a mocked BM25 retriever"""
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = [
        Document(
            page_content="Network connectivity issues",
            metadata={"ProductID": "P001"}
        )
    ]
    retriever.k = 5
    return retriever


@pytest.fixture
def mock_ensemble_retriever():
    """Fixture providing a mocked ensemble retriever"""
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = [
        Document(
            page_content="Router XYZ troubleshooting guide",
            metadata={"ProductID": "P001"}
        ),
        Document(
            page_content="Switch ABC configuration steps",
            metadata={"ProductID": "P002"}
        )
    ]
    return retriever


@pytest.fixture
def patch_csv_loader():
    """Patch the CSVLoader to return mock documents"""
    with patch("src.document_loader.CSVLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_docs = [
            Document(page_content="Doc 1 content", metadata={"key": "value1"}),
            Document(page_content="Doc 2 content", metadata={"key": "value2"})
        ]
        mock_loader.load.return_value = mock_docs
        mock_loader_cls.return_value = mock_loader
        yield mock_loader


@pytest.fixture
def patch_exists_true():
    """Patch os.path.exists to return True"""
    with patch("os.path.exists", return_value=True) as mock_exists:
        yield mock_exists


@pytest.fixture
def patch_exists_false():
    """Patch os.path.exists to return False"""
    with patch("os.path.exists", return_value=False) as mock_exists:
        yield mock_exists


@pytest.fixture
def patch_csv_read():
    """Patch pandas.read_csv to return mock dataframes"""
    with patch("pandas.read_csv") as mock_read_csv:
        yield mock_read_csv


@pytest.fixture
def patch_makedirs():
    """Patch os.makedirs to prevent directory creation during tests"""
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def patch_remove():
    """Patch os.remove to prevent file deletion during tests"""
    with patch("os.remove") as mock_remove:
        yield mock_remove

    