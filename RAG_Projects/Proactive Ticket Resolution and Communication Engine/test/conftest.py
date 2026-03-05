import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

class MockEmbeddings:
    """Mock embedding model for testing vector stores"""
    def embed_documents(self, texts):
        # Return mock embeddings of the right shape (4 dimensions for simplicity)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    
    def embed_query(self, text):
        # Return a mock embedding for a query
        return [0.1, 0.2, 0.3, 0.4]

@pytest.fixture
def mock_openai_api_key():
    return "sk-mock-api-key"

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test data files"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def metadata_csv(test_data_dir):
    """Create a test metadata CSV file"""
    metadata_file = test_data_dir / "metadata.csv"
    
    metadata_df = pd.DataFrame({
        'TicketID': [1, 2, 3],
        'customerID': [101, 102, 103],
        'locationID': [1001, 1002, 1003],
        'type': ['complaint', 'inquiry', 'complaint'],
        'clusterID': [1, 2, 1],
        'estimated_resolution_time': [24, 12, 48]
    })
    
    metadata_df.to_csv(metadata_file, index=False)
    return str(metadata_file)

@pytest.fixture
def description_csv(test_data_dir):
    """Create a test description CSV file"""
    description_file = test_data_dir / "description.csv"
    
    description_df = pd.DataFrame({
        'TicketID': [1, 2, 3],
        'description': [
            'This is a short description for ticket 1.',
            'This is a medium length description for ticket 2 that has a bit more detail about the issue.',
            'This is a longer description for ticket 3 that contains multiple sentences. '
            'It describes an issue in more detail. The customer is experiencing problems with their service. '
            'They have attempted several troubleshooting steps already.'
        ]
    })
    
    description_df.to_csv(description_file, index=False)
    return str(description_file)

@pytest.fixture
def long_description_csv(test_data_dir):
    """Create a test description CSV file with a very long description"""
    description_file = test_data_dir / "long_description.csv"
    
    # Create a description that's over 1500 characters
    long_text = "This is a very long description. " * 100
    
    description_df = pd.DataFrame({
        'TicketID': [1],
        'description': [long_text]
    })
    
    description_df.to_csv(description_file, index=False)
    return str(description_file)

@pytest.fixture
def mock_embeddings():
    """Create a mock embedding model"""
    return MockEmbeddings()

@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store"""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [
        (MagicMock(metadata={'ticket_id': '1', 'location_id': '1001', 'estimated_resolution_time': '24'}), 0.8),
        (MagicMock(metadata={'ticket_id': '2', 'location_id': '1002', 'estimated_resolution_time': '12'}), 0.7),
        (MagicMock(metadata={'ticket_id': '3', 'location_id': '1001', 'estimated_resolution_time': '48'}), 0.6)
    ]
    return mock_vs

@pytest.fixture
def mock_summarization_chain():
    """Create a mock summarization chain"""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "This is a summarized description."
    return mock_chain