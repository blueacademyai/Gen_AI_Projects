import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager

class TestVectorStoreManager:
    
    def test_init(self, mock_embeddings, test_data_dir):
        """Test VectorStoreManager initialization"""
        index_path = test_data_dir / "test_index"
        manager = VectorStoreManager(str(index_path), mock_embeddings)
        
        assert manager.index_path == index_path
        assert manager.embedding_model == mock_embeddings
        assert manager.vectorstore is None
    
    @patch('src.vector_store.FAISS')
    def test_create_vectorstore(self, mock_faiss, mock_embeddings, test_data_dir, tmp_path):
        """Test creating a vector store with both valid and empty document cases"""
        # Test Case 1: Successful vector store creation with valid documents
        mock_faiss.from_documents.return_value = MagicMock()
        
        index_path = test_data_dir / "test_index"
        manager = VectorStoreManager(str(index_path), mock_embeddings)
        
        documents = [
            Document(page_content="Test content 1", metadata={"key": "value"}),
            Document(page_content="Test content 2", metadata={"key": "value"})
        ]
        
        manager.create_vectorstore(documents)
        
        mock_faiss.from_documents.assert_called_once_with(documents, mock_embeddings)
        manager.vectorstore.save_local.assert_called_once_with(str(index_path))
        
        # Reset mock for next test case
        mock_faiss.reset_mock()
        
        # Test Case 2: Empty documents list raises ValueError
        mock_embeddings_empty = type('MockEmbeddings', (), {
            'embed_documents': lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts],
            'embed_query': lambda self, text: [0.1, 0.2, 0.3]
        })()
        
        index_path_empty = tmp_path / "vectorstore"
        vector_store_manager = VectorStoreManager(str(index_path_empty), mock_embeddings_empty)
        
        with pytest.raises(ValueError) as exc_info:
            vector_store_manager.create_vectorstore(documents=[])
        
        assert str(exc_info.value) == "Documents list cannot be empty", "Expected ValueError with correct message"

    @pytest.fixture
    def mock_embeddings(self):
        """Minimal mock embeddings object."""
        return type('MockEmbeddings', (), {
            'embed_documents': lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts],
            'embed_query': lambda self, text: [0.1, 0.2, 0.3]
        })()

    def test_append_vectorstore_uninitialized(self, tmp_path, mock_embeddings):
        """
        Test ValueError handling in append_vectorstore when vector store is not initialized.
        Verifies that a ValueError is raised with the correct message.
        """
        # Initialize VectorStoreManager with a temporary directory and mock embeddings
        index_path = tmp_path / "vectorstore"
        vector_store_manager = VectorStoreManager(str(index_path), mock_embeddings)

        # Test with valid data but uninitialized vector store
        new_data = {
            'description': 'Valid description',
            'TicketID': '1',
            'locationID': 'LOC001',
            'estimated_resolution_time': '2'
        }
        with pytest.raises(ValueError) as exc_info:
            vector_store_manager.append_vectorstore(new_data)

        # Verify the exception message
        assert str(exc_info.value) == "Vector store not initialized. Call create_vectorstore first.", \
            "Expected ValueError with correct message for uninitialized vector store"

    def test_append_vectorstore_missing_fields(self, tmp_path, mock_embeddings):
        """
        Test ValueError handling in append_vectorstore when required fields are missing.
        Verifies that a ValueError is raised with the correct message for each missing field.
        """
        # Initialize VectorStoreManager and create a vector store to avoid uninitialized error
        index_path = tmp_path / "vectorstore"
        vector_store_manager = VectorStoreManager(str(index_path), mock_embeddings)
        from langchain_core.documents import Document
        vector_store_manager.create_vectorstore([Document(page_content="Test", metadata={})])

        # Test data missing 'TicketID'
        new_data = {
            'description': 'Valid description',
            'locationID': 'LOC001',
            'estimated_resolution_time': '2'
        }
        with pytest.raises(ValueError) as exc_info:
            vector_store_manager.append_vectorstore(new_data)

        # Verify the exception message
        assert str(exc_info.value) == "Missing required field: TicketID", \
            "Expected ValueError with correct message for missing TicketID"

    def test_append_vectorstore_too_short_description(self, tmp_path, mock_embeddings):
        """
        Test ValueError handling in append_vectorstore when description is too short.
        Verifies that a ValueError is raised with the correct message.
        """
        # Initialize VectorStoreManager and create a vector store to avoid uninitialized error
        index_path = tmp_path / "vectorstore"
        vector_store_manager = VectorStoreManager(str(index_path), mock_embeddings)
        from langchain_core.documents import Document
        vector_store_manager.create_vectorstore([Document(page_content="Test", metadata={})])

        # Test data with too short description
        new_data = {
            'description': 'Hi',
            'TicketID': '1',
            'locationID': 'LOC001',
            'estimated_resolution_time': '2'
        }
        with pytest.raises(ValueError) as exc_info:
            vector_store_manager.append_vectorstore(new_data)

        # Verify the exception message
        assert str(exc_info.value) == "Description is too short for meaningful vector representation", \
            "Expected ValueError with correct message for too short description"
        
    
    @pytest.fixture
    def mock_embeddings(self):
        """Minimal mock embeddings object."""
        return type('MockEmbeddings', (), {
            'embed_documents': lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts],
            'embed_query': lambda self, text: [0.1, 0.2, 0.3]
        })()

    @patch('src.vector_store.FAISS')  # Adjust the module path if needed
    def test_get_vectorstore(self, mock_faiss, mock_embeddings, tmp_path):
        """
        Test get_vectorstore method for:
        1. FileNotFoundError when index path does not exist.
        2. Returning already loaded vector store.
        """
        # Test case 1: Verify FileNotFoundError when index path does not exist
        index_path = tmp_path / "non_existent_index"
        manager = VectorStoreManager(str(index_path), mock_embeddings)

        with pytest.raises(FileNotFoundError) as exc_info:
            manager.get_vectorstore()
        assert str(exc_info.value) == f"Vector store index not found at {index_path}", \
            "Expected FileNotFoundError with correct message for non-existent index"

        # Test case 2: Verify returning already loaded vector store
        index_path = tmp_path / "test_index"
        manager = VectorStoreManager(str(index_path), mock_embeddings)
        
        # Set the vectorstore directly
        mock_vs = MagicMock()
        manager.vectorstore = mock_vs
        
        result = manager.get_vectorstore()
        
        assert result == mock_vs, "Expected already loaded vector store to be returned"
        mock_faiss.load_local.assert_not_called(), "FAISS.load_local should not be called when vector store is already loaded"