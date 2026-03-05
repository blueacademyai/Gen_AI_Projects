import os
import pytest
from unittest.mock import patch, MagicMock

# Define mock classes for imports
class MockFAISS:
    @classmethod
    def from_documents(cls, documents, embedding_model):
        mock = MagicMock()
        mock.save_local = MagicMock()
        return mock
    
    @classmethod
    def load_local(cls, folder_path, embeddings, index_name):
        return MagicMock()

class MockOpenAIEmbeddings:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]

# Apply patches for imports
patch("langchain_community.vectorstores.FAISS", MockFAISS).start()
patch("langchain_openai.OpenAIEmbeddings", MockOpenAIEmbeddings).start()

# Clear any existing patches
patch.stopall()

# Now apply fresh patches
patch("src.vector_store.FAISS", MockFAISS).start()
patch("src.vector_store.OpenAIEmbeddings", MockOpenAIEmbeddings).start()

# Now import the class we want to test
from src.vector_store import SocialMediaVectorStore


class TestSocialMediaVectorStore:
    def test_init(self, mock_embeddings):
        """Test initialization with default and custom parameters."""
        # Default initialization 
        vs = SocialMediaVectorStore()
        assert vs.index_path == "faiss_index"
        assert vs.index_name == "support_docs"
        assert vs.vectorstore is None
        
        # Custom initialization
        vs = SocialMediaVectorStore(
            embedding_model=mock_embeddings,
            index_path="custom_path",
            index_name="custom_index"
        )
        assert vs.embedding_model == mock_embeddings
        assert vs.index_path == "custom_path"
        assert vs.index_name == "custom_index"
    
    def test_create_vectorstore(self, sample_documents, mock_embeddings):
        """Test vector store creation from documents."""
        # Mock FAISS.from_documents directly
        with patch("src.vector_store.FAISS.from_documents") as mock_from_documents:
            mock_vs = MagicMock()
            mock_from_documents.return_value = mock_vs
            
            vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
            result = vs.create_vectorstore(sample_documents)
            
            # Check FAISS.from_documents was called correctly
            mock_from_documents.assert_called_once()
            
            # Check instance state was updated
            assert result is mock_vs
            assert vs.vectorstore is mock_vs
            
            # Test with empty document list
            result = vs.create_vectorstore([])
            assert result is None
    
    @patch("os.makedirs")
    def test_save_vectorstore(self, mock_makedirs, mock_embeddings):
        """Test saving vector store to disk."""
        vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
        
        # Test when vectorstore is None
        assert vs.save_vectorstore() is False
        
        # Test successful save
        mock_faiss = MagicMock()
        vs.vectorstore = mock_faiss
        
        result = vs.save_vectorstore()
        
        mock_makedirs.assert_called_once_with("faiss_index", exist_ok=True)
        assert mock_faiss.save_local.called
        assert result is True
        
        # Test exception handling
        mock_faiss.save_local.side_effect = Exception("Save error")
        result = vs.save_vectorstore()
        assert result is False
    
    @patch("os.path.exists")
    def test_load_vectorstore(self, mock_exists, mock_embeddings):
        """Test loading vector store from disk."""
        mock_exists.return_value = True
        
        # Mock FAISS.load_local
        with patch("src.vector_store.FAISS.load_local") as mock_load_local:
            mock_vs = MagicMock()
            mock_load_local.return_value = mock_vs
            
            vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
            result = vs.load_vectorstore()
            
            # Check path existence check was correct
            mock_exists.assert_called_once_with(os.path.join("faiss_index", "support_docs.faiss"))
            
            # Check results
            assert result is mock_vs
            assert vs.vectorstore is mock_vs
            
            # Test when file doesn't exist
            mock_exists.return_value = False
            result = vs.load_vectorstore()
            assert result is None
    
    def test_get_embedding_for_text(self, mock_embeddings):
        """Test getting embeddings for text."""
        vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
        embedding = vs.get_embedding_for_text("Test query")
        
        mock_embeddings.embed_query.assert_called_once_with("Test query")
        assert embedding == [0.1, 0.2, 0.3, 0.4]
        
        # Test exception handling
        mock_embeddings.embed_query.side_effect = Exception("Embedding error")
        with pytest.raises(Exception):
            vs.get_embedding_for_text("Test query")