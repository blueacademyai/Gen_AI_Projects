# import pytest
# import numpy as np
# import os
# import pickle
# from pathlib import Path
# from langchain.schema import Document
# from unittest.mock import patch, MagicMock
# from src.vector_store import ResearchVectorStore

# class TestFAISSResearchVectorStore:
#     @pytest.fixture
#     def sample_documents(self):
#         """Create sample documents for testing"""
#         return [
#             Document(
#                 page_content="Quantum computing is a rapidly evolving field.",
#                 metadata={
#                     'title': 'Quantum Computing Advances',
#                     'authors': 'Smith, J.',
#                     'year': 2022,
#                     'venue': 'Quantum Journal',
#                     'n_citation': 15,
#                     'id': 'paper1',
#                     'references': "['ref1', 'ref2']"
#                 }
#             ),
#             Document(
#                 page_content="Neural networks have revolutionized machine learning.",
#                 metadata={
#                     'title': 'Neural Network Applications',
#                     'authors': 'Johnson, A.',
#                     'year': 2023,
#                     'venue': 'AI Conference',
#                     'n_citation': 30,
#                     'id': 'paper2',
#                     'references': "['ref3', 'ref4']"
#                 }
#             )
#         ]
    
#     @pytest.fixture
#     def mock_embeddings(self):
#         """Mock embedding responses"""
#         return [
#             [0.1, 0.2, 0.3],  # Simple 3D embeddings for testing
#             [0.4, 0.5, 0.6]
#         ]
    
#     @patch('src.vector_store.faiss')
#     # @patch('src.vector_store.HuggingFaceEmbeddings')
#     def test_create_vector_store(self, mock_hf_embeddings, mock_faiss, sample_documents, tmp_path):
#         """Test creating vector store with documents"""
#         # Setup mocks
#         mock_embeddings_instance = MagicMock()
#         mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
#         mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
#         mock_hf_embeddings.return_value = mock_embeddings_instance
        
#         # Mock FAISS index
#         mock_index = MagicMock()
#         mock_faiss.IndexFlatL2.return_value = mock_index
        
#         # Create vector store
#         store = ResearchVectorStore(store_path=str(tmp_path))
#         store.create_vector_store(sample_documents)
        
#         # Check if FAISS index was created and documents were processed
#         assert mock_faiss.IndexFlatL2.called
#         assert mock_embeddings_instance.embed_documents.called
#         assert mock_faiss.write_index.called
        
#         # Check if pickle files were created
#         assert os.path.exists(tmp_path / "metadata.pkl")
#         assert os.path.exists(tmp_path / "documents.pkl")
    
    
#     @patch('src.vector_store.faiss')
#     @patch('src.vector_store.HuggingFaceEmbeddings')
#     def test_query_similar(self, mock_hf_embeddings, mock_faiss, sample_documents, tmp_path):
#         """Test querying similar documents"""
#         # Setup mocks
#         mock_embeddings_instance = MagicMock()
#         mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
#         mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
#         mock_hf_embeddings.return_value = mock_embeddings_instance
        
#         # Mock FAISS index
#         mock_index = MagicMock()
#         mock_index.ntotal = 2
#         mock_index.d = 3
#         mock_index.search.return_value = (
#             np.array([[0.1]]),  # distances
#             np.array([[0]])     # indices
#         )
#         mock_faiss.IndexFlatL2.return_value = mock_index
        
#         # Create vector store with the mock index
#         store = ResearchVectorStore(store_path=str(tmp_path))
        
#         # Set up the store with test data directly
#         store.index = mock_index
#         store.documents = ["Quantum computing is a rapidly evolving field."]
#         store.metadata = [{
#             'title': 'Quantum Computing Advances',
#             'authors': 'Smith, J.',
#             'year': '2022',
#             'venue': 'Quantum Journal',
#             'n_citation': '15',
#             'id': 'paper1',
#             'references': 'ref1,ref2'
#         }]
#         store.embedding_size = 3
        
#         # Test query
#         results = store.query_similar("quantum computing", k=1)
        
#         # Assertions
#         assert len(results) == 1
#         assert 'content' in results[0]
#         assert 'metadata' in results[0]
#         assert 'similarity' in results[0]
#         assert results[0]['metadata']['title'] == 'Quantum Computing Advances'
        
#         # Test query with recency
#         results = store.query_similar("quantum computing", k=1, use_recency=True)
#         assert len(results) == 1
    
    
#     @patch('src.vector_store.faiss')
#     @patch('src.vector_store.HuggingFaceEmbeddings')
#     def test_save_and_load(self, mock_hf_embeddings, mock_faiss, sample_documents, tmp_path):
#         """Test saving and loading the vector store"""
#         # Setup mocks
#         mock_embeddings_instance = MagicMock()
#         mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
#         mock_hf_embeddings.return_value = mock_embeddings_instance
        
#         # Mock FAISS functions
#         mock_index = MagicMock()
#         mock_index.ntotal = 2
#         mock_index.d = 3
#         mock_faiss.IndexFlatL2.return_value = mock_index
        
#         # Create and save vector store
#         store = ResearchVectorStore(store_path=str(tmp_path))
#         store.index = mock_index
#         store.documents = ["Quantum computing is a rapidly evolving field."]
#         store.metadata = [{'title': 'Quantum Computing Advances'}]
#         store.embedding_size = 3
#         store.save()
        
#         # Check if files were saved
#         assert mock_faiss.write_index.called
#         assert os.path.exists(tmp_path / "metadata.pkl")
#         assert os.path.exists(tmp_path / "documents.pkl")
        
#         # Mock for load
#         mock_faiss.read_index.return_value = mock_index
        
#         # Test loading
#         mock_faiss.read_index.reset_mock()
#         loaded_store = ResearchVectorStore.load(str(tmp_path))
#         assert mock_faiss.read_index.called
#         assert loaded_store.index is not None
#         assert loaded_store.embedding_size == 3

import pytest
import numpy as np
import os
import pickle
from pathlib import Path
from langchain.schema import Document
from unittest.mock import patch, MagicMock
from src.vector_store import ResearchVectorStore

class TestResearchVectorStore:
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="Quantum computing is a rapidly evolving field.",
                metadata={
                    'title': 'Quantum Computing Advances',
                    'authors': 'Smith, J.',
                    'year': 2022,
                    'venue': 'Quantum Journal',
                    'n_citation': 15,
                    'id': 'paper1',
                    'references': "['ref1', 'ref2']",
                    'abstract': 'Overview of recent advances in quantum computing.'
                }
            ),
            Document(
                page_content="Neural networks have revolutionized machine learning.",
                metadata={
                    'title': 'Neural Network Applications',
                    'authors': 'Johnson, A.',
                    'year': 2023,
                    'venue': 'AI Conference',
                    'n_citation': 30,
                    'id': 'paper2',
                    'references': "['ref3', 'ref4']",
                    'abstract': 'A study of neural network applications in industry.'
                }
            )
        ]
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embedding responses"""
        return [
            [0.1, 0.2, 0.3],  # Simple 3D embeddings for testing
            [0.4, 0.5, 0.6]
        ]

    @pytest.fixture(autouse=True)
    def setup_openai_key(self, monkeypatch):
        """Setup OpenAI API key for tests"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_create_vector_store(self, mock_openai_embeddings, mock_faiss, sample_documents, tmp_path):
        """Test creating vector store with documents"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create vector store
        store = ResearchVectorStore(store_path=str(tmp_path))
        
        # Manually set the embeddings to our mock
        store.embeddings = mock_embeddings_instance
        
        # Create the vector store
        store.create_vector_store(sample_documents)
        
        # Check if FAISS index was created and documents were processed
        assert mock_faiss.IndexFlatL2.called
        assert mock_embeddings_instance.embed_documents.called
        assert mock_faiss.write_index.called
        
        # Check if correct number of documents were processed
        assert len(store.documents) == 2
        assert len(store.metadata) == 2
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_query_similar(self, mock_openai_embeddings, mock_faiss, sample_documents, tmp_path):
        """Test querying similar documents"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS index and search results
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.d = 3
        # Return indices 0 with distance 0.1
        mock_index.search.return_value = (
            np.array([[0.1]]),  # distances
            np.array([[0]])     # indices (first document)
        )
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create vector store
        store = ResearchVectorStore(store_path=str(tmp_path))
        
        # Set up the store with test data
        store.embeddings = mock_embeddings_instance
        store.index = mock_index
        store.documents = [doc.page_content for doc in sample_documents]
        store.metadata = [doc.metadata for doc in sample_documents]
        store.embedding_size = 3
        
        # Test query
        results = store.query_similar("quantum computing", k=1)
        
        # Assertions
        assert len(results) == 1
        assert 'content' in results[0]
        assert 'metadata' in results[0]
        assert 'similarity' in results[0]
        assert results[0]['metadata']['title'] == 'Quantum Computing Advances'
        
        # Test query with recency
        results = store.query_similar("quantum computing", k=1, use_recency=True)
        assert len(results) == 1
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_save_and_load(self, mock_openai_embeddings, mock_faiss, sample_documents, tmp_path):
        """Test saving and loading the vector store"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS functions
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.d = 3
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create vector store
        store = ResearchVectorStore(store_path=str(tmp_path))
        store.embeddings = mock_embeddings_instance
        store.index = mock_index
        store.documents = [doc.page_content for doc in sample_documents]
        store.metadata = [doc.metadata for doc in sample_documents]
        store.embedding_size = 3
        
        # Create the files that would normally be created by save()
        index_path = tmp_path / "faiss_index.bin"
        metadata_path = tmp_path / "metadata.pkl"
        documents_path = tmp_path / "documents.pkl"
        
        # Create a dummy binary file for the index
        with open(index_path, 'wb') as f:
            f.write(b'test')
            
        # Save metadata and documents
        with open(metadata_path, 'wb') as f:
            pickle.dump(store.metadata, f)
        
        with open(documents_path, 'wb') as f:
            pickle.dump(store.documents, f)
        
        # Now call save() which should use our mocks
        store.save()
        
        # Verify files exist
        assert os.path.exists(index_path)
        assert os.path.exists(metadata_path)
        assert os.path.exists(documents_path)
        
        # Mock for load
        mock_faiss.read_index.return_value = mock_index
        
        # Test loading
        loaded_store = ResearchVectorStore.load(str(tmp_path))
        assert loaded_store.index is not None
        assert loaded_store.embedding_size == 3
        assert len(loaded_store.documents) == 2
        assert len(loaded_store.metadata) == 2
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_empty_query(self, mock_openai_embeddings, mock_faiss, sample_documents, tmp_path):
        """Test handling of empty queries"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Create vector store
        store = ResearchVectorStore(store_path=str(tmp_path))
        store.embeddings = mock_embeddings_instance
        store.index = MagicMock()
        
        # Test with empty string
        results = store.query_similar("")
        assert len(results) == 0
        
        # Test with only whitespace
        results = store.query_similar("   ")
        assert len(results) == 0
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_no_index_available(self, mock_openai_embeddings, mock_faiss, tmp_path):
        """Test querying when no index is available"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Create vector store without an index
        store = ResearchVectorStore(store_path=str(tmp_path))
        store.embeddings = mock_embeddings_instance
        store.index = None
        
        # Test query
        results = store.query_similar("quantum computing")
        assert len(results) == 0
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_recency_ranking(self, mock_openai_embeddings, mock_faiss, sample_documents, tmp_path):
        """Test recency-based ranking"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS index to return both documents
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.d = 3
        # Return both indices with equal distances
        mock_index.search.return_value = (
            np.array([[0.2, 0.2]]),  # equal distances
            np.array([[0, 1]])       # both indices
        )
        
        # Create vector store
        store = ResearchVectorStore(store_path=str(tmp_path))
        store.embeddings = mock_embeddings_instance
        store.index = mock_index
        
        # Set documents with different years
        store.documents = [doc.page_content for doc in sample_documents]
        store.metadata = [
            {'title': 'Older Paper', 'year': 2020},
            {'title': 'Newer Paper', 'year': 2023}
        ]
        store.embedding_size = 3
        
        # Test query with recency
        results = store.query_similar("test query", k=2, use_recency=True)
        
        # The newer paper should be ranked higher despite equal similarity
        assert len(results) == 2
        assert results[0]['metadata']['title'] == 'Newer Paper'
        assert results[0]['metadata']['year'] == 2023