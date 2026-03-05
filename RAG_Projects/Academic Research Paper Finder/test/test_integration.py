import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import os
from unittest.mock import patch, MagicMock
from src.document_loader import ResearchPaperLoader
from src.vector_store import ResearchVectorStore
from src.retriever import ResearchPaperRetriever
import pickle

class TestResearchPaperIntegration:
    @pytest.fixture
    def sample_data_directory(self, tmp_path):
        """Create sample data directory with research papers CSV"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Sample paper data
        papers_data = {
            'title': [
                'Transformer Models in NLP: A Survey',
                'Quantum Computing for Cryptography',
                'Gene Sequencing Algorithms Using AI'
            ],
            'abstract': [
                'This paper provides a comprehensive survey of transformer models in NLP.',
                'Applications of quantum computing in modern cryptography systems.',
                'Novel approaches to gene sequencing using artificial intelligence.'
            ],
            'authors': [
                'Smith, J.; Johnson, A.',
                'Brown, R.; Davis, M.',
                'Wilson, C.; Moore, T.'
            ],
            'year': [2022, 2021, 2023],
            'venue': [
                'Journal of NLP',
                'Quantum Computing Conference',
                'Bioinformatics Journal'
            ],
            'n_citation': [120, 75, 30],
            'references': [
                'ref1,ref2,ref3',
                'ref4,ref5',
                'ref6,ref7,ref8'
            ],
            'id': ['paper1', 'paper2', 'paper3']
        }
        
        # Create CSV file
        df = pd.DataFrame(papers_data)
        csv_path = data_dir / "papers.csv"
        df.to_csv(csv_path, index=False)
        
        return data_dir
    
    @pytest.fixture(autouse=True)
    def setup_openai_key(self, monkeypatch):
        """Setup OpenAI API key for tests"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_full_pipeline(self, mock_openai_embeddings, mock_faiss, sample_data_directory, tmp_path):
        """Test complete pipeline with document loading, embedding, and retrieval"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_index.d = 3
        # Return index 0 (first document) for our test query
        mock_index.search.return_value = (
            np.array([[0.1]]),  # distances
            np.array([[0]])     # indices (first document)
        )
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Set up vector store directory
        faiss_store_dir = tmp_path / "faiss_store"
        os.makedirs(faiss_store_dir, exist_ok=True)
        
        # 1. Load documents
        loader = ResearchPaperLoader(str(sample_data_directory / "papers.csv"))
        documents = loader.create_documents()
        
        assert len(documents) == 3
        
        # 2. Create vector store
        vector_store = ResearchVectorStore(store_path=str(faiss_store_dir))
        
        # Manually set our mocks
        vector_store.embeddings = mock_embeddings_instance
        
        # Create the vector store
        vector_store.create_vector_store(documents)
        
        # Set the index directly to our mock
        vector_store.index = mock_index
        
        # 3. Create retriever
        retriever = ResearchPaperRetriever(vector_store)
        
        # 4. Test retrieval - should return the first document (index 0)
        results = retriever.retrieve_papers("transformer models in NLP")
        
        assert len(results) == 1
        assert "Transformer Models in NLP" in results[0]['title']
        assert results[0]['authors'] == 'Smith, J.; Johnson, A.'
        assert 'abstract' in results[0]
        
        # 5. Test retrieval with recency
        # Mock the search result again
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances for all 3 docs
            np.array([[0, 1, 2]])        # indices of all 3 docs
        )
        
        results_with_recency = retriever.retrieve_papers_with_recency("transformer models")
        
        # Sort results by year (most recent first) - this happens in the app
        sorted_results = sorted(results_with_recency, key=lambda x: x['year'], reverse=True)
        
        # The most recent should be 2023 (paper3)
        assert len(sorted_results) > 0  # Make sure we have results
        assert sorted_results[0]['year'] == 2023
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_query_validation(self, mock_openai_embeddings, mock_faiss, sample_data_directory, tmp_path):
        """Test query validation in retriever"""
        # Setup minimal mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create vector store
        vector_store = ResearchVectorStore(store_path=str(tmp_path))
        vector_store.embeddings = mock_embeddings_instance
        vector_store.index = mock_index
        
        # Create retriever
        retriever = ResearchPaperRetriever(vector_store)
        
        # Test empty query
        with pytest.raises(ValueError) as e:
            retriever.retrieve_papers("")
        assert "Query cannot be empty" in str(e.value)
        
        # Test short query
        with pytest.raises(ValueError) as e:
            retriever.retrieve_papers("ai")
        assert "Query too short" in str(e.value)
    
    @patch('src.vector_store.faiss')
    @patch('src.vector_store.OpenAIEmbeddings')
    def test_loading_existing_store(self, mock_openai_embeddings, mock_faiss, sample_data_directory, tmp_path):
        """Test loading an existing vector store"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_index.d = 3
        mock_faiss.read_index.return_value = mock_index
        
        # Create store path and required files
        store_path = tmp_path / "faiss_store"
        store_path.mkdir()
        
        # Create dummy files
        (store_path / "faiss_index.bin").write_bytes(b"test")
        
        # Create metadata and documents pickle files
        metadata = [
            {'abstract': 'Paper 1 content', 'year': 2022},
            {'abstract': 'Paper 2 content', 'year': 2023}
        ]
        documents = [
            "title: Paper 1",
            "title: Paper 2"
        ]
        
        with open(store_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
            
        with open(store_path / "documents.pkl", 'wb') as f:
            pickle.dump(documents, f)
        
        # Load the store
        loaded_store = ResearchVectorStore.load(str(store_path))
        
        # Check if it loaded correctly
        assert loaded_store.index is not None
        assert len(loaded_store.metadata) == 2
        assert len(loaded_store.documents) == 2
        
        # Create retriever with loaded store
        retriever = ResearchPaperRetriever(loaded_store)
        
        # Set up mock for search
        loaded_store.index.search.return_value = (
            np.array([[0.1]]),  # distances
            np.array([[0]])     # indices (first document)
        )
        
        # Try retrieving
        loaded_store.embeddings = mock_embeddings_instance
        results = retriever.retrieve_papers("test query with enough chars")
        
        assert len(results) == 1

        assert results[0]['title'] == 'Paper 1'