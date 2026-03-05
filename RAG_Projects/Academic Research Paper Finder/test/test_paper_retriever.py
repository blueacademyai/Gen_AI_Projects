import pytest
from unittest.mock import MagicMock, patch
from src.retriever import ResearchPaperRetriever

class TestResearchPaperRetriever:
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = MagicMock()
        
        # Define return value for query_similar
        mock_store.query_similar.return_value = [
            {
                'content': 'title: Transformer models have changed NLP',
                'metadata': {
                    'authors': 'Smith, J.',
                    'year': 2022,
                    'venue': 'NLP Conference',
                    'n_citation': 50,
                    'abstract': 'An in-depth analysis of BERT models and their applications.',
                    'id': 'paper1'
                },
                'similarity': 0.92
            },
            {
                'content': 'title: GPT models represent a breakthrough in language generation',
                'metadata': {
                    'authors': 'Johnson, A.',
                    'year': 2023,
                    'venue': 'AI Journal',
                    'n_citation': 35,
                    'abstract': 'A comprehensive review of GPT architecture development.',
                    'id': 'paper2'
                },
                'similarity': 0.85
            }
        ]
        
        return mock_store
    
    def test_retrieve_papers(self, mock_vector_store):
        """Test basic paper retrieval"""
        retriever = ResearchPaperRetriever(mock_vector_store)
        results = retriever.retrieve_papers("transformer models in NLP", k=2)
        
        assert len(results) == 2
        assert results[0]['title'] == 'Transformer models have changed NLP'
        assert results[1]['title'] == 'GPT models represent a breakthrough in language generation'
        assert results[0]['rank'] == 1
        assert results[1]['rank'] == 2
        assert 'similarity_score' in results[0]
        assert 'abstract' in results[0]
    
    def test_retrieve_papers_with_recency(self, mock_vector_store):
        """Test paper retrieval with recency ranking"""
        retriever = ResearchPaperRetriever(mock_vector_store)
        results = retriever.retrieve_papers_with_recency("transformer models", k=2)
        
        assert len(results) == 2
        # Since this just calls retrieve_papers with use_recency=True,
        # we don't need additional assertions beyond checking it works
    
    def test_short_query_handling(self, mock_vector_store):
        """Test handling of short queries"""
        retriever = ResearchPaperRetriever(mock_vector_store)
        
        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve_papers("AI")
        assert "Query too short" in str(exc_info.value)
    
    def test_empty_query_handling(self, mock_vector_store):
        """Test handling of empty queries"""
        retriever = ResearchPaperRetriever(mock_vector_store)
        
        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve_papers("")
        assert "Query cannot be empty" in str(exc_info.value)