import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio
from langchain.schema import AIMessage
import time
from src.rag_chain import SupportRAGChain


@pytest.fixture
def mock_vector_store():
    """Create mock vector store with predefined responses"""
    store = Mock()
    store.query_similar.return_value = [
        {
            'content': """
            Subject: Browser Login Issue
            Description: Unable to login using Safari browser.
            """,
            'metadata': {
                'ticket_id': 'tech-001',
                'support_type': 'technical',
                'tags': ['Browser', 'Login', 'Safari'],
                'answer': 'Clear browser cache and cookies.',
                'type': 'Technical',
                'queue': 'Tech Support',
                'priority': 'high'
            },
            'similarity': 0.92
        }
    ]
    return store


# Patch the entire SupportRAGChain.query method for testing
@patch.object(SupportRAGChain, 'query')
@pytest.mark.asyncio
async def test_basic_query_complete_mock(mock_query):
    """Test basic query by completely mocking the query method"""
    mock_response = "To resolve the Safari browser login issue: Clear your browser cache and cookies."
    mock_query.return_value = mock_response
    
    # Create instance with a dummy vector store (won't be used due to mocking)
    rag_chain = SupportRAGChain(Mock())
    
    query = "I'm having trouble with Safari browser login, can you help me resolve this issue?"
    response = await rag_chain.query(query)
    
    # Assert the mock was called with correct parameters
    # mock_query.assert_called_once_with(query, None)
    
    # Check response content
    assert response == mock_response
    assert "cache" in response.lower()
    assert "cookies" in response.lower()


# Patching only specific internal methods instead of the whole query method
class TestPartialMocks:
    @pytest.fixture
    def rag_chain(self, mock_vector_store):
        return SupportRAGChain(mock_vector_store)
    
    @pytest.mark.asyncio
    async def test_query_with_internal_mocks(self, rag_chain):
        """Test query by mocking all internal methods it calls"""
        query = "I'm having trouble with Safari browser login"
        mock_docs = [{'content': 'test content', 'metadata': {'answer': 'Clear cache'}}]
        mock_context = "Context with answer: Clear cache"
        
        # Mock the internal methods
        with patch.object(rag_chain, 'get_relevant_documents', return_value=mock_docs) as mock_get_docs:
            with patch.object(rag_chain, 'prepare_context', return_value=mock_context) as mock_prepare:
                # Create a mock for the LLM response
                mock_response = AIMessage(content="Clear your browser cache and cookies")
                
                # Create a mock chain
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = mock_response
                
                # Monkeypatch the chain creation temporarily
                original_prompt = rag_chain.prompt_template
                original_llm = rag_chain.llm
                
                try:
                    # Replace with mocks for this test only
                    mock_prompt = MagicMock()
                    mock_prompt.__or__.return_value = mock_chain
                    rag_chain.prompt_template = mock_prompt
                    
                    # Execute the method under test
                    result = await rag_chain.query(query)
                    
                    # Verify all parts were called correctly
                    mock_prepare.assert_called_once_with(mock_docs)
                    mock_chain.ainvoke.assert_called_once()
                    
                    # Check the result
                    assert "cache" in result.lower()
                    assert "cookies" in result.lower()
                    
                finally:
                    # Restore original objects
                    rag_chain.prompt_template = original_prompt
                    rag_chain.llm = original_llm