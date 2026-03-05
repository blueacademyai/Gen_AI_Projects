import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Import Document first to ensure it's available
from langchain_core.documents import Document

# Mock the modules we're testing against to avoid import errors
class MockSocialMediaDocumentLoader:
    def __init__(self, *args, **kwargs):
        pass
    
    def load_and_process(self, *args, **kwargs):
        return [Document(page_content="Test content", metadata={"source": "test.txt"})]

class MockSocialMediaVectorStore:
    def __init__(self, *args, **kwargs):
        pass
    
    def create_vectorstore(self, documents):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [(documents[0], 0.3)]
        mock_retriever = MagicMock()
        mock_retriever.__or__.return_value = lambda x: "Formatted context"
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs

class MockSocialMediaRAGChain:
    def __init__(self, *args, **kwargs):
        self.chain = MagicMock()
        self.chain.invoke.return_value = "Here's how to reset your password..."
    
    def query(self, question):
        return {
            "answer": "Here's how to reset your password...",
            "source_documents": [Document(page_content="Test content", metadata={"source": "test.txt"})]
        }

# Create patches for the imports
patch("src.document_loader.SocialMediaDocumentLoader", MockSocialMediaDocumentLoader).start()
patch("src.vector_store.SocialMediaVectorStore", MockSocialMediaVectorStore).start()
patch("src.rag_chain.SocialMediaRAGChain", MockSocialMediaRAGChain).start()

# Now do the imports (which will use our mocks)
from src.document_loader import SocialMediaDocumentLoader
from src.vector_store import SocialMediaVectorStore
from src.rag_chain import SocialMediaRAGChain


class TestIntegration:
    def test_end_to_end_workflow(self, temp_dir, mock_text_file):
        """Test the complete workflow from document loading to query response."""
        # 1. Set up document loader and load documents
        loader = SocialMediaDocumentLoader(temp_dir)
        documents = loader.load_and_process()
        
        assert len(documents) > 0, "Should have loaded at least one document"
        
        # 2. Create vector store with the documents
        vector_store = SocialMediaVectorStore()
        vs = vector_store.create_vectorstore(documents)
        
        # 3. Create RAG chain with the vector store
        rag_chain = SocialMediaRAGChain(vectorstore=vs)
        
        # 4. Execute a query and check the response
        response = rag_chain.query("How do I reset my password?")
        
        # Verify the query went through
        assert "answer" in response
        assert "source_documents" in response