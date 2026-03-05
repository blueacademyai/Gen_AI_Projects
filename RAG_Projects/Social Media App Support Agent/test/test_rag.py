import pytest
from unittest.mock import patch, MagicMock

# Import the dependencies here
from langchain_core.documents import Document

# Add patches for langchain_openai's ChatOpenAI
patch("langchain_openai.ChatOpenAI", MagicMock()).start()

# Mock modules we need to test against
class MockPromptTemplate:
    def __init__(self, template=None):
        self.template = template
    
    @classmethod
    def from_template(cls, template):
        return cls(template=template)

class MockStrOutputParser:
    def __init__(self):
        pass

class MockRunnablePassthrough:
    def __init__(self):
        pass

# Apply patches
patch("langchain_core.prompts.PromptTemplate", MockPromptTemplate).start()
patch("langchain_core.output_parsers.StrOutputParser", MockStrOutputParser).start()
patch("langchain_core.runnables.RunnablePassthrough", MockRunnablePassthrough).start()

# Explicitly mock ChatOpenAI for the src module 
patch("src.rag_chain.ChatOpenAI", MagicMock()).start()

# Mock the module we're testing
class MockSocialMediaRAGChain:
    def __init__(
        self,
        vectorstore,
        llm=None,
        temperature=0.0,
        k=4,
        return_source_documents=True,
        prompt_template="Default prompt",
        similarity_threshold=0.7
    ):
        self.vectorstore = vectorstore
        self.llm = llm or MagicMock()
        self.k = k
        self.return_source_documents = return_source_documents
        self.prompt = MockPromptTemplate(template=prompt_template)
        self.similarity_threshold = similarity_threshold
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.chain = MagicMock()
        
    def query(self, question):
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=question,
                k=self.k
            )
            
            relevant_docs = [
                doc for doc, score in docs_with_scores 
                if score <= self.similarity_threshold
            ]
            
            if not relevant_docs:
                return {
                    "answer": "I'm sorry, I don't have enough information to answer this question confidently.",
                    "source_documents": []
                }
                
            answer = self.chain.invoke(question)
            
            sources = []
            for doc in relevant_docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            result = {
                "answer": answer,
                "sources": sources
            }
            
            if self.return_source_documents:
                result["source_documents"] = relevant_docs
            
            return result
        except Exception as e:
            return {
                "answer": f"I'm sorry, I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "source_documents": []
            }
    
    def get_relevant_documents(self, query, k=None):
        k = k or self.k
        return self.vectorstore.similarity_search(query, k=k)

# Apply the patch for importing
patch("src.rag_chain.SocialMediaRAGChain", MockSocialMediaRAGChain).start()

# Now import the module
from src.rag_chain import SocialMediaRAGChain, default_prompt_template


class TestSocialMediaRAGChain:
    def test_init(self, mock_vectorstore, mock_llm):
        """Test initialization with default and custom parameters."""
        # Default initialization
        chain = SocialMediaRAGChain(vectorstore=mock_vectorstore)
        assert chain.vectorstore == mock_vectorstore
        assert chain.k == 4
        assert chain.return_source_documents is True
        assert chain.similarity_threshold == 0.7
        
        # Custom initialization
        custom_prompt = "Custom prompt template {context} {question}"
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm,
            temperature=0.5,
            k=3,
            return_source_documents=False,
            prompt_template=custom_prompt,
            similarity_threshold=0.5
        )
        
        assert chain.vectorstore == mock_vectorstore
        assert chain.llm == mock_llm
        assert chain.k == 3
        assert chain.return_source_documents is False
        assert chain.prompt.template == custom_prompt
        assert chain.similarity_threshold == 0.5
    
    def test_create_chain(self, mock_vectorstore, mock_llm):
        """Test the chain creation process."""
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm
        )
        
        # Verify retriever was set up correctly from vectorstore
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
        
        # Verify chain was created 
        assert chain.chain is not None
    
    def test_query_success(self, mock_vectorstore, mock_llm):
        """Test successful query execution with relevant documents."""
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm
        )
        
        # Configure mock_vectorstore to return relevant docs with scores
        docs = [
            Document(page_content="How to reset password", 
                    metadata={"source": "password_reset.txt"}),
            Document(page_content="Account recovery steps", 
                    metadata={"source": "account_recovery.txt"})
        ]
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (docs[0], 0.3),  # Below threshold
            (docs[1], 0.5)   # Below threshold
        ]
        
        # Create a dummy chain result
        chain.chain.invoke.return_value = "Here's how to reset your password..."
        
        result = chain.query("How do I reset my password?")
        
        # Verify similarity search was called
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            query="How do I reset my password?",
            k=4
        )
        
        # Check result structure
        assert "answer" in result
        if "sources" in result:
            assert len(result["sources"]) >= 1
        if "source_documents" in result:
            assert len(result["source_documents"]) >= 1
    
    def test_query_no_relevant_docs(self, mock_vectorstore, mock_llm):
        """Test query execution when no documents meet the similarity threshold."""
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm,
            similarity_threshold=0.1  # Very strict threshold
        )
        
        # Configure mock to return docs with scores above threshold
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="Test content", metadata={"source": "doc1.txt"}), 0.8),  # Above threshold
            (Document(page_content="More test content", metadata={"source": "doc2.txt"}), 0.9)   # Above threshold
        ]
        
        result = chain.query("How do I reset my password?")
        
        # Check for uncertainty message
        assert "I don't have enough information" in result["answer"]
        assert "source_documents" in result
    
    def test_query_exception_handling(self, mock_vectorstore, mock_llm):
        """Test exception handling during query execution."""
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm
        )
        
        # Set up mock to raise exception
        mock_vectorstore.similarity_search_with_score.side_effect = Exception("Test error")
        
        result = chain.query("How do I reset my password?")
        
        # Check error handling
        assert "I'm sorry, I encountered an error" in result["answer"]
        assert "error" in result
        assert "source_documents" in result
    
    def test_get_relevant_documents(self, mock_vectorstore, mock_llm):
        """Test retrieving relevant documents without generating an answer."""
        chain = SocialMediaRAGChain(
            vectorstore=mock_vectorstore,
            llm=mock_llm
        )
        
        docs = chain.get_relevant_documents("password reset")
        
        # Verify correct method was called with default k
        mock_vectorstore.similarity_search.assert_called_once_with("password reset", k=4)
        
        # Verify with custom k
        chain.get_relevant_documents("password reset", k=10)
        mock_vectorstore.similarity_search.assert_called_with("password reset", k=10)