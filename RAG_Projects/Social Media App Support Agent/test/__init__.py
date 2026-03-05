"""
Mock implementations for testing the Social Media Support Agent.

This module provides mock implementations of external dependencies to allow
for isolated testing without requiring actual API calls or external libraries.
"""

from unittest.mock import MagicMock
import sys


class MockDocument:
    """Mock implementation of LangChain Document."""
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        
    def copy(self):
        """Return a copy of the document."""
        return MockDocument(self.page_content, self.metadata.copy())


class MockDocumentLoader:
    """Mock implementation of LangChain document loaders."""
    def __init__(self, file_path=None, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
    
    def load(self):
        """Return mock loaded documents."""
        return [MockDocument(
            page_content="This is mock content", 
            metadata={"source": str(self.file_path)}
        )]


class MockTextSplitter:
    """Mock implementation of LangChain text splitter."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def split_documents(self, documents):
        """Return mock split documents."""
        return documents


class MockEmbeddings:
    """Mock implementation of LangChain embeddings."""
    def embed_query(self, text):
        """Return mock embedding for query."""
        return [0.1] * 1536
    
    def embed_documents(self, documents):
        """Return mock embeddings for documents."""
        return [[0.1] * 1536 for _ in documents]


class MockVectorStore:
    """Mock implementation of LangChain vector store."""
    def __init__(self, embedding_function=None, **kwargs):
        self.embedding_function = embedding_function
        self.kwargs = kwargs
        self.index = MagicMock()
        self.docstore = MagicMock()
        self.docstore.index_to_id = {}
    
    @classmethod
    def from_documents(cls, documents, embedding, **kwargs):
        """Create a mock vector store from documents."""
        return cls(embedding_function=embedding, documents=documents, **kwargs)
    
    def as_retriever(self, **kwargs):
        """Return a mock retriever."""
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = [
            MockDocument("Relevant content 1", {"source": "doc1.txt"}),
            MockDocument("Relevant content 2", {"source": "doc2.txt"})
        ]
        return mock_retriever
    
    def similarity_search(self, query, k=4, **kwargs):
        """Return mock similar documents."""
        return [
            MockDocument("Similar content 1", {"source": "doc1.txt"}),
            MockDocument("Similar content 2", {"source": "doc2.txt"})
        ][:k]
    
    def similarity_search_with_score(self, query, k=4, **kwargs):
        """Return mock documents with scores."""
        return [
            (MockDocument("Similar content 1", {"source": "doc1.txt"}), 0.9),
            (MockDocument("Similar content 2", {"source": "doc2.txt"}), 0.8)
        ][:k]


class MockChatOpenAI:
    """Mock implementation of ChatOpenAI."""
    def __init__(self, model=None, temperature=0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        """Mock generation."""
        return "This is a mock response"


class MockPromptTemplate:
    """Mock implementation of PromptTemplate."""
    @classmethod
    def from_template(cls, template):
        """Create a mock prompt template."""
        mock_template = MagicMock()
        mock_template.template = template
        return mock_template


class MockMemory:
    """Mock implementation of ConversationBufferMemory."""
    def __init__(self, **kwargs):
        self.chat_memory = MagicMock()
        self.chat_memory.messages = []
    
    def clear(self):
        """Clear the memory."""
        self.chat_memory.messages = []


class MockLLMChain:
    """Mock implementation of LLMChain."""
    def __init__(self, llm=None, prompt=None, **kwargs):
        self.llm = llm
        self.prompt = prompt
    
    def run(self, inputs):
        """Run the chain."""
        return "Mock standalone question"


class MockQAChain:
    """Mock implementation of QA chain."""
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, inputs):
        """Run the chain."""
        return {"output_text": "This is a mock answer"}


def setup_langchain_mocks():
    """Set up mocks for all langchain modules used in the project."""
    # Create module mocks
    mock_modules = {
        # Document loaders
        'langchain.document_loaders': MagicMock(),
        'langchain.document_loaders.text': MagicMock(),
        'langchain.document_loaders.TextLoader': MockDocumentLoader,
        'langchain.document_loaders.DirectoryLoader': MockDocumentLoader,
        
        # Document store
        'langchain.docstore.document': MagicMock(),
        'langchain.docstore.document.Document': MockDocument,
        
        # Text splitter
        'langchain.text_splitter': MagicMock(),
        'langchain.text_splitter.RecursiveCharacterTextSplitter': MockTextSplitter,
        
        # Vector stores
        'langchain.vectorstores': MagicMock(),
        'langchain.vectorstores.FAISS': MockVectorStore,
        'langchain.vectorstores.base': MagicMock(),
        
        # Embeddings
        'langchain.embeddings.openai': MagicMock(),
        'langchain.embeddings.openai.OpenAIEmbeddings': MockEmbeddings,
        'langchain.embeddings.base': MagicMock(),
        
        # Chat models
        'langchain.chat_models': MagicMock(),
        'langchain.chat_models.ChatOpenAI': MockChatOpenAI,
        
        # LLMs
        'langchain.llms.base': MagicMock(),
        
        # Chains
        'langchain.chains': MagicMock(),
        'langchain.chains.RetrievalQA': MagicMock(),
        'langchain.chains.question_answering': MagicMock(),
        'langchain.chains.question_answering.load_qa_chain': lambda **kwargs: MockQAChain(),
        'langchain.chains.conversational_retrieval.base': MagicMock(),
        'langchain.chains.llm': MagicMock(),
        'langchain.chains.llm.LLMChain': MockLLMChain,
        
        # Prompts
        'langchain.prompts': MagicMock(),
        'langchain.prompts.PromptTemplate': MockPromptTemplate,
        
        # Memory
        'langchain.memory': MagicMock(),
        'langchain.memory.ConversationBufferMemory': MockMemory,
        
        # FAISS
        'faiss': MagicMock(),
        'faiss.write_index': lambda *args: None,
        'faiss.read_index': lambda *args: MagicMock()
    }
    
    # Apply mocks to sys.modules
    for module_name, mock_obj in mock_modules.items():
        if '.' in module_name:
            parent_module = module_name.split('.')[0]
            if parent_module not in sys.modules:
                sys.modules[parent_module] = MagicMock()
        sys.modules[module_name] = mock_obj