import os
import sys
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


# Create mock classes for OpenAI components to avoid API key requirements
class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    
    def __or__(self, other):
        return "This is a mocked response"


class MockEmbeddings:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


# Use patch to replace actual implementations
patch("langchain_openai.OpenAIEmbeddings", MockEmbeddings).start()
patch("langchain_openai.ChatOpenAI", MockOpenAI).start()
patch("src.vector_store.OpenAIEmbeddings", MockEmbeddings).start()
patch("src.rag_chain.ChatOpenAI", MockOpenAI).start()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_documents():
    """Return sample Document objects for testing."""
    return [
        Document(
            page_content="How to reset your password on X: Go to Settings, select Security, choose Password, and follow the prompts.",
            metadata={"source": "password_reset.txt"}
        ),
        Document(
            page_content="Creating a new account on X requires a valid email address or phone number. Click Sign Up and follow the instructions.",
            metadata={"source": "account_creation.txt"}
        ),
        Document(
            page_content="To delete your account, go to Settings > Your Account > Deactivate your account. Note this action has a 30-day recovery window.",
            metadata={"source": "account_deletion.txt"}
        )
    ]