import pytest
import os
import pickle
import warnings
from unittest.mock import patch, MagicMock
from src.vector_store import VectorStoreManager
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from src.vector_store import VectorStoreManager

@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress specific warnings during tests."""
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.vectorstores.faiss")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield


@pytest.fixture
def mock_documents():
    """Fixture providing sample documents for testing"""
    incident_docs = [
        Document(
            page_content="Network connectivity issue with Router XYZ",
            metadata={"ProductID": "P001", "CustomerID": "C001"}
        ),
        Document(
            page_content="Slow connection on Switch ABC",
            metadata={"ProductID": "P002", "CustomerID": "C002"}
        )
    ]

    tech_docs = [
        Document(
            page_content="Router XYZ troubleshooting steps",
            metadata={"ProductID": "P001", "ProductInformation": "Router XYZ"}
        ),
        Document(
            page_content="Switch ABC configuration guide",
            metadata={"ProductID": "P002", "ProductInformation": "Switch ABC"}
        )
    ]

    return incident_docs, tech_docs


@pytest.fixture
def mock_embeddings():
    """Fixture providing a mocked OpenAIEmbeddings object"""
    embeddings = MagicMock()
    return embeddings


@pytest.fixture
def mock_faiss():
    """Fixture providing a mocked FAISS vectorstore"""
    faiss = MagicMock()
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = [
        Document(
            page_content="Network troubleshooting guide",
            metadata={"ProductID": "P001"}
        )
    ]
    faiss.as_retriever.return_value = retriever
    return faiss


@pytest.fixture
def mock_bm25():
    """Fixture providing a mocked BM25Retriever object"""
    bm25 = MagicMock()
    bm25.k = 5
    bm25.get_relevant_documents.return_value = [
        Document(
            page_content="Network connectivity troubleshooting",
            metadata={"ProductID": "P001"}
        )
    ]
    return bm25


@pytest.fixture
def mock_ensemble():
    """Fixture providing a mocked EnsembleRetriever object"""
    ensemble = MagicMock()
    ensemble.get_relevant_documents.return_value = [
        Document(
            page_content="Router XYZ troubleshooting guide",
            metadata={"ProductID": "P001"}
        ),
        Document(
            page_content="Switch ABC configuration steps",
            metadata={"ProductID": "P002"}
        )
    ]
    return ensemble

def test_vectorstore_manager_init(mock_documents, mock_embeddings):
    """Test basic initialization of VectorStoreManager."""
    incident_docs, tech_docs = mock_documents
    
    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                    manager = VectorStoreManager(
                        incident_docs=incident_docs,
                        tech_docs=tech_docs,
                        top_k=3
                    )
                    
                    assert manager.k == 3
                    assert manager.incident_docs == incident_docs
                    assert manager.tech_docs == tech_docs
                    assert manager.incident_faiss_path == "index/incident_faiss"
                    assert manager.tech_faiss_path == "index/tech_faiss"


def test_vectorstore_initialization(mock_documents, mock_embeddings):
    """Test the initialization of VectorStoreManager"""
    incident_docs, tech_docs = mock_documents

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                    manager = VectorStoreManager(
                        incident_docs=incident_docs,
                        tech_docs=tech_docs
                    )

                    assert manager.incident_docs == incident_docs
                    assert manager.tech_docs == tech_docs
                    assert manager.k == 5 

@pytest.fixture
def mock_documents():
    """Fixture providing mock incident and tech documents."""
    incident_docs = [Document(page_content="Incident test", metadata={"ProductID": "P001"})]
    tech_docs = [Document(page_content="Technical test", metadata={"ProductID": "P002"})]
    return incident_docs, tech_docs

@pytest.fixture
def mock_embeddings():
    """Fixture providing mock OpenAIEmbeddings."""
    return MagicMock()

@pytest.fixture
def mock_faiss():
    """Fixture providing mock FAISS vector store."""
    mock = MagicMock(spec=FAISS)
    mock.as_retriever.return_value = MagicMock() 
    return mock

@pytest.fixture
def mock_bm25():
    """Fixture providing mock BM25 retriever."""
    return MagicMock(spec=BM25Retriever)

def test_initialize_vector_stores_existing_indexes(mock_documents, mock_embeddings, mock_faiss, mock_bm25):
    """
    Test _initialize_vector_stores when both FAISS and BM25 indexes exist for incident and tech documents.
    
    Verifies that:
    1. Existing FAISS and BM25 indexes are loaded correctly
    2. Ensemble retrievers are created for both incident and tech stores
    3. The loaded vector stores and retrievers are properly assigned to class attributes
    4. The correct methods are called with expected frequency
    """
    incident_docs, tech_docs = mock_documents

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch("os.path.exists", side_effect=[True, True, True, True]):
                    with patch.object(VectorStoreManager, "_load_faiss_vectorstore", return_value=mock_faiss) as mock_load_faiss:
                        with patch.object(VectorStoreManager, "_load_bm25", return_value=mock_bm25) as mock_load_bm25:
                            with patch.object(VectorStoreManager, "_create_ensemble") as mock_create_ensemble:
                                mock_ensemble = MagicMock(spec=EnsembleRetriever)
                                mock_create_ensemble.return_value = mock_ensemble

                                manager = VectorStoreManager(
                                    incident_docs=incident_docs,
                                    tech_docs=tech_docs
                                )

                                manager._initialize_vector_stores()

                                assert mock_load_faiss.call_count == 2, "Should load FAISS for both incident and tech stores"
                                assert mock_load_bm25.call_count == 2, "Should load BM25 for both incident and tech stores"
                                assert mock_create_ensemble.call_count == 2, "Should create ensemble retrievers for both stores"

                                assert manager.incident_faiss_vectorstore == mock_faiss, "Incident FAISS vectorstore should be set"
                                assert manager.tech_faiss_vectorstore == mock_faiss, "Tech FAISS vectorstore should be set"
                                assert manager.incident_ensemble_retriever == mock_ensemble, "Incident ensemble retriever should be set"
                                assert manager.tech_ensemble_retriever == mock_ensemble, "Tech ensemble retriever should be set"

                                mock_create_ensemble.assert_any_call(
                                    mock_faiss.as_retriever.return_value, mock_bm25
                                )

def test_create_incident_store(mock_documents, mock_embeddings, mock_faiss):
    """Test the _create_incident_store function"""
    incident_docs, _ = mock_documents

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch("os.makedirs"):
                    with patch("src.vector_store.FAISS.from_documents", return_value=mock_faiss):
                        with patch("src.vector_store.BM25Retriever.from_documents") as mock_bm25_cls:
                            with patch("builtins.open", MagicMock()):
                                with patch("pickle.dump"):
                                    with patch.object(VectorStoreManager, "_create_ensemble"):
                                        mock_bm25 = MagicMock()
                                        mock_bm25_cls.return_value = mock_bm25

                                        with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                                            manager = VectorStoreManager(
                                                incident_docs=incident_docs,
                                                tech_docs=[]
                                            )

                                            manager._create_incident_store()

                                            mock_faiss.save_local.assert_called_once()
                                            mock_bm25_cls.assert_called_once_with(incident_docs)
                                            assert mock_bm25.k == manager.k

def test_retrieve_documents(mock_documents, mock_embeddings, mock_ensemble):
    """Test the retrieve_documents function"""
    incident_docs, tech_docs = mock_documents

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                    manager = VectorStoreManager(
                        incident_docs=incident_docs,
                        tech_docs=tech_docs
                    )

                    manager.incident_ensemble_retriever = mock_ensemble
                    manager.tech_ensemble_retriever = mock_ensemble

                    mock_docs = [
                        Document(
                            page_content="Router troubleshooting",
                            metadata={"ProductID": "P001"}
                        ),
                        Document(
                            page_content="Switch configuration",
                            metadata={"ProductID": "P002"}
                        )
                    ]
                    mock_ensemble.get_relevant_documents.return_value = mock_docs

                    result = manager.retrieve_documents(
                        query="network connectivity issue",
                        product_id="P001",
                        store_type="incident"
                    )

                    mock_ensemble.get_relevant_documents.assert_called_once_with("network connectivity issue")

                    assert len(result) == 1
                    assert result[0].metadata["ProductID"] == "P001"
                    
                    mock_ensemble.reset_mock()
                    mock_ensemble.get_relevant_documents.return_value = [
                        Document(
                            page_content="Router troubleshooting",
                            metadata={"ProductID": "P003"}
                        ),
                        Document(
                            page_content="Switch configuration",
                            metadata={"ProductID": "P004"}
                        )
                    ]
                    
                    with patch("builtins.print") as mock_print:
                        result = manager.retrieve_documents(
                            query="network connectivity issue",
                            product_id="P001",
                            store_type="incident"
                        )
                    
                        mock_ensemble.get_relevant_documents.assert_called_once_with("network connectivity issue")
                        
                        assert 'No documents found matching product ID: P001' in result

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                    manager = VectorStoreManager(
                        incident_docs=incident_docs,
                        tech_docs=tech_docs
                    )

                    mock_error_ensemble = MagicMock()
                    mock_error_ensemble.get_relevant_documents.side_effect = Exception("Retrieval error")
                    manager.incident_ensemble_retriever = mock_error_ensemble

                    result = manager.retrieve_documents("query", "P001", "incident")
                    assert isinstance(result, Exception)
                    assert "Retrieval error" in str(result)



def test_create_tech_store(mock_documents, mock_embeddings, mock_faiss):
    """Test the _create_tech_store function"""
    _, tech_docs = mock_documents

    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch("os.makedirs"):
                    with patch("src.vector_store.FAISS.from_documents", return_value=mock_faiss):
                        with patch("src.vector_store.BM25Retriever.from_documents") as mock_bm25_cls:
                            with patch("builtins.open", MagicMock()):
                                with patch("pickle.dump"):
                                    with patch.object(VectorStoreManager, "_create_ensemble"):
                                        mock_bm25 = MagicMock()
                                        mock_bm25_cls.return_value = mock_bm25

                                        with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                                            manager = VectorStoreManager(
                                                incident_docs=[],
                                                tech_docs=tech_docs
                                            )

                                            manager._create_tech_store()

                                            mock_faiss.save_local.assert_called_once()
                                            mock_bm25_cls.assert_called_once_with(tech_docs)
                                            assert mock_bm25.k == manager.k


def test_create_ensemble():
    """Test the _create_ensemble function"""
    mock_faiss_retriever = MagicMock()
    mock_bm25_retriever = MagicMock()
    
    with patch("src.vector_store.EnsembleRetriever") as mock_ensemble_cls:
        mock_ensemble = MagicMock()
        mock_ensemble_cls.return_value = mock_ensemble
        
        with patch("src.vector_store.OpenAIEmbeddings"):
            with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
                with patch("src.vector_store.load_dotenv"):
                    with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                        manager = VectorStoreManager(incident_docs=[], tech_docs=[])
                        
                        result = manager._create_ensemble(mock_faiss_retriever, mock_bm25_retriever)
                        
                        mock_ensemble_cls.assert_called_once_with(
                            retrievers=[mock_faiss_retriever, mock_bm25_retriever],
                            weights=[0.8, 0.2]
                        )
                        assert result == mock_ensemble
                        
                        mock_ensemble_cls.reset_mock()
                        
                        custom_weights = [0.6, 0.4]
                        result = manager._create_ensemble(
                            mock_faiss_retriever, 
                            mock_bm25_retriever,
                            weights=custom_weights
                        )
                        
                        mock_ensemble_cls.assert_called_once_with(
                            retrievers=[mock_faiss_retriever, mock_bm25_retriever],
                            weights=custom_weights
                        )


def test_load_faiss_vectorstore(mock_embeddings):
    """Test the _load_faiss_vectorstore function"""
    with patch("src.vector_store.FAISS.load_local") as mock_load_local:
        mock_vectorstore = MagicMock()
        mock_load_local.return_value = mock_vectorstore
        
        with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
            with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
                with patch("src.vector_store.load_dotenv"):
                    with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                        manager = VectorStoreManager(incident_docs=[], tech_docs=[])
                        
                        result = manager._load_faiss_vectorstore("test/path")
                        
                        mock_load_local.assert_called_once_with(
                            "test/path",
                            mock_embeddings,
                            allow_dangerous_deserialization=True
                        )
                        assert result == mock_vectorstore
                        
                        mock_load_local.side_effect = Exception("FAISS loading error")
                        with pytest.raises(Exception) as excinfo:
                            manager._load_faiss_vectorstore("test/path")
                        assert "FAISS loading error" in str(excinfo.value)


def test_retrieve_documents_validation(mock_documents, mock_embeddings):
    """Test input validation and store selection in retrieve_documents method."""
    incident_docs, tech_docs = mock_documents
    
    with patch("src.vector_store.OpenAIEmbeddings", return_value=mock_embeddings):
        with patch("src.vector_store.os.getenv", return_value="mock_api_key"):
            with patch("src.vector_store.load_dotenv"):
                with patch.object(VectorStoreManager, "_initialize_vector_stores"):
                    manager = VectorStoreManager(
                        incident_docs=incident_docs,
                        tech_docs=tech_docs
                    )
                    
                    incident_retriever = MagicMock()
                    tech_retriever = MagicMock()
                    manager.incident_ensemble_retriever = incident_retriever
                    manager.tech_ensemble_retriever = tech_retriever
                    
                    incident_docs = [
                        Document(
                            page_content="Incident document 1",
                            metadata={"ProductID": "P001"}
                        ),
                        Document(
                            page_content="Incident document 2",
                            metadata={"ProductID": "P002"}
                        )
                    ]
                    
                    tech_docs = [
                        Document(
                            page_content="Tech document 1",
                            metadata={"ProductID": "P001"}
                        ),
                        Document(
                            page_content="Tech document 2",
                            metadata={"ProductID": "P003"}
                        )
                    ]
                    
                    incident_retriever.get_relevant_documents.return_value = incident_docs
                    tech_retriever.get_relevant_documents.return_value = tech_docs
                    
                    manager.retrieve_documents("test query", "P001", "incident")
                    incident_retriever.get_relevant_documents.assert_called_once_with("test query")
                    tech_retriever.get_relevant_documents.assert_not_called()
                    
                    incident_retriever.reset_mock()
                    tech_retriever.reset_mock()
                    
                    manager.retrieve_documents("test query", "P001", "tech")
                    tech_retriever.get_relevant_documents.assert_called_once_with("test query")
                    incident_retriever.get_relevant_documents.assert_not_called()
                    
                    incident_retriever.reset_mock()
                    tech_retriever.reset_mock()
                    
                    result = manager.retrieve_documents("test query", "P001", "incident")
                    incident_retriever.get_relevant_documents.assert_called_once()
                    assert len(result) == 1
                    assert result[0].metadata["ProductID"] == "P001"
                    
                    incident_retriever.reset_mock()
                    incident_retriever.get_relevant_documents.return_value = [
                        Document(
                            page_content="Incident document 3",
                            metadata={"ProductID": "P005"}
                        ),
                        Document(
                            page_content="Incident document 4",
                            metadata={"ProductID": "P006"}
                        )
                    ]
                    
                    with patch("builtins.print") as mock_print:
                        result = manager.retrieve_documents("test query", "P001", "incident")
                        
                        incident_retriever.get_relevant_documents.assert_called_once()
                        assert isinstance(result, str)
                       
                    result = manager.retrieve_documents("", "P001", "incident")
                    assert isinstance(result, ValueError)
                    assert "non-empty string" in str(result)
                    
                    result = manager.retrieve_documents("test query", "P001", "invalid_store")
                    assert isinstance(result, ValueError)
                    assert "must be 'incident' or 'tech'" in str(result)