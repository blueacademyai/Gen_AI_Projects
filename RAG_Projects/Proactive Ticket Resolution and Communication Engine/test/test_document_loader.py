import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src import document_loader
from src.document_loader import DocumentLoader, CombinedDocumentLoader
from langchain_core.documents import Document

class TestDocumentLoader:
    
    def test_init(self, metadata_csv, description_csv, mock_openai_api_key):
        """Test DocumentLoader initialization"""
        loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
        assert str(loader.metadata_path) == metadata_csv
        assert str(loader.description_path) == description_csv
        assert loader.summarization_chain is not None
    
    def test_load_csv_file(self, metadata_csv, mock_openai_api_key):
        """Test loading a CSV file"""
        loader = DocumentLoader(metadata_csv, metadata_csv, mock_openai_api_key)
        df = loader.load_csv_file(loader.metadata_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Based on our test fixture
        assert 'TicketID' in df.columns
    
    
    @patch('src.document_loader.DocumentLoader.summarize_description')
    def test_prepare_documents(self, mock_summarize, metadata_csv, description_csv, mock_openai_api_key):
        """Test preparing Document objects from DataFrame rows"""
        mock_summarize.return_value = "Test summary"
        
        loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
        data = loader.load_csv_data()
        documents = loader.prepare_documents(data)
        
        assert len(documents) == 3
        assert documents[0].page_content == "Test summary"
        assert documents[0].metadata['ticket_id'] == "1"
        assert documents[0].metadata['location_id'] == "1001"
        assert documents[0].metadata['estimated_resolution_time'] == "24"
    
    def test_summarize_description_short_and_empty(self, metadata_csv, description_csv, mock_openai_api_key):
        """Test that short and empty descriptions aren't summarized"""
        doc_loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
        with patch.object(doc_loader, 'summarization_chain') as mock_chain:
            mock_chain.invoke.return_value = "Mocked summary"
            
            # Test short description
            short_description = "This is a short description."
            result = doc_loader.summarize_description(short_description)
            assert result == short_description
            mock_chain.invoke.assert_not_called()
            
            # Test empty string
            empty_description = ""
            result = doc_loader.summarize_description(empty_description)
            assert result == ""
            mock_chain.invoke.assert_not_called()
            
            # Test whitespace-only string
            whitespace_description = "   "
            result = doc_loader.summarize_description(whitespace_description)
            assert result == ""
            mock_chain.invoke.assert_not_called()
            
            # Test non-string input
            non_string_description = None
            result = doc_loader.summarize_description(non_string_description)
            assert result == ""
            mock_chain.invoke.assert_not_called()
    
    def test_summarize_description_long(self, metadata_csv, description_csv, mock_openai_api_key):
        """Test summarization of long descriptions when summarization succeeds"""
        doc_loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
        long_description = "A" * 2000  # Exceeds 1500 character limit
        expected_summary = "Summarized text"
        
        with patch.object(doc_loader, 'summarization_chain') as mock_chain:
            mock_chain.invoke.return_value = expected_summary
            result = doc_loader.summarize_description(long_description)
            assert result == expected_summary
            mock_chain.invoke.assert_called_once_with(long_description)

class TestCombinedDocumentLoader:
    
    @patch('src.document_loader.DocumentLoader')
    def test_init(self, mock_doc_loader, mock_openai_api_key):
        """Test CombinedDocumentLoader initialization"""
        combined_loader = CombinedDocumentLoader(
            "active_meta.csv", "active_desc.csv", 
            "historic_meta.csv", "historic_desc.csv",
            mock_openai_api_key
        )
        
        assert mock_doc_loader.call_count == 2
        assert combined_loader.active_loader is not None
        assert combined_loader.historic_loader is not None
    
    @patch('src.document_loader.DocumentLoader')
    def test_load_active_documents(self, mock_doc_loader, mock_openai_api_key):
        """Test loading active documents"""
        mock_instance = MagicMock()
        mock_instance.load_documents.return_value = ["doc1", "doc2"]
        mock_doc_loader.return_value = mock_instance
        
        combined_loader = CombinedDocumentLoader(
            "active_meta.csv", "active_desc.csv", 
            "historic_meta.csv", "historic_desc.csv",
            mock_openai_api_key
        )
        
        result = combined_loader.load_active_documents()
        
        assert combined_loader.active_loader.load_documents.called
        assert result == ["doc1", "doc2"]
    

    def test_load_documents_functionality(self, tmp_path):
        """
        Test the functionality of the load_documents method in DocumentLoader.
        Verifies that the method loads and processes CSV data into a list of Document objects.
        """
        # Create temporary CSV files for testing
        metadata_file = tmp_path / "metadata.csv"
        description_file = tmp_path / "descriptions.csv"

        # Sample metadata CSV content
        metadata_content = """TicketID,locationID,estimated_resolution_time
            1,LOC001,2
            2,LOC002,4"""
                
        # Sample description CSV content
        description_content = """TicketID,description
            1,This is a test description for ticket 1
            2,This is a longer description for ticket 2 that needs summarization"""

        # Write content to temporary CSV files
        metadata_file.write_text(metadata_content)
        description_file.write_text(description_content)

        # Initialize DocumentLoader with test files and a dummy API key
        loader = DocumentLoader(
            metadata_path=str(metadata_file),
            description_path=str(description_file),
            openai_api_key="dummy_key"  # Replace with a valid key for actual testing
        )

        # Call the load_documents method
        documents = loader.load_documents()

        # Assertions to verify functionality
        assert isinstance(documents, list), "Output should be a list"
        assert len(documents) == 2, "Should return exactly 2 Document objects"
        assert all(isinstance(doc, Document) for doc in documents), "All items should be Document objects"
        
        # Check first document
        assert documents[0].page_content == "This is a test description for ticket 1", "Description should match original (no summarization needed)"
        assert documents[0].metadata == {
            'ticket_id': '1',
            'location_id': 'LOC001',
            'estimated_resolution_time': '2'
        }, "Metadata for first document should match input"

        # Check second document (no summarization due to short length, assuming <1500 chars)
        assert documents[1].page_content == "This is a longer description for ticket 2 that needs summarization", "Description should match original (no summarization needed)"
        assert documents[1].metadata == {
            'ticket_id': '2',
            'location_id': 'LOC002',
            'estimated_resolution_time': '4'
        }, "Metadata for second document should match input"

    @pytest.fixture
    def setup_csv_files(self, tmp_path):
        """
        Fixture to create temporary CSV files for testing.
        Returns paths to metadata and description files for both active and historic data.
        """
        # Active ticket files
        active_metadata_file = tmp_path / "active_metadata.csv"
        active_description_file = tmp_path / "active_descriptions.csv"
        
        # Historic ticket files
        historic_metadata_file = tmp_path / "historic_metadata.csv"
        historic_description_file = tmp_path / "historic_descriptions.csv"

        # Sample metadata CSV content
        metadata_content = """TicketID,locationID,estimated_resolution_time
            1,LOC001,2"""
        
        # Sample description CSV content
        description_content = """TicketID,description
            1,Test description for ticket 1"""

        # Write content to temporary CSV files
        active_metadata_file.write_text(metadata_content)
        active_description_file.write_text(description_content)
        historic_metadata_file.write_text(metadata_content)
        historic_description_file.write_text(description_content)

        return {
            "active_metadata": active_metadata_file,
            "active_description": active_description_file,
            "historic_metadata": historic_metadata_file,
            "historic_description": historic_description_file
        }

    def test_load_historic_documents_functionality(self, setup_csv_files):
        """
        Test the functionality of the load_historic_documents method in CombinedDocumentLoader.
        Verifies that the method loads and returns historic ticket documents correctly.
        """
        # Initialize CombinedDocumentLoader with test files and a dummy API key
        loader = CombinedDocumentLoader(
            active_metadata_path=str(setup_csv_files["active_metadata"]),
            active_description_path=str(setup_csv_files["active_description"]),
            historic_metadata_path=str(setup_csv_files["historic_metadata"]),
            historic_description_path=str(setup_csv_files["historic_description"]),
            openai_api_key="dummy_key"  # Replace with a valid key if summarization is needed
        )

        # Call the load_historic_documents method
        documents = loader.load_historic_documents()

        # Assertions to verify functionality
        assert isinstance(documents, list), "Output should be a list"
        assert len(documents) == 1, "Should return exactly 1 Document object"
        assert all(isinstance(doc, Document) for doc in documents), "All items should be Document objects"
        
        # Check document content and metadata
        assert documents[0].page_content == "Test description for ticket 1", "Description should match original"
        assert documents[0].metadata == {
            'ticket_id': '1',
            'location_id': 'LOC001',
            'estimated_resolution_time': '2'
        }, "Metadata should match input"

    def test_load_all_documents_functionality(self, setup_csv_files):
        """
        Test the functionality of the load_all_documents method in CombinedDocumentLoader.
        Verifies that the method loads and returns both active and historic ticket documents correctly.
        """
        # Initialize CombinedDocumentLoader with test files and a dummy API key
        loader = CombinedDocumentLoader(
            active_metadata_path=str(setup_csv_files["active_metadata"]),
            active_description_path=str(setup_csv_files["active_description"]),
            historic_metadata_path=str(setup_csv_files["historic_metadata"]),
            historic_description_path=str(setup_csv_files["historic_description"]),
            openai_api_key="dummy_key"  # Replace with a valid key if summarization is needed
        )

        # Call the load_all_documents method
        active_docs, historic_docs = loader.load_all_documents()

        # Assertions for active documents
        assert isinstance(active_docs, list), "Active documents should be a list"
        assert len(active_docs) == 1, "Should return exactly 1 active Document object"
        assert all(isinstance(doc, Document) for doc in active_docs), "All active items should be Document objects"
        assert active_docs[0].page_content == "Test description for ticket 1", "Active description should match original"
        assert active_docs[0].metadata == {
            'ticket_id': '1',
            'location_id': 'LOC001',
            'estimated_resolution_time': '2'
        }, "Metadata for active document should match input"

        # Assertions for historic documents
        assert isinstance(historic_docs, list), "Historic documents should be a list"
        assert len(historic_docs) == 1, "Should return exactly 1 historic Document object"
        assert all(isinstance(doc, Document) for doc in historic_docs), "All historic items should be Document objects"
        assert historic_docs[0].page_content == "Test description for ticket 1", "Historic description should match original"
        assert historic_docs[0].metadata == {
            'ticket_id': '1',
            'location_id': 'LOC001',
            'estimated_resolution_time': '2'
        }, "Metadata for historic document should match input"