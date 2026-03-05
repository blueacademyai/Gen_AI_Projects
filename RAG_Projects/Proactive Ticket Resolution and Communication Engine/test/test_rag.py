import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.rag_chain import RAGProcessor
from pathlib import Path

class TestRAGProcessor:
    
    def test_init(self, metadata_csv, description_csv, mock_vectorstore, mock_openai_api_key):
        """Test RAGProcessor initialization"""
        processor = RAGProcessor(
            metadata_csv, 
            description_csv, 
            mock_vectorstore,  # active_vectorstore
            mock_vectorstore,  # history_vectorstore 
            mock_openai_api_key
        )
        
        assert str(processor.metadata_path) == metadata_csv
        assert str(processor.description_path) == description_csv
        assert processor.active_vectorstore == mock_vectorstore
        assert processor.history_vectorstore == mock_vectorstore

    def test_perform_similarity_search(self, metadata_csv, description_csv, mock_vectorstore, mock_openai_api_key):
        """Test similarity search with all retrieval modes and error handling"""
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        # Test Case 1: Location-based search succeeds
        mock_vectorstore.similarity_search_with_score.return_value = [
            (MagicMock(metadata={'ticket_id': '1', 'location_id': '1001', 'estimated_resolution_time': '24'}), 0.8),
            (MagicMock(metadata={'ticket_id': '2', 'location_id': '1002', 'estimated_resolution_time': '12'}), 0.7),
            (MagicMock(metadata={'ticket_id': '3', 'location_id': '1001', 'estimated_resolution_time': '48'}), 0.6)
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 2  # Only location_id 1001 should be returned
        for doc, score in filtered_docs:
            assert doc.metadata['location_id'] == '1001'
            assert score < 0.99
        mock_vectorstore.similarity_search_with_score.assert_called_with(
            "Test description with sufficient length for validation", k=15
        )
        
        # Reset mock for next test case
        mock_vectorstore.similarity_search_with_score.reset_mock()
        
        # Test Case 2: Location-based search fails, falls back to stricter similarity search
        mock_vectorstore.similarity_search_with_score.side_effect = [
            [],  # No location-based matches
            [  # Stricter similarity matches
                (MagicMock(metadata={'ticket_id': '4', 'location_id': '1003', 'estimated_resolution_time': '36'}), 0.75),
                (MagicMock(metadata={'ticket_id': '5', 'location_id': '1004', 'estimated_resolution_time': '18'}), 0.79)
            ]
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 2
        for doc, score in filtered_docs:
            assert score < 0.8  # Stricter threshold
        assert mock_vectorstore.similarity_search_with_score.call_count == 2
        
        # Reset mock for next test case
        mock_vectorstore.similarity_search_with_score.reset_mock()
        mock_vectorstore.similarity_search_with_score.side_effect = None
        
        # Test Case 3: Both location-based and stricter similarity fail, falls back to broader search
        mock_vectorstore.similarity_search_with_score.side_effect = [
            [],  # No location-based matches
            [],  # No stricter similarity matches
            [  # Broader search matches
                (MagicMock(metadata={'ticket_id': '6', 'location_id': '1005', 'estimated_resolution_time': '24'}), 0.9),
                (MagicMock(metadata={'ticket_id': '7', 'location_id': '1006', 'estimated_resolution_time': '48'}), 0.95)
            ]
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 2
        for doc, score in filtered_docs:
            assert score < 0.99  # Relaxed threshold
        assert mock_vectorstore.similarity_search_with_score.call_count == 3
        assert mock_vectorstore.similarity_search_with_score.call_args_list[2][1]['k'] == 20  # Broader search uses k=20
        
        # Reset mock for next test case
        mock_vectorstore.similarity_search_with_score.reset_mock()
        mock_vectorstore.similarity_search_with_score.side_effect = None
        
        # Test Case 4: No relevant documents found
        mock_vectorstore.similarity_search_with_score.side_effect = [
            [],  # No location-based matches
            [],  # No stricter similarity matches
            []   # No broader search matches
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 0
        assert mock_vectorstore.similarity_search_with_score.call_count == 3

        # Test Case 5: Location-based search raises exception
        mock_vectorstore.similarity_search_with_score.reset_mock()
        mock_vectorstore.similarity_search_with_score.side_effect = [
            Exception("Location search error"),  # Location-based search fails
            [  # Stricter similarity matches
                (MagicMock(metadata={'ticket_id': '8', 'location_id': '1007', 'estimated_resolution_time': '30'}), 0.76),
                (MagicMock(metadata={'ticket_id': '9', 'location_id': '1008', 'estimated_resolution_time': '15'}), 0.78)
            ]
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 2
        for doc, score in filtered_docs:
            assert score < 0.8
        assert mock_vectorstore.similarity_search_with_score.call_count == 2

        # Reset mock for next test case
        mock_vectorstore.similarity_search_with_score.reset_mock()
        mock_vectorstore.similarity_search_with_score.side_effect = None
        
        # Test Case 6: Stricter similarity search raises exception
        mock_vectorstore.similarity_search_with_score.side_effect = [
            [],  # No location-based matches
            Exception("Stricter search error"),  # Stricter similarity search fails
            [  # Broader search matches
                (MagicMock(metadata={'ticket_id': '10', 'location_id': '1009', 'estimated_resolution_time': '20'}), 0.92),
                (MagicMock(metadata={'ticket_id': '11', 'location_id': '1010', 'estimated_resolution_time': '40'}), 0.94)
            ]
        ]
        
        filtered_docs, _ = processor.perform_similarity_search(
            "Test description with sufficient length for validation", mock_vectorstore, 1001
        )
        
        assert len(filtered_docs) == 2
        for doc, score in filtered_docs:
            assert score < 0.99
        assert mock_vectorstore.similarity_search_with_score.call_count == 3
        
        # Reset mock for next test case
        mock_vectorstore.similarity_search_with_score.reset_mock()
        mock_vectorstore.similarity_search_with_score.side_effect = None
    
    
    def test_create_new_ticket(self, metadata_csv, description_csv, mock_vectorstore, mock_openai_api_key):
        """Test creating a new ticket dictionary"""
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        with patch.object(RAGProcessor, 'summarize_description') as mock_summarize:
            mock_summarize.return_value = "Summarized description"
            
            ticket = processor.create_new_ticket(
                ticket_id=101,
                customer_id=201,
                location_id=1001,
                description="Original description that is long enough to pass validation",
                estimated_time=24,
                clusterID=3
            )
            
            assert ticket['TicketID'] == 101
            assert ticket['customerID'] == 201
            assert ticket['locationID'] == 1001
            assert ticket['description'] == "Summarized description"
            assert ticket['estimated_resolution_time'] == 24
            assert ticket['clusterID'] == 3
            assert ticket['type'] == "complaint"
    
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_append_to_csv_file(self, mock_to_csv, mock_read_csv, metadata_csv, description_csv, mock_vectorstore, mock_openai_api_key):
        """Test appending a new ticket to CSV files"""
        mock_metadata_df = pd.DataFrame({
            'TicketID': [1, 2],
            'customerID': [101, 102],   
            'locationID': [1001, 1002],
            'type': ['complaint', 'inquiry'],
            'clusterID': [1, 2],
            'estimated_resolution_time': [24, 12]
        })
        
        mock_description_df = pd.DataFrame({
            'TicketID': [1, 2],
            'description': ['Desc 1', 'Desc 2']
        })
        
        # Set up the mock to return our test DataFrames
        mock_read_csv.side_effect = [mock_metadata_df, mock_description_df]
        
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        new_ticket = {
            'TicketID': 3,
            'customerID': 103,
            'locationID': 1003,
            'type': 'complaint',
            'description': 'New description',
            'clusterID': 1,
            'estimated_resolution_time': 48
        }
        
        processor.append_to_csv_file(new_ticket)
        
        # Verify read_csv was called twice (once for metadata, once for description)
        assert mock_read_csv.call_count == 2
        
        # Verify to_csv was called twice (once for metadata, once for description)
        assert mock_to_csv.call_count == 2
    
    def test_load_active_data(self, metadata_csv, description_csv, mock_vectorstore, mock_openai_api_key):
        """Test loading active data from CSV files"""
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        # Test Case 1: Successful loading of active data
        result = processor.load_active_data()
        
        assert isinstance(result, pd.DataFrame)
        assert 'TicketID' in result.columns
        assert 'description' in result.columns
        
        # Test Case 2: FileNotFoundError for missing metadata CSV
        processor.metadata_path = Path("nonexistent_metadata.csv")
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.load_active_data()
        assert "nonexistent_metadata.csv" in str(exc_info.value)
        
    
    @patch('src.rag_chain.RAGProcessor.load_active_data')
    @patch('src.rag_chain.RAGProcessor.perform_similarity_search')
    @patch('src.rag_chain.RAGProcessor.create_new_ticket')
    def test_get_estimated_resolution_time_active(self, mock_create, mock_search, mock_load,
                                                metadata_csv, description_csv,
                                                mock_vectorstore, mock_openai_api_key):
        """Test estimating resolution time using active tickets"""
        # Mock active data
        mock_load.return_value = pd.DataFrame({
            'TicketID': [1, 2, 3],
            'customerID': [101, 102, 103]
        })

        # Mock active search returning matches
        filtered_docs = [
            (MagicMock(metadata={'ticket_id': '1', 'location_id': '1001', 'estimated_resolution_time': '24'}), 0.8),
            (MagicMock(metadata={'ticket_id': '3', 'location_id': '1001', 'estimated_resolution_time': '48'}), 0.6)
        ]
        # Simulate active search returning matches
        mock_search.return_value = (filtered_docs, None)

        # Mock create_new_ticket return value
        mock_create.return_value = {
            'TicketID': 4,
            'customerID': 104,
            'locationID': 1001,
            'type': 'complaint',
            'description': 'Test description',
            'clusterID': 5,
            'estimated_resolution_time': 36
        }

        # Initialize processor
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )

        # Call the method
        result_type, result_ticket = processor.get_estimated_resolution_time(
            description="Test description with sufficient length for validation",
            location_id=1001
        )

        # Assertions
        assert result_type == 'new_act_ticket'
        assert result_ticket['TicketID'] == 4
        assert result_ticket['estimated_resolution_time'] == 36  # Average of 24 and 48
        assert mock_search.call_args_list[0][0][1] == mock_vectorstore  # Call uses active vectorstore
        assert mock_create.called
    
    @patch('src.rag_chain.RAGProcessor.load_active_data')
    @patch('src.rag_chain.RAGProcessor.perform_similarity_search')
    @patch('src.rag_chain.RAGProcessor.create_new_ticket')
    def test_get_estimated_resolution_time_history(self, mock_create, mock_search, mock_load, 
                                                  metadata_csv, description_csv, 
                                                  mock_vectorstore, mock_openai_api_key):
        """Test estimating resolution time using historical tickets when no active matches"""
        mock_load.return_value = pd.DataFrame({
            'TicketID': [1, 2, 3],
            'customerID': [101, 102, 103]
        })
        
        history_docs = [
            (MagicMock(metadata={'ticket_id': '101', 'location_id': '1001', 'estimated_resolution_time': '36'}), 0.7),
        ]
        
        # No active matches, but history matches
        mock_search.side_effect = [([], None), (history_docs, None)]
        
        mock_create.return_value = {'TicketID': 4, 'type': 'history_based'}
        
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        result_type, result_ticket = processor.get_estimated_resolution_time(
            description="Test description with sufficient length for validation",
            location_id=1001
        )
        
        assert result_type == 'new_hs_ticket'
        assert mock_search.call_count == 2  # Called for both active and history
        assert mock_create.called
    
    


    @pytest.fixture
    def rag_processor(self, tmp_path):
        """Fixture to initialize RAGProcessor with dummy paths and vector stores."""
        metadata_path = tmp_path / "metadata.csv"
        description_path = tmp_path / "descriptions.csv"
        # Create empty CSV files to avoid FileNotFoundError
        metadata_path.write_text("TicketID,customerID,locationID\n1,1,1")
        description_path.write_text("TicketID,description\n1,test")
        return RAGProcessor(
            metadata_path=str(metadata_path),
            description_path=str(description_path),
            active_vectorstore=None,
            history_vectorstore=None,
            openai_api_key="dummy_key"  # Dummy key, as summarization is not tested
        )

    def test_validate_query(self, rag_processor):
        """
        Test exception handling in _validate_query method.
        Verifies ValueError is raised for empty/None queries and queries shorter than 20 characters.
        """
        # Test case 1: Empty query
        with pytest.raises(ValueError) as exc_info:
            rag_processor._validate_query("")
        assert str(exc_info.value) == "Query cannot be empty", \
            "Expected ValueError with correct message for empty query"

        # Test case 2: None query
        with pytest.raises(ValueError) as exc_info:
            rag_processor._validate_query(None)
        assert str(exc_info.value) == "Query cannot be empty", \
            "Expected ValueError with correct message for None query"

        # Test case 3: Whitespace-only query
        with pytest.raises(ValueError) as exc_info:
            rag_processor._validate_query("   ")
        assert str(exc_info.value) == "Query cannot be empty", \
            "Expected ValueError with correct message for whitespace-only query"

        # Test case 4: Too short query
        with pytest.raises(ValueError) as exc_info:
            rag_processor._validate_query("Short query")
        assert str(exc_info.value) == "Query too short. Please provide more details.", \
            "Expected ValueError with correct message for query shorter than 20 characters"

        # Test case 5: Valid query (no exception)
        try:
            rag_processor._validate_query("This is a valid query with sufficient length")
        except ValueError:
            pytest.fail("Valid query should not raise ValueError")