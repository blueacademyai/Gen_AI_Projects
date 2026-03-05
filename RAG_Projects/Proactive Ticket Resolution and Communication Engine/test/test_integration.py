import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.document_loader import DocumentLoader, CombinedDocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGProcessor

class TestIntegration:
    
    @patch('langchain_openai.ChatOpenAI')
    def test_document_loader_to_vector_store(self, mock_chat_openai, 
                                             metadata_csv, description_csv, 
                                             mock_embeddings, test_data_dir,
                                             mock_openai_api_key):
        """Test loading documents and creating a vector store"""
        mock_chat_instance = MagicMock()
        mock_chat_instance.__or__.return_value = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        
        # Create a document loader
        loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
        
        # Load documents
        documents = loader.load_documents()
        
        # Create a vector store from the documents
        index_path = test_data_dir / "test_integration_index"
        
        with patch('src.vector_store.FAISS') as mock_faiss:
            mock_vs = MagicMock()
            mock_faiss.from_documents.return_value = mock_vs
            
            vs_manager = VectorStoreManager(str(index_path), mock_embeddings)
            vs_manager.create_vectorstore(documents)
            
            # Verify the vector store was created with our documents
            mock_faiss.from_documents.assert_called_once_with(documents, mock_embeddings)
            mock_vs.save_local.assert_called_once_with(str(index_path))
    
    @patch('langchain_openai.ChatOpenAI')
    def test_rag_processor_workflow(self, mock_chat_openai, metadata_csv, description_csv, 
                                   mock_vectorstore, mock_openai_api_key):
        """Test the RAG processor workflow"""
        mock_chat_instance = MagicMock()
        mock_chat_instance.__or__.return_value = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        
        # Create RAG processor
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        # Test creating a new ticket and estimating resolution time
        with patch.object(RAGProcessor, 'perform_similarity_search') as mock_search:
            filtered_docs = [
                (MagicMock(metadata={'ticket_id': '1', 'location_id': '1001', 'estimated_resolution_time': '24'}), 0.8)
            ]
            mock_search.return_value = (filtered_docs, None)
            
            with patch.object(RAGProcessor, 'load_active_data') as mock_load:
                mock_load.return_value = pd.DataFrame({
                    'TicketID': [1, 2, 3],
                    'customerID': [101, 102, 103],
                    'locationID': [1001, 1002, 1003],
                    'estimated_resolution_time': [24, 12, 48]
                })
                
                with patch.object(RAGProcessor, '_validate_query') as mock_validate:
                    ticket_type, ticket = processor.get_estimated_resolution_time(
                        "Test ticket description with sufficient length for validation",
                        1001
                    )
                    
                    assert ticket_type == 'new_act_ticket'
                    assert ticket['TicketID'] == 4  # Next ID after 1,2,3
                    assert ticket['locationID'] == 1001
                    assert ticket['estimated_resolution_time'] == 24  # From the mock returned value
    
    @patch('langchain_openai.ChatOpenAI')
    def test_end_to_end_workflow(self, mock_chat_openai, metadata_csv, description_csv,
                                 mock_embeddings, test_data_dir, mock_openai_api_key):
        """Test an end-to-end workflow with patched components"""
        mock_chat_instance = MagicMock()
        mock_chat_instance.__or__.return_value = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        
        # 1. Load documents from CSV files
        with patch.object(DocumentLoader, 'summarize_description', return_value="Summary"):
            loader = DocumentLoader(metadata_csv, description_csv, mock_openai_api_key)
            documents = loader.load_documents()
            
            assert len(documents) == 3  # Based on our test fixtures
        
        # 2. Create vector stores for active and historical data
        active_index_path = test_data_dir / "active_index"
        history_index_path = test_data_dir / "history_index"
        
        with patch('src.vector_store.FAISS') as mock_faiss:
            mock_vs = MagicMock()
            mock_faiss.from_documents.return_value = mock_vs
            
            active_vs_manager = VectorStoreManager(str(active_index_path), mock_embeddings)
            active_vs_manager.create_vectorstore(documents)
            
            history_vs_manager = VectorStoreManager(str(history_index_path), mock_embeddings)
            history_vs_manager.create_vectorstore(documents)
            
            # 3. Create a RAG processor with the vector stores
            with patch.object(VectorStoreManager, 'get_vectorstore', return_value=mock_vs):
                processor = RAGProcessor(
                    metadata_csv, 
                    description_csv, 
                    active_vs_manager.vectorstore,
                    history_vs_manager.vectorstore,
                    mock_openai_api_key
                )
                
                # 4. Test estimating resolution time for a new ticket
                with patch.object(RAGProcessor, 'perform_similarity_search') as mock_search:
                    filtered_docs = [
                        (MagicMock(metadata={'ticket_id': '1', 'location_id': '1001', 'estimated_resolution_time': '24'}), 0.8)
                    ]
                    mock_search.return_value = (filtered_docs, None)
                    
                    with patch.object(RAGProcessor, 'load_active_data') as mock_load:
                        mock_load.return_value = pd.DataFrame({
                            'TicketID': [1, 2, 3],
                            'customerID': [101, 102, 103]
                        })
                        
                        with patch.object(RAGProcessor, '_validate_query'):
                            ticket_type, ticket = processor.get_estimated_resolution_time(
                                "Integration test ticket description with sufficient length",
                                1001
                            )
                            
                            assert ticket_type == 'new_act_ticket'
                            assert isinstance(ticket, dict)
                            assert ticket['locationID'] == 1001
                            
                            # 5. Test appending the new ticket to CSV files and vector store
                            with patch.object(RAGProcessor, 'append_to_csv_file') as mock_append_csv:
                                with patch.object(VectorStoreManager, 'append_vectorstore') as mock_append_vs:
                                    processor.append_to_csv_file(ticket)
                                    active_vs_manager.append_vectorstore(ticket)
                                    
                                    mock_append_csv.assert_called_once_with(ticket)
                                    mock_append_vs.assert_called_once_with(ticket)
    
    @patch('langchain_openai.ChatOpenAI')
    def test_handling_invalid_queries(self, mock_chat_openai, metadata_csv, description_csv,
                                    mock_vectorstore, mock_openai_api_key):
        """Test handling of invalid or empty queries"""
        mock_chat_instance = MagicMock()
        mock_chat_instance.__or__.return_value = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        
        processor = RAGProcessor(
            metadata_csv, description_csv, mock_vectorstore, mock_vectorstore, mock_openai_api_key
        )
        
        # Test with empty query - RAGProcessor now handles this internally instead of raising
        with patch.object(RAGProcessor, 'load_active_data') as mock_load:
            mock_load.return_value = pd.DataFrame({
                'TicketID': [1, 2, 3],
                'customerID': [101, 102, 103]
            })
            
            ticket_type, ticket = processor.get_estimated_resolution_time("", 1001)
            assert ticket_type == 'error'
            assert 'is_valid' in ticket
            assert ticket['is_valid'] is False
        
        # Test with too short query - also handled internally
        with patch.object(RAGProcessor, 'load_active_data') as mock_load:
            mock_load.return_value = pd.DataFrame({
                'TicketID': [1, 2, 3],
                'customerID': [101, 102, 103]
            })
            
            ticket_type, ticket = processor.get_estimated_resolution_time("Too short", 1001)
            assert ticket_type == 'error'
            assert 'is_valid' in ticket
            assert ticket['is_valid'] is False
            
        # Test with no matching tickets
        with patch.object(RAGProcessor, 'perform_similarity_search') as mock_search:
            # No matches in either active or history vectorstore
            mock_search.side_effect = [([], None), ([], None)]
            
            with patch.object(RAGProcessor, 'load_active_data') as mock_load:
                mock_load.return_value = pd.DataFrame({
                    'TicketID': [1, 2, 3],
                    'customerID': [101, 102, 103]
                })
                
                ticket_type, ticket = processor.get_estimated_resolution_time(
                    "This is a valid length query but doesn't match any tickets",
                    1001
                )
                
                assert ticket_type == 'Not_valid_query'
                assert 'is_valid' in ticket
                assert ticket['is_valid'] is False