import pytest
import pandas as pd
from pathlib import Path
from langchain.schema import Document
from src.document_loader import ResearchPaperLoader

class TestResearchPaperLoader:
    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Create a sample CSV file for testing"""
        data = {
            'title': ['Quantum Computing Basics', 'Deep Learning Applications'],
            'abstract': ['Introduction to quantum computing', 'Overview of deep learning in practice'],
            'authors': ['Smith, J.', 'Johnson, A.'],
            'year': [2022, 2023],
            'venue': ['Journal of Quantum Computing', 'AI Conference'],
            'n_citation': [10, 25],
            'references': ['ref1,ref2', 'ref3,ref4'],
            'id': ['paper1', 'paper2']
        }
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_papers.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def test_load_papers(self, sample_csv_data):
        """Test loading papers from CSV file"""
        loader = ResearchPaperLoader(sample_csv_data)
        documents = loader.create_documents()
        

        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert isinstance(documents[1], Document)
        
        # Check content of first document
        assert "Quantum Computing Basics" in documents[0].page_content
        assert "Smith, J." in documents[0].metadata['authors']
        assert documents[0].metadata['year'] == str(2022)
        
        # Check content of second document
        assert "Deep Learning Applications" in documents[1].page_content
        assert documents[1].metadata['n_citation'] == str(25)
    
    def test_create_documents(self, sample_csv_data):
        """Test create_documents method (which calls load_papers)"""
        loader = ResearchPaperLoader(sample_csv_data)
        documents = loader.create_documents()
        
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_file_not_found(self, tmp_path):
        """Test handling of non-existent file"""
        nonexistent_path = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            ResearchPaperLoader(str(nonexistent_path))