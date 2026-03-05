import pytest
import pandas as pd
from pathlib import Path
import os

@pytest.fixture
def sample_papers():
    """Sample research paper data for testing"""
    return [
        {
            "title": "Transformer Models in Natural Language Processing",
            "abstract": "This paper provides a comprehensive overview of transformer models.",
            "authors": "Smith, J.; Johnson, A.",
            "year": 2022,
            "venue": "Journal of NLP",
            "n_citation": 150,
            "references": "ref1,ref2,ref3",
            "id": "paper001"
        },
        {
            "title": "Quantum Computing Algorithms for Cryptography",
            "abstract": "This study examines quantum computing applications in cryptography.",
            "authors": "Brown, R.; Davis, M.",
            "year": 2021,
            "venue": "Quantum Computing Conference",
            "n_citation": 75,
            "references": "ref4,ref5",
            "id": "paper002" 
        },
        {
            "title": "Gene Sequencing Using Deep Learning",
            "abstract": "Novel approaches to gene sequencing using neural networks.",
            "authors": "Wilson, C.; Moore, T.",
            "year": 2023,
            "venue": "Bioinformatics Journal",
            "n_citation": 45,
            "references": "ref6,ref7",
            "id": "paper003"
        }
    ]

@pytest.fixture
def test_environment(sample_papers, tmp_path):
    """Create test environment with research paper data files"""
    # Create directories
    data_dir = tmp_path / "data"
    vector_store_dir = tmp_path / "chroma_db"
    data_dir.mkdir()
    vector_store_dir.mkdir()

    # Create CSV file with sample papers
    papers_df = pd.DataFrame(sample_papers)
    csv_path = data_dir / "papers.csv"
    papers_df.to_csv(csv_path, index=False)

    return {
        "tmp_dir": str(tmp_path),
        "data_dir": str(data_dir),
        "vector_store_dir": str(vector_store_dir),
        "csv_path": str(csv_path)
    }

@pytest.fixture
def mock_openai_env(monkeypatch):
    """Mock OpenAI environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    return [
        [0.1, 0.2, 0.3],  # Simplified 3D embeddings for testing
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]