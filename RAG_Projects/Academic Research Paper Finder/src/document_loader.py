from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import logging
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from uuid import uuid4

logger = logging.getLogger(__name__)

class ResearchPaperLoader:
    """
    A class to load and process research papers from CSV files.
    
    This loader processes academic research papers and converts them into
    a standardized document format for the vector store and retrieval system.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.
        
        Args:
            data_path (str): Path to the CSV file containing research papers
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")

    def create_documents(self) -> List[Document]:
        """
        Load research papers from CSV and convert to LangChain Document objects.
        source_column: title
        metadata_columns: "abstract", "authors", "n_citation", "references", "venue", "year", "id"
        Returns:
            List[Document]: List of Document objects representing research papers
        """
        try:
            # Load data from CSV
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} papers from {self.data_path}")

            ## list of columns to use for metadata and page content
            source_column = "title"
            metadata_columns = ["abstract", "authors", "n_citation", "references", "venue", "year", "id"]

            loader = CSVLoader(file_path=self.data_path, 
                               source_column=source_column, 
                               metadata_columns=metadata_columns, 
                               encoding="utf-8")
            
            # Convert DataFrame to list of Documents
            documents = loader.load()
            
            logger.info(f"Created {len(documents)} Document objects")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading papers: {str(e)}")
            raise