from typing import List, Dict, Any
import logging
from src.vector_store import ResearchVectorStore

logger = logging.getLogger(__name__)

class ResearchPaperRetriever:
    """
    A class for retrieving relevant research papers based on semantic queries using FAISS.
    
    This retriever uses FAISS vector similarity search to find papers that are semantically
    similar to the user's query, with optional recency-based ranking.
    """
    
    def __init__(self, vector_store: ResearchVectorStore):
        """
        Initialize the retriever with a FAISS vector store.
        
        Args:
            vector_store (FAISSResearchVectorStore): Vector store containing research paper embeddings
        """
        self.vector_store = vector_store

    def retrieve_papers(
        self, 
        query: str, 
        k: int = 5, 
        use_recency: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant research papers for a given query.
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return value error.
        - Shrot queries (less than 3 characters) MUST return value error.
        Args:
            query (str): User's research query
            k (int): Number of papers to retrieve
            use_recency (bool): Whether to factor in recency when ranking results
            
        Returns:
            List[Dict[str, Any]]: List of relevant research papers with metadata
            
        Raises:
            ValueError: If query is empty or too short
        
        formatted_results
            {
                'rank': 1,
                'title': "title",
                'authors': 'authors',
                'year': year,
                'venue': 'venue',
                'citations': 'n_citation',
                'abstract': 'abstract'),
                'similarity_score': 0.70,
                'paper_id': 'id'            
            }
        """
        # Input validation
        if not query:
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 3:
            raise ValueError("Query too short. Please provide a more specific query.")
            
        logger.info(f"Retrieving papers for query: {query}")
        
        # Get similar papers from vector store
        results = self.vector_store.query_similar(query, k, use_recency)
        
        # Format results for better presentation
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            
            title = result["content"]
        
            # Process year - ensure it's an integer if possible
            year = metadata.get('year', 'Unknown')
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            
            # Create formatted result
            formatted_results.append({
                'rank': i,
                'title': title[7:],
                'authors': metadata.get('authors', 'Unknown'),
                'year': year,
                'venue': metadata.get('venue', 'Unknown'),
                'citations': metadata.get('n_citation', 0),
                'abstract': metadata.get('abstract', 'Abstract not available'),
                'similarity_score': result['similarity'],
                'paper_id': metadata.get('id', '')
            })

        return formatted_results
    
    def retrieve_papers_with_recency(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve research papers with recency factored into ranking.
        
        Args:
            query (str): User's research query
            k (int): Number of papers to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant research papers with metadata in descending order of publication year
        """
        formatted_results = self.retrieve_papers(query, k, use_recency=True)
        sorted_results = sorted(formatted_results, key=lambda x: x['year'], reverse=True)
        return sorted_results