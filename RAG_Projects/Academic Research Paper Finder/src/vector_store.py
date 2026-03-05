from typing import List, Dict, Any, Optional
import os
import faiss
import numpy as np
import pickle
from pathlib import Path
import logging
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class ResearchVectorStore:
    """
    A class to manage the vector store for research papers using FAISS.
    
    This class handles the creation, storage, and retrieval of vector embeddings
    for academic research papers.
    """
    
    def __init__(self, store_path: str):
        """
        Initialize the vector store with FAISS and embeddings model.
        
        Args:
            store_path (str): Path where the vector store will be saved
        """
        self.store_path = Path(store_path)
        
        # Create a directory for the store if it doesn't exist
        self.store_path.mkdir(exist_ok=True, parents=True)
        
        # Choose embedding model based on environment variables
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize FAISS index, documents, and metadata
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_size = None
    
    def _get_embedding_size(self)   -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            int: Dimensionality of the embeddings
        """
        e = self.embeddings.embed_query("test")
        return len(e)
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create vector store from research paper documents.
        
        Args:
            documents (List[Document]): List of research paper documents
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Store documents and metadata
        self.documents = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]
        embd = self.embeddings.embed_documents(self.documents)
        # Get embedding dimension
        self.embedding_size = self._get_embedding_size()
        logger.info(f"Embedding size: {self.embedding_size}")
    
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_size)
            
        # Add embeddings to index
        faiss.normalize_L2(np.array(embd).astype('float32'))
        self.index.add(np.array(embd).astype('float32'))
        
        # logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to index")
        
        # Save the vector store
        self.save()
        logger.info(f"Vector store created with {self.index.ntotal} vectors")

    def save(self) -> None:
        """Save the vector store to disk"""
        index_path = self.store_path / "faiss_index.bin"
        metadata_path = self.store_path / "metadata.pkl"
        documents_path = self.store_path / "documents.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and documents
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Vector store saved to {self.store_path}")

    @classmethod
    def load(cls, store_path: str) -> 'ResearchVectorStore':
        """
        Load vector store from disk.
        
        Args:
            store_path (str): Path to the stored vector data
            
        Returns:
            FAISSResearchVectorStore: Loaded vector store
        """
        instance = cls(store_path)
        
        index_path = Path(store_path) / "faiss_index.bin"
        metadata_path = Path(store_path) / "metadata.pkl"
        documents_path = Path(store_path) / "documents.pkl"
        
        # Check if all required files exist
        if not all(p.exists() for p in [index_path, metadata_path, documents_path]):
            raise FileNotFoundError(f"Missing required files in {store_path}")
        
        # Load FAISS index
        instance.index = faiss.read_index(str(index_path))
        
        # Load metadata and documents
        with open(metadata_path, 'rb') as f:
            instance.metadata = pickle.load(f)
        
        with open(documents_path, 'rb') as f:
            instance.documents = pickle.load(f)
        
        # Get embedding dimensions from index
        instance.embedding_size = instance.index.d
        
        logger.info(f"Vector store loaded from {store_path} with {instance.index.ntotal} vectors")
        return instance

    def query_similar(self, query: str, k: int = 5, use_recency: bool = False) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar research papers.
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning but DO NOT raise an exception
        - Non-existent support types MUST return an empty list with an appropriate warning
    
        Args:
            query (str): Query text to find similar documents
            k (int): Number of similar documents to return
            use_recency (bool): Whether to factor in recency of papers when ranking
                When True, results are ranked using a combined score that weighs both 
                semantic similarity (70%) and publication recency (30%).
                
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata
            
        Notes:
            When use_recency=True, a combined score is calculated as follows:
            1. Semantic similarity score: Cosine similarity between query and paper (0-1)
            2. Recency score: Normalized score based on publication year
            recency_score = (paper_year - (current_year - 30)) / 30
            This creates a 0-1 score where recent papers score higher
            3. Combined score: 0.7 * similarity_score + 0.3 * recency_score
            4. Results are sorted by this combined score in descending order
            
            If paper year is missing, only the similarity score is used.

        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
            # raise ValueError("Query cannot be empty")

        if self.index is None or self.index.ntotal == 0:
            logger.warning("No index available for querying")
            return []
        
        # Get more results if we need to rerank by recency
        fetch_k = k * 3 if use_recency else k
        fetch_k = min(fetch_k, self.index.ntotal)  # Don't fetch more than available
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # Search the index
        distances, indices = self.index.search(query_embedding_np, fetch_k)
        
        # Convert distances to similarities (1 - normalized_distance)
        similarities = 1 - distances[0] / 2  # Normalized L2 distance is between 0-2
    
        # Gather results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx == -1:  # FAISS may return -1 for not enough results
                continue
                
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarity)
            })
        
        # Apply recency boost if requested
        if use_recency and results:
            import datetime
            current_year = datetime.datetime.now().year
            
            # Calculate a combined score based on similarity and recency
            for result in results:
                year = result['metadata'].get('year')
                if year and isinstance(year, (int, float, str)):
                    # Normalize year to a 0-1 scale (assuming papers are from last 30 years)
                    try:
                        year_int = int(year)
                        recency_score = (year_int - (current_year - 30)) / 30
                        recency_score = max(0, min(1, recency_score))  # Clamp between 0 and 1
                        
                        # Combined score (70% similarity, 30% recency)
                        result['combined_score'] = 0.7 * result['similarity'] + 0.3 * recency_score
                    except (ValueError, TypeError):
                        result['combined_score'] = result['similarity']
                else:
                    # If no year available, use similarity score only
                    result['combined_score'] = result['similarity']
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
        else:
            # Sort by similarity only
            results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k
        return results[:k]