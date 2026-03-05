"""
Vector Store for Social Media App Support Agent

This module provides functionality for creating and managing a FAISS vector store for the Social Media App support chatbot. It handles document embedding, vector store 
creation, saving, loading, and querying.
"""

import os
from typing import List, Dict, Optional, Any, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

class SocialMediaVectorStore:
    """
    A class for managing a FAISS vector store for social media support documentation.
    
    This class provides a complete interface for working with vector embeddings and similarity search:
    - Creating vector stores from documents
    - Persisting vector stores to disk and loading them
    - Querying for similar documents with optional filtering
    - Retrieving documents with similarity scores
    - Generating embeddings for arbitrary text
    
    Attributes:
        embedding_model (Embeddings): Model for creating vector embeddings
        index_path (str): Directory path where the FAISS index is stored
        index_name (str): Name of the FAISS index file
        vectorstore (Optional[FAISS]): The FAISS vector store instance
    """
    
    def __init__(
        self,
        embedding_model: Optional[Embeddings] = None,
        index_path: str = "faiss_index",
        index_name: str = "support_docs"
    ):
        """
        Initialize the vector store manager.
        
        What this does:
        1. Sets the embedding model (uses OpenAIEmbeddings by default if none provided)
        2. Sets the index path and name for storing/retrieving the vector database
        3. Initializes the vectorstore attribute to None (will be populated later)
        
        Args:
            embedding_model (Optional[Embeddings]): The embedding model to use (defaults to OpenAIEmbeddings)
            index_path (str): Directory to store the FAISS index (default: "faiss_index")
            index_name (str): Name of the index (default: "support_docs")
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.index_path = index_path
        self.index_name = index_name
        self.vectorstore = None
        
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from the provided documents.
        
        What this does:
        1. Validates that documents list is not empty
        2. Uses FAISS.from_documents method to:
           a. Generate embeddings for each document using the embedding model
           b. Create a searchable vector index from these embeddings
        3. Stores the created vector store in the instance variable
        4. Returns the vector store object
        
        Args:
            documents (List[Document]): List of Document objects to embed and store
            
        Returns:
            FAISS: The created FAISS vector store object
            
        Raises:
            Exception: If vector store creation fails
            
        Note:
            Returns None if documents list is empty.
        """
        if not documents:
            return None
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            return self.vectorstore
        except Exception as e:
            raise
    
    def save_vectorstore(self) -> bool:
        """
        Save the vector store to disk.
        
        What this does:
        1. Checks if vectorstore exists
        2. Creates the directory for the index if it doesn't exist
        3. Uses the FAISS save_local method to persist the index to disk
           with the configured path and name
        
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            Returns False immediately if vectorstore is None.
            Silently handles exceptions by returning False.
        """
        if self.vectorstore is None:
            return False
        
        try:
            os.makedirs(self.index_path, exist_ok=True)
            self.vectorstore.save_local(folder_path=self.index_path, index_name=self.index_name)
            return True
        except Exception:
            return False
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """
        Load a vector store from disk.
        
        What this does:
        1. Constructs the full path to the index file
        2. Checks if the index file exists
        3. Uses FAISS.load_local to load the vector store from disk
           with the configured embedding model
        4. Stores the loaded vector store in the instance variable
        
        Returns:
            Optional[FAISS]: The loaded FAISS vector store, or None if loading fails
            
        Note:
            Returns None if the index file doesn't exist.
            Silently handles exceptions by returning None.
        """
        index_path = os.path.join(self.index_path, f"{self.index_name}.faiss")
        
        if not os.path.exists(index_path):
            return None
        
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.embedding_model,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            
            return self.vectorstore
        except Exception:
            return None
    
    def get_embedding_for_text(self, text: str) -> List[float]:
        """
        Get the embedding vector for a piece of text.
        
        What this does:
        1. Uses the embedding model's embed_query method to:
           a. Process the input text according to the model's requirements
           b. Generate a numerical vector representation of the text
        2. Returns the raw embedding vector
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: The embedding vector as a list of floating point numbers
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            raise