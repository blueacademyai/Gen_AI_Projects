from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

class SupportVectorStore:
    """
    A class to manage the vector store for support tickets using ChromaDB.
    
    This class handles the creation, storage, and retrieval of vector embeddings
    for technical, product, and customer support tickets in separate collections.
    
    IMPORTANT:
    - Empty queries (null or whitespace-only) must be rejected with an empty result list
    - Queries shorter than 10 characters must be rejected with an empty result list
    - All metadata must be properly processed for ChromaDB compatibility
    """
    
    def __init__(self, vecstore_path):
        """Initialize the vector store with ChromaDB client and OpenAI embeddings."""
        self.vecstore_path = vecstore_path
        self.client = chromadb.PersistentClient(path=self.vecstore_path, tenant=DEFAULT_TENANT)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api)
        # self.embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        self.collections = {}

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB by converting lists to strings and ensuring valid types.
        
        ChromaDB requires all metadata values to be primitive types (str, int, float, bool).
        Lists must be converted to comma-separated strings, and None values must be handled appropriately.
        
        Args:
            metadata (Dict[str, Any]): Original metadata dictionary
            
        Returns:
            Dict[str, Any]: Processed metadata with ChromaDB-compatible types
        """
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                processed[key] = ','.join(value) if value else ''
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # Keep valid primitive types, convert None to empty string
                processed[key] = value if value is not None else ''
            else:
                # Convert any other types to strings
                processed[key] = str(value)
        return processed

    def _process_metadata_for_return(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata when retrieving from ChromaDB, converting string-lists back to actual lists.
        
        This function reverses the transformations done in _prepare_metadata() to ensure
        that metadata is returned in the expected format.
        
        Args:
            metadata (Dict[str, Any]): Metadata from ChromaDB
            
        Returns:
            Dict[str, Any]: Processed metadata with proper types
        """
        processed = metadata.copy()
        if 'tags' in processed and isinstance(processed['tags'], str):
            # Convert comma-separated tags back to list
            processed['tags'] = [tag.strip() for tag in processed['tags'].split(',') if tag.strip()]
        return processed

    def create_vector_store(self, documents_by_type: Dict[str, List[Document]]) -> None:
        """
        Create vector store collections from documents, organized by support type.
        
        Args:
            documents_by_type (Dict[str, List[Document]]): Dictionary of documents organized by support type
        """
        # Create collection for each support type
        for support_type, docs in documents_by_type.items():
            if not docs:
                logger.warning(f"No documents found for {support_type} support")
                continue
                
            collection_name = f"support_tickets_{support_type}"
            
            # Create or get collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"support_type": support_type}
            )
            
            # Prepare documents for insertion
            ids = [str(doc.metadata['ticket_id']) for doc in docs]
            texts = [doc.page_content for doc in docs]
            metadatas = [self._prepare_metadata(doc.metadata) for doc in docs]
            
            # Generate embeddings and add to collection
            embeddings = self.embeddings.embed_documents(texts)
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            self.collections[support_type] = collection
            logger.info(f"Created collection for {support_type} support with {len(docs)} documents")

    def save_local(self) -> None:
        """
        Save the vector store to local storage.
        
        Args:
            directory (str): Directory path to save the vector store
        """
        # os.makedirs("chroma/STRS", exist_ok=True)
        # self.client.persist()
        logger.info(f"Vector store saved to {self.vecstore_path}")


    @classmethod
    def load_local(cls, directory: str) -> 'SupportVectorStore':
        """
        Load a vector store from local storage.
        
        Args:
            directory (str): Directory path containing the vector store
            
        Returns:
            SupportVectorStore: Loaded vector store instance
        """
        # Create new instance with the directory
        instance = cls(vecstore_path=directory)
        
        # Load all collections
        collections = instance.client.list_collections()
        for collection in collections:
            metadata = collection.metadata or {}
            support_type = metadata.get('support_type')
            if support_type:
                instance.collections[support_type] = collection

        if len(collections)>0:    
            logger.info(f"Loaded vector store from {directory} with {len(collections)} collections")

            return instance
        else:
            return None

    def query_similar(
        self, 
        query: str, 
        support_type: str = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning but DO NOT raise an exception
        - Non-existent support types MUST return an empty list with an appropriate warning
        
        Args:
            query (str): Query text to find similar documents
            support_type (str, optional): Specific support type to query. If None, queries all types
            k (int): Number of similar documents to return per collection
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata, each containing:
            - 'content': Document content
            - 'metadata': Document metadata
            - 'similarity': Similarity score (1 - distance)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

                # Handle nonexistent support type
        if support_type and support_type not in self.collections:
            logger.warning(f"Support type '{support_type}' not found")
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        results = []
        
        collections_to_query = (
            [self.collections[support_type]] if support_type in self.collections
            else self.collections.values()
        )
        
        for collection in collections_to_query:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            for doc, metadata, distance in zip(
                response['documents'][0],
                response['metadatas'][0],
                response['distances'][0]
            ):
                results.append({
                    'content': doc,
                    'metadata': self._process_metadata_for_return(metadata),
                    'similarity': 1 - distance
                })
        
        # Sort by similarity and return top k overall
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def get_support_types(self) -> List[str]:
        """
        Get list of available support types in the vector store.
        
        Returns:
            List[str]: List of support type names
        """
        return list(self.collections.keys())