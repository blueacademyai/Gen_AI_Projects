from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from pathlib import Path

class VectorStoreManager:
    """
    A class for managing FAISS vector stores for document retrieval.
    
    This class handles the creation, loading, and updating of FAISS vector stores, which enable efficient similarity search across document embeddings. It provides functionality to:
        1. Create new vector stores from documents
        2. Load existing vector stores from disk
        3. Append new documents to existing vector stores
        4. Save vector stores to disk for persistence
    
    The vector stores created and managed by this class are used for semantic similarity searches across document collections, enabling efficient retrieval of relevant documents
    based on their meaning rather than keyword matching.
    
    Attributes:
        index_path (Path): Path where the FAISS index is stored on disk
        embedding_model (Embeddings): Model used to generate vector embeddings for documents
        vectorstore (FAISS): The actual FAISS vector store instance, loaded when needed
    """
    def __init__(self, index_path: str, embedding_model: Embeddings):
        """
        Initialize the VectorStoreManager with an index path and embedding model.
        
        This constructor:
        1. Converts the index path string to a Path object
        2. Stores a reference to the embedding model
        3. Initializes the vectorstore attribute to None (lazy loading)
        
        Args:
            index_path (str): Path where the FAISS index will be stored
                This should be a directory path where the vector store files will be saved
            embedding_model (Embeddings): Model used to generate embeddings for documents
                Must implement the Langchain Embeddings interface with an embed_documents method
        """
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.vectorstore = None

    def create_vectorstore(self, documents: list) -> None:
        """
        Create a FAISS vector store from documents.
        
        This method:
        1. Validates that the documents list is not empty by raising ValueError that "Documents list cannot be empty" if it is empty.
        2. Creates a new FAISS vector store using the provided documents
        3. Embeds all documents using the embedding model
        4. Saves the created vector store to the index path location
        
        Args:
            documents (list): List of Document objects to add to the vector store
                Each Document object should have:
                - page_content: The text content to be embedded
                - metadata: A dictionary containing relevant metadata
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        self.vectorstore.save_local(str(self.index_path))

    def append_vectorstore(self, new_data: dict) -> None:
        """
        Append a new ticket to the vector store.
        
        This method:
        1. Checks if the vector store is initialized
        2. Validates required fields in the input data
        3. Checks if the description is meaningful enough
        4. Creates a new Document object from the ticket data
        5. Adds the document to the existing vector store
        6. Saves the updated vector store to disk
        
        Required fields in new_data:
        - description: The text content to be embedded
        - TicketID: Unique identifier for the ticket
        - locationID: Location identifier 
        - estimated_resolution_time: Estimated resolution time
        
        Args:
            new_data (dict): Dictionary containing the new ticket data with all required fields
            
        Raises:
            ValueError: If vector store is not initialized ("Vector store not initialized. Call create_vectorstore first."), 
            required fields are missing ("Missing required field: {field}"),
            description is too short ("Description is too short for meaningful vector representation"), 
            Exception: For other unexpected errors during the process
        """
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Call create_vectorstore first.")
            
            # Validate required fields
            required_fields = ['description', 'TicketID', 'locationID', 'estimated_resolution_time']
            for field in required_fields:
                if field not in new_data:
                    raise ValueError(f"Missing required field: {field}")
            
            description = new_data.get('description', '')
            if not description or len(description.strip()) < 5:
                raise ValueError("Description is too short for meaningful vector representation")
                
            new_doc = Document(
                page_content=description,
                metadata={
                    'ticket_id': str(new_data['TicketID']),
                    'location_id': str(new_data['locationID']),
                    'estimated_resolution_time': str(new_data['estimated_resolution_time'])
                }
            )
            self.vectorstore.add_documents([new_doc])
            self.vectorstore.save_local(str(self.index_path))
        except ValueError as e:
            print(f"Error adding document to vector store: {e}")
            raise  # Re-raise to let caller handle it
        except Exception as e:
            print(f"Unexpected error adding document to vector store: {e}")
            raise  # Re-raise to let caller handle it

    def get_vectorstore(self):
        """
        Load and return the FAISS vector store.
        
        This method implements lazy loading:
        1. If the vector store is already loaded, it returns the existing instance
        2. If not loaded, it checks if the index exists at the specified path, if not, raise FileNotFoundError that Vector store index not found at {self.index_path}
        3. Loads the vector store from disk using the embedding model
        
        The method enables safe access to the vector store without requiring
        explicit loading by the client code.
        
        Returns:
            FAISS: The loaded FAISS vector store ready for similarity searches
        """
        if self.vectorstore is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Vector store index not found at {self.index_path}")
            self.vectorstore = FAISS.load_local(str(self.index_path), self.embedding_model, allow_dangerous_deserialization=True)
        return self.vectorstore