from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
import logging
import jq
from uuid import uuid4
from functools import partial

logger = logging.getLogger(__name__)

class SupportDocumentLoader:
    """
    A class to load and process support tickets from JSON and XML files using LangChain loaders.
    
    This loader uses LangChain's JSONLoader and custom XML loading to process support tickets and
    converts them into a standardized document format for the RAG system.
    
    IMPORTANT: 
    - Even when using LangChain loaders, you MUST use the custom get_json_content and 
      get_json_metadata functions to ensure consistent document formatting
    - Ensure all ticket IDs are unique across the entire dataset
    - The format of ticket IDs must follow the pattern: "{support_type}_{original_id}" for JSON
      and "{support_type}_xml_{original_id}" for XML files
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.
        
        Args:
            data_path (str): Directory path containing support ticket files
            
        Raises:
            FileNotFoundError: If the specified data path does not exist
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")
            
        # Support types mapping
        self.support_types = {
            'technical': ['Technical Support_tickets.json', 'Technical Support_tickets.xml'],
            'product': ['Product Support_tickets.json', 'Product Support_tickets.xml'],
            'customer': ['Customer Service_tickets.json', 'Customer Service_tickets.xml']
        }

    def get_json_content(self, data: Dict[str, Any]) -> str:
        """
        Format JSON data into a standardized content string.
        
        This function MUST produce content in the exact format shown below to ensure
        consistent document formatting across the system.
        
        Args:
            data (Dict[str, Any]): Raw JSON data
            
        Returns:
            str: Formatted content string with the exact format:
            
            Subject: {data.get('subject', '')}
            Description: {data.get('body', '')}
            Resolution: {data.get('answer', '')}
            Type: {data.get('type', '')}
            Queue: {data.get('queue', '')}
            Priority: {data.get('priority', '')}
        """
        return f"""
        Subject: {data.get('subject', '')}
        Description: {data.get('body', '')}
        Resolution: {data.get('answer', '')}
        Type: {data.get('type', '')}
        Queue: {data.get('queue', '')}
        Priority: {data.get('priority', '')}
        """

    def get_json_metadata(self, record: Dict[str, Any], support_type: str = None) -> Dict[str, Any]:
        """
        Extract metadata from JSON data.
        
        This function MUST produce metadata with all the required fields shown below.
        The 'ticket_id' MUST follow the format "{support_type}_{original_id}" to ensure
        proper document identification.
        
        Args:
            record (Dict[str, Any]): Raw JSON record
            support_type (str, optional): Type of support (technical, product, customer)
            
        Returns:
            Dict[str, Any]: Extracted metadata with the exact format:
            {
                'ticket_id': "{support_type}_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from JSON ("Ticket ID" field)
                'support_type': str,          # Type of support (technical, product, customer)
                'type': str,                  # Type field from original data
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags from tag_1 through tag_8
                'source': 'json',             # Source format identifier
                'subject': str,               # Subject field for content formatting
                'body': str,                  # Body field for content formatting
                'answer': str                 # Answer field for content formatting
            }
            
        Raises:
            ValueError: If support_type is not provided
        """
        # Get the correct support type, either from parameter or from record
        actual_support_type = support_type or record.get('support_type')
        if not actual_support_type:
            # Default to a value or raise an error
            raise ValueError("Support type not provided")
        
        # Extract tags, filtering out NaN values
        tags = []
        for i in range(1, 9):
            tag = record.get(f'tag_{i}')
            if tag and tag != "NaN" and str(tag).lower() != 'nan':
                tags.append(tag)

        # Generate a unique ID based on ticket ID and support type
        original_id = record.get('Ticket ID') or str(uuid4())
        unique_id = f"{actual_support_type}_{original_id}"

        return {
            'ticket_id': unique_id,  # Using the unique ID
            'original_ticket_id': original_id,  # Keep original ID for reference
            'support_type': actual_support_type,
            'type': record.get('type'),
            'queue': record.get('queue'),
            'priority': record.get('priority'),
            'language': record.get('language'),
            'tags': tags,
            'source': 'json',  # Track the source format
            # Include the original data for content formatting
            'subject': record.get('subject'),
            'body': record.get('body'),
            'answer': record.get('answer')
        }

    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Load tickets from an XML file.
        
        XML tickets MUST be processed to follow the same content and metadata format
        as JSON tickets, with the only difference being the 'ticket_id' format and
        'source' field.
        
        Args:
            file_path (Path): Path to the XML file
            support_type (str): Type of support (technical, product, customer)
            
        Returns:
            List[Document]: List of Document objects with the following format:
            
            Content format:
            Subject: {ticket_elem.findtext('subject')}
            Description: {ticket_elem.findtext('body')}
            Resolution: {ticket_elem.findtext('answer')}
            Type: {ticket_elem.findtext('type')}
            Queue: {ticket_elem.findtext('queue')}
            Priority: {ticket_elem.findtext('priority')}
            
            Metadata format:
            {
                'ticket_id': "{support_type}_xml_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from XML
                'support_type': str,          # Type of support
                'type': str,                  # Type field
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags
                'source': 'xml'               # Source format identifier
            }
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        processed_tickets = []
        for ticket_elem in root.findall('.//Ticket'):
            # Extract and process tags
            tags = []
            for i in range(1, 9):
                tag = ticket_elem.findtext(f'tag_{i}')
                if tag and tag.lower() != 'nan':
                    tags.append(tag)
            
            # Generate a unique ID
            original_id = ticket_elem.findtext('TicketID') or str(uuid4())
            unique_id = f"{support_type}_xml_{original_id}"

            # Create document with proper metadata
            processed_ticket = Document(
                page_content=f"""
                Subject: {ticket_elem.findtext('subject')}
                Description: {ticket_elem.findtext('body')}
                Resolution: {ticket_elem.findtext('answer')}
                Type: {ticket_elem.findtext('type')}
                Queue: {ticket_elem.findtext('queue')}                
                Priority: {ticket_elem.findtext('priority')}
                """,
                metadata={
                    'ticket_id': unique_id,  # Using the unique ID with format support_type_xml_original_id
                    'original_ticket_id': ticket_elem.findtext('TicketID'),
                    'support_type': support_type,
                    'type': ticket_elem.findtext('type'),
                    'queue': ticket_elem.findtext('queue'),
                    'priority': ticket_elem.findtext('priority'),
                    'language': ticket_elem.findtext('language'),
                    'tags': tags,
                    'source': 'xml'  # Track the source format
                }
            )
            processed_tickets.append(processed_ticket)
        
        return processed_tickets
    def load_tickets(self) -> Dict[str, List[Document]]:
        """
        Load all support tickets using LangChain loaders, organized by support type.
        
        IMPORTANT:
        - When using JSONLoader, you MUST create a custom function that properly passes
          the support_type parameter to get_json_metadata
        - Validate that all ticket IDs are unique across the entire dataset
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Documents as values
            
        Raises:
            ValueError: If duplicate ticket IDs are found
        """
        documents_by_type = {
            'technical': [],
            'product': [],
            'customer': []
        }
        
        # Process files for each support type
        for support_type, filenames in self.support_types.items():
            for filename in filenames:
                file_path = self.data_path / filename
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                try:
                    if file_path.suffix.lower() == '.json':
                        def metadata_transform(record, metadata=None):
                            return self.get_json_metadata(record, support_type)
                        # metadata_func = partial(self.get_json_metadata, support_type=support_type)
                        
                        # Configure JSON loader
                        loader = JSONLoader(
                            file_path=str(file_path),
                            jq_schema='.[]',  # Extract all elements from root array
                            content_key=None,
                            text_content=False,
                            metadata_func=metadata_transform  # Use the partial function with support_type
                        )
                        documents = loader.load()
                        
                        # Format content for JSON documents
                        for doc in documents:
                            doc.page_content = self.get_json_content(doc.metadata)
                            
                    elif file_path.suffix.lower() == '.xml':
                        # Load XML documents with support type
                        documents = self.load_xml_tickets(file_path, support_type)
                        
                    documents_by_type[support_type].extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue
        
        # Validate unique IDs
        all_ids = set()
        for docs in documents_by_type.values():
            for doc in docs:
                ticket_id = doc.metadata['ticket_id']
                if ticket_id in all_ids:
                    logger.error(f"Duplicate ticket ID found: {ticket_id}")
                    raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
                all_ids.add(ticket_id)
        
        return documents_by_type
    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Load and process all support tickets into LangChain Document objects.
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Document objects as values
        """

        return self.load_tickets()