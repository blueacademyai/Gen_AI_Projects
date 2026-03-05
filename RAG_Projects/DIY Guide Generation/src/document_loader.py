from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema import Document
import logging
from uuid import uuid4
import json

logger = logging.getLogger(__name__)

class SupportDocumentLoader:
    """
    A class to load and process support tickets from JSON and XML files.
    
    This loader processes support tickets and converts them into a standardized document format
    for the RAG system, with embeddings created only for subject and body fields.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.
        
        Args:
            data_path (str): Directory path containing support ticket files
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

    def get_embedding_content(self, data: Dict[str, Any]) -> str:
        """
        Extract subject and body for embedding.
        
        Args:
            data (Dict[str, Any]): Ticket data
            
        Returns:
            str: Combined subject and body for embedding
        """
        subject = data.get('subject', '')
        body = data.get('body', '')
        return f"{subject}\n{body}"

    def get_metadata(self, record: Dict[str, Any], support_type: str, source: str) -> Dict[str, Any]:
        """
        Extract metadata from record data.
        
        Args:
            record (Dict[str, Any]): Raw record data
            support_type (str): Type of support
            source (str): Source file type (json/xml)
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Extract tags, filtering out NaN values
        tags = []
        for i in range(1, 9):
            tag = record.get(f'tag_{i}')
            if tag and tag != "NaN" and str(tag).lower() != 'nan':
                tags.append(tag)

        # Generate a unique ID
        original_id = record.get('Ticket ID')

        return {
            'ticket_id': original_id,
            'support_type': support_type,
            'type': record.get('type'),
            'queue': record.get('queue'),
            'priority': record.get('priority'),
            'language': record.get('language', 'en'),
            'tags': tags,
            'subject': record.get('subject', ''),
            'body': record.get('body', ''),
            'answer': record.get('answer', ''),
            'source': source
        }

    def load_json_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Load tickets from a JSON file.
        
        Args:
            file_path (Path): Path to the JSON file
            support_type (str): Type of support
            
        Returns:
            List[Document]: List of Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for record in data:
            metadata = self.get_metadata(record, support_type, 'json')
            embedding_content = self.get_embedding_content(metadata)
            
            documents.append(Document(
                page_content=embedding_content,
                metadata=metadata
            ))
        
        return documents

    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Load tickets from an XML file.
        
        Args:
            file_path (Path): Path to the XML file
            support_type (str): Type of support
            
        Returns:
            List[Document]: List of Document objects
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        documents = []
        for ticket_elem in root.findall('.//Ticket'):
            # Extract ticket data
            ticket_data = {
                'Ticket ID': ticket_elem.findtext('TicketID'),
                'subject': ticket_elem.findtext('subject'),
                'body': ticket_elem.findtext('body'),
                'answer': ticket_elem.findtext('answer'),
                'type': ticket_elem.findtext('type'),
                'queue': ticket_elem.findtext('queue'),
                'priority': ticket_elem.findtext('priority'),
                'language': ticket_elem.findtext('language')
            }
            
            # Extract tags
            for i in range(1, 9):
                tag = ticket_elem.findtext(f'tag_{i}')
                if tag:
                    ticket_data[f'tag_{i}'] = tag
            
            # Get metadata and embedding content
            metadata = self.get_metadata(ticket_data, support_type, 'xml')
            embedding_content = self.get_embedding_content(metadata)
            
            documents.append(Document(
                page_content=embedding_content,
                metadata=metadata
            ))
        
        return documents

    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Load all support tickets, organized by support type.
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Documents as values
        """
        documents_by_type = {support_type: [] for support_type in self.support_types}
        
        # Process files for each support type
        for support_type, filenames in self.support_types.items():
            for filename in filenames:
                file_path = self.data_path / filename
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                try:
                    if file_path.suffix.lower() == '.json':
                        documents = self.load_json_tickets(file_path, support_type)
                    elif file_path.suffix.lower() == '.xml':
                        documents = self.load_xml_tickets(file_path, support_type)
                    else:
                        logger.warning(f"Unsupported file type: {file_path}")
                        continue
                        
                    documents_by_type[support_type].extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue
        
        # Validate unique IDs
        # all_ids = set()
        # for docs in documents_by_type.values():
        #     for doc in docs:
        #         ticket_id = doc.metadata['ticket_id']
        #         if ticket_id in all_ids:
        #             logger.error(f"Duplicate ticket ID found: {ticket_id}")
        #             raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
        #         all_ids.add(ticket_id)
        
        return documents_by_type