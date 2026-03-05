"""
Document Loader for Social Media App Support Agent

This module provides utilities for loading and processing documentation files for the Social Media App support chatbot. It handles reading files from various formats, preprocessing text, and chunking documents
into appropriate sizes for the vectorstore.
"""

import os
import glob
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader

class SocialMediaDocumentLoader:
    """
    Loads, cleans, and processes social media documentation files for use in AI support agents.
    
    This class reads text files from a directory, cleans their content by removing unnecessary elements, and splits documents into optimally sized chunks for efficient vector embedding
    and retrieval. It handles common issues in social media documentation like non-breaking spaces and standardized footers.
    
    Attributes:
        data_dir (Path): Path to the directory containing documentation files.
        chunk_size (int): Maximum size of each text chunk in characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
        min_chunk_size (int): Minimum size for a chunk to be considered valid.
        text_splitter (RecursiveCharacterTextSplitter): Utility to split text into chunks.
    """
    
    def __init__(
        self, 
        data_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50
    ):
        """
        Initialize the SocialMediaDocumentLoader with specified parameters.
        
        What this does:
        1. Converts the data_dir string to a Path object for better file operations
        2. Stores chunking parameters as instance variables
        3. Creates a RecursiveCharacterTextSplitter with the specified parameters
        4. Sets up specific separators for smart document splitting
        
        Args:
            data_dir (str): Directory path where documentation files are stored.
            chunk_size (int, optional): Maximum size of each text chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 200.
            min_chunk_size (int, optional): Minimum size for a valid chunk. Defaults to 50.
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def load_documents(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Loads raw documents from files matching a pattern in the specified directory.
        
        What this does:
        1. Uses glob to find all files matching the pattern in data_dir
        2. Returns an empty list immediately if no matching files are found
        3. For each file path:
           a. Creates a TextLoader with UTF-8 encoding
           b. Loads document(s) from the file
           c. Adds the source filename to metadata for each document
           d. Adds documents to the result list
        4. Silently skips files that cause exceptions during loading
        
        Args:
            file_pattern (str, optional): Glob pattern to match files. Defaults to "*.txt".
            
        Returns:
            List[Document]: List of Document objects containing the loaded content.
                            Returns an empty list if no files are found or if errors occur during loading.
                            
        Note:
            Each document's metadata will include the source filename.
            Continues processing even if some files raise exceptions during loading.
        """
        file_paths = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        if not file_paths:
            return []
        
        documents = []
        for file_path in file_paths:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                file_docs = loader.load()
                
                for doc in file_docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                
                documents.extend(file_docs)
            except Exception:
                continue
                
        return documents
    
    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text content to improve quality for embedding.
        
        What this does:
        1. Replaces non-breaking spaces (\xa0) with regular spaces
        2. Normalizes all whitespace by:
           a. Splitting text into tokens
           b. Joining tokens with single spaces
        3. Removes common footer text by:
           a. Checking if the text contains "Did someone say"
           b. If found, splits at this phrase and keeps only the content before it
        
        Args:
            text (str): The original text to clean.
            
        Returns:
            str: Cleaned and normalized text with common issues fixed.
            
        Note:
            This function:
            1. Replaces non-breaking spaces (\xa0) with regular spaces
            2. Normalizes whitespace by joining split tokens
            3. Removes common footer text starting with "Did someone say"
        """
        # Replace non-breaking spaces and normalize whitespace
        text = " ".join(text.replace("\xa0", " ").split())
        
        # Fix encoding issues for special characters
        text = text.replace("â", "'")
        text = text.replace("â", "'")
        text = text.replace("â", "-")
        
        # Normalize common Unicode characters
        replacements = {
            "Ã§": "ç",
            "Ã¨": "è",
            "Ã©": "é",
            "Ãª": "ê",
            "Ã±": "ñ",
            "Ø§Ù": "ال",
            "ÙØ§": "فا",
            "Ø¨Ù": "بي",
            "×¢×": "עב",
            "×¨××ª": "רית"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove common footer sections - looking for multiple potential markers
        footer_markers = [
            "Share this Post",
            "About the company",
            "X platform X.com",
            "© 2023 X Corp",
            "Did someone say â¦ cookies?",
            "Cookies MStV Transparenzangaben",
            "How To Contact Us"
        ]
        
        for marker in footer_markers:
            if marker in text:
                text = text.split(marker)[0]
                break
        
        # Remove navigation elements
        nav_patterns = [
            "Skip to main content",
            "Help Center",
            "Contact our support team",
            "Verify your account",
            "Access your account",
            "FAQ"
        ]
        
        for pattern in nav_patterns:
            text = text.replace(pattern, "")
        
        # Remove duplicate content that appears after cleaning
        # (Sometimes content repeats in different sections)
        paragraphs = text.split('\n\n')
        unique_paragraphs = []
        
        for p in paragraphs:
            if p and p not in unique_paragraphs:
                unique_paragraphs.append(p)
        
        text = '\n\n'.join(unique_paragraphs)
        
        # Remove excessive whitespace
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Processes documents by cleaning text and splitting into optimally sized chunks.
        
        What this does:
        1. Initializes an empty list for processed documents
        2. For each input document:
           a. Cleans the document text using the clean_text method
           b. Creates a new Document with cleaned text and original metadata
           c. Splits the document into chunks using the text_splitter
           d. Filters out chunks smaller than min_chunk_size
           e. Adds valid chunks to the result list
        
        Args:
            documents (List[Document]): List of Document objects to process.
            
        Returns:
            List[Document]: List of processed Document objects split into chunks.
                           Chunks smaller than min_chunk_size are filtered out.
        """
        processed_docs = []
        
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            cleaned_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
            chunks = self.text_splitter.split_documents([cleaned_doc])
            
            # Filter out chunks that are too small
            valid_chunks = [chunk for chunk in chunks if len(chunk.page_content) >= self.min_chunk_size]
            processed_docs.extend(valid_chunks)
            
        return processed_docs
    
    def load_and_process(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Performs end-to-end document loading and processing in a single operation.
        
        What this does:
        1. Calls load_documents with the provided file pattern
        2. Passes the loaded documents to process_documents
        3. Returns the processed document chunks
        
        This method chains the two main operations (loading and processing) for convenience.
        
        Args:
            file_pattern (str, optional): Glob pattern to match files. Defaults to "*.txt".
            
        Returns:
            List[Document]: List of processed Document objects ready for indexing.
                           Returns processed chunks from all successfully loaded documents.
        """
        documents = self.load_documents(file_pattern)
        return self.process_documents(documents)