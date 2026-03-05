from typing import List, Dict, Any
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import asyncio
import logging

from .vector_store import SupportVectorStore
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportRAGChain:
    """
    A class implementing the Retrieval-Augmented Generation (RAG) chain for support tickets.
    
    This class combines vector similarity search with LLM-based generation to provide
    relevant and contextual responses to support queries.
    
    IMPORTANT:
    - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
    - Queries shorter than 10 characters MUST be rejected with the EXACT error message: 
      "Query too short. Please provide more details."
    - Context preparation MUST follow the exact format specified in _prepare_context
    """
    
    def __init__(self, vector_store: SupportVectorStore):
        """
        Initialize the RAG chain with a vector store and LLM.
        
        Args:
            vector_store (SupportVectorStore): Vector store containing support tickets
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            openai_api_key=openai_api
        )
        
        # Define the RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a support specialist. Review the provided document carefully and extract a step-by-step guide point wise to solve the user's problem. Present these instructions in a clear, sequential format that's easy to follow. Focus only on the solution steps found in the context.
            Do not answer outside of the context.
            
            Context:
            {context}
            """),
            ("user", "{query}")
        ])

    def get_relevant_documents(
        self, 
        query: str, 
        support_type: str = None, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant support tickets for a given query.
        
        IMPORTANT:
        - Empty queries or queries shorter than 10 characters MUST be rejected with ValueError
        - The exact error message should be: "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            k (int): Number of documents to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata
            
        Raises:
            ValueError: If query is empty or too short (less than 10 characters)
        """
        # Input validation
        if not query or len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")
            
        logger.info(f"Retrieving documents for query: {query}")
        return self.vector_store.query_similar(query, support_type, k)

    def prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare retrieved documents into a formatted context string.
        
        IMPORTANT:
        - The context MUST be formatted with the EXACT format shown below
        - Each document must include: Support Type, Tags, and Content
        - When no documents are found, return "No relevant support tickets found."
        
        Args:
            documents (List[Dict[str, Any]]): Retrieved similar documents
            
        Returns:
            str: Formatted context string with the exact format:
            
            Ticket {i}:
            Support Type: subject_type
            Tags: tags
            answer: answer
            Content: content of document
        """
        if not documents:
            return "No relevant support tickets found."
        
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"""
            Ticket {i}:
            Support Type: {doc['metadata'].get('support_type', 'Unknown')}
            Tags: {', '.join(doc['metadata'].get('tags', []))}
            Answer: {''.join(doc['metadata'].get('answer', []))}
            Content: {doc['content']}
            """)
        
        return "\n".join(context_parts)

    async def query(
        self, 
        query: str, 
        support_type: str = None
    ) -> str:
        """
        Generate a response to a support query using RAG.
        
        IMPORTANT:
        - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries with only whitespace MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries shorter than 10 characters MUST be rejected with the EXACT error message:
          "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            
        Returns:
            str: Generated response based on relevant support tickets
            
        Raises:
            ValueError: With message "Query cannot be empty" if query is empty or whitespace only
            ValueError: With message "Query too short. Please provide more details." if query is shorter than 10 chars
            Exception: If there's an error generating the response
        """
        try:
            # Input validation
            if not query:
                # return "Query cannot be empty"
                raise ValueError("Query cannot be empty")
            if query.strip() == "":
                raise ValueError("Query cannot be empty")
            if len(query.strip()) < 10:
                # retur
                raise ValueError("Query too short. Please provide more details.")

            # Get relevant documents
            documents = self.get_relevant_documents(query, support_type)
            
            # Prepare context from documents
            context = self.prepare_context(documents)
        
            # Generate response using LLM
            chain = self.prompt_template | self.llm
            # from langchain.schema.runnable import RunnableLambda
            # chain = self.prompt_template | RunnableLambda(self.llm.ainvoke)
            
            response = await chain.ainvoke({
                "context": context,
                "query": query
            })
            
            logger.info(f"Generated response for query: {query}")
            logger.debug(f"Response content: {response.content}")
        
            return response.content
            
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise