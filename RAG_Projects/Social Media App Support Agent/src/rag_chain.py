"""
RAG Chain for Social Media App Support Agent

This module implements a streamlined Retrieval-Augmented Generation (RAG) chain for the Social Media App support chatbot. It combines document retrieval with LLM generation
to create accurate, context-aware responses to user queries using LangChain Expression Language (LCEL).
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Default prompt template
default_prompt_template = """
You are a helpful customer support agent for X (formerly known as Twitter), a social media platform. 
Your goal is to provide accurate, concise, and helpful responses to user queries based on X's official documentation.

Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep your answers helpful, accurate, and to the point. Provide step-by-step instructions when appropriate.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:
"""

class SocialMediaRAGChain:
    """
    A streamlined class implementing RAG (Retrieval-Augmented Generation) for social media support using LCEL.
    
    This class creates a complete RAG pipeline that enhances LLM responses with retrieved knowledge.
    It retrieves relevant documents from a vector store based on user queries, formats them as context,
    and generates accurate, contextually-informed responses for social media platform support questions.
    
    Attributes:
        vectorstore (VectorStore): Vector database for document retrieval
        llm (BaseLanguageModel): Language model for response generation
        k (int): Number of documents to retrieve per query
        return_source_documents (bool): Whether to include source documents in response
        prompt (PromptTemplate): Template used for response generation
        similarity_threshold (float): Minimum relevance score for retrieved documents
        retriever: Document retrieval component from vectorstore
        chain: The assembled LCEL pipeline for end-to-end RAG
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        llm: Optional[BaseLanguageModel] = None,
        temperature: float = 0.0,
        k: int = 4,
        return_source_documents: bool = True,
        prompt_template: str = default_prompt_template,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the RAG chain with LCEL.
        
        What this does:
        1. Stores the vector store for document retrieval
        2. Sets up the language model (defaults to ChatOpenAI with gpt-4o)
        3. Configures RAG parameters (k, return_source_documents, similarity_threshold)
        4. Creates a PromptTemplate from the provided template
        5. Initializes the retriever from the vector store with search parameters
        6. Calls _create_chain() to assemble the LCEL pipeline
        
        Args:
            vectorstore (VectorStore): The vector store for document retrieval
            llm (Optional[BaseLanguageModel]): The language model to use (defaults to ChatOpenAI)
            temperature (float): Temperature setting for the LLM (default: 0.0 for deterministic responses)
            k (int): Number of documents to retrieve from the vector store (default: 4)
            return_source_documents (bool): Whether to return source documents with responses (default: True)
            prompt_template (str): The template to use for generating responses (default: default_prompt_template)
            similarity_threshold (float): Threshold for document relevance (default: 0.8)
        """
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=temperature)
        self.k = k
        self.return_source_documents = return_source_documents
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.similarity_threshold = similarity_threshold
        
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        self._create_chain()
    
    def _create_chain(self):
        """
        Create the LCEL RAG chain.
        
        What this does:
        1. Defines a helper function 'format_docs' to convert document list to string format
        2. Assembles the LCEL pipeline with these components:
           a. A dictionary with context (from retriever + formatting) and passthrough question
           b. The prompt template to structure the input for the LLM
           c. The language model to generate a response
           d. A string output parser to extract the final text
        
        Note:
           This is an internal method called during initialization.
        """
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the chain with a user question.
        
        What this does:
        1. Uses similarity_search_with_score to get documents with relevance scores
        2. Filters documents based on the similarity threshold
        3. If no documents meet the threshold, returns a message indicating uncertainty
        4. Otherwise, invokes the LCEL chain with the question to generate an answer
        5. Extracts source information from document metadata
        6. Returns a dictionary with the answer and optional source information
        
        Args:
            question (str): The user's question about social media platform usage
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "answer": The generated response text
                - "sources": List of unique source names
                - "source_documents": List of relevant Document objects (if return_source_documents is True)
                - "error": Error message (if an exception occurs)
                
        Note:
            If an exception occurs during processing, returns an error message with the exception details.
        """
        try:
            
            # Apply similarity threshold filtering
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=question,
                k=self.k
            )
            
            # Apply the similarity threshold (lower score is better in FAISS)
            relevant_docs = [
                doc for doc, score in docs_with_scores 
                if score <= self.similarity_threshold
            ]
            
            # If no documents meet the threshold, return a response indicating uncertainty
            if not relevant_docs:
                return {
                    "answer": "I'm sorry, I don't have enough information to answer this question confidently.",
                    "source_documents": []
                }
                
            # Use the LCEL chain to generate a response
            answer = self.chain.invoke(question)
            
            # Extract source information for citation
            sources = []
            for doc in relevant_docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            result = {
                "answer": answer,
                "sources": sources
            }
            
            if self.return_source_documents:
                result["source_documents"] = relevant_docs
            
            return result
        except Exception as e:
            return {
                "answer": f"I'm sorry, I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "source_documents": []
            }
    
    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query without generating an answer.
        
        What this does:
        1. Uses the k parameter from the method call or falls back to the instance's k
        2. Calls the vectorstore's similarity_search method with the query and k
        3. Returns the list of retrieved documents without filtering by threshold
        
        This is useful for debugging or when only document retrieval is needed.
        
        Args:
            query (str): The query string to find relevant documents
            k (Optional[int]): Number of documents to retrieve (defaults to self.k)
            
        Returns:
            List[Document]: List of relevant Document objects from the vector store
        """
        k = k or self.k
        return self.vectorstore.similarity_search(query, k=k)