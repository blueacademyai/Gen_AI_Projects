from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGChain:
    """
    Implements a Retrieval-Augmented Generation (RAG) chain for technical support queries.
    
    This class builds a processing chain that combines retrieved technical documentation 
    and incident records with a language model (using gpt-4o) to generate comprehensive 
    solution guides for technical support queries.
    
    The chain uses a specialized prompt template that can prioritize solution steps from
    technical documentation or incident records based on availability and relevance.
    The resulting output is formatted as a clear, step-by-step solution guide tailored
    to the user's specific question.
    """
    def __init__(self, template_name="tech_incident_template", model="gpt-4o"):
        """
    Initialize a RAG (Retrieval-Augmented Generation) chain for technical support.
    
    This constructor creates a complete processing pipeline for answering technical
    support queries by combining retrieved documentation with a language model.
    It performs the following setup operations:
    
    1. Stores the template name for reference and potential customization
    2. Defines a comprehensive prompt template that guides the model on how to
    process technical documentation and incident records
    3. Creates a ChatPromptTemplate from the text template for structured prompting
    4. Initializes the ChatOpenAI language model with the specified model name
    and zero temperature for deterministic responses
    5. Sets up a string output parser to format the final responses
    6. Constructs a runnable chain that:
    - Takes retrieved context from both technical docs and incident records
    - Inserts this context into the prompt template
    - Passes the populated prompt to the language model
    - Parses the output into a final response
    
    Args:
        template_name (str, optional): Name identifier for the prompt template.
            Used for reference and potential template switching. 
            Defaults to "tech_incident_template".
        model (str, optional): Identifier for the OpenAI model to use.
            Specifies which language model will generate responses.
            Defaults to "gpt-4o".
    
    Attributes:
        template_name (str): Name of the prompt template
        template (str): Raw text template with placeholders for context
        prompt (ChatPromptTemplate): Structured prompt template object
        llm (ChatOpenAI): Language model interface
        output_parser (StrOutputParser): Parser for processing model outputs
        chain (RunnableSequence): Complete processing pipeline
    
    Raises:
        Exception: Re-raises any exceptions that occur during initialization, including:
            - Template formatting errors
            - API authentication issues
            - Model availability problems
            - Chain construction failures
            
    prompt_template = Analyze the provided technical documentation and incident records to address the user's question.

                    Question: {query}

                    Technical Context: {Technical_context}

                    Incident Records: {Incident_records}

                    Output Instructions:
                    1. If Both Technical context and Incident records contains steps, use those steps to create a solution guide.
                    2. If technical context contains steps, use those steps to create a solution guide.
                    3. If Incident Records context contains steps, return the Incident Records.

                    NOTE: If query is not related to networking or Technical context or Incident records or query is too short or half, 
                        then do not generate guide. say no related document found related to the product.
                   

                    Response Format:
                    - Provide clear, actionable steps
                    - Label the source of steps (Technical Context or Incident Records)
        """
        try:
            self.template_name = template_name
            
            # Template to process technical docs and incident records
            self.template = """Analyze the provided technical documentation and incident records to address the user's question.
 
                        Question: {query}
 
                        Technical Context: {Technical_context}
 
                        Incident Records: {Incident_records}
 
                        Output Instructions:
                        1. If Both Technical context and Incident records contains steps, use those steps to create a solution guide.
                        2. If technical context contains steps, use those steps to create a solution guide.
                        3. If Incident Records context contains steps, return the Incident Records.
 
                        NOTE: If query is not related to networking or Technical context or Incident records,
                            then do not generate guide. say no related document found related to the product.
                       
 
                        Response Format:
                        - Provide clear, actionable steps
                        - Label the source of steps (Technical Context or Incident Records)
                        """

            # Set up prompt template
            self.prompt = ChatPromptTemplate.from_template(self.template)
            
            # Initialize language model
            self.llm = ChatOpenAI(model=model, temperature=0)
            
            # Set up output parser
            self.output_parser = StrOutputParser()
            
            # Define the chain
            self.chain = (
                RunnablePassthrough.assign(
                    Technical_context=lambda x: x["tech_results"],
                    Incident_records=lambda x: x["incident_results"]
                )
                | self.prompt
                | self.llm
                | self.output_parser
            )

        except Exception as e:
            raise e

    def run(self, query, tech_results, incident_results):
        """
        Generate a solution guide based on technical documentation and incident records.
        
        This method processes the user's technical question along with retrieved 
        contextual documents (both technical and incident-related) through the language 
        model chain to create a comprehensive solution.
        
        Args:
            query (str): The user's technical question
            tech_results (list or str): Retrieved technical documentation
            incident_results (list or str): Retrieved incident records
            
        Returns:
            str: Formatted solution guide with step-by-step instructions
                If the query is unrelated to networking, technical context,
                or incident records, or if the query is too short,
                returns a message indicating no relevant documents were found.
        
        Raises:
            ValueError: If query is not a string
                Raises with message: "Query must be a string"
            ValueError: If query is empty
                Raises with message: "Query cannot be empty"
            Exception: Re-raises any other exceptions that occur during processing
                This includes errors in chain invocation or response formatting
        """
        try:
            
            # Input validation
            if not isinstance(query, str):
                raise ValueError("Query must be a string")
            
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Check if we have any relevant documents
            has_tech_results = tech_results and not (isinstance(tech_results, str) and "No related document found" in tech_results)
            has_incident_results = incident_results and not (isinstance(incident_results, str) and "No related document found" in incident_results)
            
            # If no relevant documents found, provide fallback response
            if not has_tech_results and not has_incident_results:
                return self._generate_fallback_response(query)
            
            # Prepare input for the chain
            input_dict = {
                "query": query,
                "tech_results": tech_results,
                "incident_results": incident_results
            }
            
            # Invoke the chain
            response = self.chain.invoke(input_dict)
            
            return response

        except Exception as e:
            raise e
    
    def _generate_fallback_response(self, query):
        """
        Generate a fallback response when no relevant documents are found.
        
        This method suggests resources user might consult for further assistance
        when no specific documentation or incident records match the user's request.
        
        Args:
            query (str): The user's technical question
            
        Returns:
            str: Fallback response with general advice related to the query domain
                ("### No Relevant Documentation Found\n\n"
                "Sorry, we couldn't find specific documentation for your query. "
                "Please try rephrasing your question or contact technical support for assistance.")
            
        fallback_template = "
        No specific documentation was found for the user's query about networking equipment.
        Please Suggests resources they might consult for further assistance for that specific product.
        
        User query: {query}
        
        Generate a helpful response that:
        1. Says that specific documentation for their exact issue wasn't found
        2. Suggests resources they might consult for further assistance
        
        "
        """
        # Create a prompt for generating generic advice
        fallback_template = """
        No specific documentation was found for the user's query about networking equipment.
        Please Suggests resources they might consult for further assistance for that specific product.
        
        User query: {query}
        
        Generate a helpful response that:
        1. Says that specific documentation for their exact issue wasn't found
        2. Suggests resources they might consult for further assistance
        
        """
        
        # Create a simple chain for the fallback
        fallback_prompt = ChatPromptTemplate.from_template(fallback_template)
        fallback_chain = fallback_prompt | self.llm | self.output_parser
        
        # Generate fallback response
        try:
            fallback_response = fallback_chain.invoke({"query": query})
            return "### No Specific Documentation Found\n\n" + fallback_response
        except Exception:
            # If even the fallback fails, return a simple message
            return (
                "### No Relevant Documentation Found\n\n"
                "Sorry, we couldn't find specific documentation for your query. "
                "Please try rephrasing your question or contact technical support for assistance."
            )