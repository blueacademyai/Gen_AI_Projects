import pandas as pd
import numpy as np
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

class RAGProcessor:
    """
    A simplified class for processing and managing service tickets using RAG.
    
    This class combines vector-based similarity search across active and historical
    ticket databases to provide intelligent ticket resolution time estimation.
    """
    
    def __init__(self, metadata_path: str, description_path: str, active_vectorstore, history_vectorstore, openai_api_key: str):
        """
        Initialize the RAGProcessor with file paths, vector stores, and OpenAI API key.
        
        Parameters:
            metadata_path (str): Path to the metadata CSV file.
            description_path (str): Path to the description CSV file.
            active_vectorstore: Vector store containing active ticket embeddings.
            history_vectorstore: Vector store containing historical ticket embeddings.
            openai_api_key (str): OpenAI API key for authentication.
        
        Notes:
            - Converts string paths to Path objects
            - Sets up a summarization chain using:
              - A prompt template for concise summaries preserving technical details
              - ChatOpenAI LLM instance using gpt-4o model with temperature=0
              - RunnablePassthrough to forward text directly to the chain
              - StrOutputParser to format the output as a string
        """
        self.metadata_path = Path(metadata_path)
        self.description_path = Path(description_path)
        self.active_vectorstore = active_vectorstore
        self.history_vectorstore = history_vectorstore

        prompt_template = """Write a concise summary of the following while preserving technical details:
        "{text}"
        CONCISE SUMMARY (In 2 lines): """
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)
        
        self.summarization_chain = (
            {"text": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
    
    def _validate_query(self, query: str):
        """
        Validate the query string.
        
        Parameters:
            query (str): The query string to validate.
        
        Raises:
            ValueError: If the query is empty, prints:
                       "Query cannot be empty"
            ValueError: If the query is too short, prints:
                       "Query too short. Please provide more details."
        
        Notes:
            - Checks if query is None, empty, or only whitespace
            - Checks if trimmed query length is less than 20 characters
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 20:
            raise ValueError("Query too short. Please provide more details.")
    
    def summarize_description(self, description: str) -> str:
        """
        Summarize a description if it exceeds 1500 characters.
        
        Parameters:
            description (str): The description text to potentially summarize.
        
        Returns:
            str: The summarized description if original is over 1500 characters,
                 otherwise returns the original description. Returns an empty string
                 if the input is not a string.
        
        Notes:
            - Checks if input is a string and longer than 1500 characters
            - Uses the pre-configured summarization_chain to create summaries
            - Strips whitespace from the generated summary
            - Falls back to the original description if summarization fails
            - Returns the original description if it's a string but not long enough to summarize
            - Returns an empty string if the input is not a string
        """
        if isinstance(description, str) and len(description) > 1500:
            try:
                summary = self.summarization_chain.invoke(description)
                return summary.strip()
            except:
                return description
        return description if isinstance(description, str) else ""

    def perform_similarity_search(self, description: str, vectorstore, location_id: int) -> tuple:
        """
        Perform a similarity search with proper location filtering.
        
        Parameters:
            description (str): The query description to search for.
            vectorstore: The vector store to search in.
            location_id (int): Location ID for filtering relevant tickets.
        
        Returns:
            tuple: A tuple containing:
                  - List of (document, score) tuples for matching documents
                  - None (placeholder for future extension)
        
        Notes:
            - Validates the query using _validate_query
            - Converts location_id to string for consistent comparison
            - Attempts three different search approaches in sequence:
              1. First approach: Location-filtered similarity search with k=15 and score < 0.99
                 Prints: "Found first filter {count} documents through similarity + location filtering"
              2. Second approach: Stricter similarity threshold (score < 0.8) without location filtering
                 Prints: "Found second {count}, {location_id-score_pairs}"
              3. Third approach: Broader search with k=20 and relaxed threshold (score < 0.99)
                 Prints: "Found third {count} documents by broader search"
            - Returns empty list if no matches found or on error
            - Prints specific error messages for each approach:
              - "Error in location-filtered search: {error}"
              - "Error in manual filtering: {error}"
              - "Error in broad search: {error}"
              - "Error in similarity search: {error}"
        """
        try:
            self._validate_query(description)
            str_location_id = str(location_id)  # Convert once for consistent comparison
            
            try:
                # First approach: Find documents similar to the query at the specific location
                matching_docs = vectorstore.similarity_search_with_score(
                    description,
                    k=15,
                )
                location_matches = [
                    (doc, score) for doc, score in matching_docs 
                    if doc.metadata.get('location_id') == str_location_id and score < 0.99
                ]
                if location_matches:
                    print(f"Found first filter {len(location_matches)} documents through similarity + location filtering")
                    return location_matches, None
            except Exception as e:
                print(f"Error in location-filtered search: {str(e)}")
            
            # Second approach: Search all docs with stricter similarity threshold
            try:
                all_docs = vectorstore.similarity_search_with_score(description, k=15)
                similar_docs = [
                    (doc, score) for doc, score in all_docs 
                    if score < 0.8  # Stricter similarity threshold
                ]
                if similar_docs and len(similar_docs) > 0:
                    print(f"Found second {len(similar_docs)},", [(doc.metadata.get('location_id'),score) for doc, score in similar_docs])
                    return similar_docs, None
            except Exception as e:
                print(f"Error in manual filtering: {str(e)}")
            
            # Third approach: Broaden search
            try:
                broader_docs = vectorstore.similarity_search_with_score(description, k=20)
                matching_docs = [
                    (doc, score) for doc, score in broader_docs 
                    if score < 0.99  # Relaxed similarity threshold
                ]
                if matching_docs and len(matching_docs) > 0:
                    print(f"Found third {len(matching_docs)} documents by broader search")
                    return matching_docs, None
            except Exception as e:
                print(f"Error in broad search: {str(e)}")
            
            return [], None
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return [], None

    def create_new_ticket(self, ticket_id: int, customer_id: int, location_id: int, description: str, estimated_time: int, clusterID: int) -> dict:
        """
        Create a new ticket dictionary with standard fields.
        
        Parameters:
            ticket_id (int): Unique identifier for the ticket.
            customer_id (int): ID of the customer who submitted the ticket.
            location_id (int): ID of the location associated with the ticket.
            description (str): Description of the issue.
            estimated_time (int): Estimated resolution time in hours.
            clusterID (int): ID of the cluster this ticket belongs to.
        
        Returns:
            dict: A dictionary containing all ticket information with standardized fields.
        
        Notes:
            - Validates the query description using _validate_query
            - Summarizes the description if it's too long
            - Returns a dictionary with the following keys:
              - TicketID
              - customerID
              - locationID
              - type (hardcoded to "complaint")
              - description (summarized)
              - clusterID
              - estimated_resolution_time
        """
        self._validate_query(description)
        
        summarized_description = self.summarize_description(description)  # Added summarization
        
        return {
            "TicketID": ticket_id,
            "customerID": customer_id,
            "locationID": location_id,
            "type": "complaint",
            "description": summarized_description,  # Using summarized description
            "clusterID": clusterID,
            "estimated_resolution_time": estimated_time
        }

    def append_to_csv_file(self, new_data: dict) -> None:
        """
        Append a new record to the active CSV files.
        
        Parameters:
            new_data (dict): Dictionary containing the new ticket data to append.
        
        Notes:
            - Reads the metadata CSV file using utf-8 encoding
            - Creates a metadata entry excluding the 'description' field
            - Appends the metadata entry to the metadata DataFrame
            - Writes the updated metadata DataFrame back to the CSV file
            - Reads the description CSV file using utf-8 encoding
            - Creates a description entry with only 'TicketID' and 'description' fields
            - Appends the description entry to the description DataFrame
            - Writes the updated description DataFrame back to the CSV file
            - Prints: "Data successfully added to {metadata_path} and {description_path}"
        """
        metadata_df = pd.read_csv(self.metadata_path, encoding='utf-8')
        
        metadata_entry = {k: v for k, v in new_data.items() if k != 'description'}
        metadata_df = pd.concat([metadata_df, pd.DataFrame([metadata_entry])], ignore_index=True)
        
        metadata_df.to_csv(self.metadata_path, index=False, encoding='utf-8')
        
        description_df = pd.read_csv(self.description_path, encoding='utf-8')
        
        description_entry = {
            "TicketID": new_data["TicketID"],
            "description": new_data["description"]
        }
        description_df = pd.concat([description_df, pd.DataFrame([description_entry])], ignore_index=True)
        
        description_df.to_csv(self.description_path, index=False, encoding='utf-8')
        
        print(f"Data successfully added to {self.metadata_path} and {self.description_path}")

    def load_active_data(self) -> pd.DataFrame:
        """
        Load current active data from metadata and description CSV files.
        
        Returns:
            pd.DataFrame: A merged DataFrame containing ticket metadata and descriptions.
        
        Notes:
            - Loads metadata and description DataFrames from CSV files using utf-8 encoding
            - Merges the DataFrames on 'TicketID' with a left join
            - Keeps only the 'TicketID' and 'description' columns from the description DataFrame
            - Fills any missing descriptions with empty string
            - Prints the first 5 rows of the merged DataFrame using head(5)
            - Returns the merged DataFrame
        """
        metadata_df = pd.read_csv(self.metadata_path, encoding='utf-8')
        description_df = pd.read_csv(self.description_path, encoding='utf-8')
        
        merged_df = pd.merge(
            metadata_df, 
            description_df[['TicketID', 'description']], 
            on='TicketID', 
            how='left'
        )
        
        merged_df['description'] = merged_df['description'].fillna("")
        print(merged_df.head(5))
        return merged_df

    def get_estimated_resolution_time(self, description: str, location_id: int) -> tuple:
        """
        Determine estimated resolution time using active and historical data.
        
        Parameters:
            description (str): Query description for the new ticket.
            location_id (int): Location ID for filtering relevant tickets.
                
        Returns:
            tuple: (ticket_type, new_ticket) where:
                  - ticket_type: String indicating the source of the match ('new_act_ticket', 
                    'new_hs_ticket', 'Not_valid_query', or 'error')
                  - new_ticket: Dictionary containing the new ticket information
        
        Notes:
            - Validates the query using _validate_query
            - Prints: "Processing query for location_id {location_id}: {first 100 chars of description}..."
            - Loads active data to determine next ticket ID
            - Searches for matches in active tickets first, then historical tickets
            - If active matches found:
              - Prints: "Found {count} matching active tickets for location {location_id}"
              - Calculates average resolution time from active tickets
              - clusterID = 5  # Default
              - Creates new ticket with type 'new_act_ticket'
            - If only historical matches found:
              - Prints: "Found {count} matching historical tickets for location {location_id}"
              - Calculates average resolution time from historical tickets
              - Creates new ticket with type 'new_hs_ticket'
            - If no matches found:
              - Prints: "No matching tickets found and query does not appear service-related"
              - Creates special "invalid query" ticket with type 'Not_valid_query'
            - On error:
              - Prints: "Error in get_estimated_resolution_time: {error}"
              - Creates error ticket with type 'error'
            - For invalid and error tickets, sets is_valid flag to False
        """
        try:
            self._validate_query(description)
            
            print(f"Processing query for location_id {location_id}: {description[:100]}...")
            
            active_data = self.load_active_data()
            
            active_docs, _ = self.perform_similarity_search(description, self.active_vectorstore, location_id)
            print("active_data", active_docs)

            history_docs, _ = self.perform_similarity_search(description, self.history_vectorstore, location_id)
            
            if active_docs and len(active_docs) > 0:
                print(f"Found {len(active_docs)} matching active tickets for location {location_id}")
                all_est_times = [float(doc.metadata.get('estimated_resolution_time', 24)) for doc, _ in active_docs]
                estimated_resolution_time = int(np.mean(all_est_times)) if all_est_times else 24
                
                ticket_id = int(active_data['TicketID'].max()) + 1 if not active_data.empty else 1
                customer_id = int(active_data['customerID'].max()) + 1 if not active_data.empty else 1
                clusterID = 5  # Default
                
                new_ticket = self.create_new_ticket(
                    ticket_id, customer_id, location_id, description, 
                    estimated_resolution_time, clusterID
                )
                return 'new_act_ticket', new_ticket
            elif history_docs and len(history_docs) > 0:
                print(f"Found {len(history_docs)} matching historical tickets for location {location_id}")
                # Calculate average resolution time from historical tickets
                all_est_times = [float(doc.metadata.get('estimated_resolution_time', 24)) for doc, _ in history_docs]
                estimated_resolution_time = int(np.mean(all_est_times)) if all_est_times else 24
                
                # Create a new ticket based on historical ticket matches
                ticket_id = int(active_data['TicketID'].max()) + 1 if not active_data.empty else 1
                customer_id = int(active_data['customerID'].max()) + 1 if not active_data.empty else 1
                clusterID = 5  # Default
                
                new_ticket = self.create_new_ticket(
                    ticket_id, customer_id, location_id, description, 
                    estimated_resolution_time, clusterID
                )
                return 'new_hs_ticket', new_ticket
            else:
                print('No matching tickets found and query does not appear service-related')
                
                ticket_id = int(active_data['TicketID'].max()) + 1 if not active_data.empty else 1
                customer_id = int(active_data['customerID'].max()) + 1 if not active_data.empty else 1
                
                invalid_ticket = {
                    "TicketID": ticket_id,
                    "customerID": customer_id,
                    "locationID": location_id,
                    "type": "invalid_query",
                    "description": "Please provide specific details about your technical issue. Our system is designed to address service-related inquiries only.",
                    "clusterID": 0,  # Special cluster ID for invalid queries
                    "estimated_resolution_time": 0,
                    "is_valid": False  # Flag to indicate this is not a valid ticket
                }
                
                return 'Not_valid_query', invalid_ticket
        except Exception as e:
            print(f"Error in get_estimated_resolution_time: {str(e)}")
            error_ticket = {
                "TicketID": 0,
                "customerID": 0,
                "locationID": location_id if isinstance(location_id, int) else 0,
                "type": "error",
                "description": f"An error occurred: {str(e)}",
                "clusterID": 0,
                "estimated_resolution_time": 0,
                "is_valid": False,
                "error_message": str(e)
            }
            
            return 'error', error_ticket