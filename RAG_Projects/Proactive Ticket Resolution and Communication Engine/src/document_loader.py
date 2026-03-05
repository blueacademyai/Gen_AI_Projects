import pandas as pd
import warnings  # Add this import to suppress deprecation warnings
from langchain_core.documents import Document
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

class DocumentLoader:
    """
    A class for loading, processing, and summarizing CSV document data.
    
    This class handles loading ticket data from CSV files, summarizing long descriptions using an LLM, and converting the data into Document objects for use in vector stores.
    
    It provides functionality to:
    1. Load and merge CSV data from metadata and description files
    2. Summarize lengthy descriptions using OpenAI's LLM
    3. Convert processed data into Document objects with appropriate metadata
    
    Attributes:
        metadata_path (Path): Path object pointing to the CSV file containing ticket metadata
        description_path (Path): Path object pointing to the CSV file containing ticket descriptions
        summarization_chain: LangChain LCEL pipeline for text summarization
    """
    def __init__(self, metadata_path: str, description_path: str, openai_api_key: str):
        """
        Initializes the class instance with metadata and description file paths, OpenAI API key,
        and sets up a summarization chain using LangChain components.
        
        Parameters:
            metadata_path (str): Path to the metadata file.
            description_path (str): Path to the description file.
            openai_api_key (str): OpenAI API key for authentication.
        
        Raises:
            FileNotFoundError: If metadata_path doesn't exist, prints:
                                "Metadata path {metadata_path} does not exist"
            FileNotFoundError: If description_path doesn't exist, prints:
                                "Description path {description_path} does not exist"
        
        Notes:
            - Converts string paths to Path objects
            - Validates existence of both files before proceeding
            - Creates a summarization chain with the following components:
                - A prompt template asking to write a concise summary
                - ChatOpenAI LLM instance using gpt-4o model with temperature=0
                - RunnablePassthrough to forward text directly to the chain
                - StrOutputParser to format the output as a string
        """

        self.metadata_path = Path(metadata_path)
        self.description_path = Path(description_path)
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata path {metadata_path} does not exist")
            
        if not self.description_path.exists():
            raise FileNotFoundError(f"Description path {description_path} does not exist")
        
        prompt_template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)
        
        self.summarization_chain = (
            {"text": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )

    def load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.
        
        Parameters:
            file_path (Path): Path object pointing to the CSV file to be loaded.
        
        Returns:
            pd.DataFrame: Pandas DataFrame containing the data from the CSV file.
        
        Notes:
            - Uses UTF-8 encoding when reading the file
            - The method directly returns the result of pd.read_csv
        """
        return pd.read_csv(file_path, encoding='utf-8')

    def load_csv_data(self) -> pd.DataFrame:
        """
        Loads and merges data from the metadata and description CSV files.

        Returns:
            pd.DataFrame: A merged DataFrame containing metadata with descriptions.

        Raises:
            ValueError: If 'TicketID' column is missing in metadata file, prints:
                        "Missing required column 'TicketID' in metadata file {self.metadata_path}"
            ValueError: If 'TicketID' or 'description' columns are missing in description file, prints:
                        "Missing required columns in description file {self.description_path}"

        Notes:
            - Calls self.load_csv_file() to load both CSV files
            - Performs a left join on 'TicketID' column to merge the DataFrames
            - Fills any missing description values with empty strings
            - Returns the merged DataFrame with all metadata columns and the description column
        """
        metadata_df = self.load_csv_file(self.metadata_path)
        descriptions_df = self.load_csv_file(self.description_path)
        
        # Check that required columns exist
        if 'TicketID' not in metadata_df.columns:
            raise ValueError(f"Missing required column 'TicketID' in metadata file {self.metadata_path}")
        if 'TicketID' not in descriptions_df.columns or 'description' not in descriptions_df.columns:
            raise ValueError(f"Missing required columns in description file {self.description_path}")
        
        merged_df = pd.merge(metadata_df, descriptions_df[['TicketID', 'description']], on='TicketID', how='left')
        
        merged_df['description'] = merged_df['description'].fillna("")
        
        return merged_df

    def summarize_description(self, description: str) -> str:
        """
        Summarize a ticket description if it exceeds a certain length.
        
        This method:
        1. Validates that the input is a valid string
        2. Checks if the description is empty
        3. Determines if the description needs summarization (>1500 characters)
        4. If needed, uses a summarization chain to create a shorter version
        5. Handles exceptions by falling back to truncation
        
        Args:
            description (str): The ticket description text to potentially summarize
                If not a string, an empty string will be returned
                If empty or whitespace only, an empty string will be returned
        
        Returns:
            str: The summarized or original description:
                - Empty string if input is not a string or is empty
                - Summarized text if input is >1500 characters and summarization succeeds
                - Original text if input is ≤1500 characters
                - Truncated text (first 1000 chars + "...") if summarization fails and text is >1000 chars
                - Original text if summarization fails and text is ≤1000 chars
        
        Raises:
            No exceptions are raised; all errors are caught and handled internally
            If an error occurs, prints "Error during description summarization: {error_message}"
        """
        try:
            if not isinstance(description, str):
                return ""
                
            if not description.strip():
                return ""
                
            if len(description) > 1500:
                summary = self.summarization_chain.invoke(description)
                return summary.strip()
            return description
        except Exception as e:
            print(f"Error during description summarization: {e}")
            # On error, return a truncated version of the original description
            return description[:1000] + "..." if len(description) > 1000 else description

    def prepare_documents(self, data: pd.DataFrame) -> list:
        """
        Converts DataFrame rows into Document objects with summarized descriptions and metadata.
        
        Parameters:
            data (pd.DataFrame): DataFrame containing ticket data with descriptions.
        
        Returns:
            list: A list of Document objects, each containing a summarized description 
                    and relevant metadata from the corresponding row.
        
        Notes:
            - Iterates through each row in the provided DataFrame
            - Gets the description field using row.get('description', '') with an empty string as default
            - Calls self.summarize_description() to potentially summarize long descriptions
            - Creates a Document object for each row with:
                - page_content: the summarized description text
                - metadata: a dictionary containing:
                    - ticket_id: the row's TicketID converted to string
                    - location_id: the row's locationID converted to string
                    - estimated_resolution_time: the row's estimated_resolution_time converted to string
            - Appends each Document to a list which is returned at the end
        """
        documents = []
        for _, row in data.iterrows():
            description = row.get('description', '')
            summarized_description = self.summarize_description(description)
            doc = Document(
                page_content=summarized_description,
                metadata={
                    'ticket_id': str(row['TicketID']),
                    'location_id': str(row['locationID']),
                    'estimated_resolution_time': str(row['estimated_resolution_time'])
                }
            )
            documents.append(doc)
        return documents

    def load_documents(self) -> list:
        """
        Loads CSV data and converts it into a list of Document objects.
        
        Returns:
            list: A list of Document objects created from the CSV data.
        
        Notes:
            - Calls self.load_csv_data() to load and merge the metadata and description CSVs
            - Passes the resulting DataFrame to self.prepare_documents() to convert rows to Document objects
            - Returns the list of Document objects directly from prepare_documents
        """
        data = self.load_csv_data()
        return self.prepare_documents(data)


class CombinedDocumentLoader:
    """
    A class that combines active and historic ticket data for processing.
    
    This class manages the loading and processing of both active and historic ticket data by combining metadata and description files for each category. It creates separate
    DocumentLoader instances for active and historic data, allowing them to be processed individually or together.
    
    The class provides methods to:
    1. Load only active ticket documents
    2. Load only historic ticket documents
    3. Load both sets of documents together
    
    Attributes:
        active_loader (DocumentLoader): DocumentLoader instance for active ticket data
        historic_loader (DocumentLoader): DocumentLoader instance for historic ticket data
    """
    
    def __init__(self, active_metadata_path: str, active_description_path: str, historic_metadata_path: str, historic_description_path: str, openai_api_key: str):
        """
        Initializes the class with both active and historic document loaders.
        
        Parameters:
            active_metadata_path (str): Path to the active tickets metadata CSV file.
            active_description_path (str): Path to the active tickets description CSV file.
            historic_metadata_path (str): Path to the historic tickets metadata CSV file.
            historic_description_path (str): Path to the historic tickets description CSV file.
            openai_api_key (str): OpenAI API key for authentication.
        
        Notes:
            - Creates two DocumentLoader instances:
                1. self.active_loader: For loading active ticket data
                2. self.historic_loader: For loading historic ticket data
            - Both loaders are initialized with their respective metadata and description paths
            - The same openai_api_key is passed to both loaders
        """
        # Create loaders for active and historic data
        self.active_loader = DocumentLoader(
            metadata_path=active_metadata_path,
            description_path=active_description_path,
            openai_api_key=openai_api_key
        )
        
        self.historic_loader = DocumentLoader(
            metadata_path=historic_metadata_path,
            description_path=historic_description_path,
            openai_api_key=openai_api_key
        )
    
    def load_active_documents(self) -> list:
        """
        Loads and returns active ticket documents.
        
        Returns:
            list: A list of Document objects created from the active ticket data.
        
        Notes:
            - Calls load_documents() method on the self.active_loader instance
            - Returns the list of Document objects directly
        """
        return self.active_loader.load_documents()
    
    def load_historic_documents(self) -> list:
        """
        Loads and returns historic ticket documents.
        
        Returns:
            list: A list of Document objects created from the historic ticket data.
        
        Notes:
            - Calls load_documents() method on the self.historic_loader instance
            - Returns the list of Document objects directly
        """
        return self.historic_loader.load_documents()
            
    def load_all_documents(self) -> tuple:
        """
        Loads and returns both active and historic ticket documents.
        
        Returns:
            tuple: A tuple containing two lists:
                    - First element: List of Document objects from active tickets
                    - Second element: List of Document objects from historic tickets
        
        Notes:
            - Calls self.load_active_documents() to retrieve active ticket documents
            - Calls self.load_historic_documents() to retrieve historic ticket documents
            - Returns both lists as a tuple in the order (active_docs, historic_docs)
        """
        active_docs = self.load_active_documents()
        historic_docs = self.load_historic_documents()
        return active_docs, historic_docs