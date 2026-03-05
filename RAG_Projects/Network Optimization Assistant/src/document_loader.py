import pandas as pd
import os
from langchain_community.document_loaders import CSVLoader  

class Guide_DocumentLoader:

    def __init__(self,
                 tech_src_path="data/src_tech_records.csv",
                 tech_meta_path="data/metadata_tech_records.csv",
                 incident_src_path="data/src_incident_records.csv",
                 incident_meta_path="data/metadata_incident_records.csv"):
        """
        Initialize the document loader with paths to technical documentation and incident record CSV files.
        
        This constructor sets up the infrastructure for loading and processing two distinct
        types of documents:
        
        1. Technical guides: Product manuals, technical documentation, and solution guides
        2. Incident records: Customer support tickets, problem reports, and resolution histories
        
        It configures the file paths for both content and metadata CSV files, defines the
        column structures for each document type, creates temporary file paths for processed
        data, and validates that all required files exist.
        
        Args:
            tech_src_path (str, optional): Path to the CSV file containing the content
                of technical documentation. Defaults to "data/src_tech_records.csv".
            tech_meta_path (str, optional): Path to the CSV file containing metadata
                for technical documentation, such as product IDs, tags, and document types.
                Defaults to "data/metadata_tech_records.csv".
            incident_src_path (str, optional): Path to the CSV file containing the content
                of customer support tickets and incident reports.
                Defaults to "data/src_incident_records.csv".
            incident_meta_path (str, optional): Path to the CSV file containing metadata
                for incident records, including customer IDs, product IDs, and timestamps.
                Defaults to "data/metadata_incident_records.csv".
        
        Attributes:
            tech_src_path (str): Path to technical content CSV
            tech_meta_path (str): Path to technical metadata CSV
            incident_src_path (str): Path to incident content CSV
            incident_meta_path (str): Path to incident metadata CSV
            incident_content_col (str): Column name for incident content ('ProblemDescription')
            incident_metadata_cols (list): Column names for incident metadata fields = ['CustomerID', 'ProductID', 'ProductInfo', 'SolutionDetails',
                                         'Status', 'Tags', 'Timestamp', 'DocID']
            tech_content_col (str): Column name for technical content ('step_description')
            tech_metadata_cols (list): Column names for technical metadata fields = ['ProductID', 'ProductInformation', 'SolutionSteps',
                                     'TechnicalTags', 'DocumentType']
            temp_incident_path (str): Temporary file path for processed incident data
            temp_tech_path (str): Temporary file path for processed technical data
            
        Raises:
            FileNotFoundError: If any of the required CSV files cannot be found at the
                specified paths. Raises with message: "Required file not found: {path}"
            Exception: Re-raises any other exceptions that occur during initialization,
                such as permission errors or file system issues
        
        Note:
            This method only validates file existence and sets up the configuration.
            Actual file loading and processing is performed by other methods.
        """
        try:
            # Store file paths
            self.tech_src_path = tech_src_path
            self.tech_meta_path = tech_meta_path
            self.incident_src_path = incident_src_path
            self.incident_meta_path = incident_meta_path

            # Define content and metadata column names
            self.incident_content_col = 'ProblemDescription'
            self.incident_metadata_cols = ['CustomerID', 'ProductID', 'ProductInformation', 'SolutionDetails',
                                         'Status', 'Tags', 'Timestamp', 'DocID']
            self.tech_content_col = 'step_description'
            self.tech_metadata_cols = ['ProductID', 'ProductInformation', 'SolutionSteps',
                                     'TechnicalTags', 'DocumentType']

            # Define temp file paths for processed data
            self.temp_incident_path = "data/filtered_temp_incident_record.csv"
            self.temp_tech_path = "data/filtered_temp_tech_doc.csv"

            # Validate file existence
            for path in [self.tech_src_path, self.tech_meta_path, self.incident_src_path, self.incident_meta_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required file not found: {path}")

        except Exception as e:
            raise e

    def load_incident_documents(self):
        """
        Load and prepare customer support ticket documents from CSV files for retrieval.
        
        This method performs a complete ETL (Extract, Transform, Load) process for incident records by:
        
        1. Extracting data from two separate CSV files:
        - Source file containing problem descriptions (incident content)
        - Metadata file containing supplementary information like customer and product details
        2. Transforming the data:
        - Validating that both files have matching row counts
        - Concatenating the dataframes horizontally
        - Selecting only the required columns
        - Creating a temporary combined CSV file
        3. Loading the data:
        - Using CSVLoader to convert the data into Document objects
        - Adding metadata fields to each Document for retrieval and filtering
        
        The method creates Document objects where:
        - The page_content field contains the problem description
        - The metadata field contains customer ID, product ID, solution details, etc.
        
        Returns:
            list: A list of Document objects, each representing a customer support ticket
                with problem description as content and all metadata fields attached.
        
        Raises:
            ValueError: If the number of rows in the source and metadata files doesn't match.
                Raises with message: "Row count mismatch: {len(incident_src)} in source vs {len(incident_meta)} in metadata"
            FileNotFoundError: If the CSV files cannot be accessed during processing
            pandas.errors.ParserError: If the CSV files are malformed or corrupt
            PermissionError: If there are file system permission issues
            Exception: Re-raises any other exceptions that occur during processing
        
        Note:
            This method creates a temporary file at the path specified by self.temp_incident_path
            and always attempts to clean up this file in the finally block, even if an exception occurs.
            Any exceptions during the cleanup process are suppressed to ensure they don't mask
            the original exception.
        """
        try:
            # Load source data
            incident_src = pd.read_csv(self.incident_src_path)

            # Load metadata
            incident_meta = pd.read_csv(self.incident_meta_path)

            # Validate row counts match
            if len(incident_src) != len(incident_meta):
                raise ValueError(f"Row count mismatch: {len(incident_src)} in source vs {len(incident_meta)} in metadata")

            # Concatenate dataframes horizontally
            incident_records = pd.concat([incident_src, incident_meta], axis=1)

            # Select required columns
            all_cols = self.incident_metadata_cols + [self.incident_content_col]
            incident_records = incident_records[all_cols]

            # Create temporary directory if needed
            os.makedirs(os.path.dirname(self.temp_incident_path), exist_ok=True)

            # Save to temporary file
            incident_records.to_csv(self.temp_incident_path, index=False)

            # Load documents using CSVLoader
            incident_loader = CSVLoader(
                file_path=self.temp_incident_path,
                source_column=self.incident_content_col,
                metadata_columns=self.incident_metadata_cols
            )

            documents = incident_loader.load()

            return documents

        except Exception as e:
            raise e
        finally:
            # Clean up temporary file
            if os.path.exists(self.temp_incident_path):
                try:
                    os.remove(self.temp_incident_path)
                except Exception as e:
                    raise e

    def load_tech_documents(self):
        """
        Load and prepare technical documentation from CSV files for retrieval.
        
        This method performs a complete ETL (Extract, Transform, Load) process for technical documentation by:
        
        1. Extracting data from two separate CSV files:
        - Source file containing step descriptions and technical instructions
        - Metadata file containing supplementary information like product IDs and document types
        2. Transforming the data:
        - Validating that both files have matching row counts
        - Concatenating the dataframes horizontally
        - Selecting only the required columns
        - Creating a temporary combined CSV file
        3. Loading the data:
        - Using CSVLoader to convert the data into Document objects
        - Adding metadata fields to each Document for retrieval and filtering
        
        The method creates Document objects where:
        - The page_content field contains the technical step descriptions
        - The metadata field contains product ID, product information, document type, etc.
        
        Returns:
            list: A list of Document objects, each representing a technical guide entry
                with step descriptions as content and all metadata fields attached.
        
        Raises:
            ValueError: If the number of rows in the source and metadata files doesn't match.
                Raises with message: "Row count mismatch: {len(tech_src)} in source vs {len(tech_meta)} in metadata"
            FileNotFoundError: If the CSV files cannot be accessed during processing
            pandas.errors.ParserError: If the CSV files are malformed or corrupt
            PermissionError: If there are file system permission issues
            Exception: Re-raises any other exceptions that occur during processing
        
        Note:
            This method creates a temporary file at the path specified by self.temp_tech_path
            and always attempts to clean up this file in the finally block, even if an exception occurs.
            Any exceptions during the cleanup process are suppressed to ensure they don't mask
            the original exception.
        """
        try:
            # Load source data
            tech_src = pd.read_csv(self.tech_src_path)

            # Load metadata
            tech_meta = pd.read_csv(self.tech_meta_path)

            # Validate row counts match
            if len(tech_src) != len(tech_meta):
                raise ValueError(f"Row count mismatch: {len(tech_src)} in source vs {len(tech_meta)} in metadata")

            # Concatenate dataframes horizontally
            tech_docs = pd.concat([tech_src, tech_meta], axis=1)

            # Select required columns
            all_cols = self.tech_metadata_cols + [self.tech_content_col]
            tech_docs = tech_docs[all_cols]

            # Create temporary directory if needed
            os.makedirs(os.path.dirname(self.temp_tech_path), exist_ok=True)

            # Save to temporary file
            # Note: temp_tech_path is being tested in unit tests to verify correct file handling
            # This path is critical for proper ETL processing and document loading
            tech_docs.to_csv(self.temp_tech_path, index=False)

            # Load documents using CSVLoader
            tech_loader = CSVLoader(
                file_path=self.temp_tech_path,
                source_column=self.tech_content_col,
                metadata_columns=self.tech_metadata_cols
            )

            documents = tech_loader.load()

            return documents

        except Exception as e:
            raise e
        finally:
            # Clean up temporary file
            if os.path.exists(self.temp_tech_path):
                try:
                    os.remove(self.temp_tech_path)
                except Exception as e:
                    raise e

    def load_all_documents(self):
        """
        Loads both customer support tickets and technical guides in one go.

        This function is a shortcut to load both types of documents—customer support tickets and technical guides—at the same time. It calls the separate loading functions for each type and returns both sets of documents, making it easier to set up the system with all data at once.

        Returns:
            tuple: Two lists—(incident_documents, technical_documents). The first has customer support ticket documents, and the second has technical guide documents.

        Raises:
            Exception: Re-raises any exceptions from load_incident_documents() or load_tech_documents()
            This includes all errors from those methods such as FileNotFoundError, ValueError, etc.
    
        """
        incident_docs = self.load_incident_documents()
        tech_docs = self.load_tech_documents()

        return incident_docs, tech_docs
