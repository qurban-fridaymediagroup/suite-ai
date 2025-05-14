import os
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import openai
import pinecone

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the embedding model
index_name = "text-embedding-3-small"

class PineconeManager:
    def __init__(self):
        """Initialize the Pinecone client and connect to the index."""
        # Get Pinecone API key from environment variables
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "suiteai-index")
        self.dimension = 1536  # Dimension of text-embedding-3-small embeddings

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)

        # Check if index exists, if not create it
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )

        # Connect to the index
        self.index = pinecone.Index(self.index_name)

    def generate_embedding(self, text):
        """Generate embedding for a given text."""
        if not text or not isinstance(text, str):
            return None

        try:
            # Generate embedding using OpenAI API
            response = openai.embeddings.create(
                model=index_name,
                input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def record_exists(self, id):
        """Check if a record with the given ID exists in Pinecone."""
        try:
            # Fetch the vector by ID
            result = self.index.fetch(ids=[id])
            # If the ID exists in the result, the record exists
            return id in result.get('vectors', {})
        except Exception as e:
            print(f"Error checking if record exists: {e}")
            return False

    def upsert_record(self, id, text, metadata=None):
        """
        Upsert a record to Pinecone.

        Args:
            id (str): Unique identifier for the record
            text (str): Text to generate embedding for
            metadata (dict, optional): Additional metadata for the record

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if record already exists
            if self.record_exists(id):
                print(f"Record with ID {id} already exists. Skipping.")
                return True

            # Generate embedding
            embedding = self.generate_embedding(text)
            if not embedding:
                print(f"Failed to generate embedding for text: {text}")
                return False

            # Prepare metadata if not provided
            if metadata is None:
                metadata = {}

            # Add text to metadata for easier retrieval
            metadata['text'] = text

            # Upsert the vector
            self.index.upsert(vectors=[(id, embedding, metadata)])
            print(f"Successfully upserted record with ID {id}")
            return True
        except Exception as e:
            print(f"Error upserting record: {e}")
            return False

    def search(self, query, top_k=10, filter=None):
        """
        Search for similar records in Pinecone.

        Args:
            query (str): Query text
            top_k (int, optional): Number of results to return. Defaults to 10.
            filter (dict, optional): Metadata filter. Defaults to None.

        Returns:
            list: List of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                print(f"Failed to generate embedding for query: {query}")
                return []

            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )

            return results.get('matches', [])
        except Exception as e:
            print(f"Error searching in Pinecone: {e}")
            return []

    def batch_upsert(self, records):
        """
        Batch upsert records to Pinecone.

        Args:
            records (list): List of tuples (id, text, metadata)

        Returns:
            int: Number of successfully upserted records
        """
        success_count = 0
        for record in records:
            id, text, metadata = record
            if self.upsert_record(id, text, metadata):
                success_count += 1

        return success_count

    def load_and_upsert_from_csv(self, csv_path, columns_to_process=None):
        """
        Load data from CSV and upsert to Pinecone.

        Args:
            csv_path (str): Path to the CSV file
            columns_to_process (list, optional): List of columns to process. 
                If None, processes all columns.

        Returns:
            dict: Dictionary with counts of records processed for each column
        """
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            print(f"CSV loaded successfully with {len(df)} rows")

            # If columns_to_process is not specified, use default columns
            if columns_to_process is None:
                columns_to_process = [
                    "Classification/Brand/Cost Center/Class", "AccountType", "Subsidiary", 
                    "Department", "Location", "BudgetCategory", "Currency", 
                    "AcctNumber", "Account Name", "Customer", "Vendor"
                ]

            results = {}

            # Process each column
            for column in columns_to_process:
                if column in df.columns:
                    print(f"Processing column: {column}")
                    # Get unique values for this column
                    unique_values = df[column].dropna().astype(str).unique().tolist()

                    # Prepare records for batch upsert
                    records = []
                    for value in unique_values:
                        # Create a unique ID for each record
                        id = f"{column}_{value}".replace(" ", "_").lower()
                        # Add metadata
                        metadata = {
                            'column': column,
                            'value': value
                        }
                        records.append((id, value, metadata))

                    # Batch upsert records
                    success_count = self.batch_upsert(records)
                    results[column] = {
                        'total': len(unique_values),
                        'success': success_count
                    }
                    print(f"  Processed {success_count}/{len(unique_values)} unique values for {column}")
                else:
                    print(f"Column {column} not found in CSV")
                    results[column] = {
                        'total': 0,
                        'success': 0,
                        'error': 'Column not found'
                    }

            return results
        except Exception as e:
            print(f"Error loading and upserting from CSV: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize the PineconeManager
    pinecone_manager = PineconeManager()

    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'Netsuite info all final data.csv')

    # Load and upsert data from CSV
    results = pinecone_manager.load_and_upsert_from_csv(csv_path)

    # Print results
    print("\nResults:")
    for column, stats in results.items():
        print(f"{column}: {stats}")

    # Example search
    query = "marketing department"
    print(f"\nSearching for: {query}")
    search_results = pinecone_manager.search(query, top_k=5)

    # Print search results
    print("\nSearch Results:")
    for i, result in enumerate(search_results):
        print(f"{i+1}. ID: {result['id']}")
        print(f"   Score: {result['score']}")
        print(f"   Metadata: {result['metadata']}")
        print()
