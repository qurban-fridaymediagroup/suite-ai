import os
import pandas as pd
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
import unicodedata
import re
import time

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

class PineconeManager:
    def __init__(self):
        """Initialize the Pinecone client and connect to the index."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "suiteai-index")
        self.dimension = 1536  # Dimension for text-embedding-3-small

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)

        # Check if index exists, if not create it
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")

    def sanitize_id(self, id_str):
        """Sanitize ID to ensure it only contains ASCII characters."""
        # Normalize to decompose accented characters (e.g., ü -> u)
        normalized = unicodedata.normalize('NFKD', str(id_str)).encode('ascii', 'ignore').decode('ascii')
        # Replace invalid characters with underscores, keep alphanumeric and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', normalized).lower()
        # Remove multiple consecutive underscores and strip leading/trailing underscores
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        # Ensure ID is not empty
        if not sanitized:
            sanitized = "unknown"
        return sanitized

    def generate_embedding(self, text):
        """Generate embedding for a given text."""
        if not text or not isinstance(text, str):
            return None
        try:
            time.sleep(0.1)  # Avoid OpenAI rate limits
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for text '{text}': {e}")
            return None

    def batch_upsert(self, records, batch_size=100):
        """Batch upsert records to Pinecone in chunks."""
        success_count = 0
        vectors = []

        for i, (id, text, metadata) in enumerate(records):
            # Ensure ID is sanitized (redundancy for safety)
            sanitized_id = self.sanitize_id(id)
            embedding = self.generate_embedding(text)
            if embedding:
                if metadata is None:
                    metadata = {}
                metadata['text'] = text
                vectors.append((sanitized_id, embedding, metadata))
            else:
                print(f"Skipping record with ID {sanitized_id} due to embedding failure")

            # Upsert in batches
            if len(vectors) >= batch_size or i == len(records) - 1:
                try:
                    time.sleep(0.5)  # Avoid Pinecone rate limits
                    self.index.upsert(vectors=vectors)
                    success_count += len(vectors)
                    print(f"Upserted batch of {len(vectors)} records")
                    vectors = []
                except Exception as e:
                    print(f"Error upserting batch: {e}")
                    for vec_id, _, _ in vectors:
                        print(f"Problematic ID: {vec_id}")
                    vectors = []  # Clear vectors to avoid retrying failed batch

        return success_count

    def load_and_upsert_from_csv(self, csv_path, columns_to_process=None):
        """Load data from CSV and upsert to Pinecone."""
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV loaded with {len(df)} rows")

            # Use all columns if none specified
            if columns_to_process is None:
                columns_to_process = df.columns.tolist()

            results = {}
            for column in columns_to_process:
                if column in df.columns:
                    print(f"Processing column: {column}")
                    unique_values = df[column].dropna().astype(str).unique().tolist()
                    records = [
                        (self.sanitize_id(f"{column}_{value}"), value, {'column': column, 'value': value})
                        for value in unique_values
                    ]
                    success_count = self.batch_upsert(records)
                    results[column] = {
                        'total': len(unique_values),
                        'success': success_count
                    }
                    print(f"Processed {success_count}/{len(unique_values)} values for {column}")
                else:
                    print(f"Column {column} not found in CSV")
                    results[column] = {'total': 0, 'success': 0, 'error': 'Column not found'}

            return results
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return {'error': str(e)}

    def list_vectors(self, limit=10):
        """List vectors in the index for verification."""
        try:
            result = self.index.query(vector=[0]*self.dimension, top_k=limit, include_metadata=True)
            return result.get('matches', [])
        except Exception as e:
            print(f"Error listing vectors: {e}")
            return []

if __name__ == "__main__":
    
    # Initialize PineconeManager
    pinecone_manager = PineconeManager()

    # Path to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'Netsuite info all final data.csv')

    # Upsert all columns
    results = pinecone_manager.load_and_upsert_from_csv(csv_path)

    # Print results
    print("\nResults:")
    for column, stats in results.items():
        print(f"{column}: {stats}")

    print("\nVectors containing 'FMG':")
for vector in pinecone_manager.list_vectors_by_value("FMG"):
    print(f"ID: {vector['id']}, Metadata: {vector['metadata']}")
  