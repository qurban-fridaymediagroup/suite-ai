import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone
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
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")

    def sanitize_id(self, id_str):
        """Sanitize ID to ensure it only contains ASCII characters."""
        normalized = unicodedata.normalize('NFKD', str(id_str)).encode('ascii', 'ignore').decode('ascii')
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', normalized).lower()
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
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

    def exact_search(self, query, column=None, top_k=5):
        """Search for exact matches in Pinecone using metadata filter."""
        try:
            # Prepare filter for column and exact value (case-insensitive)
            filter_dict = {"value": {"$eq": query}}
            if column:
                filter_dict["column"] = column

            # Query Pinecone with a dummy vector to use metadata filter
            result = self.index.query(
                vector=[0] * self.dimension,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return result.get('matches', [])
        except Exception as e:
            print(f"Error performing exact search: {e}")
            return []

    def semantic_search(self, query, column=None, top_k=5):
        """Perform semantic search in Pinecone."""
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []

        try:
            # Prepare filter for column if specified
            filter_dict = {"column": column} if column else None

            # Query Pinecone
            result = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return result.get('matches', [])
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

# -- Streamlit UI --
st.title("🔍 Semantic Search across NetSuite Columns in Pinecone")

# Cache-clearing button
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

try:
    # Initialize PineconeManager
    pinecone_manager = PineconeManager()

    # List of searchable columns
    searchable_columns = [
        "Classification/Brand/Cost Center/Class", "AccountType", "Subsidiary", "Department",
        "Location", "BudgetCategory", "Currency", "AcctNumber", "Account Name", "Customer", "Vendor"
    ]

    # Select column
    selected_column = st.selectbox("Select a column to search", ["All Columns"] + searchable_columns)

    # Input query
    query = st.text_input("Enter search query")

    if query:
        # Check for exact matches in Pinecone
        exact_matches = None
        if selected_column != "All Columns":
            exact_matches = pinecone_manager.exact_search(query, column=selected_column, top_k=10)
            if exact_matches:
                st.success(f"Found exact match for: {query}")
                # Convert exact matches to DataFrame
                exact_data = []
                for match in exact_matches:
                    metadata = match.get('metadata', {})
                    exact_data.append({
                        'Column': metadata.get('column', 'N/A'),
                        'Value': metadata.get('value', 'N/A'),
                        'Similarity': 1.0
                    })
                exact_df = pd.DataFrame(exact_data)
                st.dataframe(exact_df[['Column', 'Value', 'Similarity']])
        
        # Perform semantic search if no exact matches or searching all columns
        if not exact_matches or selected_column == "All Columns":
            column_filter = selected_column if selected_column != "All Columns" else None
            results = pinecone_manager.semantic_search(query, column=column_filter, top_k=10)
            
            if results:
                # Convert Pinecone results to DataFrame
                result_data = []
                for match in results:
                    metadata = match.get('metadata', {})
                    result_data.append({
                        'Column': metadata.get('column', 'N/A'),
                        'Value': metadata.get('value', 'N/A'),
                        'Similarity': match.get('score', 0.0)
                    })
                result_df = pd.DataFrame(result_data)
                
                if result_df.iloc[0]['Similarity'] >= 0.999:  # Treat very high similarity as near-exact
                    st.success(f"Found near-exact match for: {query}")
                else:
                    st.success(f"Top {len(result_df)} similar matches for: {query}")
                
                st.dataframe(result_df[['Column', 'Value', 'Similarity']])
            else:
                st.warning("No matches found in Pinecone.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")