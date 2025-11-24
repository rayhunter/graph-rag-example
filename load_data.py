"""
This script loads, processes, and visualizes documents from a list of URLs.
It includes functions for fetching URLs, cleaning and preprocessing documents,
and adding them to a graph vector store.
"""
import os
import json
import warnings
import cassio
import tempfile
from pathlib import Path
from google.cloud import storage

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    AsyncHtmlLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    HtmlLinkExtractor,
    KeybertLinkExtractor,
    GLiNERLinkExtractor,
)
from langchain_community.document_transformers import BeautifulSoupTransformer

from util.config import LOGGER, OPENAI_API_KEY, ASTRA_DB_DATABASE_ID, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ENDPOINT, MOVIE_NODE_TABLE
from util.scrub import clean_and_preprocess_documents
from util.visualization import visualize_graph_text
from dotenv import load_dotenv
load_dotenv()

# Suppress all of the Langchain beta and other warnings
warnings.filterwarnings("ignore", lineno=0)

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Add debug logging before initialization
print("Checking Astra DB credentials:")
print(f"Database ID: {ASTRA_DB_DATABASE_ID[:8]}..." if ASTRA_DB_DATABASE_ID else "None")
print(f"Token starts with: {ASTRA_DB_APPLICATION_TOKEN[:15]}..." if ASTRA_DB_APPLICATION_TOKEN else "None")
print(f"API Endpoint: {ASTRA_DB_ENDPOINT[:30]}..." if ASTRA_DB_ENDPOINT else "None")

if not ASTRA_DB_DATABASE_ID or not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ENDPOINT:
    raise ValueError("Missing required Astra DB credentials")


# Initialize Astra connection using Cassio
# Modern cassio (0.1.10+) uses Data API instead of old bundle approach
# Default keyspace for Astra DB is typically "default_keyspace"
cassio.init(
    database_id=ASTRA_DB_DATABASE_ID,
    token=ASTRA_DB_APPLICATION_TOKEN,
    keyspace="default_keyspace"  # Astra DB default keyspace
)

# Initialize the graph vector store
store = CassandraGraphVectorStore(
    embedding=embeddings,
    keyspace="default_keyspace",  # Must match cassio.init() keyspace
    table_name=MOVIE_NODE_TABLE
)


def download_gcs_files(bucket_name, folder_prefix, local_dir):
    """
    Downloads files from Google Cloud Storage bucket to local directory.
    
    Parameters:
    bucket_name (str): Name of the GCS bucket
    folder_prefix (str): Folder path in the bucket (e.g., 'documents/')
    local_dir (str): Local directory to download files to
    
    Returns:
    list: List of downloaded file paths
    """
    downloaded_files = []
    
    try:
        # Initialize GCS client (uses GOOGLE_APPLICATION_CREDENTIALS env var)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs in the folder
        blobs = bucket.list_blobs(prefix=folder_prefix)
        
        for blob in blobs:
            # Skip if it's a folder marker or empty
            if blob.name.endswith('/') or blob.size == 0:
                continue
                
            # Get file extension
            file_ext = Path(blob.name).suffix.lower()
            
            # Only process PDFs and Markdown files
            if file_ext not in ['.pdf', '.md', '.markdown']:
                print(f"Skipping {blob.name} (unsupported format)")
                continue
            
            # Create local file path
            local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
            
            # Download the file
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_file_path)
            downloaded_files.append(local_file_path)
            
        print(f"Downloaded {len(downloaded_files)} files from GCS")
        return downloaded_files
        
    except Exception as e:
        LOGGER.error(f"Error downloading from GCS: {e}")
        raise


def load_documents_from_files(file_paths):
    """
    Load documents from local file paths (PDFs and Markdown).
    
    Parameters:
    file_paths (list): List of file paths to load
    
    Returns:
    list: List of loaded documents
    """
    documents = []
    
    for file_path in file_paths:
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                print(f"Loading PDF: {os.path.basename(file_path)}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
            elif file_ext in ['.md', '.markdown']:
                print(f"Loading Markdown: {os.path.basename(file_path)}")
                # Use simple TextLoader for markdown files - no external dependencies
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                # Set source metadata
                for doc in docs:
                    doc.metadata['source_type'] = 'markdown'
                    doc.metadata['file_name'] = os.path.basename(file_path)
                documents.extend(docs)
                
        except Exception as e:
            LOGGER.error(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(documents)} document pages/sections")
    return documents


def main():
    """
    Main function to load, process, and visualize documents from Google Cloud Storage.

    This function downloads documents from GCS, transforms and cleans them,
    splits them into chunks, and adds them to a graph vector store.
    It also visualizes the documents as a text-based graph.
    """
    try:
        # Get GCS configuration from environment variables
        gcs_bucket = os.getenv("GCS_BUCKET_NAME")
        gcs_folder = os.getenv("GCS_FOLDER_PREFIX", "")  # Optional, defaults to bucket root
        
        if not gcs_bucket:
            raise ValueError("GCS_BUCKET_NAME environment variable is not set")
        
        print(f"Loading documents from GCS bucket: {gcs_bucket}/{gcs_folder}")
        
        # Create temporary directory for downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from GCS
            file_paths = download_gcs_files(gcs_bucket, gcs_folder, temp_dir)
            
            if not file_paths:
                raise ValueError("No PDF or Markdown files found in the specified GCS location")
            
            # Load documents from downloaded files
            documents = load_documents_from_files(file_paths)
            
            if not documents:
                raise ValueError("No documents could be loaded from the files")

            # Process documents in chunks of 10
            chunk_size = 10
            for i in range(0, len(documents), chunk_size):
                print(f"\nProcessing documents {i + 1} to {min(i + chunk_size, len(documents))}...")
                document_chunk = documents[i:i + chunk_size]

                # Extract keywords and topics using KeyBERT
                print("Extracting keywords...")
                transformer = LinkExtractorTransformer([
                    KeybertLinkExtractor(),
                ])
                document_chunk = transformer.transform_documents(document_chunk)

                # Clean and preprocess documents
                # For PDFs and Markdown, minimal cleaning is needed
                # document_chunk = clean_and_preprocess_documents(document_chunk)

                # Split documents into chunks
                print("Splitting into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=64,
                )
                document_chunk = text_splitter.split_documents(document_chunk)
                
                # Extract named entities and create graph links
                # Customize these labels based on your document content
                print("Extracting entities for graph links...")
                ner_extractor = GLiNERLinkExtractor([
                    "Person", "Organization", "Location", "Product", 
                    "Technology", "Concept", "Topic", "Category"
                ])
                transformer = LinkExtractorTransformer([ner_extractor])
                document_chunk = transformer.transform_documents(document_chunk)

                # Add documents to the graph vector store
                print(f"Adding {len(document_chunk)} chunks to vector store...")
                store.add_documents(document_chunk)

                # Visualize the graph text for the current chunk
                visualize_graph_text(document_chunk)
                
            print(f"\n✅ Successfully processed {len(documents)} documents into graph vector store!")

    except Exception as e:
        LOGGER.error("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
