import os
import argparse
from dublin_data_processor import DublinDataProcessor
from dublin_vector_db import DublinVectorDB
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Database connection setup
db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
db_connection = f"postgres://postgres:testpass123@{db_host}:5432/postgres"

def process_directory(base_dir):
    """Process all supported files in the directory structure."""
    processor = DublinDataProcessor()
    all_chunks = []
    
    # Define supported extensions and their directories
    data_dirs = {
        'pdf': Path(base_dir) / 'raw_pdfs',
        'csv': Path(base_dir) / 'raw_csv',
        'json': Path(base_dir) / 'raw_json'
    }
    
    for format_type, directory in data_dirs.items():
        if directory.exists():
            print(f"\nProcessing {format_type.upper()} files from {directory}")
            files = list(directory.glob(f'*.{format_type}'))
            
            if not files:
                print(f"No {format_type} files found in {directory}")
                continue
                
            print(f"Found {len(files)} {format_type} files")
            
            for file_path in tqdm(files, desc=f"Processing {format_type} files"):
                try:
                    chunks = processor.process_file(str(file_path), format_type)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return all_chunks

def ingest_data(data_dir, db_connection, batch_size=100):
    """Ingest documents from multiple formats into the vector database."""
    processor = DublinDataProcessor()
    db = DublinVectorDB(db_connection)
    
    print("Setting up database schema...")
    db.setup_database()
    
    # Process all documents
    chunks = process_directory(data_dir)
    print(f"\nProcessed total {len(chunks)} chunks from all documents")
    
    if not chunks:
        print("No documents were processed successfully!")
        return
    
    # Generate embeddings with batching
    print("\nGenerating embeddings...")
    embeddings = processor.generate_embeddings(chunks)
    
    if not embeddings or len(embeddings) != len(chunks):
        print("Error: Embedding generation failed or mismatch in counts")
        return
    
    # Store in database with progress tracking
    print("\nStoring in database...")
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    with tqdm(total=len(chunks), desc="Storing chunks") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Create document entry
            doc_metadata = batch_chunks[0].metadata
            doc_id = db.store_document(
                title=doc_metadata["title"],
                source=doc_metadata["source"],
                document_type=doc_metadata["document_type"]
            )
            
            # Store chunks and embeddings
            db.store_chunks(doc_id, batch_chunks, batch_embeddings)
            pbar.update(len(batch_chunks))
    
    print("\nData ingestion complete!")

def main():
    parser = argparse.ArgumentParser(description="Dublin RAG Assistant - Multi-format Data Ingestion")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into the database")
    parser.add_argument("--data_dir", type=str, help="Base directory containing data folders")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    args = parser.parse_args()

    if args.ingest:
        if not args.data_dir or not os.path.isdir(args.data_dir):
            print(f"Error: Invalid data directory: {args.data_dir}")
            return
        ingest_data(args.data_dir, db_connection, args.batch_size)

if __name__ == "__main__":
    main()