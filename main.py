# main.py
import os
import argparse
from dublin_data_processor import DublinDataProcessor
from dublin_vector_db import DublinVectorDB
from dublin_rag import DublinRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ingest_data(data_dir, db_connection):
    """Ingest Dublin planning documents into the vector database."""
    processor = DublinDataProcessor(os.getenv("OPENAI_API_KEY"))
    db = DublinVectorDB(db_connection)
    
    # Set up database schema
    db.setup_database()
    
    # Process documents
    chunks = processor.process_directory(data_dir)
    print(f"Processed {len(chunks)} chunks from documents")
    
    # Generate embeddings
    embeddings = processor.generate_embeddings(chunks)
    
    # Store in database
    for i, chunk in enumerate(chunks):
        doc_id = db.store_document(
            title=chunk.metadata.get("title", "Unknown"),
            source=chunk.metadata.get("source", "Unknown"),
            document_type=chunk.metadata.get("document_type", "Planning Document")
        )
        db.store_chunks(doc_id, [chunk], [embeddings[i]])
    
    print("Data ingestion complete!")

def main():
    parser = argparse.ArgumentParser(description="Dublin RAG Assistant")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into the vector database")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing planning documents")
    
    args = parser.parse_args()
    
    db_connection = os.getenv("DATABASE_URL")
    
    if args.ingest:
        ingest_data(args.data_dir, db_connection)
    else:
        print("Starting Dublin RAG Assistant...")
        # This would be handled by the web app in production

if __name__ == "__main__":
    main()