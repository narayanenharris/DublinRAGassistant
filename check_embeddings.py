import psycopg
from dotenv import load_dotenv
import os
import sys

def check_embedding_status():
    """Check the status of embedding generation in the database."""
    load_dotenv()
    
    # Get connection string from environment
    db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@127.0.0.1:5432/postgres")
    print(f"Attempting to connect to database...")
    
    try:
        # Test connection first
        with psycopg.connect(db_url) as conn:
            print("Database connection successful!")
            
            # Check if tables exist
            table_check = conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents'
                )
            """).fetchone()[0]
            
            if not table_check:
                print("Error: 'documents' table not found! Have you run the data ingestion?")
                return False
                
            # Check document count
            print("\nQuerying database...")
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            if doc_count == 0:
                print("No documents found in database. Please run data ingestion first.")
                return False
            
            # Check chunks and embeddings
            chunk_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as chunks_with_embeddings,
                    COUNT(CASE WHEN embedding IS NULL THEN 1 END) as chunks_without_embeddings
                FROM chunks
            """).fetchone()
            
            # Get distribution by document
            doc_stats = conn.execute("""
                SELECT 
                    d.title,
                    COUNT(c.id) as chunk_count,
                    COUNT(c.embedding) as chunks_with_embeddings
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.title
            """).fetchall()
            
            print("\n=== Embedding Generation Status ===")
            print(f"Total documents: {doc_count}")
            print(f"Total chunks: {chunk_stats[0]}")
            print(f"Chunks with embeddings: {chunk_stats[1]}")
            print(f"Chunks without embeddings: {chunk_stats[2]}")
            print("\n=== Document-level Status ===")
            
            for doc in doc_stats:
                print(f"\nDocument: {doc[0]}")
                print(f"Total chunks: {doc[1]}")
                print(f"Chunks with embeddings: {doc[2]}")
                if doc[1] != doc[2]:
                    print(f"Missing embeddings: {doc[1] - doc[2]}")
            
            # Check if all embeddings are generated
            return chunk_stats[1] == chunk_stats[0]
            
    except psycopg.OperationalError as e:
        print(f"\nDatabase connection failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Is the database running? Run: docker ps")
        print("2. Check your .env file has correct DATABASE_URL")
        print("3. Ensure PostgreSQL is running on port 5432")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check if data ingestion was completed successfully")
        return False

if __name__ == "__main__":
    print("\n=== Database Embedding Check ===")
    all_complete = check_embedding_status()
    print(f"\nReady to run app: {'Yes' if all_complete else 'No - embeddings incomplete'}")
    sys.exit(0 if all_complete else 1)