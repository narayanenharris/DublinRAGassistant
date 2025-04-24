import numpy as np
import json
import psycopg
from tqdm import tqdm
import os
from dotenv import load_dotenv

def load_embeddings_to_db():
    # Load environment variables
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")
    
    print("\n=== Loading Embeddings into PostgreSQL ===")
    
    # Load data
    embeddings = np.load('embeddings.npy')
    with open('chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Get actual embedding dimension
    embedding_dim = embeddings[0].shape[0]
    print(f"Loaded {len(embeddings)} embeddings (dimension: {embedding_dim}) and {len(chunks)} chunks")
    
    try:
        with psycopg.connect(db_url) as conn:
            # Create tables with correct vector dimension (384)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    content TEXT,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Insert document
            doc_id = conn.execute("""
                INSERT INTO documents (title, source)
                VALUES ('Dublin Planning Documents', 'Azure Processed')
                RETURNING id
            """).fetchone()[0]
            
            # Insert chunks with embeddings in batches
            print("\nInserting chunks and embeddings...")
            batch_size = 100
            
            # Prepare the batch insert query
            insert_query = """
                INSERT INTO chunks (document_id, content, embedding)
                VALUES (%s, %s, %s)
            """
            
            with conn.cursor() as cur:
                for i in tqdm(range(0, len(chunks), batch_size)):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]
                    
                    # Process each item in the batch
                    for chunk, emb in zip(batch_chunks, batch_embeddings):
                        cur.execute(insert_query, (doc_id, chunk, emb.tolist()))
                    
                    conn.commit()  # Commit after each batch
                
                # Final commit
                conn.commit()
            
            print("\nSuccessfully loaded all data into PostgreSQL!")
            
            # Verify
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            print(f"Total chunks in database: {count}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTrying to get more error details...")
        try:
            print(f"First embedding dimension: {embeddings[0].shape}")
            print(f"Number of chunks: {len(chunks)}")
        except Exception as debug_e:
            print(f"Debug error: {str(debug_e)}")

if __name__ == "__main__":
    load_embeddings_to_db()