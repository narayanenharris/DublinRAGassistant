import os
import psycopg
from pgvector.psycopg import register_vector
from typing import List, Dict

class DublinVectorDB:
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def setup_database(self):
        """Set up the database schema for storing Dublin planning documents."""
        try:
            with psycopg.connect(self.connection_string) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        source TEXT NOT NULL,
                        document_type TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        embedding vector(1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                try:
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
                        ON chunks USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100)
                    """)
                except Exception as e:
                    print(f"Warning: Index creation failed: {e}")
                
                print("Database setup completed successfully")
        except Exception as e:
            print(f"Error setting up database: {e}")
            raise
    
    def store_document(self, title: str, source: str, document_type: str) -> int:
        try:
            with psycopg.connect(self.connection_string) as conn:
                cursor = conn.execute(
                    "INSERT INTO documents (title, source, document_type) VALUES (%s, %s, %s) RETURNING id",
                    (title, source, document_type)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error storing document: {e}")
            raise
    
    def store_chunks(self, document_id: int, chunks: List, embeddings: List) -> None:
        try:
            with psycopg.connect(self.connection_string) as conn:
                register_vector(conn)
                chunk_data = [
                    (document_id, chunk.page_content, chunk.metadata.get("page", 0), embedding)
                    for chunk, embedding in zip(chunks, embeddings)
                ]
                conn.executemany(
                    """
                    INSERT INTO chunks (document_id, content, page_number, embedding) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    chunk_data
                )
        except Exception as e:
            print(f"Error storing chunks: {e}")
            raise
    
    def query_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        try:
            with psycopg.connect(self.connection_string) as conn:
                register_vector(conn)               
                cursor = conn.execute("""
                    SELECT 
                        c.content, 
                        c.page_number, 
                        d.title, 
                        d.source,
                        c.embedding <-> %s as distance
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY distance ASC
                    LIMIT %s
                """, (query_embedding, limit))
                
                return [
                    {
                        "content": row[0],
                        "page_number": row[1],
                        "title": row[2],
                        "source": row[3],
                        "similarity_score": row[4]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"Error querying similar chunks: {e}")
            raise