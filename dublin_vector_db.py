# dublin_vector_db.py
import os
import psycopg
from pgvector.psycopg import register_vector
from typing import List, Dict

class DublinVectorDB:
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def setup_database(self):
        """Set up the database schema for storing Dublin planning documents."""
        with psycopg.connect(self.connection_string) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    document_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    content TEXT,
                    page_number INTEGER,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster similarity search
            try:
                conn.execute("CREATE INDEX ON chunks USING ivfflat (embedding vector_l2_ops)")
            except Exception as e:
                print(f"Index creation failed: {e}")
    
    def store_document(self, title: str, source: str, document_type: str):
        """Store document metadata and return document ID."""
        with psycopg.connect(self.connection_string) as conn:
            cursor = conn.execute(
                "INSERT INTO documents (title, source, document_type) VALUES (%s, %s, %s) RETURNING id",
                (title, source, document_type)
            )
            return cursor.fetchone()[0]
    
    def store_chunks(self, document_id: int, chunks, embeddings):
        """Store document chunks and their embeddings."""
        with psycopg.connect(self.connection_string) as conn:
            register_vector(conn)
            
            for i, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO chunks (document_id, content, page_number, embedding) VALUES (%s, %s, %s, %s)",
                    (
                        document_id, 
                        chunk.page_content, 
                        chunk.metadata.get("page", 0), 
                        embeddings[i]
                    )
                )
    
    def query_similar(self, query_embedding, limit=5):
        """Find chunks similar to the query."""
        with psycopg.connect(self.connection_string) as conn:
            register_vector(conn)
            
            cursor = conn.execute("""
                SELECT c.content, c.page_number, d.title, d.source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.embedding <-> %s
                LIMIT %s
            """, (query_embedding, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "content": row[0],
                    "page_number": row[1],
                    "title": row[2],
                    "source": row[3]
                })
            
            return results