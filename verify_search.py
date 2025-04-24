from sentence_transformers import SentenceTransformer
import psycopg
import numpy as np
from typing import List, Dict
import os
from dotenv import load_dotenv

def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Perform semantic search using the query."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")
    
    print(f"\n=== Searching for: {query} ===")
    
    # Generate embedding for the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    
    try:
        with psycopg.connect(db_url) as conn:
            # Simplified query with lower similarity threshold
            results = conn.execute("""
                SELECT 
                    c.content,
                    d.title,
                    d.source,
                    1 - (c.embedding <-> %s::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                AND 1 - (c.embedding <-> %s::vector) > 0.1  -- Lowered threshold
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            matches = []
            for content, title, source, similarity in results.fetchall():
                # Clean and format the content
                clean_content = content.replace("\n", " ").strip()
                if "|" in clean_content:
                    parts = clean_content.split("|")
                    if len(parts) > 1:
                        clean_content = parts[1].strip()
                
                matches.append({
                    "content": clean_content,
                    "title": title,
                    "source": source,
                    "similarity": float(similarity)
                })
            
            return matches
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
            