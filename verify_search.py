import logging
from sentence_transformers import SentenceTransformer
import psycopg
from typing import List, Dict
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def semantic_search(query: str, top_k: int = 5) -> List[Dict]:

    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")
    logger.info(f"Searching for: {query}")
    logger.info(f"Using database URL: {db_url}")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query)
        logger.info(f"Generated embedding of size: {len(query_embedding)}")
        
        with psycopg.connect(db_url) as conn:
            count_result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            logger.info(f"Found {count_result[0]} documents in database")
            
            results = conn.execute("""
                SELECT 
                    text_content,
                    metadata->>'title' as title,
                    metadata->>'source' as source,
                    1 - (embedding <-> %s::vector) as similarity
                FROM documents
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <-> %s::vector) > 0.3
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            matches = []
            for content, title, source, similarity in results.fetchall():
                clean_content = content.replace("\n", " ").strip()
                matches.append({
                    "content": clean_content[:1000],
                    "title": title or "Untitled",
                    "source": source or "Unknown",
                    "similarity": float(similarity)
                })
            
            logger.info(f"Found {len(matches)} matches with similarity > 0.3")
            return matches
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.exception("Full traceback:")
        return []

if __name__ == "__main__":
    print("\n=== Testing Semantic Search ===")
    test_query = "Dublin Development Plan"
    results = semantic_search(test_query)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {result['title']}")
            print(f"Similarity: {result['similarity']:.2%}")
            print(f"Content Preview: {result['content'][:200]}...")
    else:
        print("\nNo results found. Checking database connection...")
        try:
            with psycopg.connect(os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                print(f"Database connection successful. Found {count} documents.")
        except Exception as e:
            print(f"Database connection failed: {str(e)}")