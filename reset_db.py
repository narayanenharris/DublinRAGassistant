import psycopg
import os
from dotenv import load_dotenv

def reset_database():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")
    
    print("\n=== Resetting Database Tables ===")
    
    try:
        with psycopg.connect(db_url) as conn:
            print("Dropping existing tables...")
            conn.execute("""
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS documents;
            """)
            conn.commit()
            print("Tables dropped successfully!")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    reset_database()