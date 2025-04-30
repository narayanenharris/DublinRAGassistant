# Dublin RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for Dublin city planning documents, powered by PostgreSQL vector search and FastAPI.

## Features

-  Semantic search through Dublin planning documents
-  AI-powered document analysis
-  Web interface for easy access
-  Vector-based similarity search
-  Real-time query processing

## Setup Options

### Option 1: Docker Setup (Recommended)

1. Build and run using Docker Compose:
```powershell
docker-compose up --build
```

2. Initialize the database (in a new terminal):
```powershell
docker exec -it dublinrag-app python reset_db.py
docker exec -it dublinrag-app python load_to_postgres.py
```

3. Access the application at: http://localhost:8000

### Option 2: Local Setup

1. Clone the repository:
```powershell
git clone https://github.com/narayanenharris/DublinRAGassistant.git
cd DublinRAGassistant
```

2. Install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Start PostgreSQL:
```powershell
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=testpass123 -e POSTGRES_DB=postgres ankane/pgvector
```

4. Initialize the database:
```powershell
python reset_db.py
python load_to_postgres.py
```

5. Run the application:
```powershell
uvicorn app:app --reload
```

## Environment Variables

Create a `.env` file with:
```
DATABASE_URL=postgres://postgres:testpass123@localhost:5432/postgres
```

For Docker, the URL will be:
```
DATABASE_URL=postgres://postgres:testpass123@postgres:5432/postgres
```

## Technology Stack

- FastAPI
- PostgreSQL with pgvector
- Sentence Transformers
- Docker & Docker Compose
- Python 3.10+

## Project Repository

GitHub: [https://github.com/narayanenharris/DublinRAGassistant](https://github.com/narayanenharris/DublinRAGassistant)