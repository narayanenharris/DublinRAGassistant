import os
import time
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dublin_rag import DublinRAG
from dotenv import load_dotenv
from verify_search import semantic_search
from metrics import RAGMetrics
import psycopg
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Dublin RAG Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates and components
templates = Jinja2Templates(directory="templates")
db_connection = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@127.0.0.1:5432/postgres")
rag = DublinRAG(connection_string=db_connection)
metrics = RAGMetrics()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "title": "Dublin RAG Assistant",
            "system_metrics": metrics.get_system_stats()
        }
    )

@app.get("/health")
async def health_check():
    try:
        db_status = "unknown"
        doc_count = 0
        
        try:
            container_db_url = "postgres://postgres:testpass123@db:5432/postgres"
            with psycopg.connect(container_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents")
                    doc_count = cur.fetchone()[0]
                    db_status = "connected"
        except Exception as db_error:
            logger.error(f"Database connection failed: {db_error}")
            db_status = f"error: {str(db_error)}"

        return {
            "status": "healthy" if db_status == "connected" else "unhealthy",
            "timestamp": time.time(),
            "database": {
                "status": db_status,
                "document_count": doc_count
            },
            "container_info": {
                "database": "dublinragassistant-db-1",
                "api": "dublinragassistant-app-1"
            }
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
    
@app.post("/query")
async def process_query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        query = request.query.strip()
        logger.info(f"Processing query {request_id}: {query}")

        if len(query) < 10:
            metrics.log_short_query(query)
            return {
                "answer": generate_suggestion_response(query),
                "sources": [],
                "metrics": {
                    "query_length": len(query),
                    "processing_time": time.time() - start_time,
                    "status": "short_query"
                }
            }

        search_start = time.time()
        context_results = semantic_search(query, top_k=request.top_k)
        search_time = time.time() - search_start
        
        if not context_results:
            metrics.log_empty_results(query)
            return {
                "answer": generate_no_results_response(query),
                "sources": [],
                "metrics": {
                    "search_time": search_time,
                    "status": "no_results"
                }
            }

        processing_start = time.time()
        answer = await process_results(
            context_results,
            query,
            processing_time=time.time() - start_time
        )
        
        sources = [{
            "title": result["title"],
            "excerpt": result["content"][:200] + "...",
            "relevance": f"{result['similarity']:.2%} match"
        } for result in context_results[:3]]

        total_time = time.time() - start_time
        query_metrics = metrics.log_query_metrics(
            query=query,
            results=context_results,
            start_time=start_time,
            search_time=search_time,
            processing_time=total_time - search_time
        )

        return {
            "request_id": request_id,
            "answer": answer,
            "sources": sources,
            "metrics": query_metrics
        }

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error processing query {request_id}: {str(e)}")
        metrics.log_error(query, str(e), error_time)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error processing your request",
                "error": str(e),
                "request_id": request_id
            }
        )

async def process_results(results: List[Dict], query: str, processing_time: float) -> str:
    findings = []
    for result in results:
        content = result["content"].replace("\n", " ").strip()
        if "|" in content:
            content = content.split("|")[1].strip()
        if len(content) > 50:
            findings.append({
                "content": content,
                "similarity": result["similarity"],
                "source": result["title"]
            })

    findings.sort(key=lambda x: x["similarity"], reverse=True)
    
    response = f"""# Analysis of {query}

## Key Findings

"""
    for i, finding in enumerate(findings[:3], 1):
        response += f"{i}. {finding['content']}\n   _(Relevance: {finding['similarity']:.2%})_\n\n"

    response += f"""## Performance Metrics 
- Processing Time: {processing_time:.3f}s
- Results Found: {len(findings)}
- Average Relevance: {sum(f['similarity'] for f in findings)/len(findings):.2%}

## Summary 

Based on Dublin's planning documents, these findings from {len(set(f['source'] for f in findings))} different sources indicate:
- Key policy directions and implementations
- Specific development guidelines
- Important considerations for urban planning

## Next Steps 

Would you like more specific information about any of these aspects?
"""
    return response

def generate_suggestion_response(query: str) -> str:
    return f"""# Query Too Short

## Suggestion 

Your query "{query}" is too short for me to provide a meaningful response. To get better results, please:

1. Be more specific about what you want to know
2. Include more context in your question
3. Use complete sentences

## Examples of Good Queries 

- "What are the height restrictions for buildings in Dublin city center?"
- "Explain the sustainable development goals in Dublin's latest planning framework"
- "What are the requirements for converting residential buildings to commercial use?"
- "Tell me about Dublin's housing development policies for 2024"

Would you like to try asking a more detailed question? """

def generate_no_results_response(query: str) -> str:
    """Generate response for queries with no results."""
    return f"""# No Results Found

## Sorry! 

I couldn't find any relevant information about "{query}" in Dublin's planning documents.

## Suggestions 

Try:
1. Using different keywords or terms
2. Checking for typos in your query
3. Rephrasing your question
4. Looking for related topics instead

## Popular Topics 

You might be interested in:
- Dublin Development Plan
- Zoning Regulations
- Building Heights
- Sustainable Development
- Housing Policies

Would you like to ask about any of these topics? """



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )