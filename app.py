import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dublin_rag import DublinRAG
from dotenv import load_dotenv
from verify_search import semantic_search

# Load environment variables
load_dotenv()

app = FastAPI(title="Dublin RAG Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize RAG system
db_connection = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@127.0.0.1:5432/postgres")
rag = DublinRAG(connection_string=db_connection)

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "Dublin RAG Assistant"}
    )

def process_results(results, query):
    """Process search results into a structured response"""
    # Extract relevant information from results
    findings = []
    for result in results:
        content = result["content"].replace("\n", " ").strip()
        if "|" in content:
            content = content.split("|")[1].strip()
        if len(content) > 50:
            findings.append(content)

    # Create structured response
    response = f"""# Analysis of {query}

## Key Findings 📋

"""
    for i, finding in enumerate(findings[:3], 1):
        response += f"{i}. {finding}\n\n"

    response += """## Summary 🎯

Based on Dublin's planning documents, these findings indicate:
- Key policy directions and implementations
- Specific development guidelines
- Important considerations for urban planning

## Next Steps 🔍

Would you like more specific information about any of these aspects?
"""
    return response

@app.post("/query")
async def query(request: QueryRequest):
    try:
        query = request.query.strip().lower()
        
        # Handle very short queries
        if len(query) < 10:
            return {
                "answer": f"""# Understanding Your Query

I notice you've asked about "{request.query}". To provide you with the most helpful information, 
could you please specify what aspect you'd like to know about? For example:

## Popular Topics About Dublin:

1. **Urban Development**
   - City planning and zoning
   - Future development projects
   - Neighborhood regeneration

2. **Housing**
   - Housing policies
   - Residential development
   - Affordable housing initiatives

3. **Infrastructure**
   - Transportation
   - Public facilities
   - Environmental projects

4. **Community Planning**
   - Local area plans
   - Public spaces
   - Community facilities

Please feel free to ask about any of these specific areas.""",
                "sources": []
            }

        # Get search results
        context_results = semantic_search(query, top_k=8)
        
        if not context_results:
            return {
                "answer": f"""# No Direct Matches Found

I couldn't find specific information about "{request.query}". Try:

1. **Be More Specific**
   - Use exact terms from Dublin's planning documents
   - Specify time periods (e.g., 2022-2028)
   - Name specific areas or policies

2. **Popular Topics**
   - Dublin housing policies
   - City center development
   - Environmental initiatives
   - Transportation planning""",
                "sources": []
            }

        # Generate response
        answer = process_results(context_results, query)
        
        # Format sources
        sources = [{
            "title": result["title"],
            "excerpt": result["content"][:200] + "...",
            "relevance": f"{result['similarity']:.2%} match"
        } for result in context_results[:3]]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "answer": "I encountered an issue processing your request. Please try rephrasing your question.",
            "sources": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)