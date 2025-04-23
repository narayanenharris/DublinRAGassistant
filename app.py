# app.py
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dublin_rag import DublinRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Dublin RAG Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize RAG system
rag = DublinRAG(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    connection_string=os.getenv("DATABASE_URL")
)

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "Dublin RAG Assistant"}
    )

@app.post("/query")
async def query(request: QueryRequest):
    response = rag.generate_answer(request.query)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)