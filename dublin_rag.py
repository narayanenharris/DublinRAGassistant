from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from local_embedding_model import LocalEmbeddingModel
from dublin_vector_db import DublinVectorDB
from langchain_community.llms import Ollama  # Add this for local LLM

class DublinRAG:
    def __init__(self, connection_string):
        self.vector_db = DublinVectorDB(connection_string)
        self.embeddings_model = LocalEmbeddingModel()
        # Initialize local LLM (using Mistral)
        self.llm = Ollama(model="mistral")  
        
        self.template = """You are a helpful assistant specialized in Dublin urban planning and development.
Answer the question based ONLY on the provided context. If you cannot find the answer in the context, say "I don't have enough information about this in my current database." Do not make up or infer information.

Context:
{context}

Question: {question}

Answer the question in a helpful, comprehensive way. Cite the specific document title and page number when providing information.
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)
   
    def retrieve(self, query: str, limit=5) -> List[Dict]:
        """Retrieve relevant documents for the query."""
        query_embedding = self.embeddings_model.embed_query(query)
        results = self.vector_db.query_similar(query_embedding, limit)
        return results
    
    def generate_answer(self, query: str) -> Dict:
        """Generate an answer using RAG."""
        # Retrieve relevant context
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have any relevant information in my database to answer this question.",
                "sources": []
            }
        
        # Format context for the prompt
        context_text = "\n\n".join([
            f"Document: {doc['title']}, Page: {doc['page_number']}\n{doc['content']}" 
            for doc in retrieved_docs
        ])
        
        # Generate answer using local LLM
        try:
            response = self.llm.invoke(
                self.prompt.format(context=context_text, question=query)
            )
            
            return {
                "answer": response,
                "sources": [{"title": doc["title"], "page": doc["page_number"]} for doc in retrieved_docs]
            }
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                "answer": "An error occurred while generating the answer.",
                "sources": []
            }