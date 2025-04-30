import logging
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from local_embedding_model import LocalEmbeddingModel
from dublin_vector_db import DublinVectorDB
from langchain_community.llms import Ollama


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DublinRAG:
    def __init__(self, connection_string):
        logger.info("Initializing DublinRAG...")
        self.vector_db = DublinVectorDB(connection_string)
        self.embeddings_model = LocalEmbeddingModel()
        try:
            self.llm = Ollama(
                model="mistral",
                temperature=0.7,
                num_ctx=2048,
                timeout=30
            )
            logger.info("Successfully initialized Ollama LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise
        
        self.template = """You are a helpful assistant specialized in Dublin urban planning and development.
Answer the question based ONLY on the provided context. If you cannot find the answer in the context, say "I don't have enough information about this in my current database." Do not make up or infer information.

Context:
{context}

Question: {question}

Answer the question in a helpful, comprehensive way. Include specific citations from the documents provided.
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)
    def retrieve(self, query: str, limit=5) -> List[Dict]:
        logger.info(f"Processing query: {query}")
        try:
            query_embedding = self.embeddings_model.embed_query(query)
            results = self.vector_db.query_similar(query_embedding, limit)
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return []
    
    def generate_answer(self, query: str) -> Dict:
        logger.info(f"Generating answer for: {query}")
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I don't have any relevant information in my database to answer this question.",
                "sources": []
            }
        try:
            context_text = "\n\n".join([
                f"Document: {doc['title']}, Page: {doc.get('page_number', 'N/A')}\n{doc['content']}" 
                for doc in retrieved_docs
            ])
            logger.info("Context prepared successfully")
            logger.info("Sending request to Ollama...")
            response = self.llm.invoke(
                self.prompt.format(context=context_text, question=query)
            )
            
            if not response:
                logger.error("Received empty response from LLM")
                return {
                    "answer": "The system generated an empty response. Please try rephrasing your question.",
                    "sources": []
                }
            
            logger.info("Successfully generated response")
            return {
                "answer": str(response), 
                "sources": [{
                    "title": doc["title"], 
                    "page": doc.get("page_number", "N/A")
                } for doc in retrieved_docs]
            }
            
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": []
            }