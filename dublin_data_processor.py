import os
from tqdm import tqdm
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from local_embedding_model import LocalEmbeddingModel
import json
import csv
from pathlib import Path
import psycopg
from sentence_transformers import SentenceTransformer
import torch
import gc
from dotenv import load_dotenv

class DublinDataProcessor:
    def __init__(self):
        load_dotenv()
        self.db_url = os.getenv("DATABASE_URL", "postgres://postgres:testpass123@localhost:5432/postgres")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.batch_size = 32 if torch.cuda.is_available() else 16

    def process_file(self, file_path: str, format_type: str):
        if format_type == 'pdf':
            return self.load_pdf(file_path)
        elif format_type == 'csv':
            return self.load_csv(file_path)
        elif format_type == 'json':
            return self.load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")


    def load_pdf(self, file_path: str):
        try:
            pdf = PdfReader(file_path)
            pages = []
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                metadata = {
                    "source": file_path,
                    "title": os.path.basename(file_path),
                    "page": page_num,
                    "document_type": "Development Plan" if "Development Plan" in os.path.basename(file_path) else "Planning Document"
                }
                doc = type('Document', (), {
                    'page_content': text,
                    'metadata': metadata
                })
                pages.append(doc)
            chunks = self.text_splitter.split_documents(pages)
            return chunks
        except Exception as e:
            print(f"Error loading or processing PDF {file_path}: {e}")
            return []
        

    def load_csv(self, file_path: str):
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata = {
                    "source": file_path,
                    "title": Path(file_path).name,
                    "page": 1,
                    "document_type": "CSV Data"
                }
                chunks.append(type('Document', (), {
                    'page_content': json.dumps(row),
                    'metadata': metadata
                }))
        return chunks    


    def load_json(self, file_path: str):
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metadata = {
                "source": file_path,
                "title": Path(file_path).name,
                "page": 1,
                "document_type": "JSON Data"
            }
            if isinstance(data, list):
                for item in data:
                    chunks.append(type('Document', (), {
                        'page_content': json.dumps(item),
                        'metadata': metadata
                    }))
            else:
                chunks.append(type('Document', (), {
                    'page_content': json.dumps(data),
                    'metadata': metadata
                }))
        return chunks    

    def process_directory(self, directory_path: str):

        all_chunks = []
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at {directory_path}")
            return []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    try:
                        chunks = self.load_pdf(file_path)
                        if chunks:
                            all_chunks.extend(chunks)
                            print(f"Successfully processed {file_path}, found {len(chunks)} chunks.")
                        else:
                            print(f"No chunks generated for {file_path} (possibly empty or error during load).")
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")
        return all_chunks


    def generate_embeddings(self, chunks):
        if not chunks:
            print("No chunks provided to generate embeddings.")
            return [] 
        try:
            total_chunks = len(chunks)
            print(f"\nGenerating embeddings for {total_chunks} chunks...")
            embeddings_list = []
            for i in tqdm(range(0, total_chunks, self.batch_size), desc="Processing batches"):
                batch = chunks[i:i + self.batch_size]
                texts = [chunk.page_content for chunk in batch]
                try:
                    batch_embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                    embeddings_list.extend(batch_embeddings)
                    with psycopg.connect(self.db_url) as conn:
                        with conn.cursor() as cur:
                            for chunk, embedding in zip(batch, batch_embeddings):
                                cur.execute("""
                                    INSERT INTO documents (text_content, metadata, embedding)
                                    VALUES (%s, %s, %s)
                                """, (
                                    chunk.page_content,
                                    json.dumps(chunk.metadata),
                                    embedding.tolist()
                                ))
                        conn.commit()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()  
                except Exception as e:
                    print(f"\nError in batch {i}: {str(e)}")
                    continue          
            print(f"\nEmbeddings generated and stored: {len(embeddings_list)}/{total_chunks}")
            return embeddings_list
            
        except Exception as e:
            print(f"Error in embedding generation process: {str(e)}")
            return []