from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import time
import json
import os
import psutil
from typing import List, Dict

def process_chunks(input_file: str, output_file: str, batch_size: int = 32):
    """Process document chunks and generate embeddings."""
    
    print("\n=== System Information ===")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    print("\n=== Loading Model ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\n=== Loading Data ===")
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    total_chunks = len(chunks)
    print(f"\n=== Processing {total_chunks} chunks in batches of {batch_size} ===")
    
    start_time = time.time()
    embeddings = []
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
        
        if i % (batch_size * 100) == 0:
            print(f"\nProgress: {i}/{total_chunks} chunks")
            print(f"Memory used: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")
    
    duration = time.time() - start_time
    
    print(f"\n=== Results ===")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average speed: {total_chunks/duration:.2f} chunks/second")
    
    # Save embeddings
    np.save(output_file, np.array(embeddings))
    print(f"\nEmbeddings saved to: {output_file}")

if __name__ == "__main__":
    process_chunks('chunks.json', 'embeddings.npy', batch_size=32)