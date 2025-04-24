import requests
import json
from tqdm import tqdm
import gc
import time
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import concurrent.futures
import torch

class LocalEmbeddingModel:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/embeddings"
        self.max_retries = 3
        self.retry_delay = 2
        # Optimize batch size for GPU
        self.batch_size = 64 if torch.cuda.is_available() else 32
        # Set CUDA device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure Ollama to use GPU
        self.headers = {
            "Content-Type": "application/json",
            "X-Ollama-Tags": "cuda"
        }
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_query(self, text: str) -> Optional[List[float]]:
        """Generate embeddings using GPU acceleration."""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "nomic-embed-text",
                    "prompt": text,
                    "options": {
                        "num_gpu": 1,  # Use GPU
                        "num_thread": 8  # Optimize thread count
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"\nError: API returned status code {response.status_code}")
                return None

        except Exception as e:
            print(f"\nError in embed_query: {str(e)}")
            print("Retrying...")
            raise

    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings with GPU optimization."""
        total_texts = len(texts)
        embeddings = []
        start_time = time.time()

        # Larger batch size for GPU processing
        batch_size = self.batch_size * 2
        
        print(f"\nProcessing {total_texts} texts with GPU acceleration")
        print(f"Batch size: {batch_size}")

        with tqdm(total=total_texts, desc="Generating embeddings") as pbar:
            # Process in larger batches for GPU
            for i in range(0, total_texts, batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                # Process batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(self.embed_query, text) for text in batch]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            embedding = future.result()
                            if embedding:
                                batch_embeddings.append(embedding)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\nError in batch processing: {str(e)}")
                
                embeddings.extend(batch_embeddings)
                
                # Update progress with GPU stats
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
                    pbar.set_postfix({
                        'GPU Memory': f'{gpu_memory:.1f}MB',
                        'chunks/s': f'{len(embeddings)/(time.time()-start_time):.2f}'
                    })
                
                # Memory management
                if i % (batch_size * 4) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        print(f"\nCompleted in {time.time() - start_time:.2f}s")
        if torch.cuda.is_available():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        
        return embeddings