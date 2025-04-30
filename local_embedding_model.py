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
        self.api_url = "http://172.206.80.163:11434/api/embeddings"
        self.max_retries = 3
        self.retry_delay = 2
        self.batch_size = 128 if torch.cuda.is_available() else 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.headers = {
            "Content-Type": "application/json",
            "X-Ollama-Tags": "cuda"
        }
        
        self._log_device_info()
    
    def _log_device_info(self):
        print(f"\n=== Device Information ===")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            print(f"Batch Size: {self.batch_size}")
        print("==THE=END==\n")


    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        total_texts = len(texts)
        embeddings = []
        start_time = time.time()
        failed_count = 0
        batch_size = self.batch_size * 2
        
        print(f"\nProcessing {total_texts} texts on Azure VM")
        print(f"Batch size: {batch_size}")

        with tqdm(total=total_texts, desc="Generating embeddings") as pbar:
            for i in range(0, total_texts, batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    futures = [executor.submit(self.embed_query, text) for text in batch]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            embedding = future.result()
                            if embedding:
                                batch_embeddings.append(embedding)
                            else:
                                failed_count += 1
                            pbar.update(1)
                        except Exception as e:
                            print(f"\nError in batch processing: {str(e)}")
                            failed_count += 1
                embeddings.extend(batch_embeddings)
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
                    pbar.set_postfix({
                        'GPU Memory': f'{gpu_memory:.1f}MB',
                        'chunks/s': f'{len(embeddings)/(time.time()-start_time):.2f}',
                        'failed': failed_count
                    })
                
                if i % batch_size == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        print(f"\n=== Embedding Generation Summary ===")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Successful embeddings: {len(embeddings)}/{total_texts}")
        print(f"Failed embeddings: {failed_count}")
        if torch.cuda.is_available():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        
        return embeddings