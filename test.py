import torch
from sentence_transformers import SentenceTransformer
import sentence_transformers
import psutil
import time
import transformers
import huggingface_hub

def check_versions():
    print("\n=== Package Versions ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub.__version__}")
    print(f"Sentence Transformers: {sentence_transformers.__version__}")

def test_cpu_performance():
    check_versions()
    
    print("\n=== System Information ===")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
    
    print("\n=== Loading Model ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test embedding generation
    test_texts = ["Test sentence " + str(i) for i in range(100)]
    
    print("\n=== Generating Embeddings ===")
    start_time = time.time()
    embeddings = model.encode(test_texts, batch_size=32, show_progress_bar=True)
    duration = time.time() - start_time
    
    print(f"\nResults:")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Sentences per second: {len(test_texts)/duration:.2f}")
    print(f"Embedding dimension: {embeddings.shape}")

if __name__ == "__main__":
    test_cpu_performance()