from dublin_data_processor import DublinDataProcessor
import json
from tqdm import tqdm
import os

def prepare_chunks():
    print("\n=== Processing Documents ===")
    processor = DublinDataProcessor()
    
    # Check if data directory exists
    if not os.path.exists('./data'):
        print("Error: './data' directory not found!")
        return
    
    print("Processing documents from ./data directory...")
    chunks = processor.process_directory('./data')
    
    if not chunks:
        print("No chunks were generated!")
        return
    
    print(f"\nTotal chunks generated: {len(chunks)}")
    
    # Save chunks to JSON
    output_file = 'chunks.json'
    print(f"\nSaving chunks to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([c.page_content for c in chunks], f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(chunks)} chunks to {output_file}")
    except Exception as e:
        print(f"Error saving chunks: {str(e)}")

if __name__ == "__main__":
    prepare_chunks()