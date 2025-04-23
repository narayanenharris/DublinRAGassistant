# dublin_data_processor.py
import os
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader # Corrected import if needed, TextLoader not used here
from langchain_openai import OpenAIEmbeddings
import getpass # For API key input if not using environment variables

class DublinDataProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        # Ensure you have the openai package installed and the key is valid
        try:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:
            print(f"Error initializing OpenAI Embeddings. Check your API key and installation: {e}")
            # Handle the error appropriately, maybe raise it or exit
            raise e # Or sys.exit(1) after importing sys

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def load_pdf(self, file_path: str):
        """Load a PDF document and split it into chunks."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Extract metadata about the document
            doc_metadata = {
                "source": file_path,
                "title": os.path.basename(file_path),
                # Simple check for document type - adjust as needed
                "document_type": "Development Plan" if "Development Plan" in os.path.basename(file_path) else "Planning Document"
            }

            # Add metadata to each page
            for page in pages:
                page.metadata.update(doc_metadata)
                # Ensure page number exists and increments correctly
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            return chunks
        except Exception as e:
            print(f"Error loading or processing PDF {file_path}: {e}")
            return [] # Return empty list on error to avoid crashing the whole process

    def process_directory(self, directory_path: str):
        """Process all PDF files in a directory."""
        all_chunks = []
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at {directory_path}")
            return []

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'): # Use lower() for case-insensitivity
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    try:
                        chunks = self.load_pdf(file_path)
                        if chunks: # Only extend if chunks were successfully created
                           all_chunks.extend(chunks)
                           print(f"Successfully processed {file_path}, found {len(chunks)} chunks.")
                        else:
                           print(f"No chunks generated for {file_path} (possibly empty or error during load).")
                    except Exception as e:
                        # Catch potential errors during the load_pdf call itself
                        print(f"Failed to process {file_path}: {e}")
        return all_chunks

    def generate_embeddings(self, chunks):
        """Generate embeddings for document chunks."""
        if not chunks:
            print("No chunks provided to generate embeddings.")
            return []
        try:
            # Note: Embedding many chunks might take time and cost API credits
            print(f"Generating embeddings for {len(chunks)} chunks...")
            # The original list comprehension is fine for smaller numbers of chunks
            # For very large numbers, consider batching requests if the embedding provider supports it
            embeddings_list = [self.embeddings.embed_query(chunk.page_content) for chunk in chunks]
            print("Embeddings generated successfully.")
            return embeddings_list
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [] # Return empty list on error

# --- Example Usage ---
if __name__ == "__main__":
    # --- IMPORTANT: Handle your API Key securely! ---
    # Best practice: Use environment variables
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("OpenAI API key not found in environment variables.")
        # Fallback to prompt, but avoid hardcoding keys!
        # api_key = getpass.getpass("Enter your OpenAI API Key: ")
        print("Please set the OPENAI_API_KEY environment variable.")
        exit(1) # Exit if no key is available

    # Define the path to your PDF directory
    # This path is relative to where you RUN the script
    pdf_directory = "data/raw_pdfs"

    # Create an instance of the processor
    try:
        processor = DublinDataProcessor(openai_api_key=api_key)

        # Process the directory
        print(f"\nStarting processing in directory: {os.path.abspath(pdf_directory)}") # Show absolute path for clarity
        document_chunks = processor.process_directory(pdf_directory)

        if document_chunks:
            print(f"\nSuccessfully processed directory. Total chunks found: {len(document_chunks)}")

            # Optional: Generate embeddings for the chunks
            # embeddings = processor.generate_embeddings(document_chunks)
            # if embeddings:
            #     print(f"Successfully generated {len(embeddings)} embeddings.")
            #     # Now you can use the chunks and embeddings (e.g., store in a vector database)
            # else:
            #     print("Failed to generate embeddings.")

        else:
            print(f"\nNo document chunks were generated. Check the directory '{pdf_directory}' and ensure it contains valid PDF files.")

    except Exception as e:
        print(f"\nAn error occurred during processor initialization or processing: {e}")