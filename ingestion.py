import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pandas as pd

# Load environment variables
load_dotenv()

# --- Configuration ---
VECTOR_DB_PATH = "faiss_index"
DOCS_FOLDER = "docs"
TEMP_DIR = "temp_processing"

def load_document(file_path):
    """Selects the correct loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext == ".xlsx":
        # Convert Excel to CSV temporarily, then load CSV
        os.makedirs(TEMP_DIR, exist_ok=True)
        csv_filepath = os.path.join(TEMP_DIR, os.path.basename(file_path).replace(".xlsx", ".csv"))
        pd.read_excel(file_path, engine='openpyxl').to_csv(csv_filepath, index=False)
        return CSVLoader(csv_filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def create_vector_store():
    """Loads documents, adds metadata tags for access control, chunks, embeds, and saves the FAISS index."""
    
    documents = []
    
    if not os.path.exists(DOCS_FOLDER):
        print(f"Error: '{DOCS_FOLDER}' folder not found. Please create it and add your documents.")
        return

    print("--- Starting Document Loading and Tagging ---")
    
    for file in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file)
        
        try:
            print(f"Processing: {file}")
            
            # ACCESS CONTROL LOGIC: Tag documents based on filename keyword
            if "Manager" in file or "Recruitment" in file or "Confidential" in file:
                role_access = "Manager"
            else:
                role_access = "Employee"
            
            loader = load_document(file_path)
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata['role_access'] = role_access
                doc.metadata['filename'] = os.path.basename(file_path)
            
            documents.extend(loaded_docs)

        except ValueError as e:
            print(f"Skipping file {file}: {e}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # 1. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"\nSuccessfully loaded {len(documents)} document pages/records. Split into {len(chunks)} chunks for indexing.")
    
    # 2. Create Embeddings (Local Hugging Face Model)
    print("Initializing local Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

    # 3. Create and Save Vector Store (FAISS)
    print("Generating embeddings and creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"\n--- SUCCESS: FAISS index saved to '{VECTOR_DB_PATH}' ---")

if __name__ == "__main__":
    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_store()
    else:
        print(f"\n'{VECTOR_DB_PATH}' folder already exists. Delete it manually to re-create the index.")