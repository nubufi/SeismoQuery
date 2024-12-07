import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pymupdf4llm as pm

# Path to the Earthquake Code document
INPUT_FILE = "app/data/TBDY2018.pdf"
DB_PATH = "app/data/tbdy2018.db"
LLAMA_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_split_code(file_path, chunk_size=1000, chunk_overlap=100):
    """
    Load the earthquake code text file and split it into chunks.
    """
    text = pm.to_markdown(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.create_documents([text])
    return documents

def embed_and_store(documents, db_path):
    """
    Embed the chunks and store them in a FAISS vector database.
    """
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=LLAMA_MODEL_NAME)
    
    print("Storing embeddings in FAISS...")
    vector_store = FAISS.from_documents(documents, embeddings)

    print(f"Saving FAISS index to {db_path}...")
    vector_store.save_local(db_path)
    print("FAISS index saved!")

if __name__ == "__main__":
    # Ensure the database path exists
    os.makedirs(DB_PATH, exist_ok=True)

    # Load and split the document
    print(f"Loading and splitting the document from {INPUT_FILE}...")
    docs = load_and_split_code(INPUT_FILE)

    # Embed and store in FAISS
    embed_and_store(docs, DB_PATH)
