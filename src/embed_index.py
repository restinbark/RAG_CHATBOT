import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def embed_and_index(data_path="data/filtered_complaints.csv", index_path="vector_store/faiss_index"):
    # Load data
    df = pd.read_csv(data_path)
    
    # Combine narrative with product info for traceability
    documents = df["cleaned_narrative"].tolist()
    products = df["Product"].tolist()
    
    # Chunking
    from chunking import chunk_texts
    chunks = chunk_texts(documents)
    
    # Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # FAISS indexing
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save vector index
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "faiss.index"))
    
    # Save metadata (so we know which chunk belongs to what)
    with open(os.path.join(index_path, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    
    print("Vector store and metadata saved!")

if __name__ == "__main__":
    embed_and_index()
