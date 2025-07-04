import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_vector_store(path="vector_store/faiss_index"):
    index_file = os.path.join(path, "faiss.index")
    meta_file = os.path.join(path, "metadata.pkl")

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing FAISS index at: {index_file}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Missing metadata file at: {meta_file}")

    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

def embed_query(query, model):
    return model.encode([query])[0]

def retrieve_top_k(query_embedding, index, chunks, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question, model_pipeline):
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, say so.

Context:
{context}

Question:
{question}

Answer:"""
    result = model_pipeline(prompt)
    return result[0]['generated_text']

def ask_question(question, index_path="vector_store/faiss_index", product_filter="All"):
    # Load vector store and metadata
    index, chunks = load_vector_store(index_path)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

    query_embedding = embed_query(question, embed_model)
    retrieved_chunks = retrieve_top_k(query_embedding, index, chunks, k=10)

    # Apply product filter if selected
    if product_filter and product_filter != "All":
        filtered_chunks = [c for c in retrieved_chunks if product_filter.lower() in c.lower()]
        if not filtered_chunks:
            filtered_chunks = retrieved_chunks[:3]  # fallback if nothing matches
    else:
        filtered_chunks = retrieved_chunks

    context = "\n\n".join(filtered_chunks)
    answer = generate_answer(context, question, llm)

    return answer, filtered_chunks
