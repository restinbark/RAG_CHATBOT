import sys
import os
import streamlit as st

# ✅ Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag_pipeline import ask_question

# --- Streamlit UI Setup ---
st.set_page_config(page_title="CrediTrust Chatbot", layout="wide")
st.title("💬 CrediTrust Complaint Answering Chatbot")
st.markdown("Ask questions about customer complaints and explore filtered results.")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
index_path = st.sidebar.text_input("Vector store path:", value="vector_store/faiss_index")

# Product filter (editable later based on actual data)
product_filter = st.sidebar.selectbox("Filter by product (optional):", ["All", "Credit Card", "Loan", "Buy Now Pay Later", "Mortgage"])

# --- User Input ---
question = st.text_input("💡 Enter your question:")

if st.button("🔍 Ask"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Pass filter into the pipeline
                answer, sources = ask_question(question, index_path=index_path, product_filter=product_filter)

                st.success("✅ Answer Generated!")

                st.subheader("📘 Answer")
                st.markdown(f"**{answer.strip()}**")

                st.subheader("📚 Retrieved Complaint Snippets")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {src[:300]}...")

            except Exception as e:
                st.error(f"❌ Error: {e}")
