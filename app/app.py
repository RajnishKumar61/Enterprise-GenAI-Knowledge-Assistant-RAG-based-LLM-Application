import sys
import os
import streamlit as st

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_pipeline.rag_chain import create_rag_chain

st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")

st.title("Enterprise GenAI Knowledge Assistant (RAG-based LLM Application)")

st.markdown(
"""
AI assistant powered by **RAG**, **OpenAI GPT-4o-mini**, and **LangChain**.  
Upload a PDF knowledge base and ask questions from it.
"""
)

# ===============================
# Upload PDF
# ===============================

st.subheader("Upload Knowledge Base (PDF)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# ===============================
# Process PDF
# ===============================

if uploaded_file is not None:

    from ingestion.document_loader import load_pdf
    from vector_store.create_vector_db import create_vector_db

    os.makedirs("data", exist_ok=True)

    file_path = "data/uploaded.pdf"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing PDF...")

    documents = load_pdf(file_path)

    create_vector_db(documents)

    st.success("Knowledge base created successfully!")

# ===============================
# Ask Question
# ===============================

st.subheader("Ask Questions From Your Knowledge Base")

rag_chain = create_rag_chain()

question = st.text_input("Ask a question")

if question:

    with st.spinner("Generating answer..."):

        response = rag_chain.invoke({"input": question})

        st.markdown("### Answer")
        st.write(response["answer"])
