import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_pipeline.rag_chain import create_rag_chain
from ingestion.document_loader import load_pdf
from embeddings.text_splitter import split_documents

st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")

st.title("Enterprise GenAI Knowledge Assistant (RAG-based LLM Application)")

st.subheader("Upload Knowledge Base (PDF)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)

    file_path = "data/uploaded.pdf"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing PDF...")

    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    # ✅ create rag chain ONLY here
    rag_chain = create_rag_chain(chunks)

    st.success("Knowledge base ready!")

    query = st.text_input("Ask a question:")

    if query:
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"]) 
        