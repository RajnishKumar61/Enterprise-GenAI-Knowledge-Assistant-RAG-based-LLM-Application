from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from embeddings.text_splitter import split_documents


def create_vector_db(documents):

    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("vector_store")

    return vector_store
