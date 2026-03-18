from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_retriever():
    # ✅ Use OpenAI embeddings (cloud-friendly)
    embeddings = OpenAIEmbeddings()

    # ✅ Load FAISS vector store
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ✅ Create retriever
    retriever = vector_store.as_retriever()

    return retriever