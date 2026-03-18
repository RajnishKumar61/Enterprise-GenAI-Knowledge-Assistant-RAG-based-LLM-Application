from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_retriever(documents):
    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store.as_retriever()