import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.retriever import get_retriever

# Load environment variables
load_dotenv()


def create_rag_chain():

    # Get retriever from vector database
    retriever = get_retriever()

    # LLM model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant.

        Answer the question based only on the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    # Document chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Retrieval chain
    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain


if __name__ == "__main__":

    rag_chain = create_rag_chain()

    question = input("Ask a question: ")

    response = rag_chain.invoke({
        "input": question
    })

    print("\nAnswer:\n")
    print(response["answer"])
