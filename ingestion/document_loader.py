from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# compatibility function
def load_documents(file_path):
    return load_pdf(file_path)
