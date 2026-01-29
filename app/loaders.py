from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_promptior_docs():
    """
    Loads content from Promptior website and splits it into chunks.
    """
    urls = [
        "https://promptior.ai/",
    ]

    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)
    return split_docs
