from pathlib import Path
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader


BASE_PATH = Path(__file__).parent / "docs"

# Document loader (from docs folder)

def load_promptior_text_docs():
    documents = []

    for file_name in [
        "promptior_about.txt",
        "promptior_clients.txt",
    ]:
        file_path = BASE_PATH / file_name
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_name}
            )
        )

    return documents


# Web data loader (from promptior website)

def load_promptior_web_docs():
    loader = WebBaseLoader("https://promptior.ai")
    return loader.load()
