from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(folder: str) -> list[Document]:
    """Load documents from the specified folder. Supports PDF, TXT, and DOCX files."""
    loaders = [
        DirectoryLoader(path=folder, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(path=folder, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(path=folder, glob="*.docx", loader_cls=Docx2txtLoader),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def hash_text(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()
