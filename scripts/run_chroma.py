from rag.backends.chroma_rag import RagWithChroma

if __name__ == "__main__":
    rag = RagWithChroma()

    rag.sync_documents()

    rag.chat()
