from langchain_chroma import Chroma

from rag.configs import ChromaConfig
from rag.base_rag import BaseRag
from rag.utils import hash_text, load_documents, split_documents


class RagWithChroma(BaseRag):
    def __init__(
        self, folder: str = "documents", chroma_config: ChromaConfig = ChromaConfig()
    ):
        super().__init__(folder=folder)

        self.config = chroma_config
        self.vectordb = Chroma(
            persist_directory=self.config.persist_directory,
            collection_name=chroma_config.collection_name,
            embedding_function=self.embeddings,
        )

    def sync_documents(self):
        documents = load_documents(self.folder)
        chunks = split_documents(documents=documents)

        newly_added_chunk = []
        all_docs = self.vectordb.get()
        existing_chunk_hashes = {m.get("chunk_id") for m in all_docs.get("metadatas")}
        for chunk in chunks:
            chunk_hash = hash_text(text=chunk.page_content)
            if chunk_hash not in existing_chunk_hashes:
                chunk.metadata.update(
                    {
                        "chunk_id": chunk_hash,
                        "source": chunk.metadata.get("source", "unknown"),
                    }
                )
                newly_added_chunk.append(chunk)

        if newly_added_chunk:
            self.vectordb.add_documents(newly_added_chunk)
            print(f"Added {len(newly_added_chunk)} new chunks to the vector store.")

    def get_vectordb_as_retriever(self):
        return self.vectordb.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={
                "k": self.config.k_documents,
                "score_threshold": self.config.score_threshold,
            },
        )


if __name__ == "__main__":
    rag = RagWithChroma()

    rag.sync_documents()

    rag.chat()
