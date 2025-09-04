from pydantic import BaseModel


class ChromaConfig(BaseModel):
    persist_directory: str = "db/chroma_db"
    collection_name: str = "documents"
    search_type: str = "similarity_score_threshold"
    k_documents: int = 3
    score_threshold: float = 0.2
