from app.config import VECTOR_STORE
from app.vector_store.faiss_store import FaissVectorStore


def create_vector_store():
    if VECTOR_STORE == "faiss":
        return FaissVectorStore()

    if VECTOR_STORE == "milvus":
        from app.vector_store.milvus_store import MilvusVectorStore
        return MilvusVectorStore()

    raise ValueError(
        f"Unsupported VECTOR_STORE: {VECTOR_STORE}. "
        f"Expected 'faiss' or 'milvus'."
    )