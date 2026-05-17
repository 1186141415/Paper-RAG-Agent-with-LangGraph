# app/vector_store/faiss_store.py

import faiss
import numpy as np

from app.llm_utils import get_embedding
from app.logger_config import setup_logger
from app.vector_store.base import BaseVectorStore


logger = setup_logger()


class FaissVectorStore(BaseVectorStore):
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embeddings = None

    def build(self, chunks: list[dict]) -> None:
        """
        Build FAISS index from document chunks.

        chunks format:
        [
            {
                "text": "...",
                "source": "Paper1.pdf"
            }
        ]
        """
        self.chunks = chunks

        if not self.chunks:
            logger.warning("[FaissVectorStore.build] no chunks provided")
            self.index = None
            self.embeddings = None
            return

        texts = [c["text"] for c in self.chunks]

        embeddings = [get_embedding(t) for t in texts]
        self.embeddings = np.vstack(embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        logger.info(
            f"[FaissVectorStore.build] index built, "
            f"chunks={len(self.chunks)}, dim={dim}"
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search top-k related chunks from FAISS.

        Return format must stay compatible with RAGSystem:
        [
            {
                "text": "...",
                "source": "...",
                "distance": 1.23,
                "retrieval_rank": 1
            }
        ]
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been built.")

        if not self.chunks:
            logger.warning("[FaissVectorStore.search] no chunks available")
            return []

        k = min(k, len(self.chunks))

        query_vec = get_embedding(query).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vec, k)

        results = []

        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            idx = int(idx)

            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = dict(self.chunks[idx])
            chunk["distance"] = float(distance)
            chunk["retrieval_rank"] = rank

            results.append(chunk)

        if results:
            best_distance = min(c["distance"] for c in results)
            logger.info(
                f"[FaissVectorStore.search] query='{query}', "
                f"returned={len(results)}, best_distance={best_distance:.4f}"
            )
        else:
            logger.warning(
                f"[FaissVectorStore.search] query='{query}', no valid chunks returned"
            )

        return results