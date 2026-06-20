# app/vector_store/faiss_store.py

import faiss
import numpy as np

from app.llm_utils import get_embedding
from app.logger_config import setup_logger
from app.vector_store.base import BaseVectorStore


logger = setup_logger()

SOURCE_FILTER_OVER_FETCH_FACTOR = 20


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

    def _collect_search_results(
        self,
        distances,
        indices,
        k: int,
        source: str | None,
    ) -> list[dict]:
        results = []

        for idx, distance in zip(indices[0], distances[0]):
            idx = int(idx)

            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = dict(self.chunks[idx])

            if source is not None and chunk.get("source") != source:
                continue

            chunk["distance"] = float(distance)
            chunk["retrieval_rank"] = len(results) + 1
            results.append(chunk)

            if len(results) >= k:
                break

        return results

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[dict]:
        """
        Search top-k related chunks from FAISS.

        When source is set, over-fetch from FAISS then post-filter in Python so
        top-k results are restricted to that document.

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

        if source is None:
            fetch_k = k
        else:
            fetch_k = min(
                len(self.chunks),
                max(k, k * SOURCE_FILTER_OVER_FETCH_FACTOR),
            )

        query_vec = get_embedding(query).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vec, fetch_k)

        results = self._collect_search_results(distances, indices, k, source)

        if source is not None and len(results) < k and fetch_k < len(self.chunks):
            distances, indices = self.index.search(query_vec, len(self.chunks))
            results = self._collect_search_results(distances, indices, k, source)

        if results:
            best_distance = min(c["distance"] for c in results)
            logger.info(
                f"[FaissVectorStore.search] query='{query}', "
                f"source={source!r}, returned={len(results)}, "
                f"best_distance={best_distance:.4f}"
            )
        else:
            logger.warning(
                f"[FaissVectorStore.search] query='{query}', source={source!r}, "
                f"no valid chunks returned"
            )

        return results