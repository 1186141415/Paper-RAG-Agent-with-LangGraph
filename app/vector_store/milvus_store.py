from pymilvus import MilvusClient

from app.llm_utils import get_embedding
from app.logger_config import setup_logger
from app.vector_store.base import BaseVectorStore
from app.config import (
    MILVUS_LITE_URI,
    MILVUS_COLLECTION_NAME,
    MILVUS_METRIC_TYPE,
)

logger = setup_logger()

class MilvusVectorStore(BaseVectorStore):
    def __init__(
        self,
        uri: str = MILVUS_LITE_URI,
        collection_name: str = MILVUS_COLLECTION_NAME,
        metric_type: str = MILVUS_METRIC_TYPE,
        drop_old: bool = True,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.drop_old = drop_old
        self.client = MilvusClient(uri=self.uri)
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        """
        Build Milvus collection from document chunks.

        Current experiment version:
        - rebuilds the whole collection each time
        - keeps the same output format as FaissVectorStore.search()
        """
        self.chunks = chunks

        if not self.chunks:
            logger.warning("[MilvusVectorStore.build] no chunks provided")
            return

        texts = [c["text"] for c in self.chunks]
        embeddings = [get_embedding(t).tolist() for t in texts]

        dim = len(embeddings[0])

        if self.drop_old and self.client.has_collection(
                collection_name=self.collection_name
        ):
            self.client.drop_collection(
                collection_name=self.collection_name
            )
            logger.info(
                f"[MilvusVectorStore.build] dropped old collection: "
                f"{self.collection_name}"
            )

        if not self.client.has_collection(collection_name=self.collection_name):
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=dim,
                    metric_type=self.metric_type,
                )
            except TypeError:
                # Compatible fallback for pymilvus versions that do not accept metric_type here.
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=dim,
                )

            logger.info(
                f"[MilvusVectorStore.build] collection created: "
                f"{self.collection_name}, dim={dim}"
            )

        data = []

        for i, (chunk, vector) in enumerate(zip(self.chunks, embeddings)):
            data.append(
                {
                    "id": i,
                    "vector": vector,
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", "unknown"),
                    "chunk_id": i,
                }
            )

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        logger.info(
            f"[MilvusVectorStore.build] inserted chunks={len(data)} "
            f"into collection={self.collection_name}"
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search top-k related chunks from Milvus.

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
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Milvus collection has not been built: {self.collection_name}"
            )

        query_vec = get_embedding(query).tolist()

        raw_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=k,
            output_fields=["text", "source", "chunk_id"],
        )

        results = []

        if not raw_results:
            logger.warning(
                f"[MilvusVectorStore.search] query='{query}', empty search result"
            )
            return results

        for rank, item in enumerate(raw_results[0], start=1):
            entity = item.get("entity", {})

            chunk = {
                "source": entity.get("source", "unknown"),
                "text": entity.get("text", ""),
                "chunk_id": entity.get("chunk_id"),
                "distance": float(item.get("distance", 0.0)),
                "retrieval_rank": rank,
            }

            results.append(chunk)

        if results:
            best_distance = min(c["distance"] for c in results)
            logger.info(
                f"[MilvusVectorStore.search] query='{query}', "
                f"returned={len(results)}, best_distance={best_distance:.4f}"
            )
        else:
            logger.warning(
                f"[MilvusVectorStore.search] query='{query}', no valid chunks returned"
            )

        return results