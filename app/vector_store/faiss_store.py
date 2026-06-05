# app/vector_store/faiss_store.py

import faiss
import numpy as np

from app.llm_utils import get_embedding, get_embeddings
from app.logger_config import setup_logger
from app.vector_store.base import BaseVectorStore


logger = setup_logger()


class FaissVectorStore(BaseVectorStore):
    """按论文 source 物理分片：每个 source 建一个独立的 `IndexFlatL2`。

    - `sources=None`：跨所有分片检索后合并取全局 top-k（与单一大索引等价，
      因为每个分片各取 k 个，全局 top-k 必然落在这些候选里）；
    - `sources` 指定：只在对应分片检索，从向量空间层面隔离不同论文，
      缓解同领域多篇论文「跨论文 chunk 混淆」。
    """

    def __init__(self):
        self.chunks = []
        self.dim = None
        self.indexes = {}        # source -> faiss.IndexFlatL2
        self.shard_chunks = {}   # source -> list[chunk dict]

    def build(self, chunks: list[dict]) -> None:
        """
        Build per-source FAISS shards from document chunks.

        chunks format:
        [
            {
                "text": "...",
                "source": "Paper1.pdf"
            }
        ]
        """
        self.chunks = chunks
        self.indexes = {}
        self.shard_chunks = {}
        self.dim = None

        if not self.chunks:
            logger.warning("[FaissVectorStore.build] no chunks provided")
            return

        texts = [c["text"] for c in self.chunks]

        # 批量 + 缓存：相同 chunk 在 reload / 重启时不再重复请求 embedding API
        embeddings = get_embeddings(texts).astype("float32")
        self.dim = int(embeddings.shape[1])

        # 按 source 分组，每篇论文一个独立索引
        by_source: dict[str, list] = {}
        for chunk, vec in zip(self.chunks, embeddings):
            src = chunk.get("source", "unknown")
            by_source.setdefault(src, []).append((chunk, vec))

        for src, items in by_source.items():
            vecs = np.vstack([v for _, v in items]).astype("float32")
            index = faiss.IndexFlatL2(self.dim)
            index.add(vecs)
            self.indexes[src] = index
            self.shard_chunks[src] = [c for c, _ in items]

        logger.info(
            f"[FaissVectorStore.build] built {len(self.indexes)} per-source shards, "
            f"chunks={len(self.chunks)}, dim={self.dim}"
        )

    def search(self, query: str, k: int = 5, sources=None) -> list[dict]:
        """
        Search top-k related chunks from FAISS shards.

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
        if not self.indexes:
            raise RuntimeError("FAISS index has not been built.")

        if not self.chunks:
            logger.warning("[FaissVectorStore.search] no chunks available")
            return []

        target = [s for s in (sources or self.indexes.keys()) if s in self.indexes]
        if not target:
            # 指定的 source 都不存在，回退到全部分片（与 RAG 层 fallback 双保险）
            target = list(self.indexes.keys())

        query_vec = get_embedding(query).reshape(1, -1).astype("float32")

        candidates = []
        for src in target:
            index = self.indexes[src]
            shard = self.shard_chunks[src]
            kk = min(k, len(shard))
            if kk <= 0:
                continue

            distances, indices = index.search(query_vec, kk)

            for dist, i in zip(distances[0], indices[0]):
                i = int(i)
                if i < 0 or i >= len(shard):
                    continue
                chunk = dict(shard[i])
                chunk["distance"] = float(dist)
                candidates.append(chunk)

        # 跨分片合并：按 distance 升序取 top-k，统一重排 retrieval_rank
        candidates.sort(key=lambda c: c["distance"])

        results = []
        for rank, chunk in enumerate(candidates[:k], start=1):
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        if results:
            best_distance = min(c["distance"] for c in results)
            logger.info(
                f"[FaissVectorStore.search] query='{query}', shards={len(target)}, "
                f"returned={len(results)}, best_distance={best_distance:.4f}"
            )
        else:
            logger.warning(
                f"[FaissVectorStore.search] query='{query}', no valid chunks returned"
            )

        return results
