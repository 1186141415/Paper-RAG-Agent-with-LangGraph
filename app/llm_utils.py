import hashlib
import json
import os
import threading
import time

import numpy as np
from openai import OpenAI

from app.logger_config import setup_logger

logger = setup_logger()


from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_BATCH_SIZE,
    LLM_API_TIMEOUT,
)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=LLM_API_TIMEOUT,
    max_retries=2,
)

client2 = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    timeout=LLM_API_TIMEOUT,
    max_retries=2,
)


# -------------------- Embedding 磁盘缓存 --------------------
# 轻量 JSON 缓存，key = sha256(EMBEDDING_MODEL + ":" + text)。
# 目的：reload_kb / 重启后，相同 chunk 直接命中缓存，不再重复消耗 embedding API。
# 缓存随模型名绑定，换 EMBEDDING_MODEL 自然失效，避免维度 / 语义错配。
_cache_lock = threading.Lock()
_embedding_cache: dict | None = None


def _cache_key(text: str) -> str:
    return hashlib.sha256(f"{EMBEDDING_MODEL}:{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    if os.path.exists(EMBEDDING_CACHE_PATH):
        try:
            with open(EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
            logger.info(f"[embedding_cache] loaded {len(_embedding_cache)} cached vectors")
        except Exception as e:
            logger.warning(f"[embedding_cache] failed to load cache, starting empty: {e}")
            _embedding_cache = {}
    else:
        _embedding_cache = {}

    return _embedding_cache


def _save_cache() -> None:
    if _embedding_cache is None:
        return
    try:
        cache_dir = os.path.dirname(EMBEDDING_CACHE_PATH)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f)
    except Exception as e:
        logger.warning(f"[embedding_cache] failed to save cache: {e}")


def _embed_api_batch(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """调用 embedding API，input 为 list。重试后仍失败则降级逐条，保证可用性。"""
    for attempt in range(max_retries):
        try:
            response = client2.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            # 按 index 排序，确保返回顺序与输入对齐
            ordered = sorted(response.data, key=lambda d: d.index)
            return [list(d.embedding) for d in ordered]
        except Exception as e:
            logger.warning(
                f"[get_embeddings] batch embedding failed, retry {attempt + 1}/{max_retries}: {e}"
            )
            time.sleep(2)

    logger.warning("[get_embeddings] batch failed after retries, falling back to per-item")
    return [get_embedding(t, use_cache=False).tolist() for t in texts]


def get_embeddings(texts: list[str], use_cache: bool = True, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """批量获取 embedding，带磁盘缓存。返回 np.ndarray，形状 (n, dim)，dtype float32。

    - 命中缓存的 chunk 不再打 API；
    - 未命中的按 batch_size 分批请求（OpenAI-compatible input=list）；
    - 批量调用失败自动降级逐条。
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    cache = _load_cache() if use_cache else {}

    results: list = [None] * len(texts)
    missing_idx: list[int] = []
    missing_texts: list[str] = []

    for i, t in enumerate(texts):
        key = _cache_key(t)
        if use_cache and key in cache:
            results[i] = cache[key]
        else:
            missing_idx.append(i)
            missing_texts.append(t)

    if missing_texts:
        logger.info(
            f"[get_embeddings] total={len(texts)}, "
            f"cache_hit={len(texts) - len(missing_texts)}, to_embed={len(missing_texts)}"
        )
        with _cache_lock:
            for start in range(0, len(missing_texts), batch_size):
                batch = missing_texts[start:start + batch_size]
                vectors = _embed_api_batch(batch)
                for j, vec in enumerate(vectors):
                    idx = missing_idx[start + j]
                    results[idx] = vec
                    if use_cache:
                        cache[_cache_key(missing_texts[start + j])] = vec
            if use_cache:
                _save_cache()
    else:
        logger.info(f"[get_embeddings] all {len(texts)} vectors served from cache")

    return np.array(results, dtype="float32")


def get_embedding(text, max_retries=3, use_cache=False):
    """单条 embedding（保留原有签名与行为）。

    默认 use_cache=False：检索阶段 query 每次不同，不入缓存。
    构建索引走 get_embeddings 批量 + 缓存。
    """
    if use_cache:
        cached = _load_cache().get(_cache_key(text))
        if cached is not None:
            return np.array(cached, dtype="float32")

    for attempt in range(max_retries):
        try:
            response = client2.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            vec = np.array(response.data[0].embedding, dtype="float32")
            if use_cache:
                with _cache_lock:
                    _load_cache()[_cache_key(text)] = vec.tolist()
                    _save_cache()
            return vec

        except Exception as e:
            logger.warning(f"Embedding failed, retry: {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    logger.error("Embedding failed after all retries")
    raise RuntimeError(f"Embedding failed after all retries")
