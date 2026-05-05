import numpy as np
import time
from openai import OpenAI

from app.logger_config import setup_logger

logger = setup_logger()


from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    CHAT_MODEL,
    EMBEDDING_MODEL
)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

client2 = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL
)


def get_embedding(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client2.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return np.array(response.data[0].embedding, dtype="float32")

        except Exception as e:
            logger.warning(f"Embedding failed, retry: {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    logger.error("Embedding failed after all retries")
    raise RuntimeError(f"Embedding failed after all retries")