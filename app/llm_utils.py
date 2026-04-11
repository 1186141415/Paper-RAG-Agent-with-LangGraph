import numpy as np
import time
from openai import OpenAI

from app.logger_config import setup_logger

logger = setup_logger()

# import os  #弃用

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
            # print(f"Embedding失败，第{attempt + 1}次重试...")
            logger.warning(f"Embedding failed, retry: {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    # print("Embedding最终失败，返回零向量")
    # return np.zeros(1536, dtype="float32")  # embedding维度
    logger.error("Embedding failed after all retries")
    raise RuntimeError(f"Embedding failed after all retries")


def decide_tool(query):
    prompt = f"""  
    You are an AI assistant.

    Decide whether the following question needs document retrieval.

    Question:
    {query}

    Answer ONLY:
    - "RAG" if it needs document-based answer
    - "LLM" if it can be answered directly   
              """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
