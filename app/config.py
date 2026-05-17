import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_BASE_URL = os.getenv('EMBEDDING_BASE_URL', 'https://api.shubiaobiao.com/v1')

ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
MCP_SEARCH_URL = os.getenv(
    "MCP_SEARCH_URL",
    "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp"
)

MCP_SEARCH_RECENCY = os.getenv('MCP_SEARCH_RECENCY', 'oneMonth')
MCP_SEARCH_CONTENT_SIZE = os.getenv('MCP_SEARCH_CONTENT_SIZE', 'medium')
MCP_SEARCH_LOCATION = os.getenv('MCP_SEARCH_LOCATION', 'us')

CHAT_MODEL = os.getenv('CHAT_MODEL', 'deepseek-chat')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
DATA_DIR = os.getenv('DATA_DIR', 'data')


VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()

# 注意：不要命名为 MILVUS_URI，避免和 pymilvus 内部配置冲突
MILVUS_LITE_URI = os.getenv("MILVUS_LITE_URI", "./milvus_demo.db")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "paper_rag_chunks")
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2")