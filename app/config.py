import os

from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_BASE_URL = os.getenv('EMBEDDING_BASE_URL', 'https://api.shubiaobiao.com/v1')

CHAT_MODEL = os.getenv('CHAT_MODEL', 'deepseek-chat')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
DATA_DIR = os.getenv('DATA_DIR', 'data')
