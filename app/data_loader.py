import os
from pypdf import PdfReader
import re

# Character-based chunking parameters.
# For English academic papers, 200 characters is usually too short and may break
# method descriptions or experiment conclusions into fragmented pieces.
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


def split_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.

    Current implementation uses character length instead of token length.
    The default chunk size is tuned for English academic papers:
    - 700 characters keeps more complete local context than 200 characters.
    - 120 characters overlap helps preserve continuity across chunks.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # 多行空白换成一行
    text = re.sub(r'\s+', ' ', text)  # 将所有连续空白字符（空格、制表符、换行等）替换成单个空格，实现“规范化空白”。
    return text

def load_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            text = clean_text(text)

            documents.append({
                "text": text,
                "source": filename
            })

    return documents

def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append({
                    "text": text,
                    "source": filename
                })

    return documents

def process_documents(documents):
    all_chunks = []

    for doc in documents:
        chunks = split_text(
            doc["text"],
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP
        )

        for c in chunks:
            all_chunks.append({
                "text": c,
                "source": doc["source"]
            })

    return all_chunks