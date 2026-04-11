import os
from pypdf import PdfReader
import re


def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
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
        chunks = split_text(doc["text"], chunk_size=200, overlap=50)

        for c in chunks:
            all_chunks.append({
                "text": c,
                "source": doc["source"]
            })

    return all_chunks