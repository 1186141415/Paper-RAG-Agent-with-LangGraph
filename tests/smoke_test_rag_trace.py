from app.config import DATA_DIR
from app.data_loader import load_pdfs, process_documents
from app.rag_system import RAGSystem

docs = load_pdfs('D:\Paper-RAG-Agent-with-LangGraph\data')
chunks = process_documents(docs)

rag = RAGSystem(chunks)
rag.build_index()

result = rag.ask_with_trace(
    "What is the difference between paper1 and paper2?",
    chat_history=[]
)

print("=== Answer ===")
print(result["answer"])

print("\n=== Retrieved Chunks ===")
for i, chunk in enumerate(result["retrieved_chunks"], start=1):
    print(f"\n[{i}] Source: {chunk['source']}")
    print(chunk["text"][:500])