import faiss
from app.llm_utils import client, client2, get_embedding, decide_tool
import numpy as np

from app.config import (
    CHAT_MODEL
)

from app.logger_config import setup_logger

logger = setup_logger()


class RAGSystem:
    def __init__(self, chunks, top_k=20, rerank_k=10):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.index = None
        self.embeddings = None

        # self.chat_history = []  # 新增记忆

    # 把rag变成一个工具
    def rag_tool(self, query):
        return self.ask(query)

    def build_index(self):
        texts = [c["text"] for c in self.chunks]

        if self.embeddings is None:  # 加缓存避免重复计算消耗API
            embeddings = [get_embedding(t) for t in texts]
            self.embeddings = np.vstack(embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, k=5):
        query_vec = get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0]]

    def rerank(self, query, chunks):
        prompt = f"""
                You are a ranking assistant.

                Query:
                {query}

                Rank the following passages from most relevant to least relevant.

                Passages:
                """

        for i, c in enumerate(chunks):
            prompt += f"\n[{i}] {c}\n"

        prompt += "\nReturn ONLY the indices in sorted order, like [2,0,1]."

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        import ast
        try:
            return ast.literal_eval(response.choices[0].message.content)
        except:
            return list(range(len(chunks)))

    # 限制历史长度，否则会爆token
    # def trim_history(self, max_turn=3):
    #    if len(self.chat_history) > max_turn * 2:
    #        self.chat_history = self.chat_history[-max_turn * 2:]

    def ask(self, question, chat_history=None):  # 这个ask，没有返回top_k_chunks
        if chat_history is None:
            chat_history = []

        retrieved = self.retrieve(question, k=self.top_k)

        # 可以加 rerank（你已经有了）
        # context = "\n".join(retrieved)

        # rerank（用text）
        texts = [c["text"] for c in retrieved]
        sorted_indices = self.rerank(question, texts)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        # 拼context（加来源！）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        # 构造messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based on context and coversation history."
            }
        ]

        # 加历史对话
        messages.extend(chat_history)

        # 当前问题
        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        return answer

    def ask_with_trace(self, question, chat_history=None):
        if chat_history is None:
            chat_history = []

        retrieved = self.retrieve(question, k=self.top_k)

        # rerank（用 text）
        texts = [c["text"] for c in retrieved]
        sorted_indices = self.rerank(question, texts)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        # 拼 context（加来源）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based on context and conversation history."
            }
        ]

        # 保留历史对话
        messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        retrieved_chunks = []
        for c in best_chunks:
            retrieved_chunks.append({
                "source": c["source"],
                "text": c["text"],
            })

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }

    def ask_with_agent(self, question):
        decision = decide_tool(question)
        print("🧠 Decision:", decision)

        if "RAG" in decision:
            print("📚 Using RAG...")
            return self.ask(question)

        else:
            print("💬 Using LLM...")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
            )

            return response.choices[0].message.content

# if __name__ == '__main__':
#    docs = load_pdfs("data")  # 你的文件夹
#    chunks = process_documents(docs)
#
#    rag = RAGSystem(chunks)
#    rag.build_index()
#
#   answer = rag.ask("What are the differences between paper1 and paper2?")
#    print(answer)
