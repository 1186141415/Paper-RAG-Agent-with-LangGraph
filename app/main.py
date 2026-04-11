from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import DATA_DIR
from app.data_loader import load_pdfs, process_documents
from app.rag_system import RAGSystem
from app.tools import TOOLS
from app.session_manager import SessionManager
from app.graph.builder import build_agent_graph
from app.logger_config import setup_logger

logger = setup_logger()

app = FastAPI()

session_manager = SessionManager(max_turns=3)

rag = None
graph = None


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.on_event("startup")
def startup_event():
    global rag, graph

    logger.info("Loading RAG system...")

    docs = load_pdfs(DATA_DIR)
    logger.info(f"docs数量: {len(docs)}")

    chunks = process_documents(docs)
    logger.info(f"chunks数量: {len(chunks)}")

    rag = RAGSystem(chunks)
    rag.build_index()

    graph = build_agent_graph(TOOLS, rag=rag) # 初始化Graph

    logger.info("RAG + LangGraph ready!")


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        history = session_manager.get_history(req.session_id)

        state = {
            "session_id": req.session_id,
            "query": req.question,
            "chat_history": history,
        }

        result = graph.invoke(state)
        answer = result["final_answer"]

        session_manager.append_turn(
            req.session_id,
            req.question,
            answer
        )

        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        logger.exception("Error occurred in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear/{session_id}")
def clear_session(session_id: str):
    session_manager.clear_session(session_id)
    return {
        "session_id": session_id,
        "message": "session cleared"
    }