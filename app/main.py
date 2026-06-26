from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import DATA_DIR
from app.data_loader import load_pdfs, process_documents
from app.rag_system import RAGSystem
from app.tools import TOOLS
from app.session_manager import SessionManager
from app.logger_config import setup_logger

from app.graph.workflow import AgentWorkflow

logger = setup_logger()

session_manager = SessionManager(max_turns=3)

rag = None
workflow = None


def _init_rag_and_workflow() -> tuple[int, int]:
    """Load PDFs, build RAG index, and initialize AgentWorkflow. Return (total_docs, total_chunks)."""
    global rag, workflow

    logger.info("Loading RAG system...")

    docs = load_pdfs(DATA_DIR)
    logger.info(f"docs count: {len(docs)}")

    chunks = process_documents(docs)
    logger.info(f"chunks count: {len(chunks)}")

    rag = RAGSystem(chunks)
    rag.build_index()

    workflow = AgentWorkflow(TOOLS, rag=rag)

    logger.info("RAG + LangGraph ready!")
    return len(docs), len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_rag_and_workflow()
    yield
    logger.info("FastAPI lifespan shutdown.")


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        history = session_manager.get_history(req.session_id)

        result = workflow.invoke(
            session_id=req.session_id,
            query=req.question,
            chat_history=history
        )

        answer = result.get("final_answer", "")
        chunks = result.get("retrieved_chunks", [])

        decision = result.get("decision", {})
        tool_result = result.get("tool_result", {})

        agent_trace = {
            "route_decision": decision,
            "tool_used": tool_result.get("tool_name", ""),
            "tool_input": tool_result.get("tool_input", ""),
            "fallback_used": result.get("fallback_used", False),
            "context_sufficient": result.get("context_sufficient"),
            "context_metrics": result.get("context_metrics", {}),
            "error": result.get("error", ""),
            "retry_count": result.get("retry_count", 0),
            "workflow": result.get(
                "workflow_path",
                ["choose_tool", "execute_tool", "generate_answer"]
            )
        }

        session_manager.append_turn(
            req.session_id,
            req.question,
            answer
        )

        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer,
            "chunks": chunks,
            "agent_trace": agent_trace,
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


@app.post("/reload_kb")
def reload_kb():
    try:
        logger.info("Reloading knowledge base...")
        total_docs, total_chunks = _init_rag_and_workflow()
        logger.info("Knowledge base and AgentWorkflow reloaded successfully.")
        return {
            "status": "success",
            "message": f"Knowledge base reloaded. Total chunks: {total_chunks}",
            "total_docs": total_docs,
            "total_chunks": total_chunks,
        }
    except Exception as e:
        logger.exception("Error occurred in /reload_kb")
        raise HTTPException(status_code=500, detail=str(e))
