# AGENTS.md — PaperPilot

详细架构、数据契约、技术债见 **`docs/project/PROJECT_CONTEXT.md`**（重构前先读它）。本文件只放每轮必看的精简约定。

## 这个项目是什么
科研论文 RAG + 多工具 Agent 系统。运行时多进程：

| 进程 | 端口 | 职责 |
|------|------|------|
| **Vue 3** (`frontend/`) | `:5173` | 主产品界面（Stitch 设计落地） |
| **Django** (`django_shell/`) | `:8001` | REST API + 会话持久化；遗留 Django 模板仍挂在根路径 |
| **FastAPI** (`app/`) | `:8000` | AI 推理：RAG / LangGraph / 工具 |

链路：PDF → 向量检索 → LLM rerank → 双层证据门 → LangGraph 工具编排 → 答案 + Agent Trace。

## 启动命令
```bash
# 1. AI 服务（仓库根目录）
uvicorn app.main:app --reload --port 8000

# 2. Django API
cd django_shell && python manage.py runserver 8001

# 3. Vue 前端（主界面）
cd frontend && npm install && npm run dev
# → http://127.0.0.1:5173
```

- 切向量库：`.env` 里 `VECTOR_STORE=faiss|milvus`
- 可选 Spring BFF：`spring-bff/`（`:8080`），开发时 Vite 默认直连 Django
- 完整启动说明见 `README.md` §11 与 `docs/MVP_SPRINT.md`

## 分层铁律
1. Django 只做 API / 上传 / 持久化，**绝不**写 RAG/Agent/LangGraph 逻辑。
2. FastAPI 只做 AI 推理接口，不做页面。
3. Vue (`frontend/`) 只做展示与交互，通过 `/api/**` 调 Django。
4. LangGraph 只编排，RAGSystem 不关心底层是 FAISS 还是 Milvus（只用 `VectorStore.build/search`）。
5. 配置走 `app/config.py` + `.env`；日志走 `setup_logger()`。

## 红线（改之前先确认）
- calculator **绝不**用 `eval`/`exec`（只走 AST 白名单）。
- `VectorStore.search` 返回字段固定：`{source, text, distance, retrieval_rank}`，FAISS/Milvus 必须一致。
- router **不许**改写 `rag/llm/time/web_search` 的原始 query。
- **不要**移除 LangGraph 里 `fallback_used` 这个防无限循环的不变量。
- 改距离阈值 / 换 embedding 模型 / 换向量库 → 必须重评证据门阈值并跑评测集回归。
- 改任何核心数据契约字段（AgentState / agent_trace / chunk）→ 全链路 grep 同步，含 `frontend/` 与遗留 `*.html` 模板。

## 新增 LLM 调用 / 工具的模式
- LLM 调用：prompt 强约束结构化输出 + 解析侧清洗兜底 + 写入 trace（参考 `rag_system.rerank` / `nodes.choose_tool`）。
- 新工具：在 `tools.py` 注册进 `TOOLS`，并在 `choose_tool` prompt 与 `normalize_decision` 白名单同步加上；注意 `execute_tool` 按 name 分支传参。

## 当前最该改的几件事（详见 `docs/project/PROJECT_CONTEXT.md` §8）
会话状态双写不同步 / embedding 批处理与缓存已部分落地 / 单次 RAG 多次串行 LLM 调用等。

## 风格
- 简洁，不解释标准写法；只说改动的独特价值。
- 中文注释 + 英文 docstring/prompt（沿用现状）。
