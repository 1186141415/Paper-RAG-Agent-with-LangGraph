# PaperPilot — Project Context (for Cursor / AI Coding Agent)

> 这份文档不是给人浏览的 README，而是给 AI coding agent（Cursor / Claude Code 等）读的「项目说明书」。
> 目标：让 agent 在**没有上下文**的情况下，快速理解 PaperPilot 的分层结构、数据契约、控制流、已知技术债和重构红线，
> 从而能安全地重构和扩展，而不会破坏现有的运行闭环。
>
> README.md 仍然保留，作为面向人 / 面试官的项目介绍。两者职责不同：README 讲「这个项目是什么、亮点在哪」，本文件讲「代码长什么样、改的时候要小心什么」。

---

## 0. 一句话定位

PaperPilot 是一个面向**科研论文阅读与文献回溯**的轻量级 **RAG + 多工具 Agent 应用系统**：
本地 PDF → 向量检索 → LLM rerank → 双层证据充分性裁决 → LangGraph 工具编排 → FastAPI 推理服务 → Django 产品壳层展示，并保存历史会话与 Agent Trace。

它**不是**：完整 autonomous agent、完整 ReAct、生产级 SaaS、多租户 / 权限系统。当前定位是一个结构清晰、可运行、可观察、可持续迭代的 AI 应用工程样例。

---

## 1. 运行时进程模型（重要，先理解这个再读代码）

系统是**两个独立进程**，通过 HTTP 通信，不是单体应用：

| 进程 | 启动命令 | 默认端口 | 职责 |
|---|---|---|---|
| FastAPI AI Service | `uvicorn app.main:app --reload`（在仓库根目录运行）| `:8000` | AI 推理：RAG / LangGraph / 工具 / 短期会话 |
| Django Product Shell | `python manage.py runserver 8001`（在 `django_shell/` 运行）| `:8001` | 产品壳层：页面、上传、历史会话持久化 |

- Django 通过 `settings.FASTAPI_BASE_URL`（默认 `http://127.0.0.1:8000`）调用 FastAPI 的 `/ask` 和 `/reload_kb`。
- **隐含耦合（重构时注意）**：FastAPI 的 `DATA_DIR` 默认是相对路径 `"data"`（相对于 uvicorn 启动目录），Django 的 `documents/views.py` 里 `DATA_DIR = BASE_DIR.parent / "data"`。两者都指向「仓库根目录/data」**只有在 uvicorn 从仓库根目录启动时才成立**。这是一个脆弱约定，重构时建议显式统一成绝对路径配置。

---

## 2. 分层架构

```
User
  ↓
Django Product Shell        ← 页面 / 上传 / 历史会话持久化（SQLite），不写 AI 逻辑
  ↓ HTTP
FastAPI AI Service          ← /ask /reload_kb /clear，编排各层
  ↓
AgentWorkflow (LangGraph)   ← choose_tool → execute_tool →(条件边)→ llm_fallback / generate_answer
  ↓
Tools Layer                 ← rag / calculator / time / web_search / llm
  ↓
RAGSystem                   ← 检索 + rerank + 双层证据充分性 + 答案生成
  ↓
VectorStore（抽象层）        ← FaissVectorStore / MilvusVectorStore，可切换
```

**分层铁律（这是项目的核心设计原则，重构时务必保持）：**
1. Django 只做产品壳层 / 展示 / 持久化，**绝不**把 RAG / Agent / LangGraph 逻辑搬进 Django view。
2. FastAPI 只做 AI 推理服务接口层，不做页面。
3. LangGraph 只负责编排，不写具体业务检索逻辑。
4. RAGSystem 不关心底层是 FAISS 还是 Milvus，只调用 `VectorStore.build/search`。
5. 工具执行有明确输入边界（典型：calculator 用 AST 白名单，**绝不用 `eval`**）。

---

## 3. 真实目录结构

> 注意：README 第 9 节的目录树**漏掉了 `app/vector_store/`**（FAISS / Milvus 抽象层），以实际为准如下。
> 另外，本文档里某些文件在导出时带了括号（如 `views(chat).py`），它们的真实路径是 `chat/views.py`、`documents/views.py`。

```
Paper-RAG-Agent-with-LangGraph/
├─ app/                          # FastAPI + RAG + LangGraph 核心
│  ├─ main.py                    # FastAPI 入口：/ask /clear/{id} /reload_kb；startup 加载 KB
│  ├─ config.py                  # 集中读取 .env（模型 / API / MCP / 向量库 / 数据目录）
│  ├─ data_loader.py             # 加载 PDF/TXT、清洗、切分（700 字符 / 120 重叠）
│  ├─ llm_utils.py               # 两个 OpenAI 客户端 + get_embedding（带重试）
│  ├─ logger_config.py           # 统一 setup_logger()
│  ├─ rag_system.py              # RAG 核心：检索 / rerank / 双层证据门 / 答案生成
│  ├─ session_manager.py         # FastAPI 侧 in-memory 短期多轮上下文（max 3 turns）
│  ├─ tools.py                   # 工具定义 + TOOLS 列表
│  ├─ mcp_tools.py               # MCP（智谱 web_search_prime）封装，双层 JSON 解析兜底
│  ├─ graph/
│  │  ├─ builder.py              # build_agent_graph：注册节点 + 条件边 + 编译
│  │  ├─ nodes.py                # choose_tool / execute_tool / llm_fallback / generate_answer / route_after_execute
│  │  ├─ state.py                # AgentState (TypedDict, total=False)
│  │  └─ workflow.py             # AgentWorkflow 封装 graph.invoke
│  └─ vector_store/
│     ├─ base.py                 # BaseVectorStore(ABC): build / search
│     ├─ factory.py              # create_vector_store()，按 VECTOR_STORE 选择后端
│     ├─ faiss_store.py          # FaissVectorStore（IndexFlatL2）
│     └─ milvus_store.py         # MilvusVectorStore（Milvus Lite，本地）
├─ data/                         # 本地论文知识库（Paper1.pdf ...）
├─ django_shell/
│  ├─ manage.py
│  ├─ db.sqlite3
│  ├─ chat/
│  │  ├─ models.py               # ChatSession / ChatMessage
│  │  ├─ admin.py                # admin 注册
│  │  ├─ views.py                # chat_home / session_list / session_detail
│  │  ├─ urls.py
│  │  └─ services/ai_client.py   # ask_ai()：POST FastAPI /ask
│  ├─ documents/
│  │  ├─ views.py                # upload_page：保存 PDF + 调 /reload_kb
│  │  └─ urls.py
│  ├─ config/                    # settings.py / urls.py / wsgi.py
│  └─ templates/
│     ├─ chat/{chat_home,session_list,session_detail}.html
│     └─ documents/upload.html
├─ tests/                        # 冒烟测试（smoke_test_*.py）
├─ requirements.txt
└─ README.md
```

---

## 4. 模块职责清单（file-by-file）

- **app/main.py** — FastAPI 应用。`lifespan` async context manager 在启动时加载 PDF、构建 `RAGSystem` 并初始化 `AgentWorkflow`，依赖统一挂在 `app.state`（`rag` / `workflow` / `session_manager`）。`/reload_kb` 重新构建后**原子替换** `app.state.rag` / `app.state.workflow`；各端点通过 `Request` 取依赖，已无模块级全局可变状态。
- **app/config.py** — 所有配置来源。新增配置项请加在这里，不要在业务代码里散落 `os.getenv`。
- **app/data_loader.py** — `load_pdfs` / `load_documents` / `process_documents` / `split_text` / `clean_text`。切分是**字符级**（不是 token 级），默认 700/120，针对英文论文调过。
- **app/llm_utils.py** — `client`（DeepSeek，chat）+ `client2`（embedding，独立 base_url）+ `get_embedding(text)`（单条，3 次重试、sleep 2s，默认不缓存）+ `get_embeddings(texts)`（批量 + 磁盘缓存，按 `EMBEDDING_BATCH_SIZE` 分批，批量失败自动降级逐条）。构建索引走 `get_embeddings`，检索 query 走单条 `get_embedding`。
- **app/logger_config.py** — `setup_logger()` 返回名为 `"rag_agent"` 的 logger。所有模块统一用它。
- **app/rag_system.py** — 见 §6.3。核心方法：`build_index` / `retrieve` / `rerank` / `assess_context_sufficiency` / `assess_context_relevance_with_llm` / `ask_with_trace`。
- **app/session_manager.py** — `SessionManager(max_turns=3)`，`get_history / append_turn / trim_history / clear_session`，纯内存 dict。
- **app/tools.py** — `rag_tool / calculator_tool / time_tool / llm_tool` + `TOOLS` 列表。calculator 基于 `ast.parse(mode="eval")` + operator 白名单。
- **app/mcp_tools.py** — `web_search_tool(query)`，通过 `MultiServerMCPClient` 调智谱 `web_search_prime`，`asyncio.run` 同步包裹，结果做最多两次 `json.loads` 兜底，取前 5 条。
- **app/graph/builder.py** — 用工厂函数 `build_choose_tool_node(tools)` / `build_execute_tool_node(tools, rag)` 注入依赖，注册节点并连边，最后 `compile()`。
- **app/graph/nodes.py** — 路由与执行节点 + `route_after_execute` 条件函数 + `normalize_decision` / `maybe_force_web_search` / `clean_json_text` 工具函数。
- **app/graph/state.py** — `AgentState`，所有节点间传递的状态结构（见 §5.3）。
- **app/graph/workflow.py** — `AgentWorkflow(tools, rag)`，`invoke(session_id, query, chat_history)` → 拼初始 state → `graph.invoke`。
- **app/vector_store/** — `BaseVectorStore` 抽象 + FAISS / Milvus 两个实现 + `factory`。两个实现的 `search` 返回结构必须一致（见 §5.1）。
- **django chat/views.py** — `chat_home` 是主入口：POST 时调 `ask_ai`，拿到 `chunks` + `agent_trace`，落库 `ChatSession` + 两条 `ChatMessage`，渲染页面。
- **django documents/views.py** — `upload_page`：保存 PDF 到 `data/`，POST `/reload_kb`，对 `ConnectionError` / `ReadTimeout` 分别给提示。⚠️ 文件头部有一行无用 import `from urllib import response`（死代码，可删）。
- **django chat/services/ai_client.py** — `ask_ai(session_id, question)`：`requests.post(FASTAPI_BASE_URL + "/ask", timeout=120)`。

---

## 5. 关键数据契约（重构时**最不能破坏**的部分）

任何重构如果改了下面这些结构的字段名 / 形状，必须同步改掉所有上下游（RAGSystem ↔ VectorStore ↔ nodes ↔ main.py ↔ Django 模板）。建议先读这一节，再动代码。

### 5.1 chunk 与 VectorStore 接口

- `process_documents` 产出的 chunk：`{"text": str, "source": str}`
- `VectorStore.build(chunks: list[dict]) -> None`
- `VectorStore.search(query: str, k: int, sources: list[str] | None = None) -> list[dict]`（`sources` 可选：按论文 source 分片检索，`None`=检索全部；返回字段不变），每个元素**必须**含：
  ```python
  {"source": str, "text": str, "distance": float, "retrieval_rank": int}
  ```
  FAISS 与 Milvus 都遵循这个返回格式，RAGSystem / 前端展示 / Agent Trace 都依赖它。

### 5.2 工具层契约

- `TOOLS` 每项：`{"name": str, "description": str, "func": callable}`，`name ∈ {rag, calculator, time, web_search, llm}`
- 函数签名**不统一**（execute_tool 里按 name 分支调用）：
  - `rag_tool(query, rag, chat_history=None)`
  - `llm_tool(query, chat_history=None)`
  - 其余：`func(input)` 单参数
- `rag_tool` 返回 **dict**：`{"answer", "retrieved_chunks", "context_sufficient", "context_metrics"}`；其它工具返回 **str**。`generate_answer_node` 据此分支（dict → 拆字段，str → 直接当 final_answer）。

### 5.3 AgentState（TypedDict, total=False）

```
session_id: str
query: str
chat_history: list[{"role","content"}]
decision: {"tool","input","reason"}
tool_result: {"tool_name","tool_input","tool_output"}
final_answer: str
retrieved_chunks: list[dict]
context_sufficient: bool
context_metrics: dict
fallback_used: bool
retry_count: int
workflow_path: list[str]   # 实际经过的节点
error: str                 # 异常兜底用
```

### 5.4 /ask 请求与响应

请求：`{"session_id": str, "question": str, "chat_history"?: list[{"role","content"}]}`（`chat_history` 可选：调用方传入则优先作为 LLM 上下文，否则回退到 FastAPI 内存历史）
响应：
```
{
  "session_id", "question", "answer",
  "chunks": [...],            # = retrieved_chunks
  "agent_trace": {
    "route_decision": {tool,input,reason},
    "tool_used", "tool_input",
    "fallback_used", "context_sufficient",
    "context_metrics": {...},
    "error", "retry_count",
    "workflow": [...]
  }
}
```

### 5.5 context_metrics 字段（Agent Trace 可解释性的核心）

`num_chunks / min_required_chunks / best_distance / avg_top_distance / max_best_distance / max_avg_top_distance / reason / distance_gate_passed / llm_question_type / llm_gate_mode / llm_relevance_check / llm_relevance_verdict / llm_relevance_reason / llm_relevance_error / llm_soft_warning / rerank_used / rerank_fallback / rerank_error / rerank_indices / rerank_raw_output / final_sufficiency_reason`

`chat_home.html` / Agent Trace 卡片读取其中部分字段，改名前先 grep 模板。

---

## 6. 控制流

### 6.1 一次问答的完整链路

```
Django chat_home (POST)
  → ai_client.ask_ai → FastAPI POST /ask
  → session_manager.get_history(session_id)      # 取 in-memory 短期历史
  → AgentWorkflow.invoke(session_id, query, chat_history)
  → graph.invoke(state)                           # 见 6.2
  → 返回 final_answer + chunks + agent_trace
  → session_manager.append_turn(...)              # 写回 in-memory
Django 侧：保存 ChatSession + user/assistant ChatMessage 到 SQLite，渲染
```

### 6.2 LangGraph 节点图

```
START → choose_tool → execute_tool → route_after_execute
                                         ├─ (无 error) ───────────→ generate_answer → END
                                         └─ (有 error 且未 fallback) → llm_fallback → generate_answer → END
```

- `route_after_execute`：`if state.error and not state.fallback_used → "llm_fallback" else "generate_answer"`。
- `fallback_used` 的存在**防止无限 fallback 循环**——这是一个有意的不变量，不要去掉。

### 6.3 RAGSystem.ask_with_trace 内部流水线

```
retrieve(top_k=20)
  → rerank(LLM, 取 rerank_k=10，带 fallback + trace)
  → assess_context_sufficiency()            # Layer 1: 距离门（不调 LLM）
  → assess_context_relevance_with_llm()     # Layer 2: 类型化 LLM 相关性门
  → 决策 context_sufficient（见下）
      ├─ True  → 拼 context（带 [Source]）→ LLM 生成答案
      └─ False → 返回硬编码中文「证据不足」提示，不强行生成
```

`context_sufficient` 最终判定逻辑（顺序敏感）：
1. `not distance_sufficient` → **False**
2. `llm_relevance_error`（judge 失败/格式异常）→ **True**（降级到距离门，soft warning）
3. `context_relevant`（judge 判 YES）→ **True**
4. `question_type == "BROAD"` 但 judge 判 NO → **True**（只作 soft warning，避免误杀宽泛问题）
5. 其它（SPECIFIC / COMPARISON 判 NO）→ **False**（硬阻断）

### 6.4 Router 三层防护（choose_tool）

1. `clean_json_text`：剥离 ```` ```json ```` / ```` ``` ```` 代码块
2. `normalize_decision`：工具白名单校验，非法工具降级 `llm`；对 `rag/llm/time/web_search` **强制保留原始 query**（防止 router 把问题改写成答案）；calculator 允许保留抽出的表达式
3. `maybe_force_web_search`：当问题**同时**含联网信号（latest/最新/news…）和本地文档信号（paper1/论文/pdf…）时，强制路由 `web_search`

---

## 7. 关键常量与可调参数

| 位置 | 常量 | 默认值 | 说明 |
|---|---|---|---|
| data_loader.py | `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | 700 / 120 | 字符级切分 |
| rag_system.py | `RAGSystem(top_k, rerank_k)` | 20 / 10 | 召回 20，rerank 取 10 |
| rag_system.py | `MIN_CONTEXT_CHUNKS` | 2 | 距离门最少 chunk 数 |
| rag_system.py | `CONTEXT_TOP_N_FOR_AVG` | 3 | 算平均距离取前 N |
| rag_system.py | `CONTEXT_MAX_BEST_DISTANCE` | 2.2 | 距离门阈值（best）|
| rag_system.py | `CONTEXT_MAX_AVG_TOP_DISTANCE` | 2.4 | 距离门阈值（avg）|
| rag_system.py | `RELEVANCE_GATE_PREVIEW_CHUNKS` | 5 | 喂给 relevance judge 的 chunk 数 |
| config.py | `CHAT_MODEL` | `deepseek-chat` | 可换 OpenAI-compatible |
| config.py | `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| config.py | `VECTOR_STORE` | `faiss` | `faiss` / `milvus` |

⚠️ **距离阈值（2.2 / 2.4）是针对当前 embedding 模型 + FAISS L2 经验调出来的。** 一旦更换 embedding 模型或向量库（包括切到 Milvus），距离尺度会变，阈值必须重新评估，否则证据门会失准。

---

## 8. 当前已知技术债 / 重构重点（高优先级，按影响排序）

> 这些都是从现有代码里读出来的真实问题，不是臆测。重构时可优先处理。

1. ~~**会话状态双写、不同步（架构级）**~~ —（✅ 已改善）`/ask` 新增可选 `chat_history`；Django `chat_home` 现在把该会话的 SQLite 历史（最近 3 轮 = 6 条）随请求传入，FastAPI 优先用传入历史作为 LLM 上下文，统一会话来源到持久层；未传入时回退内存 `SessionManager`（向后兼容）。
   - 残留：FastAPI 内存 `SessionManager` 仍保留（兼容不传 history 的客户端），与传入历史并存；多 worker 共享仍需引入共享存储，留待后续。

2. ~~**FastAPI 启动钩子已废弃**~~ —（✅ 已解决）`on_event` 已迁移到 `lifespan` async context manager（见 `app/main.py`）。

3. ~~**模块级全局可变状态**~~ —（✅ 已解决）`rag` / `workflow` / `session_manager` 已迁移到 `app.state`，端点经 `Request` 获取，`/reload_kb` 原子替换引用。⚠️ 多 worker 间仍各自构建、不共享，留待引入共享存储时处理。

4. ~~**Embedding 无批处理、无缓存**~~ —（✅ 已解决）新增 `get_embeddings(texts)`：按 `EMBEDDING_BATCH_SIZE` 批量请求 + 按 `sha256(EMBEDDING_MODEL + text)` 磁盘缓存（`EMBEDDING_CACHE_PATH`，默认 `.embedding_cache/embeddings.json`，已 gitignore），批量调用失败自动降级逐条。`faiss` / `milvus` 的 `build` 已改用它；检索 query 仍走单条 `get_embedding`。README「构建后缓存」描述现已与代码一致。

5. **单次 RAG 问答的 LLM 调用次数偏多** — router(1) + rerank(1) + relevance gate(1) + 答案(1) ≈ 4 次串行 LLM 调用，延迟和成本高。重构可考虑：合并 rerank+relevance、并行化、或在 distance gate 足够强时跳过部分 LLM 调用。

6. **大量宽泛 `except Exception`** — 各节点 / rerank / relevance / mcp 都用兜底 except，利于 demo 健壮性但掩盖根因。重构时建议细化异常类型 + 保留 trace。

7. **`/reload_kb` 同步阻塞** — 大 PDF 上传会阻塞请求（Django 侧已用 5/180s 分离超时缓解）。重构方向：后台任务 + 任务状态查询。

8. **死代码 / 杂项** — `documents/views.py` 顶部 `from urllib import response` 无用；`RAGSystem.retrieve(k=5)` 默认值实际不被使用（调用方都传 top_k）。

9. **安全（仅本地 demo 现状，部署前必改）** — `settings.py` 里 `SECRET_KEY` 硬编码入库、`DEBUG=True`、无鉴权、无 CSRF 之外的访问控制。任何对外部署前都要处理。

10. **耦合无重试 / 熔断** — Django→FastAPI 是裸 `requests.post`，无重试 / 退避 / 熔断。

11. **无流式输出（SSE）** — 当前一次性返回，README 已列为 future work。

12. **测试** — 只有 smoke 脚本，无 pytest 配置 / CI / 断言式单测。

---

## 9. 重构红线（动手前先确认的事）

- **不要**把 RAG / Agent / LangGraph 逻辑挪进 Django view（违背分层铁律）。
- **不要**给 calculator 引入 `eval` / `exec` / 函数调用 / 属性访问（安全边界）。
- **不要**改 `VectorStore.search` 的返回字段名（FAISS/Milvus 必须一致，前端和 trace 依赖）。
- **不要**去掉 `fallback_used` 这个防循环不变量。
- **不要**让 router 改写 `rag/llm/time/web_search` 的原始 query。
- **改距离阈值 / 换 embedding 模型 / 换向量库**时，必须重新评估证据门阈值，并跑评测集回归（见 README §15 的 22 题评测集）。
- 改任何 §5 的数据契约字段，必须全链路 grep 同步（含 `*.html` 模板）。

---

## 10. 编码约定

- 配置：一律走 `app/config.py` + `.env`，不在业务代码里散落 `os.getenv`。
- 日志：一律 `from app.logger_config import setup_logger` → `logger = setup_logger()`，日志带模块前缀如 `[choose_tool_node]`、`[rerank]`、`[context_sufficiency]`。
- 注释风格：中文注释 + 英文 docstring / prompt（沿用现状即可）。
- LLM prompt：rerank / relevance / router 的 prompt 都要求**只返回结构化输出**（JSON / 两行 / 索引列表），并在解析侧做清洗 + 兜底——新增 LLM 调用请遵循同样模式（prompt 约束 + 解析兜底 + trace 记录）。
- 新增工具：在 `tools.py` 注册到 `TOOLS`，并在 `choose_tool` 的 prompt 选择指引和 `normalize_decision` 白名单里同步加上；注意 `execute_tool` 按 name 分支传参。
- 向量库后端：通过 `factory.create_vector_store()` + `VECTOR_STORE` 环境变量切换，新后端继承 `BaseVectorStore` 并保持 search 返回格式。

---

## 11. 技术栈（参考）

- 后端 / AI：Python 3.11、FastAPI、LangGraph、FAISS（IndexFlatL2）、Milvus Lite（实验）、PyPDF、OpenAI-compatible API、DeepSeek `deepseek-chat`、`text-embedding-3-small`
- Agent / 工具：LangGraph StateGraph + conditional edges、LLM Router + 工具白名单 + 关键词兜底、AST-safe calculator、MCP（智谱 `web_search_prime`，langchain-mcp-adapters）
- 产品壳层：Django + SQLite + Django Templates + requests
- 工程：python-dotenv、logging、Git、smoke tests

---

## 12. 重构路线建议（结合 README §16 Future Work 与 §8 技术债）

短期（工程健壮性）：迁移 FastAPI lifespan、去全局可变状态、embedding 批处理+缓存、统一会话来源、补 pytest。
中期（RAG 质量）：source-aware filtering / 按论文分片索引 / hybrid retrieval（解决 README §15 指出的「同领域多论文跨 chunk 混淆」瓶颈）、基于评测集回调距离阈值、Agent Trace 加 `latency_ms` / `tool_status`。
长期（能力）：Reflection node（基于 `retry_count`）+ query rewrite 重检索、SSE streaming、更多 MCP 工具、把线上 DeepSeek 换成本地/私有化模型（适配科研敏感数据）、生产级 Milvus Server。

---

*本文件随重构持续更新；内容以当前真实代码为准，与 README 出现冲突时以代码 + 本文件为准。*
