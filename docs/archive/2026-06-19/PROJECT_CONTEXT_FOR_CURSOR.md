# PaperPilot 项目上下文说明

## 1. 项目名称

PaperPilot：论文 RAG 问答与多工具 Agent 系统

原项目名：

Paper-RAG-Agent-with-LangGraph

这是一个面向科研论文阅读、文献内容回溯和论文问答场景的 AI 应用工程项目。

项目不是简单的“大模型聊天 Demo”，也不是单文件 RAG 脚本，而是一个包含以下能力的完整最小闭环系统：

- 本地 PDF 论文知识库
- 文档加载、清洗、切分
- Embedding 向量化
- FAISS / Milvus 向量检索
- LLM Rerank
- 证据充分性判断
- LangGraph Agent 工具编排
- FastAPI AI 推理服务
- Django 产品壳层
- 聊天历史保存
- Retrieved Context 展示
- Agent Trace 展示

当前项目已经具备可运行、可演示、可继续迭代的工程基础。后续重构和完善时，应优先保持系统分层清晰，而不是盲目扩展功能。

---

## 2. 项目定位

PaperPilot 的核心目标是：

帮助用户围绕本地论文库进行问答、检索、对比和知识回溯，并让系统能够展示回答依据和 Agent 执行过程。

典型使用场景包括：

- 询问某篇论文的主要贡献
- 查询某篇论文解决了什么问题
- 回溯某个概念、方法、实验结论来自哪篇论文
- 对比两篇或多篇论文的方法差异
- 查看回答所依据的论文片段
- 观察 Agent 本轮选择了什么工具、为什么选择、是否触发 fallback
- 在本地论文知识不足时调用外部搜索工具

项目强调的是 AI 应用工程能力，而不是单纯追求模型回答效果。

它重点体现：

- RAG 系统如何组织
- Agent Workflow 如何编排
- 工具调用如何加边界
- 检索证据如何展示
- 证据不足时如何拒答
- AI 推理服务和 Web 产品层如何拆分
- 项目如何从原型逐步走向可维护结构

---

## 3. 当前系统主链路

当前完整流程如下：

用户上传 PDF 论文  
→ Django 保存到 data/ 目录  
→ Django 调用 FastAPI `/reload_kb`  
→ FastAPI 重新加载论文  
→ 文本清洗和 chunk 切分  
→ 构建 Embedding  
→ 构建向量索引  
→ 重建 RAGSystem  
→ 重建 AgentWorkflow  
→ 用户发起问题  
→ Agent 选择工具  
→ RAG 检索论文片段  
→ LLM Rerank 重排序  
→ 证据充分性判断  
→ 基于上下文生成回答  
→ 返回 answer / retrieved chunks / agent trace  
→ Django 页面展示结果  
→ 保存 ChatSession / ChatMessage

系统闭环可以概括为：

上传论文  
→ 重建知识库  
→ 发起问答  
→ Agent 工具路由  
→ RAG 检索  
→ Rerank  
→ 证据充分性判断  
→ 生成回答  
→ 展示检索片段  
→ 展示 Agent Trace  
→ 保存历史会话

---

## 4. 总体架构

项目采用分层架构。

主要分为六层：

1. Django Product Shell

负责产品壳层和用户界面。

包括：

- 聊天页面
- 上传论文页面
- 历史会话列表
- 会话详情页
- 调用 FastAPI
- 保存聊天历史
- 展示 Agent Trace
- 展示 Retrieved Context

Django 不应该直接实现核心 RAG、Agent 或 LangGraph 逻辑。

2. FastAPI AI Service

负责 AI 推理服务接口。

主要接口包括：

- POST `/ask`
- POST `/reload_kb`
- POST `/clear/{session_id}`

FastAPI 是 Django 与 AI 核心能力之间的服务边界。

3. LangGraph Agent Workflow

负责 Agent 工具调用流程编排。

当前节点包括：

- choose_tool
- execute_tool
- route_after_execute
- llm_fallback
- generate_answer

LangGraph 负责状态流转、条件边和 fallback。

4. Tools Layer

负责统一封装工具能力。

当前工具包括：

- rag
- calculator
- time
- web_search
- llm

工具层要保持输入边界清晰，不能让模型随意执行危险逻辑。

5. RAGSystem

负责论文知识库问答核心逻辑。

包括：

- 文档 chunk 管理
- Embedding
- 向量检索
- LLM Rerank
- 证据充分性判断
- 上下文组装
- 基于证据回答生成
- context metrics 返回

6. Storage Layer

当前包括：

- 本地 data/ PDF 文件目录
- FAISS 本地向量索引
- Milvus Lite 实验索引
- SQLite 聊天历史数据库

---

## 5. 当前目录结构说明

项目根目录大致如下：

Paper-RAG-Agent-with-LangGraph/
    app/
        main.py
        config.py
        data_loader.py
        llm_utils.py
        logger_config.py
        rag_system.py
        session_manager.py
        tools.py
        mcp_tools.py
        graph/
            builder.py
            nodes.py
            state.py
            workflow.py

    data/
        Paper1.pdf
        Paper2.pdf
        Paper3.pdf

    django_shell/
        manage.py
        chat/
            models.py
            views.py
            urls.py
            services/
                ai_client.py
        documents/
            views.py
            urls.py
        templates/
            chat/
                chat_home.html
                session_list.html
                session_detail.html
            documents/
                upload.html

    tests/
        smoke_test_imports.py
        smoke_test_graph.py
        smoke_test_nodes.py
        smoke_test_rag_trace.py
        smoke_test_tools_with_mcp.py
        smoke_test_web_search_tool.py

    requirements.txt
    README.md

---

## 6. 核心文件职责

### app/main.py

FastAPI 服务入口。

负责：

- 初始化 RAGSystem
- 初始化 tools
- 初始化 AgentWorkflow
- 暴露 `/ask`
- 暴露 `/reload_kb`
- 暴露 `/clear/{session_id}`
- 在 reload 后重新构建知识库和 AgentWorkflow

注意：

`/reload_kb` 不只是重建 RAG，还必须同步更新 AgentWorkflow，否则上传新论文后 Agent 仍可能使用旧的 RAG 实例。

---

### app/config.py

统一读取环境变量和配置。

包括：

- DeepSeek API 配置
- Embedding API 配置
- 数据目录配置
- MCP 搜索配置
- 向量库配置
- 模型名称配置

后续重构时，所有配置项应该继续集中在 config.py 或专门的 settings 模块中，不要散落到各个业务文件。

---

### app/data_loader.py

负责数据加载和预处理。

包括：

- 加载 PDF
- 加载 TXT
- 清洗文本
- 切分 chunk
- 给 chunk 添加 source / chunk_id 等元信息

当前 chunk 默认策略：

- chunk_size = 700
- chunk_overlap = 120

这个参数是针对英文论文内容调过的，重构时不要随意改大或改小。若要调整，应配合 eval 测试。

---

### app/rag_system.py

RAG 核心文件。

负责：

- 构建向量索引
- 执行向量检索
- 执行 LLM rerank
- 判断 context 是否充分
- 生成最终回答
- 返回 retrieved chunks 和 context_metrics

当前重点能力包括：

1. FAISS Top-K 检索

默认召回 top_k = 20。

2. LLM Rerank

对初步召回结果进行重排序，取前 10 个 chunk 进入后续判断。

Rerank 层要保留鲁棒性处理：

- 清理 markdown 代码块
- 从文本中提取 bracket list
- ast.literal_eval 安全解析
- 去重
- 补齐缺失索引
- 失败时 fallback 到原始顺序
- 将 fallback 状态写入 context_metrics

3. 双层证据充分性判断

第一层：Distance Gate

基于 FAISS 距离进行粗判断。

核心指标：

- num_chunks
- best_distance
- avg_top_distance
- max_best_distance
- max_avg_top_distance
- distance_gate_passed

第二层：Typed LLM Relevance Gate

先把问题分类为：

- BROAD
- SPECIFIC
- COMPARISON

再判断当前 retrieved passages 是否足够回答问题。

不同类型问题的策略不同：

- BROAD：主要贡献、研究动机、整体方法类问题，可以更宽松
- SPECIFIC：具体事实、术语、数据集、实验指标类问题，需要直接证据
- COMPARISON：多论文对比问题，需要覆盖被对比的双方或多方

BROAD 问题中，LLM 判 NO 时可以作为 soft warning，不一定硬拒答。

SPECIFIC / COMPARISON 问题中，证据不足时应该硬阻断，不能强行生成。

4. 证据不足时拒答

如果 context_sufficient = false，系统应该明确告诉用户：

当前知识库中没有检索到足够证据，不能可靠回答。

不能为了“看起来智能”而编造答案。

---

### app/tools.py

Agent 工具定义文件。

当前工具包括：

1. rag

用于论文知识库问答。

2. calculator

安全计算器。

注意：

calculator 不能使用裸 eval。

当前应该使用 Python AST 解析，只允许：

- 数字
- 加减乘除
- 括号
- 一元正负号

禁止：

- 函数调用
- 属性访问
- import
- 变量
- 任意代码执行

3. time

用于时间查询。

4. web_search

用于外部搜索。

当前通过 MCP / 智谱 web_search_prime 接入。

5. llm

通用大模型回答工具。

---

### app/mcp_tools.py

负责 MCP Web Search 封装。

需要处理 MCP 返回结果格式不稳定的问题。

当前已有双层 JSON 解析兜底：

- 外层 MCP 返回结构
- text 字段内部可能还是 JSON 字符串

重构时要保留这类兼容逻辑。

---

### app/session_manager.py

FastAPI 侧短期会话管理。

通过 session_id 维护多轮上下文。

注意：

Django 侧保存长期聊天记录。

FastAPI 侧 session_manager 负责当前推理上下文，不等同于数据库持久化。

---

### app/graph/state.py

定义 AgentState。

AgentState 是 LangGraph 各节点之间传递状态的核心结构。

当前应包含类似字段：

- query
- chat_history
- tool
- tool_input
- tool_result
- answer
- retrieved_chunks
- route_decision
- fallback_used
- retry_count
- context_sufficient
- context_metrics
- workflow_path
- error

后续增加新节点时，应该先明确是否需要扩展 AgentState，而不是随意在 dict 中塞字段。

---

### app/graph/nodes.py

定义 LangGraph 节点函数。

当前核心节点：

1. choose_tool

负责让 LLM Router 选择工具。

Router 应返回 JSON：

- tool
- input
- reason

需要做：

- JSON 清洗
- 工具白名单校验
- 非法工具降级到 llm
- 对 rag / llm / web_search 等工具保留原始 query
- 必要时根据关键词强制 web_search

2. execute_tool

根据工具名执行对应工具。

3. llm_fallback

当工具执行失败时，由 LLM 直接兜底回答。

4. generate_answer

整理最终输出。

注意：

节点函数不应该承担过多业务逻辑。复杂逻辑应下沉到 tools 或 rag_system。

---

### app/graph/builder.py

负责构建 LangGraph StateGraph。

当前流程：

START  
→ choose_tool  
→ execute_tool  
→ route_after_execute  
→ generate_answer 或 llm_fallback  
→ END

需要保留 conditional edge。

如果 execute_tool 失败，并且 fallback_used = false，则进入 llm_fallback。

如果已经 fallback 过，不要无限循环。

---

### app/graph/workflow.py

AgentWorkflow 封装层。

负责对外提供统一 invoke 方法。

FastAPI 不应该直接操作 LangGraph 内部细节，而是通过 AgentWorkflow 调用。

---

### django_shell/chat/models.py

Django 聊天模型。

主要包括：

- ChatSession
- ChatMessage

用于保存历史会话和消息。

---

### django_shell/chat/views.py

Django 聊天页面逻辑。

负责：

- 接收用户问题
- 调用 FastAPI `/ask`
- 保存 user message
- 保存 assistant message
- 渲染回答
- 渲染 Agent Trace
- 渲染 Retrieved Context

注意：

Django view 不应该直接调用 RAGSystem。

---

### django_shell/chat/services/ai_client.py

Django 调 FastAPI 的客户端封装。

后续重构时，所有请求 FastAPI 的逻辑应集中在这里，不要散落到 views.py。

---

### django_shell/documents/views.py

文档上传页面逻辑。

负责：

- 上传 PDF
- 保存到 data/
- 调用 FastAPI `/reload_kb`
- 展示 reload 结果
- 展示当前知识库文件列表

注意：

上传后必须触发 reload，否则新论文不会进入知识库。

---

## 7. Agent Workflow 设计

当前 Agent 不是完整 autonomous agent，也不是完整 ReAct agent。

它更准确的定位是：

轻量级 Tool Calling + RAG + LangGraph Workflow。

当前流程：

START  
→ choose_tool  
→ execute_tool  
→ route_after_execute  
→ generate_answer  
→ END

异常流程：

START  
→ choose_tool  
→ execute_tool  
→ route_after_execute  
→ llm_fallback  
→ generate_answer  
→ END

关键设计点：

1. LLM Router 负责工具选择，但不能完全信任 LLM 输出。

2. Router 输出必须经过 normalize_decision。

3. 工具选择必须受白名单约束。

4. 工具执行失败后可以 fallback，但只能 fallback 一次。

5. workflow_path 必须记录实际执行路径，供前端展示。

6. Agent Trace 必须返回前端，方便调试和演示。

---

## 8. RAG 流程设计

RAG 内部流程：

用户问题  
→ 问题向量化  
→ VectorStore 检索 top_k  
→ LLM Rerank  
→ 取前若干 chunks  
→ Distance Gate  
→ Typed LLM Relevance Gate  
→ 判断 context_sufficient  
→ 如果充分，基于上下文生成回答  
→ 如果不充分，返回证据不足说明  
→ 返回 answer / chunks / context_metrics

需要注意：

RAG 回答必须优先基于 retrieved context。

如果没有足够证据，不要强行回答。

context_metrics 是系统可解释性的关键，不能在重构中删掉。

---

## 9. VectorStore 设计

项目最初使用 FAISS 本地向量索引。

当前已经开始引入 VectorStore 抽象，用于支持 FAISS 和 Milvus 后端切换。

目标接口：

- build(chunks)
- search(query, k)

当前后端：

1. FaissVectorStore

默认本地向量索引实现。

适合：

- 本地 Demo
- 快速实验
- 小型论文库

2. MilvusVectorStore

基于 Milvus Lite 的实验实现。

适合：

- 验证未来迁移向量数据库的可行性
- 接近真实 AI Agent 平台的技术结构
- 为后续 metadata filter / 多知识库隔离 / collection 管理预留结构

当前 Milvus 只是实验，不要把它误认为生产级 Milvus Server 部署。

重构时应保持：

RAGSystem 不直接依赖 FAISS 或 Milvus 具体实现，而是依赖 VectorStore 抽象。

---

## 10. API 设计

### POST /ask

执行一次 Agent 问答。

请求字段：

- session_id
- question

返回字段：

- session_id
- question
- answer
- chunks
- agent_trace

agent_trace 中应包含：

- route_decision
- tool_used
- tool_input
- fallback_used
- context_sufficient
- context_metrics
- retry_count
- workflow
- error

### POST /reload_kb

重新加载 data/ 目录下的论文文件，并重建知识库。

必须完成：

- 重新加载 PDF
- 重新切分 chunks
- 重建 embedding / vector index
- 重建 RAGSystem
- 重建 AgentWorkflow

返回字段：

- status
- message
- total_docs
- total_chunks

### POST /clear/{session_id}

清空 FastAPI 侧指定 session 的短期上下文。

---

## 11. Django 产品壳层设计

Django 当前不是核心 AI 层，而是产品展示层。

它负责让项目从“后端接口 Demo”变成“可演示产品”。

Django 页面包括：

1. Chat Home

功能：

- 输入 session_id
- 输入问题
- 展示回答
- 展示 Agent Trace
- 展示 Retrieved Context
- 展示最近会话

2. Upload Papers

功能：

- 上传 PDF
- 保存到 data/
- 调用 `/reload_kb`
- 展示当前知识库文件

3. Session History

功能：

- 查看历史会话列表

4. Session Detail

功能：

- 查看某个 session 的历史问答

后续重构时，Django 仍应保持轻量，不要把 AI 核心逻辑搬进 Django。

---

## 12. 当前工程亮点

这个项目最重要的工程亮点包括：

1. 分层清晰

Django、FastAPI、LangGraph、Tools、RAGSystem、VectorStore 各自职责相对独立。

2. Agent Trace 可解释

系统不仅返回答案，还返回工具选择、路由理由、fallback 状态、context_sufficient、context_metrics 和 workflow path。

3. RAG 防幻觉设计

通过 Distance Gate + Typed LLM Relevance Gate 判断证据是否足够。

4. Rerank 鲁棒性

LLM rerank 输出不稳定时，系统可以解析、补齐、fallback，并记录 trace。

5. 工具安全边界

calculator 不使用 eval，而是 AST 白名单解析。

6. 上传后动态更新知识库

Django 上传 PDF 后会触发 FastAPI `/reload_kb`，使新论文进入 RAG 检索链路。

7. LangGraph 条件分支

工具执行失败后进入 llm_fallback，并通过 fallback_used 防止无限循环。

8. VectorStore 抽象

为 FAISS 到 Milvus 的迁移预留结构。

---

## 13. 当前已知问题和重构方向

### 13.1 RAG 检索精度问题

当前主要瓶颈不是简单 top_k 不够，而是多篇相近论文混合检索时容易发生 source-level confusion。

例如：

- 用户问 Paper1
- 检索结果可能混入 Paper2 或 Paper3 的相似 chunk
- SPECIFIC / COMPARISON 问题尤其容易受影响

后续优先方向：

- source-aware retrieval
- 根据问题中提到的 paper name 做 metadata filter
- 支持按论文单独建索引
- 支持 hybrid retrieval
- 支持关键词 + 向量混合召回
- 在 Retrieved Context 中更清楚展示 source 分布

---

### 13.2 Eval 体系仍然偏轻量

当前已有 smoke test 和部分 eval 记录，但还不够系统化。

后续可以补：

- docs/eval/eval_questions.json
- docs/eval/eval_run_result_top_k20.md
- scripts/eval_rag.py
- expected sources
- expected keywords
- actual retrieved sources
- context_sufficient
- answer quality note

目标不是做复杂 benchmark，而是形成小规模可重复的 RAG 质量观察闭环。

---

### 13.3 Agent 能力仍然轻量

当前 Agent 更接近工具路由工作流，不是复杂 autonomous Agent。

后续可以探索：

- query rewrite
- reflection node
- retry retrieval
- 多轮问题压缩
- session memory summary
- 更丰富 MCP tools

但不要一上来做复杂 Multi-Agent。

---

### 13.4 Django 页面仍然是 Demo 级

当前 Django 主要用于演示。

后续可以优化：

- 页面布局
- Agent Trace 展示层次
- Retrieved Context 可读性
- 上传状态提示
- 文档列表管理
- 删除文档
- reload 进度反馈

但不要过早做复杂权限、多租户、SaaS 后台。

---

## 14. Cursor 重构时的硬约束

使用 Cursor 重构项目时，请遵守以下约束：

1. 不要把 RAG 核心逻辑写进 Django views。

2. 不要把 LangGraph 节点逻辑写进 FastAPI endpoint。

3. 不要删除 Agent Trace。

4. 不要删除 context_metrics。

5. 不要让 RAG 在证据不足时强行回答。

6. 不要用 eval 实现 calculator。

7. 不要把配置项散落在业务代码里。

8. 不要在没有测试的情况下随意修改 chunk_size、overlap、top_k、distance threshold。

9. 不要把 Milvus 实验直接当成生产部署。

10. 不要过早加入复杂登录、权限、多租户、前端框架。

11. 不要为了“代码少”把分层重新揉成一个大文件。

12. 新增能力前，先明确它属于哪一层：

- Django 展示层
- FastAPI 服务层
- LangGraph 编排层
- Tools 工具层
- RAGSystem 检索生成层
- VectorStore 存储检索层

---

## 15. 推荐的后续重构优先级

### P0：保持项目可运行

先保证：

- FastAPI 可以启动
- Django 可以启动
- `/ask` 可用
- `/reload_kb` 可用
- RAG 问答可用
- Agent Trace 正常返回
- Retrieved Context 正常展示

不要先大改架构导致项目跑不起来。

---

### P1：整理 VectorStore 抽象

目标：

- RAGSystem 只依赖 VectorStore 接口
- FAISS 和 Milvus 实现解耦
- search 返回结构统一
- metadata 信息保留 source / chunk_id / distance / retrieval_rank

---

### P2：提升 source-aware retrieval

目标：

当用户问题明确提到某篇论文时，优先只检索该论文或提高该论文 chunk 权重。

例如：

- “What is the main contribution of Paper1?”
- “Compare Paper2 and Paper3”
- “Does Paper1 mention XXX?”

可做：

- 从 query 中识别 paper name
- 对 source metadata 过滤
- 对 comparison 问题分别检索多个 source
- 合并结果后 rerank

---

### P3：完善 eval 闭环

目标：

建立最小但稳定的评测机制。

建议增加：

- docs/eval/eval_questions.json
- scripts/eval_rag.py
- docs/eval/eval_run_result_top_k20.md

每条 eval 记录：

- question
- question_type
- expected_source
- expected_keywords
- retrieved_sources
- context_sufficient
- answer_summary
- manual_note

---

### P4：优化 Agent Trace 前端展示

目标：

让 Django 页面更适合演示和面试讲解。

Agent Trace 可分组展示：

- Routing
- Tool Execution
- RAG Context
- Fallback
- Workflow Path

Retrieved Context 可展示：

- source
- retrieval_rank
- distance
- text snippet

---

### P5：增加轻量 Memory

不要一开始做复杂长期记忆。

可以先做：

- 多轮 chat history 截断
- 超过 N 轮后总结为 session summary
- summary 参与后续 Agent 输入

这可以作为 Agent Memory 的最小版本。

---

## 16. Cursor 执行任务时的建议方式

后续让 Cursor 改代码时，不要只说“帮我优化项目”。

应该使用类似任务描述：

### 示例任务 1：整理 VectorStore

请阅读 PROJECT_CONTEXT_FOR_CURSOR.md 和 README.md。  
现在需要重构向量检索层。  
目标是让 RAGSystem 不直接依赖 FAISS，而是依赖 VectorStore 抽象。  
请先分析当前 RAGSystem 中和 FAISS 强绑定的位置，然后给出最小改动方案。  
不要修改 Django。  
不要修改 Agent Trace 字段。  
不要改变 `/ask` 返回结构。  
先输出计划，不要直接大范围改代码。

---

### 示例任务 2：增加 source-aware retrieval

请阅读 PROJECT_CONTEXT_FOR_CURSOR.md。  
当前问题是多篇相似论文混合检索时容易出现 source-level confusion。  
请为 RAGSystem 增加一个最小版本的 source-aware retrieval。  
当用户问题中明确出现 Paper1 / Paper2 / Paper3 等 source 名称时，优先过滤对应 source 的 chunks。  
如果是 comparison 问题，分别检索双方 source，再合并结果进入 rerank。  
不要删除原有 distance gate 和 typed LLM relevance gate。  
不要改变前端展示结构。

---

### 示例任务 3：完善 eval 脚本

请基于当前项目增加一个轻量 RAG eval 脚本。  
新增 docs/eval/eval_questions.json 和 scripts/eval_rag.py。  
eval 脚本需要读取问题，调用现有 RAGSystem 或 FastAPI `/ask`，记录 retrieved sources、context_sufficient、answer 和 manual note。  
不要引入复杂 benchmark 框架。  
目标是方便人工观察 RAG 检索质量。

---

### 示例任务 4：优化 Agent Trace 页面

请优化 Django chat_home.html 中 Agent Trace 的展示方式。  
不要改 FastAPI 返回结构。  
只调整模板展示。  
目标是把 route_decision、tool_used、context_sufficient、context_metrics、workflow_path 分组展示，让页面更适合演示。  
保持简单 CSS，不引入重前端框架。

---

## 17. 项目后续发展方向

PaperPilot 后续不是要立刻变成完整科研 SaaS。

更合理的发展方向是：

1. 让 RAG 检索更准

特别是多论文场景下的 source-aware retrieval。

2. 让 Agent 执行更可解释

保留和增强 Agent Trace。

3. 让评测更可复现

用小规模 eval 闭环持续观察改动效果。

4. 让产品壳层更像真实应用

优化上传、聊天、历史、证据展示。

5. 让工程结构更适合继续扩展

保持 FastAPI / LangGraph / Tools / RAG / Django 分层。

---

## 18. 当前项目一句话总结

PaperPilot 是一个面向论文阅读场景的 RAG + Agent 应用工程项目，使用 FastAPI 提供 AI 推理服务，LangGraph 编排工具调用流程，RAGSystem 完成论文检索、rerank 和证据充分性判断，Django 提供上传、问答、历史会话、Retrieved Context 和 Agent Trace 展示。

后续重构时，最重要的不是盲目增加功能，而是在保持项目可运行的前提下，逐步提升 RAG 检索质量、Agent 可解释性、代码分层清晰度和工程可维护性。