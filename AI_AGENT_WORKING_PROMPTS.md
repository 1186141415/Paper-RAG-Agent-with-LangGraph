# AI_AGENT_WORKING_PROMPTS.md — PaperPilot AI Coding 协作提示词

本文档用于指导 Cursor / Claude Code / Codex / Gemini CLI 等 AI coding agent 参与 PaperPilot 项目开发。

使用方式：

每次开始一个新任务时，先让 AI 读取：

1. README.md
2. PROJECT_CONTEXT.md
3. AGENTS.md
4. PRODUCT_VISION.md
5. 当前任务相关代码文件

然后再让 AI 输出分析和计划。

---

## 1. 通用启动提示词

你现在要参与 PaperPilot 项目的开发。

请先阅读以下文件：

- README.md
- PROJECT_CONTEXT.md
- AGENTS.md
- PRODUCT_VISION.md

阅读后，请先不要直接改代码。

请先输出：

1. 你对这个项目的理解。
2. 当前系统的主要分层。
3. 本次任务可能影响哪些层。
4. 本次任务涉及哪些关键数据契约。
5. 你准备怎么改。
6. 你认为有哪些风险。
7. 你准备如何验证。

注意：

- 不要把 RAG / Agent / LangGraph 逻辑写进 Django view。
- 不要破坏 `/ask`、`/reload_kb`、Agent Trace、Retrieved Context。
- 不要删除 context_metrics。
- 不要让 calculator 使用 eval / exec。
- 不要随意修改 chunk_size、top_k、distance threshold。
- 不要一次性大重构。
- 请优先做最小可运行改动。
- 如果发现文档和代码不一致，以代码为准，并指出差异。

---

## 2. Cursor 使用提示词

请基于当前代码库和项目文档，帮助我完成一个小步重构任务。

要求：

1. 先扫描项目结构，不要直接改代码。
2. 找出和本任务相关的文件。
3. 给出最小改动计划。
4. 明确哪些文件会被修改。
5. 保持现有功能可运行。
6. 改完后告诉我如何手动验证。
7. 如果需要测试，请优先补轻量 smoke test 或 pytest。
8. 不要引入重型依赖。
9. 不要重写整个项目。

请按以下格式回复：

- 当前理解
- 影响范围
- 修改计划
- 风险点
- 具体改动
- 验证方式
- 后续建议

---

## 3. Claude Code 使用提示词

请以资深 AI 应用工程师 + Python 后端工程师的视角，协助我改造 PaperPilot。

你需要特别关注：

1. 架构边界是否被破坏。
2. RAG 证据链是否仍然可靠。
3. Agent Trace 是否仍然完整。
4. 错误处理是否更清晰。
5. 是否存在过度设计。
6. 是否保持项目可运行。

工作方式：

- 先读文档。
- 再读相关代码。
- 先给 plan。
- 等我确认后再改。
- 改动要小。
- 每次只解决一个核心问题。
- 修改后给出 diff 总结和验证步骤。

不要做：

- 不要把 FastAPI、Django、LangGraph、RAGSystem 混在一起。
- 不要为了优雅重构导致功能不可运行。
- 不要新增复杂抽象但没有实际收益。
- 不要删除现有 trace 字段。
- 不要把 demo 改成不可维护的半成品 SaaS。

---

## 4. Codex 使用提示词

请在当前仓库中完成一个可验证的小任务。

开始前请阅读：

- PROJECT_CONTEXT.md
- AGENTS.md
- PRODUCT_VISION.md

任务要求：

1. 只修改和任务直接相关的文件。
2. 保持现有 API 响应结构兼容。
3. 保持现有测试或 smoke test 可运行。
4. 如果发现潜在 bug，请先说明，不要擅自扩大修改范围。
5. 修改完成后，请提供：
   - 修改文件列表
   - 每个文件改了什么
   - 如何运行
   - 如何验证
   - 是否存在未解决问题

请优先使用现有项目风格，不要引入不必要的新框架。

---

## 5. Gemini CLI 使用提示词

请作为代码审查和架构分析助手，帮我分析 PaperPilot 当前任务。

请先阅读：

- PROJECT_CONTEXT.md
- AGENTS.md
- PRODUCT_VISION.md

然后输出：

1. 当前任务是否符合 PaperPilot 的长期产品方向。
2. 当前任务属于哪一层：
   - Django 产品层
   - FastAPI 服务层
   - LangGraph 编排层
   - Tools 工具层
   - RAGSystem 检索生成层
   - VectorStore 层
   - Eval / 测试层
   - 文档层
3. 当前任务是否可能破坏核心数据契约。
4. 当前任务是否有更小的实现路径。
5. 当前任务完成后如何验证。
6. 是否值得现在做，还是应该排到后面。

请重点帮我防止：

- 过早复杂化
- 为了炫技加功能
- 没有 eval 的检索优化
- 破坏原有演示闭环
- 修改范围过大

---

## 6. 每次发任务时推荐格式

以后不要只说：

“帮我优化 PaperPilot。”

要说：

任务名称：
本次要解决的问题：
为什么现在要做：
涉及文件：
不允许改的内容：
验收标准：
验证方式：

示例：

任务名称：增加 source-aware retrieval 最小版本

本次要解决的问题：
多篇相似论文混合检索时，用户问 Paper1，检索结果可能混入 Paper2 / Paper3，导致回答证据来源混乱。

为什么现在要做：
这是 PaperPilot 从面试项目升级为真实科研工具的关键问题。科研场景下，多论文对比和来源准确性比回答流畅更重要。

涉及文件：
- app/rag_system.py
- app/vector_store/
- tests/
- README.md 或 eval 文档

不允许改的内容：
- 不要修改 Django 页面结构。
- 不要删除 Agent Trace。
- 不要删除 context_metrics。
- 不要改变 /ask 响应结构。
- 不要删除原有 distance gate 和 typed LLM relevance gate。

验收标准：
- 当问题明确提到某篇论文 source 时，优先检索该 source。
- comparison 问题能分别检索双方 source，再合并结果。
- retrieved_chunks 中保留 source / distance / retrieval_rank。
- 旧的普通问答仍然可用。

验证方式：
- 运行现有 smoke test。
- 手动测试单论文问题。
- 手动测试多论文 comparison 问题。
- 记录 retrieved_sources 和 context_sufficient。

---

## 7. 当前推荐任务队列

### P0：保持系统稳定

- FastAPI lifespan 替代 on_event
- 去除 main.py 中脆弱的全局可变状态
- 统一 data/ 路径配置
- 补 /health 接口
- 保证 /ask 和 /reload_kb 稳定

### P1：RAG 质量提升

- source-aware retrieval
- eval_questions.json
- scripts/eval_rag.py
- eval_run_result.md
- hybrid retrieval 最小版本
- rerank 前后对比记录

### P2：科研产品能力

- 论文结构化笔记
- 多论文对比表
- Related Work 素材生成
- Markdown 导出
- Evidence-based summary

### P3：Agent 能力增强

- query rewrite
- retry retrieval
- session summary memory
- tool latency trace
- tool status trace
- 更丰富 MCP tools

### P4：工程化增强

- Docker
- PostgreSQL
- Redis
- SSE streaming
- 请求超时与重试
- 基础鉴权
- 日志与成本统计

---

## 8. 最重要的判断

PaperPilot 的长期价值不在于“功能堆得多”，而在于：

1. 能否准确找到论文证据。
2. 能否避免无证据生成。
3. 能否解释 Agent 为什么这样回答。
4. 能否支持真实科研工作流。
5. 能否通过 eval 持续观察质量变化。
6. 能否保持清晰、可维护、可演示的工程结构。

所有 AI coding agent 在改代码时，都必须围绕这些目标工作。