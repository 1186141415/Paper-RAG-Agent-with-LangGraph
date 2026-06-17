# SAAS_ROADMAP.md — PaperPilot SaaS 化演进路线

## TL;DR / 当前只做

- **当前阶段（Stage 0 → 1）**：运行稳定、RAG 质量、Agent Trace、eval 闭环、**会话一致性**（含 Django→FastAPI 传 `chat_history` 的 P0 最小修复）。
- **不要现在做**：多租户、K8s、Spring Boot 全量重写、复杂 RBAC、Redis / PostgreSQL / Celery / Milvus Server / 对象存储等 Stage 2/3 组件。
- **SaaS 化**是长期产品演进路线，**不是**当前重构任务；下文 Stage 2/3 与附录架构图仅供远期参考。

---

> **WARNING FOR AI AGENTS**
>
> Do **not** implement Stage 2/3 components such as Redis, PostgreSQL, Celery, Milvus Server, Object Storage, RBAC, multi-tenancy, or K8s **unless the user explicitly asks for that specific task**.
>
> **Current priority is Stage 0 → Stage 1 only.** Do not treat diagrams or tables in the appendix as implementation backlog.

---

> 本文档说明：PaperPilot **当前不是**生产级 SaaS，也**不必**为了“像 SaaS”而推翻现有轻量架构。  
> 它描述的是：在保持当前 **FastAPI AI Engine + Django 产品壳层** 可运行、可演示、可迭代的前提下，如何**逐步**演进为面向高校科研场景的 SaaS 化 AI Agent / RAG 平台。
>
> 与 `docs/project/PRODUCT_VISION.md`（产品愿景）、`docs/project/PROJECT_CONTEXT.md`（代码契约与技术债）、`docs/project/AI_AGENT_WORKING_PROMPTS.md`（协作节奏）配套阅读。  
> 出现冲突时：**数据契约与运行中的代码** 以 `docs/project/PROJECT_CONTEXT.md` 为准；**阶段优先级** 以本文 + `docs/project/PRODUCT_VISION.md` 为准。

---

## 1. 文档目的与边界

### 1.1 我们在说什么

- **SaaS 化**：多用户、可隔离的项目空间、可控权限、稳定部署、可观测、可计费（若需要），以及围绕「上传论文 → 建库 → 问答 → 追溯证据」的完整产品闭环。
- **不是**：用技术栈清单替换产品思考；也不是把当前 Demo 一次性重写成“大厂微服务”。

### 1.2 我们在不否定什么

当前架构是**有意为之**的工程样例，价值在于：

| 现状 | 价值 |
|---|---|
| FastAPI（`:8000`）+ Django（`:8001`）双进程 | AI 推理与产品壳层解耦，便于独立扩展与部署 |
| SQLite + 本地 `data/` PDF | 零运维成本、本地 Demo、面试与实验友好 |
| LangGraph 轻量 Workflow | 可解释的 Tool Calling + RAG，非过度自治 Agent |
| FAISS / Milvus Lite + `VectorStore` 抽象 | 验证检索后端可切换，为 Milvus Server 预留结构 |
| Agent Trace + `context_metrics` | 科研场景「可信 > 流畅」的可观测基础 |

**结论**：SaaS 化是**演进**，不是**否定**；每一阶段都应保持 `/ask`、`/reload_kb`、Retrieved Context、Agent Trace 可用。

### 1.3 当前阶段优先级（不变）

在触及「企业级 SaaS」之前，团队资源应优先投入：

1. **运行稳定**：启动生命周期、路径与状态管理、健康检查、超时与错误可理解。
2. **RAG 质量**：多论文 source 混淆、source-aware retrieval、阈值与评测集回归。
3. **Agent Trace**：字段稳定、可调试、可对外讲解裁决过程。
4. **Eval 闭环**：固定问题集、可重复脚本、变更前后对比记录。
5. **会话一致性**：Django 持久化历史与 FastAPI 推理历史对齐（见 §5.1 P0；完整多 worker 方案属 Stage 2）。

上述五项是 **Stage 0 → Stage 1** 的硬门槛；未达标前，不宜大规模投入多租户、K8s、复杂权限。

---

## 2. SaaS 化的核心：不是堆技术

企业级 SaaS 的竞争力来自**产品能力与运营约束**，而非组件数量。

| 维度 | 说明 | 当前 Demo 缺口（示意） |
|---|---|---|
| **用户与身份** | 谁在使用、如何注册/登录、会话归属谁 | 无登录；`session_id` 由用户手填 |
| **项目空间** | Workspace / Project / 论文库边界 | 全局单一 `data/` 目录 |
| **权限（RBAC）** | 上传、删库、导出、管理成员 | 无鉴权；`DEBUG=True` 本地假设 |
| **数据隔离** | 向量、文件、会话、Trace 按租户/项目隔离 | 全进程共享索引与 SQLite |
| **任务状态** | 上传、建库、大批量 embedding 异步可查询 | `/reload_kb` 同步阻塞 |
| **可观测性** | 日志、指标、Trace、成本、慢查询 | 有 Trace，无统一监控与成本账 |
| **稳定部署** | 多实例、滚动发布、配置与密钥管理 | 单机双进程 + 手工启动 |

长期技术组件（MySQL、Redis、Milvus Server、对象存储、任务队列等）都是为**服务上表能力**而存在；没有清晰的产品边界就引入中间件，只会增加运维负担。**当前不要为实现上表能力而提前上栈。**

---

## 3. 阶段总览：从 Demo 到企业级 SaaS

```text
Stage 0  面试 / 作品级 Demo（当前主体）          ← AI Agent 默认工作范围
    ↓    运行稳定 + RAG 质量 + Trace + eval + 会话一致性
Stage 1  作品级科研 RAG Agent（可对外试用的小团队）  ← AI Agent 默认工作范围
    ↓    source-aware retrieval、eval 闭环、embedding 缓存等
Stage 2  科研助手 SaaS 原型（课题组试点）          ← 非当前；见附录
    ↓    租户隔离 + RBAC + 异步建库 + Milvus Server 等
Stage 3  企业级高校科研 SaaS（规模化、合规、SLA）    ← 非当前；见附录
```

与 `PRODUCT_VISION.md` 的 Stage 0–3 对齐。§4–§5 为**当前与近期**；§6–§7 为**远期规划**，勿与当前排期混淆。

---

## 4. Stage 0 — 当前：轻量双进程 Demo（保持）

### 4.1 架构快照（当前真实形态）

```text
User → Django（页面、上传、SQLite 历史）
         ↓ HTTP
       FastAPI（/ask、/reload_kb、/clear）
         ↓
       LangGraph → Tools → RAGSystem → VectorStore（FAISS / Milvus Lite）
```

### 4.2 本阶段已具备

- 完整问答闭环、双层证据门、拒答策略、Agent Trace、Retrieved Context。
- PDF 上传触发知识库重建；向量后端可通过 `.env` 切换实验性 Milvus Lite。
- Smoke tests；README / PROJECT_CONTEXT 文档化数据契约。

### 4.3 本阶段明确不是

- 多租户、计费、SLA、高可用集群。
- 生产级向量库、对象存储、分布式任务队列。
- 完整 Auth / RBAC / 审计日志。

**态度**：Stage 0 是**资产**，不是「技术债羞耻」；SaaS 路线在其上叠加，而非推倒重来。

---

## 5. Stage 1 — 作品级科研 Agent（近期目标 · **当前默认排期**）

**目标用户**：硕士生、博士生、小课题组；**部署形态**：单机或单台云主机仍可接受。

### 5.1 现在要做（P0 / P1）

与 `PROJECT_CONTEXT.md` §8、`AI_AGENT_WORKING_PROMPTS.md` P0/P1 一致：

| 优先级 | 事项 | 价值 |
|---|---|---|
| P0 | FastAPI `lifespan` 替代 `on_event`；`rag`/`workflow` 迁入 `app.state` | 启动规范、为多 worker 铺路 |
| P0 | `DATA_DIR` 等路径在 config 单点绝对化 | 消除 Django / FastAPI cwd 耦合 |
| P0 | `/health` | 可探测、部署自检 |
| P0 | **Django→FastAPI 传 `chat_history`：P0 最小修复**（`/ask` 请求体必带最近 N 轮；FastAPI 以请求体为准写回 SessionManager） | 修复会话双写：FastAPI 重启后 Django 页面历史与推理上下文一致 |
| P1 | `docs/eval/eval_questions.json` + `scripts/eval_rag.py` + 结果记录 | RAG 改动可回归 |
| P1 | source-aware retrieval 最小版 | 缓解多论文 source 混淆（产品核心痛点） |
| P1 | Embedding 批处理 + 磁盘缓存（或修正 README 表述） | 降低 reload 成本与 API 费用 |
| P1 | Agent Trace 字段小步统一（如 `tool_status`） | 试用者调试体验 |

**会话相关边界（勿混淆）**：

| 事项 | 阶段 | 说明 |
|---|---|---|
| Django→FastAPI 传 `chat_history` | **Stage 1 · P0** | 最小修复，**当前应做**，不是可选项 |
| 统一会话来源 + 多 worker 共享（如 Redis / DB 只读历史） | **Stage 2** | 更大改造；在 P0 最小修复验证后再做 |

**原则**：仍可使用 SQLite、本地或单机 `data/`；不强制 MySQL / K8s。

### 5.2 未来再做（Stage 1 内可排队，非阻塞）

- 页面 UI 精修、Markdown 导出、论文结构化笔记（Evidence 绑定）。
- Hybrid retrieval（关键词 + 向量）实验版。
- `/reload_kb` 后台化（见 Stage 2 任务队列）。

---

## 6. Stage 2 — 科研助手 SaaS 原型（中期 · **非当前**）

> ⚠️ **本节为远期规划，不代表当前要实现。** 除非用户明确要求某一 Stage 2 专项（且说明验收标准），AI Agent 不得主动引入 Redis、PostgreSQL、Celery、Milvus Server、对象存储、RBAC、多租户等。

**目标用户**：课题组、实验室、企业技术调研小组；**部署形态**：Docker Compose 或少量 VM；开始要求**数据隔离**与**基础权限**。

### 6.1 产品能力目标

- **Tenant / Workspace / Project**：用户属于组织；论文库、会话、向量 collection 挂在 Project 下。
- **Auth**：注册、登录、JWT/Session；API Key 供脚本调用（可选）。
- **RBAC（最小集）**：Owner / Member / Viewer；上传与删库需写权限。
- **任务状态**：上传 PDF、解析、embedding、建索引 → `pending / running / success / failed`，前端可轮询或 SSE 推送进度。
- **可观测性 v1**：结构化日志、请求 ID、按 session/project 的 LLM 调用次数与耗时汇总（成本跟踪雏形）。

### 6.2 推荐技术映射（逐步实现，不要求一步到位）

| 能力 | 方向性技术选型 | 说明 |
|---|---|---|
| AI 推理 | **FastAPI AI Engine**（延续） | 保持 RAG/Agent/LangGraph 不进 Django；可拆为独立服务扩缩容 |
| 业务后台 | **Django**（延续）或 **DRF / Spring Boot**（备选） | 用户、项目、权限、上传元数据、任务状态；Spring 仅在有 Java 基建时考虑，**非默认、非当前** |
| 业务库 | **PostgreSQL** 或 **MySQL** | 替代 SQLite 存用户、项目、消息、任务；会话与审计 |
| 缓存 / 会话 | **Redis** | 短期对话缓存、限流、任务进度、分布式锁；**统一会话来源 + 多 worker 共享**在此阶段落地 |
| 向量 | **Milvus Server**（非 Lite） | 按 Project collection 或 partition；metadata filter（source、paper_id）。**硬约束**：从 FAISS / Milvus Lite 切换到 Milvus Server 后，距离尺度可能变化；当前 evidence gate 的 `2.2` / `2.4` 阈值**必须**基于 eval 集重新标定，**禁止**直接沿用 |
| 文件 | **Object Storage**（S3 / MinIO / 校内 OSS） | PDF 与导出物；`data/` 目录升级为桶路径 |
| 长任务 | **Async task queue**（Celery / RQ / ARQ + Redis） | `reload_kb`、批量 embedding、eval 批跑 |
| 实时 | **SSE** 优先，**WebSocket** 按需 | 流式答案、建库进度；SSE 实现成本更低 |
| 部署 | **Docker** + **CI/CD** | 镜像、环境变量、迁移脚本；仍可不使用 K8s |

### 6.3 进入 Stage 2 后优先做 vs 更晚再做（**均非当前**）

| 进入 Stage 2 后优先做（非当前） | 更晚再做（Stage 2 后期或 Stage 3） |
|---|---|
| 数据模型草图：User / Workspace / Project / Document / ChatSession | 复杂审批流、课题组计费 |
| Milvus Server 单环境 PoC；验证与 `VectorStore` 契约一致；**eval 重标定距离门阈值** | 多区域 Milvus、跨集群复制 |
| 对象存储上传 PDF；Django 只存元数据与 URL | 病毒扫描、PDF 预览服务 |
| Celery 化 `reload_kb`；任务表 + 状态 API | 自动扩缩容 worker |
| JWT + 项目级 API 鉴权 | 细粒度字段级权限、SSO（校内统一身份） |
| Docker Compose 一键起全栈 | K8s Helm、HPA、金丝雀 |

架构示意见 **附录 A**（远期参考，非实现清单）。

---

## 7. Stage 3 — 企业级高校科研 SaaS（长期 · **非当前**）

> ⚠️ **本节为远期规划。** 不要因阅读本节而提前引入 K8s、集中监控平台、计费等。

**目标**：多课题组、校内或商用部署、合规与 SLA、可运维。

### 7.1 产品与安全

- 强隔离：租户级网络/逻辑隔离、加密 at rest、备份与恢复 RPO/RTO 目标。
- 合规：数据驻留、导出与删除（被遗忘权）、操作审计、敏感课题「仅本地模型」模式。
- SLA 与支持：状态页、告警、值班 runbook。

### 7.2 技术栈深化（未来再做）

| 领域 | 方向 |
|---|---|
| 编排 | **Kubernetes**、多可用区、Ingress、密钥（Vault / K8s Secrets） |
| 观测 | **Monitoring**（Prometheus/Grafana）、集中 **logging**（Loki/ELK）、链路追踪 |
| 成本 | Token / embedding 按 Project 账单、配额与告警 |
| AI | 可插拔 LLM Provider、私有化模型、本地 embedding |
| 集成 | 校内统一认证、文献 API、Zotero 等（按需求） |

### 7.3 明确不在 Stage 3 之前过度投入

- 微服务拆分过细（「每个工具一个服务」）。
- 过早多区域、多活。
- 无 eval 支撑的检索「算法炫技」。
- 与科研主流程无关的功能堆叠（支付、营销自动化等）。

---

## 8. 现在要做 vs 未来再做（总表）

### 8.1 现在要做（Stage 0 → 1，**默认排期**）

- 运行稳定：lifespan、`app.state`、绝对路径、`/health`。
- **会话一致性（P0）**：Django→FastAPI 传 `chat_history`（最小修复，**必做**）。
- RAG 质量：eval 闭环、source-aware retrieval、证据门阈值随评测回归。
- 可解释性：Agent Trace / `context_metrics` 稳定，不删字段。
- 文档与代码一致：如 embedding 缓存表述与实现。
- 保持双进程分层；Django 仍不写 Agent 逻辑。

### 8.2 未来再做（Stage 2+）

- 多租户与 Workspace / Project 数据模型及隔离策略。
- Auth、RBAC、API Key、SSO。
- PostgreSQL/MySQL 替代 SQLite；Redis；Milvus Server；对象存储。
- **统一会话来源 + 多 worker 共享**（Redis 等，大于 P0 `chat_history` 修复）。
- 异步任务队列与任务状态 UI。
- SSE/WebSocket 流式输出。
- 集中监控、日志平台、成本仪表盘。
- Docker 生产镜像、CI/CD、K8s 与高可用。
- 计费、发票、复杂 SaaS 商业化（若有）。

### 8.3 刻意不做（除非需求明确）

- 为 SaaS 而 SaaS：无用户的「假多租户」。
- 无 eval 的大规模检索重构。
- 用 Spring Boot **全量重写**仍运行良好的 Django 壳层（无团队/Java 基建需求时）。
- Stage 0 / 1 阶段引入 K8s + 全套微服务。
- 无用户明确任务时，主动实现 WARNING 块所列 Stage 2/3 组件。

---

## 9. 迁移原则（给后续开发 / AI Agent）

1. **小步可运行**：每个 PR 后 `/ask`、`/reload_kb`、Trace、Chunks 仍可用。
2. **契约优先**：`VectorStore.search` 返回格式、`/ask` 响应、`context_metrics` 字段变更须全链路 grep（含模板）。
3. **先产品边界，后中间件**：先定义 Project 里有什么数据，再选 Milvus collection 策略。
4. **证据链不降级**：SaaS 化不得削弱证据门、拒答与 Trace。
5. **评测驱动 RAG**：换 embedding、向量库（含 FAISS → Milvus Server）、`top_k` 必须跑 eval 回归；**距离门阈值禁止跨后端直接沿用**。
6. **文档同步**：阶段切换时更新 README、`docs/project/PROJECT_CONTEXT.md`、本文 Stage 勾选状态。
7. **服从 TL;DR 与 WARNING**：未明确要求时，只做 Stage 0 → Stage 1 项。

---

## 10. 成功标准（按阶段）

| 阶段 | 技术成功标准 | 产品成功标准 |
|---|---|---|
| Stage 0 → 1 | smoke 全绿；eval 可重复；reload 不拖垮演示；FastAPI 重启后多轮上下文仍正确（`chat_history` P0） | 多论文对比误 source 明显减少；Trace 能解释拒答 |
| Stage 2 | Docker Compose 可复现；任务可查询；按 Project 隔离 | 3–5 个真实课题组试用；bad case 有记录 |
| Stage 3 | 监控告警、备份恢复演练、安全评审 | 校内/商业试点 SLA；敏感数据路径清晰 |

---

## 11. 相关文档

| 文档 | 关系 |
|---|---|
| `docs/project/PRODUCT_VISION.md` | 产品愿景与用户场景 |
| `docs/project/PROJECT_CONTEXT.md` | 代码结构、数据契约、当前技术债 |
| `docs/AGENTS.md` | Agent 每轮必守红线 |
| `docs/project/AI_AGENT_WORKING_PROMPTS.md` | 任务格式与推荐队列 |
| `README.md`（根目录） | 对外介绍与 Quick Start |

---

## 12. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-06-03 | 初版：定义 Stage 0–3、现在/未来边界、长期技术映射与 SaaS 核心能力表 |
| 2026-06-03 | 小修：TL;DR、AI Agent WARNING、会话 P0 与 Stage 2 边界、Milvus 阈值硬约束、附录弱化远期架构 |

---

## 附录 A：远期架构推演（**非当前实现**）

> ⚠️ **以下内容为 Stage 2 目标态推演，仅供架构讨论与面试讲解。**  
> **不代表当前仓库待办，AI Agent 不得据此自动引入 Redis / PostgreSQL / Milvus Server / K8s 等。**

### A.1 Stage 2 架构示意（目标态）

```text
                    ┌─────────────────┐
                    │  Web / Mobile   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │  Django / DRF / SpringBoot   │  ← Auth, RBAC, Project, 任务状态
              │  + PostgreSQL / MySQL        │
              └──────────────┬──────────────┘
                             │ HTTP / 内部 API
              ┌──────────────┴──────────────┐
              │   FastAPI AI Engine          │  ← LangGraph, RAG, Tools
              └──────────────┬──────────────┘
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    Milvus Server      Redis            Object Storage
         │                  │                  │
    Async Workers (embedding, index, eval)
```

**分层铁律不变**：业务后台不写 RAG 逻辑；FastAPI 不做页面。

**向量库切换提醒（与 §6.2 一致）**：上线 Milvus Server 时，须在 eval 集上重新标定 `CONTEXT_MAX_BEST_DISTANCE`（当前约 `2.2`）与 `CONTEXT_MAX_AVG_TOP_DISTANCE`（当前约 `2.4`），不得从 FAISS / Milvus Lite 直接复制。

### A.2 长期技术方向参考清单

以下组件在路线图中**都会出现**，但**不要求在当前仓库一次性实现**。引入每一项前，应能回答：「它服务于哪条 SaaS 核心能力？」且用户是否**明确要求**该任务。

| 技术方向 | 主要服务的 SaaS 能力 | 建议最早阶段 |
|---|---|---|
| FastAPI AI Engine | 推理隔离、独立扩缩容 | Stage 0（已有） |
| Django / DRF / Spring Boot 业务后台 | 用户、项目、权限、任务 | Stage 2 |
| PostgreSQL / MySQL | 持久化、关系查询、迁移 | Stage 2 |
| Redis | 会话、缓存、队列 broker、限流 | Stage 2 |
| Milvus Server | 规模化向量、metadata filter、多库 | Stage 2（**切换后 eval 重标定阈值**） |
| Object Storage | PDF、导出、大文件 | Stage 2 |
| Async task queue | 建库、eval、批处理 | Stage 2 |
| SSE / WebSocket | 流式回答、任务进度 | Stage 2（SSE 优先） |
| Auth / RBAC | 身份与授权 | Stage 2 |
| Tenant / Workspace / Project | 隔离与协作边界 | Stage 2 |
| Monitoring / logging / cost tracking | 可观测与成本 | Stage 2 末期 → Stage 3 |
| Docker / CI/CD / K8s | reproducible 部署与扩缩容 | Stage 2（Docker/CI）→ Stage 3（K8s） |

---

*PaperPilot 的 SaaS 化是一条**以科研证据可信为核心**的渐进路线。当前轻量架构是正确的起点；企业级能力应在 eval 与产品验证证明价值后，再按阶段叠加。*
