# AGENT_PROTOCOL.md — 多智能体协议设计（v1）

> 文档定位：这是 PaperPilot 从「单链路 RAG」升级到「多智能体论文问答」的**协议约束文档**，不是教程也不是 README。
> 它的作用是：在写代码之前，先把每个 Agent 的**职责 / 输入 / 输出 / 失败处理**和**状态契约**钉死，避免改造过程中跑偏。
> 写代码时以本文件的契约为准；本文件与代码冲突时，先改本文件达成一致，再改代码。
>
> 适用范围：仅 `app/` 下的 FastAPI + LangGraph + RAG 层。Django 产品壳层不受影响。
> 版本：v1（第一阶段，只重点优化 COMPARISON 类型问题）。
> 状态：设计草案（design draft），尚未实现。

---

## 0. 一句话目标

把当前「`choose_tool → execute_tool(rag.ask_with_trace 一次) → generate_answer`」的单链路，
在 **rag 这条路径内部**升级为 **Planner → Answer Worker → Synthesizer** 的三段式多智能体流程，
专门解决「同领域多篇论文，检索 chunk 跨论文混淆，导致 COMPARISON 问题答得差」这个已知短板。

其它工具路径（calculator / time / web_search / llm）**完全不变**。

---

## 1. v1 的目标与非目标

### 1.1 目标（v1 必须做到）

1. **按论文拆解**：把一个需要跨论文的问题，拆成「每篇论文一个子问题」，分别检索、分别回答。
2. **复用现有 RAG**：Answer Worker 直接调用现有 `RAGSystem.ask_with_trace`，**不新写检索/rerank/证据门逻辑**。
3. **聚焦单篇检索**：Worker 检索时能限定在「目标论文」范围内，避免召回到另一篇论文的 chunk。
4. **可解释**：`agent_trace` 能看到「问题被怎么拆的、每篇论文分别答了什么、最后怎么合成的」。
5. **优雅降级**：任何一个 Agent 失败，都退化成「现有单链路 RAG」的行为，绝不让整个请求 500。
6. **COMPARISON 优先**：v1 的收益主要体现在 COMPARISON 问题上；BROAD / SPECIFIC 退化成「单子任务」，等价于现状。

### 1.2 非目标（v1 明确不做，避免过度工程化）

- ❌ 不引入消息队列 / 任务总线（无 Celery / Kafka / RabbitMQ）。
- ❌ 不做并行 Worker（v1 用**顺序 for 循环**跑子任务，不上 LangGraph `Send` map-reduce）。
- ❌ 不做复杂反思循环（无多轮 self-reflection、无自动重检索回环）。
- ❌ 不做 Evidence Verifier / Citation Checker 的**实现**（只预留接口，见 §6）。
- ❌ 不大规模重写：新增 3 个节点 + 4 个状态字段 + 1 处检索过滤，其余尽量复用。
- ❌ 不改向量库后端、不改 chunk 结构、不改 `VectorStore.search` 返回字段名。

> 设计原则：**v1 先证明「按论文拆解」这件事有价值（用评测集量化），再谈并行和反思。**

---

## 2. AgentState 需要新增的字段

> 现有 `AgentState` 定义见 `app/graph/state.py`，是 `TypedDict, total=False`。下面只列**新增**字段，每个都说明「为什么需要」。

| 新字段 | 类型 | 由谁写入 | 为什么需要 |
|---|---|---|---|
| `question_type` | `str` | Planner | 取值 `BROAD` / `SPECIFIC` / `COMPARISON`。**今天这个分类藏在 `context_metrics["llm_question_type"]` 里**（由证据门产生），现在把它提升为一等状态字段，让 Planner / Synthesizer / trace 共用同一个「问题类型」真值来源，而不是各算各的。 |
| `subtasks` | `list[dict]` | Planner | 每个元素：`{"sub_id": str, "sub_question": str, "target_source": str \| None}`。这是 Planner 的拆解结果，Answer Worker 要按它逐个执行，trace 要靠它解释「问题被怎么拆的」。 |
| `sub_answers` | `list[dict]` | Answer Worker | 每个元素是一个子问题的回答结果（结构见 §4.3）。Synthesizer 的输入就是它；trace 靠它暴露「每篇论文分别答了什么、证据够不够」。 |
| `available_sources` | `list[str]` | 入口注入 / Planner 前 | 当前知识库里有哪些论文文件名（如 `["Paper1.pdf", "Paper2.pdf"]`）。**Planner 必须知道有哪些论文，才能把子问题分配给真实存在的论文**，否则模型可能凭空编出 `Paper3`。这是一个反幻觉约束。 |
| `verification` | `dict`（v1 留空不写） | （v2 的 Verifier） | **v1 预留、不使用**。给未来的 Evidence Verifier / Citation Checker 占位，见 §6。现在不写它，是为了让 state 结构在加 v2 时不需要再次变更契约。 |

**新增字段总数：4 个活跃（`question_type` / `subtasks` / `sub_answers` / `available_sources`）+ 1 个预留（`verification`）。**

> `available_sources` 怎么来：现有 `RAGSystem` 持有 `self.chunks`，每个 chunk 有 `source` 字段，对其去重排序即可得到论文列表。这是一个**只读派生**，不需要新存储。

---

## 3. Planner Agent（规划 / 分解）

### 3.1 职责
- 对**已经被路由到 rag** 的问题，做两件事：
  1. **分类**：判定 `question_type ∈ {BROAD, SPECIFIC, COMPARISON}`。
  2. **拆解**：产出 `subtasks` 列表，每个子任务绑定一个 `target_source`（目标论文）。
- 拆解规则（v1）：
  - `COMPARISON` → 为参与对比的**每篇论文**各产 1 个子问题，子问题改写成「针对单篇论文」的形式，`target_source` 指向该论文。
  - `BROAD` / `SPECIFIC` → 只产 **1 个**子任务；若问题已经点名某篇论文（如 "Paper1"），`target_source` 设为该论文；否则 `target_source = None`（全库检索，等价于现状）。
- **Planner 只做规划，不回答问题**（和现有 `choose_tool` 路由「只选工具不答题」是同一条纪律）。

### 3.2 输入
| 字段 | 来源 | 说明 |
|---|---|---|
| `query` | state | 用户原始问题 |
| `chat_history` | state | 多轮上下文（用于消解指代，如「这两篇」指哪两篇） |
| `available_sources` | state | 真实存在的论文列表，约束 `target_source` 只能取其中之一 |

### 3.3 输出（写回 state）
```json
{
  "question_type": "COMPARISON",
  "subtasks": [
    {"sub_id": "s1", "sub_question": "What is the main method/contribution of Paper1?", "target_source": "Paper1.pdf"},
    {"sub_id": "s2", "sub_question": "What is the main method/contribution of Paper2?", "target_source": "Paper2.pdf"}
  ],
  "workflow_path": [..., "planner"]
}
```
- LLM 只需返回结构化 JSON（`{question_type, subtasks}`），解析侧做清洗 + 兜底（**复用 `app/graph/nodes.py` 已有的 `clean_json_text` 思路**）。
- 约束：`subtasks` 数量在 v1 **上限建议为 3**（`MAX_SUBTASKS = 3`），防止模型对一个问题拆出过多子任务、放大 LLM 调用成本。

### 3.4 失败处理（优雅降级，绝不抛错中断）
| 失败情形 | 处理 |
|---|---|
| LLM 调用异常 / 超时 | 降级为单子任务：`subtasks = [{"sub_id":"s1","sub_question": query,"target_source": None}]`，`question_type = "UNKNOWN"`。**等价于现状的单链路 RAG。** |
| 返回 JSON 解析失败 / 格式不符 | 同上降级。模仿现有 router 的「解析失败 → 退回安全默认」。 |
| `target_source` 不在 `available_sources` 里 | 把该 `target_source` 归一为 `None`（全库检索），不让幻觉论文名流入 Worker。 |
| 拆出的 `subtasks` 超过 `MAX_SUBTASKS` | 截断到前 `MAX_SUBTASKS` 个。 |
| `subtasks` 为空 | 降级为单子任务（原始 query，全库）。 |

> 关键不变量：**Planner 失败不是 `error`，是「降级」**。不要往 `state["error"]` 写值，否则会误触发现有的 `llm_fallback`（那是给「工具执行异常」用的，不是给规划降级用的）。

---

## 4. Answer Worker Agent（子问题回答，复用现有 RAG）

### 4.1 职责
- 对 `subtasks` 里的**每一个**子任务，调用一次现有 `RAGSystem.ask_with_trace(sub_question, chat_history)`，
  但检索范围限定在 `target_source` 这篇论文内。
- v1 用**顺序 for 循环**遍历 `subtasks`（一个 `answer_worker` 节点内部循环），**不并行**。
- Worker **不新写任何检索 / rerank / 证据门逻辑**——这些全部复用 `ask_with_trace` 内部已有的：召回(top_k=20) → LLM rerank(rerank_k=10) → 距离门 → 类型化 LLM 相关性门 → 生成 / 拒答。

### 4.2 输入
| 字段 | 来源 | 说明 |
|---|---|---|
| `sub_question` | 当前 subtask | 针对单篇论文改写过的问题 |
| `target_source` | 当前 subtask | 限定检索范围；`None` 表示全库 |
| `chat_history` | state | 透传给 `ask_with_trace` |
| `rag` | 节点构造时注入 | 现有 `RAGSystem` 实例（沿用 `build_execute_tool_node(tools, rag=rag)` 的依赖注入方式） |

> **唯一需要碰检索层的改动**：给 `RAGSystem.retrieve` / `VectorStore.search` 增加一个**可选** `source` 过滤参数。
> - v1 最小做法（FAISS）：多召回一些（如 `top_k * 论文数`）后，在 Python 里按 `source` 过滤再交给 rerank——**不动索引结构**。
> - Milvus 可用 filter 表达式。
> - `VectorStore.search` 的**返回字段名不变**（`source / text / distance / retrieval_rank`），只是新增入参。这是为了不破坏现有数据契约。

### 4.3 输出（追加到 `state["sub_answers"]`）
每个子任务产出一个 sub_answer：
```json
{
  "sub_id": "s1",
  "target_source": "Paper1.pdf",
  "sub_question": "What is the main method/contribution of Paper1?",
  "answer": "....",
  "retrieved_chunks": [ {source, text, distance, retrieval_rank}, ... ],
  "context_sufficient": true,
  "context_metrics": { ... },     // 直接沿用 ask_with_trace 的 context_metrics
  "error": ""
}
```
- 前 4 个核心字段（`answer / retrieved_chunks / context_sufficient / context_metrics`）**直接来自 `ask_with_trace` 的返回**（结构见 `app/rag_system.py` 的 `ask_with_trace`），只是外面包上 `sub_id / target_source / sub_question`。

### 4.4 失败处理（局部失败不拖垮全局）
| 失败情形 | 处理 |
|---|---|
| 某个 Worker 的 `ask_with_trace` 抛异常 | 该 sub_answer 记为 `{answer: "", context_sufficient: false, error: "<异常信息>"}`，**继续跑其它子任务**。一篇论文检索挂掉，不该让整个对比失败。 |
| `context_sufficient == false`（证据门拒答） | 照常保留 `ask_with_trace` 返回的「证据不足」答案与 `context_metrics`，`context_sufficient` 标 false。交给 Synthesizer 决定怎么处理「部分有证据」。 |
| `target_source` 那篇论文检索不到任何 chunk | 等价于「证据不足」，按上一行处理。 |

> Worker 层**不触发** `llm_fallback`、**不写** `state["error"]`。失败信息留在各自的 sub_answer 里，由 Synthesizer 统一裁决。这样保住了「`fallback_used` 防循环」这个现有不变量不被污染（见 §8）。

---

## 5. Synthesizer Agent（合成最终答案）

### 5.1 职责
- 把所有 `sub_answers` 合并成一个最终答案，写入 `state["final_answer"]`。
- `COMPARISON` 时：显式做「对比 / 异同」叙述，并对每篇论文带 `[Source: xxx.pdf]` 引用——这正是单次检索做不好、而拆解后能做好的地方。
- `BROAD` / `SPECIFIC`（单子任务）时：基本是「透传 + 轻量润色」，不强行制造对比。

### 5.2 输入
| 字段 | 来源 | 说明 |
|---|---|---|
| `query` | state | 用户原始问题（合成要回答的是原问题，不是子问题） |
| `question_type` | state | 决定合成策略（对比 vs 透传） |
| `sub_answers` | state | 各论文的子答案 + 证据 + 充分性 |

### 5.3 输出（写回 state，**对齐现有 `generate_answer` 期望的字段**）
```json
{
  "final_answer": "Paper1 提出……；相比之下 Paper2……（带 [Source] 引用）",
  "retrieved_chunks": [ ...各 sub_answer 的 chunks 合并去重... ],
  "context_sufficient": true,
  "context_metrics": {
     "multi_agent": true,
     "question_type": "COMPARISON",
     "sub_metrics": [ {sub_id, target_source, context_sufficient, ...}, ... ]
  },
  "workflow_path": [..., "synthesizer"]
}
```
- **为什么要写这几个字段**：现有 `generate_answer_node`（`app/graph/nodes.py`）会从 `tool_result.tool_output` 里拆 `answer / retrieved_chunks / context_sufficient / context_metrics`。v1 让 Synthesizer **直接把这几个字段写进 state**，使下游 `generate_answer` 退化成「透传」即可，**不破坏现有响应结构**。
- `context_metrics` 在多智能体路径下是**聚合版**：保留 `question_type`，新增 `multi_agent: true` 和 `sub_metrics`（每个子任务的充分性摘要）。注意是**新增 key**，不改任何现有 key 名（Django 模板还在读旧 key，见 §8）。

### 5.4 失败处理
| 失败情形 | 处理 |
|---|---|
| Synthesizer LLM 调用异常 | **确定性兜底**：把各 `sub_answers` 按 `[Source]` 标题拼接成一个答案，不再调 LLM。保证有输出。 |
| 全部 sub_answer 都 `context_sufficient == false` | 返回与现状一致的「证据不足」中文提示（复用 `ask_with_trace` 里那段硬编码文案的语气），`context_sufficient = false`。 |
| 部分 sufficient、部分不足 | 用「有证据的子答案」合成，并**如实说明哪篇论文证据不足**（例如「关于 Paper2 的对应内容未检索到足够证据」）。诚实暴露缺口，符合本项目「不强行编」的反幻觉基调。 |

---

## 6. Evidence Verifier / Citation Checker 是否进入 v1

**结论：不进入 v1。只预留接口。**

### 6.1 为什么不进 v1
- 本质上它是一个「校验 + 可能触发重做」的反思环节，和「不做复杂反思循环」的非目标冲突。
- v1 的首要任务是**先用评测集证明「按论文拆解」本身有收益**。先加 Verifier 会把「拆解的收益」和「校验的收益」混在一起，无法归因。

### 6.2 如何预留（让 v2 加它时几乎零契约变更）
1. **状态字段已留**：`verification: dict`（§2），v1 不写。
2. **图里留好插入点**：未来 Verifier 节点固定插在 `synthesizer → generate_answer` 之间。v1 这条边先直连，v2 在中间插一个节点即可。
3. **复用现有防循环不变量**：未来若 Verifier 判定「引用不实 / 证据不足」要重做，**复用现有的 `retry_count` + `fallback_used` 机制**做「最多重试 1 次」的护栏（和 `route_after_execute` 是同一套路），不发明新机制。
4. **数据已经现成**：v1 的 Synthesizer 已经产出「带 `[Source]` 引用 + 每篇 chunks」的结果——这正是 Citation Checker 需要的输入。也就是说 **v1 已经把 Verifier 要吃的数据准备好了**，v2 只是加一个吃这份数据的节点。

> 这一点面试很加分：「我没有提前造反思循环，但我的状态机和数据结构是**为它预留好的**——加 Verifier 不需要重构，只需插一个节点。」

---

## 7. COMPARISON 问题的完整状态流转示例

以评测集 `q008`：**"What is the difference between Paper1 and Paper2?"** 为例。

### 7.1 新的图（rag 路径内部展开）
```
START
  → choose_tool                         # 现有路由：判定 tool = "rag"
  → [tool == "rag" ?]
        ├─ 否 → execute_tool → route_after_execute → (llm_fallback?) → generate_answer → END   # 旧路径，原样不动
        └─ 是 → planner → answer_worker → synthesizer → generate_answer → END                   # 新增多智能体路径
```

### 7.2 一步步的 state 变化

| 步骤 | 节点 | state 关键变化 |
|---|---|---|
| 1 | `choose_tool` | `decision = {"tool":"rag","input":"What is the difference between Paper1 and Paper2?","reason":"..."}`；`workflow_path += ["choose_tool"]` |
| 2 | 条件边 | `decision.tool == "rag"` → 进入 `planner` |
| 3 | `planner` | `available_sources = ["Paper1.pdf","Paper2.pdf"]`；`question_type = "COMPARISON"`；`subtasks = [{s1, "…Paper1?", "Paper1.pdf"}, {s2, "…Paper2?", "Paper2.pdf"}]`；`workflow_path += ["planner"]` |
| 4 | `answer_worker` (循环 1/2) | 调 `ask_with_trace("…Paper1?", source="Paper1.pdf")` → `sub_answers += [{s1, Paper1.pdf, answer1, chunks(仅Paper1), context_sufficient:true, metrics}]` |
| 5 | `answer_worker` (循环 2/2) | 调 `ask_with_trace("…Paper2?", source="Paper2.pdf")` → `sub_answers += [{s2, Paper2.pdf, answer2, chunks(仅Paper2), context_sufficient:true, metrics}]`；`workflow_path += ["answer_worker"]` |
| 6 | `synthesizer` | 输入 `(query, COMPARISON, [sa1, sa2])` → `final_answer = "<对比叙述，带 [Source: Paper1.pdf] / [Source: Paper2.pdf]>"`；`retrieved_chunks = merge(sa1.chunks, sa2.chunks)`；`context_sufficient = true`；`context_metrics = {multi_agent:true, question_type:"COMPARISON", sub_metrics:[...]}`；`workflow_path += ["synthesizer"]` |
| 7 | `generate_answer` | 透传 `final_answer` 等字段；`workflow_path += ["generate_answer"]` → END |
| 8 | `main.py` `/ask` | `agent_trace` 在现有键之外**新增** `plan`(=subtasks) / `sub_answers` 摘要；其余键不变 |

### 7.3 对比「升级前」
- 升级前：一次检索同时召回 Paper1+Paper2 的 chunk，rerank 后 top-10 可能 8 条来自 Paper1、2 条来自 Paper2，对比天然失衡。
- 升级后：两篇各自独立召回 + 各自过证据门，Synthesizer 拿到的是「两边都站得住」的子答案再对比。

---

## 8. 哪些旧字段继续复用，哪些不要动

### 8.1 继续复用（不改语义，只继续往里写）
| 字段 | 在 v1 里的角色 |
|---|---|
| `query` / `chat_history` / `session_id` | Planner、Worker 照常读取 |
| `decision` | **路由仍然先跑**；`tool=="rag"` 才进多智能体路径，其它工具走旧路 |
| `final_answer` | 由 Synthesizer 写（替代原来 rag 工具的产出位置） |
| `retrieved_chunks` | 由 Synthesizer 合并各 sub_answer 后写 |
| `context_sufficient` | 由 Synthesizer 聚合后写 |
| `context_metrics` | 由 Synthesizer 写聚合版（**只新增 key，不改旧 key 名**） |
| `workflow_path` | 各新节点追加自己的名字（`planner` / `answer_worker` / `synthesizer`） |

### 8.2 不要动（动了会破坏现有运行闭环）
| 字段 / 契约 | 为什么不能动 |
|---|---|
| `fallback_used` 的语义 | 它是 `route_after_execute` 的**防无限兜底循环不变量**。多智能体的失败一律「优雅降级」，**不准**借用这个字段，更不准制造新的回环。 |
| `tool_result` 的形状 `{tool_name, tool_input, tool_output}` | `generate_answer` + `main.py` + Django 都依赖它。v1 不删它；rag 多智能体路径改为「Synthesizer 直接写 state 字段」，让 `generate_answer` 退化为透传，从而**不需要改 `tool_result` 结构**。 |
| `context_metrics` 现有 key 名 | Django 模板（`chat_home.html` 等）在读这些 key。**只能加，不能改名/删**。 |
| `VectorStore.search` 返回字段 `{source,text,distance,retrieval_rank}` | FAISS / Milvus 必须一致，RAG / trace / 前端都依赖。`source` 过滤是**新增入参**，不动返回结构。 |
| `error` 字段的用途 | 仅表示「工具执行异常 → 触发 llm_fallback」。Planner/Worker/Synthesizer 的降级**不写** `error`。 |

---

## 9. 与现有 RAGSystem / LangGraph / agent_trace 的兼容性

### 9.1 与 `RAGSystem` 兼容
- **唯一改动**：`retrieve` / `VectorStore.search` 增加可选 `source` 入参（默认 `None` = 全库 = 现状）。
- `rerank`、`assess_context_sufficiency`、`assess_context_relevance_with_llm`、双层证据门、生成/拒答**全部不动**。
- Worker 对每个子问题调用一次 `ask_with_trace`——也就是说**现有 RAG 流水线被复用 N 次**，而不是被改写。

### 9.2 与 LangGraph 兼容（`app/graph/builder.py`）
- 新增 3 个节点：`planner`、`answer_worker`、`synthesizer`（依赖注入方式沿用现有 `build_execute_tool_node(tools, rag=rag)` 的工厂函数写法）。
- 新增 1 条条件边：`choose_tool` 之后，`decision.tool == "rag"` → `planner`，否则 → `execute_tool`（旧逻辑）。
- 新增直连边：`planner → answer_worker → synthesizer → generate_answer`。
- 旧的 `execute_tool → route_after_execute →(llm_fallback / generate_answer)` **整段保留不变**。
- `answer_worker` 内部用 Python for 循环遍历 `subtasks`（**不**用 `Send` / map-reduce），符合「v1 不并行」。

### 9.3 与 `agent_trace` 兼容（`app/main.py` 的 `/ask`）
- 现有键全部保留：`route_decision / tool_used / fallback_used / context_sufficient / context_metrics / error / retry_count / workflow`。
- **新增键**（additive，前端不读也不会坏）：`plan`（= `question_type` + `subtasks`）、`sub_answers`（每篇论文的答案 + 充分性摘要）。
- `workflow` 字段会自然出现新节点名（`planner / answer_worker / synthesizer`），正好让前端「Agent Trace 卡片」直观展示多智能体路径。

### 9.4 改动清单总览（量化「最小」）
| 文件 | 改动 |
|---|---|
| `app/graph/state.py` | +4 活跃字段 +1 预留字段 |
| `app/graph/nodes.py`（或新建 `app/graph/agents.py`） | +`planner` / `answer_worker` / `synthesizer` 三个节点 |
| `app/graph/builder.py` | +3 节点注册、+1 条件边、+若干直连边 |
| `app/rag_system.py` + `app/vector_store/*` | `retrieve`/`search` +1 可选 `source` 入参（FAISS 用 over-fetch + 后过滤） |
| `app/main.py` | `agent_trace` +2 个新键（additive） |
| `prompts`（建议新建 `app/prompts.py`） | +Planner / +Synthesizer 两段 prompt |

> 没有任何「破坏性」改动。最坏情况下（所有 Agent 都降级），系统行为 == 现有单链路 RAG。

---

## 10. 面试时如何解释这个多智能体设计

**30 秒电梯版：**
> 「我把 PaperPilot 的单链路 RAG 升级成了 Planner-Worker-Synthesizer 的多智能体。起因是我用自己的 22 题评测集发现：跨论文对比类问题（COMPARISON）答得差，因为一次检索会把两篇论文的 chunk 混在一起。我的做法是：Planner 按论文把问题拆开，Worker **复用我原来的整条 RAG 流水线**、但把检索限定在单篇论文内，Synthesizer 再把各篇的子答案合成带引用的对比。整个改造只加了 3 个节点和 4 个状态字段，任何一步失败都会优雅降级回原来的单链路。」

**面试官会追问的点，以及你要能答的「为什么」：**

1. **为什么是多智能体而不是改 prompt？**
   因为根因是「检索召回跨论文混淆」，不是「生成不会写对比」。拆成「每篇独立检索」是从**数据层**解决问题；只改 prompt 解决不了召回失衡。

2. **为什么 Worker 复用 RAGSystem，而不是写新的？**
   复用 = 双层证据门、rerank、trace 这些防幻觉与可观测能力**自动继承**，不重复造轮子，也保证行为一致、好回归。

3. **多智能体不是更慢更贵吗？**（一定会问）
   是。一次 COMPARISON 从 ~4 次 LLM 调用涨到 planner(1) + worker(2×4) + synthesizer(1)。这是**用成本换准确性**的有意权衡，且只在 rag 路径触发；简单问题不付这个成本。下一步可以用 LangGraph `Send` 把 Worker 并行化把延迟摊平。

4. **为什么 v1 不做反思 / Verifier？**
   避免把「拆解的收益」和「校验的收益」混在一起、无法用评测集归因。但我**预留了**状态字段、图插入点，并复用现成的 `retry_count` 防循环机制——加 Verifier 不需要重构，只需插一个节点。

5. **怎么证明它真的更好？**（关键，体现工程素养）
   用评测集里 5 道 COMPARISON 题（q008/q009/q010/q019/q020）做 A/B：对比单链路 vs 多智能体在「是否两篇都被引用、`expected_keywords` 命中、证据门误杀率」上的差异（评估口径见 `docs/eval/`）。

**这个设计能体现的能力关键词（贴 AI 应用工程师 JD）：**
Agent 编排（LangGraph 状态机 / 条件路由）、RAG 质量优化、可观测性（agent_trace / 可解释证据门）、面向扩展的架构设计（预留 Verifier 接口）、用评测集驱动决策（eval-driven）、工程权衡意识（成本 vs 准确性、优雅降级）。

---

## 附录 A：v1 → v2 演进路线（仅备忘，不在本次范围）
1. Worker 并行化（LangGraph `Send` map-reduce）降低 COMPARISON 延迟。
2. 加 Evidence Verifier / Citation Checker（用 §6 预留的接口）。
3. 把 Planner 的分类与现有证据门的 `llm_question_type` 合并，省掉一次 LLM 调用。
4. Planner 支持「同一篇论文内的多角度子问题」（不止按论文，还能按维度：方法 / 实验 / 局限）。

*本文件随设计推进持续更新。修改契约前，先更新本文件再改代码。*
