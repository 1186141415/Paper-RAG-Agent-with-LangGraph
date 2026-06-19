# PaperPilot 文档索引（docs/INDEX.md）

> 这是 `docs/` 的导航页。**先看这里，再看具体文档。**
> 原则：**活文档**放顶层分类目录，长期演进；**旧文档/快照**按日期归档进 `archive/`。

## 当前项目阶段
- 主线：把单链路 RAG 升级为 **Planner-Worker-Synthesizer 多智能体**，第一阶段只重点优化 COMPARISON 问题。
- 当前在 `feature/multi-agent` 分支，处于**协议设计**阶段（还没动代码）。

## 文档地图

| 文档 | 作用 | 状态 |
|---|---|---|
| [architecture/AGENT_PROTOCOL.md](architecture/AGENT_PROTOCOL.md) | 多智能体 v1 协议：每个 Agent 的职责/输入/输出/失败处理 + 状态契约 | ✅ v1 草案 |
| `architecture/ARCHITECTURE.md` | 系统整体架构图 + 分层职责 + 数据契约 | ⏳ 待写 |
| `eval/EVAL_PLAN.md` | RAG / Agent 效果评估方案（基于下面的评测集） | ⏳ 待写 |
| [eval/eval_questions.json](eval/eval_questions.json) | 22 题评测集（已分 BROAD / SPECIFIC / COMPARISON） | ✅ 活文档 |
| [eval/eval_run_result_top_k20.md](eval/eval_run_result_top_k20.md) · [top_k40](eval/eval_run_result_top_k40.md) | 历史评测运行结果 | ✅ 记录 |
| [devlog/](devlog/) | 每日开发日志（做了什么 / 踩了什么坑 / 怎么解决），按日期一文件 | ✅ 进行中 |
| [archive/2026-06-19/](archive/2026-06-19/) | 2026-06-19 之前的旧项目文档快照（已被新文档体系取代） | 🗄️ 归档 |

## 临时参考
- 在 `architecture/ARCHITECTURE.md` 写好之前，**最完整的系统说明**仍是归档里的
  [archive/2026-06-19/PROJECT_CONTEXT.md](archive/2026-06-19/PROJECT_CONTEXT.md)（注意其中有两处已过时，见该日 devlog）。

## 约定
- 改任何「契约类」内容（状态字段、数据结构、Agent 输入输出），**先改 `AGENT_PROTOCOL.md` 再改代码**。
- 每次有进展，在 `devlog/` 当天文件里追加一条。
