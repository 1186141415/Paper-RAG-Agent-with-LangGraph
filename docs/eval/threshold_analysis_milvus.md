# 双层证据门阈值分析：FAISS vs Milvus 的 L2 距离尺度

> 复现脚本：`scripts/analyze_gate_distance.py`（仓库根目录运行 `python scripts/analyze_gate_distance.py`）。
> 结论用于评估：`VECTOR_STORE` 从 `faiss` 切到 `milvus` 后，双层证据门第一层（距离门）的两个阈值是否仍然适用。

## 1. 背景

距离门用两个**绝对**距离阈值判定上下文是否充分（`app/rag_system.py`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `CONTEXT_MAX_BEST_DISTANCE` | 2.2 | 最近 chunk 的距离上限 |
| `CONTEXT_MAX_AVG_TOP_DISTANCE` | 2.4 | top-N 平均距离上限 |

这两个值是**针对 FAISS `IndexFlatL2` 经验调出**的。绝对阈值对距离尺度敏感，换向量库时若尺度变化，门就会失准。

## 2. 方法

- 知识库：`data/` 下 3 篇论文，共 **35 个 chunk**（字符级切分 700/120）。
- embedding 模型不变（`text-embedding-3-small`），FAISS 与 Milvus 复用同一份缓存向量，**保证向量完全相同**。
- 用同一组代表性问题（BROAD / SPECIFIC / COMPARISON 各一）分别在 FAISS 与 Milvus 上 `search(k=20)`，对比 `best_distance` 与 `avg_top_distance`。

## 3. 实测数据

| query | FAISS best/avg | Milvus best/avg | Milvus/FAISS ratio | √(FAISS_best) |
|---|---|---|---|---|
| What is the main contribution of this paper? | 1.480 / 1.492 | 1.217 / 1.221 | 0.822 | **1.217** |
| What dataset is used in the experiments? | 1.399 / 1.404 | 1.183 / 1.185 | 0.846 | **1.183** |
| What is the difference between paper1 and paper2? | 1.600 / 1.607 | 1.265 / 1.268 | 0.791 | **1.265** |

`mean |√(FAISS_best) − Milvus_best| = 0.0000`。

## 4. 核心结论

**Milvus L2 = √(FAISS L2)。**

- FAISS `IndexFlatL2` 返回的是**平方**欧氏距离（squared L2，不开方）；
- Milvus 的 `L2` metric 返回的是**未平方**的欧氏距离（开方后）；
- 两者向量完全相同时，`√(FAISS_best)` 三个点都精确等于 `Milvus_best`（误差 0）。

因此 FAISS↔Milvus 的距离**不是固定线性比例**（实测 ratio 0.79~0.85 只是在 1.x 距离段的局部近似，因为 `ratio = 1/√d` 随距离变化）。在阈值 2.2 处，正确换算是 `√2.2`，而不是 `2.2 × 0.82`。

## 5. 对当前系统的影响

- 当前 `.env` 跑 `VECTOR_STORE=milvus`，但代码阈值仍是 FAISS 的 `2.2 / 2.4`（squared 尺度）。
- Milvus 返回的距离在 ~1.2 量级，**远低于 2.2/2.4**，导致**距离门在 Milvus 下几乎永远 PASS**，门形同虚设。
- 这与 README §15 观察到的「22 个问题全部通过 distance gate」相互印证——部分原因正是 Milvus 距离尺度更小而阈值未随之调整。

## 6. 建议（保守，未改代码）

| 后端 | best 阈值 | avg 阈值 | 依据 |
|---|---|---|---|
| FAISS（现状） | 2.2 | 2.4 | 平方 L2，已调优 |
| Milvus（建议） | **√2.2 ≈ 1.48 → 取 1.5** | **√2.4 ≈ 1.55 → 取 1.6** | 未平方 L2，由上式换算 |

落地方式（任选其一，建议在更大评测集确认后再做）：

1. **按后端区分阈值**：在 `app/config.py` / `rag_system.py` 按 `VECTOR_STORE` 选择阈值组（推荐，避免再次换库踩同样的坑）。
2. **统一尺度**：在 `FaissVectorStore.search` 对距离开方，使两后端都用「未平方 L2」，从而共用一套阈值。
3. 暂不改：知悉当前 Milvus 下距离门偏松，主要依赖第二层 LLM 相关性门兜底。

> 本轮按 analysis 范围执行，仅给出结论与建议，**未改动 `rag_system.py` 的阈值常量**。样本规模（35 chunk / 3 query）偏小，建议结合 `docs/eval/eval_questions.json` 的完整评测集复测后再落地阈值变更，并跑回归（见 `docs/AGENTS.md` 红线：改阈值/换向量库必须重评 + 回归）。
