# Eval Run Result

## 2026-05-09 top_k=40 Re-evaluation

### Purpose

This run repeats the 22-question eval with `top_k = 40` (up from 20) to measure whether increasing the FAISS retrieval pool improves cross-paper chunk coverage, and to compare results against the earlier `top_k = 20` baseline.

The hypothesis: a larger retrieval pool gives the reranker more chances to include chunks from both papers, which should improve Paper2-specific questions and COMPARISON questions.

---

## Test Setup

| Parameter | top_k=20 Run | top_k=40 Run |
|-----------|-------------|-------------|
| FAISS retrieval | `top_k = 20` | `top_k = 40` |
| Rerank | LLM-based, `rerank_k = 10`, `temperature = 0` | same |
| Distance gate | `best <= 2.2`, `avg_top <= 2.4`, `min_chunks = 2` | same |
| Typed LLM relevance gate | BROAD / SPECIFIC / COMPARISON | same |
| BROAD soft warning | LLM gate NO → soft warning, use distance result | same |
| Papers | Paper1.pdf (steganalysis SepSteNet/DPES), Paper2.pdf (VoIP DVSF) | same |
| Questions | 22 (6 BROAD, 11 SPECIFIC, 5 COMPARISON) | same |

---

## Test Results

### Full Question Table

| # | Question (abbreviated) | Label | LLM Type | Dist Gate | LLM Gate | top_k=20 Suff? | top_k=40 Suff? | Change |
|---|----------------------|-------|----------|:---:|:---:|:---:|:---:|--------|
| q001 | Main contribution of Paper1? | BROAD | BROAD | pass | pass | YES | YES | — |
| q002 | Problem Paper1 tries to solve? | BROAD | BROAD | pass | pass | YES | YES | — |
| q003 | Summarize main findings of Paper2 | BROAD | BROAD | pass | fail | YES | YES | — |
| q004 | Research motivation behind Paper2? | BROAD | SPECIFIC | pass | fail | YES | **no** | Type reclassified BROAD→SPECIFIC, hard block applied |
| q005 | What methodology does Paper1 use? | BROAD | BROAD | pass | pass | YES | YES | — |
| q006 | Datasets used in experiments of Paper1? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q007 | Evaluation metrics reported in Paper2? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q008 | Difference between Paper1 and Paper2? | COMPARISON | COMPARISON | pass | fail | no | no | — |
| q009 | Which method performs better? | COMPARISON | COMPARISON | pass | fail | no | no | — |
| q010 | Are approaches complementary or competing? | COMPARISON | COMPARISON | pass | fail | no | no | — |
| q011 | Key limitations discussed in Paper1? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q012 | Does Paper2 propose a novel architecture? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q013 | How does Paper1 handle cold-start? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q014 | Does Paper1 mention quantum computing? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| **q015** | Does Paper2 discuss transformer architectures? | SPECIFIC | SPECIFIC | pass | **pass** | no | **YES** | top_k=40 retrieved Paper2 chunk mentioning "Transformer architecture" (LStegT model) |
| q016 | Core theoretical foundation of Paper1? | BROAD | SPECIFIC | pass | fail | no | no | — |
| q017 | Ablation studies conducted in Paper2? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q018 | How does Paper2 compare against baselines? | SPECIFIC | SPECIFIC | pass | fail | no | no | Type changed COMPARISON→SPECIFIC |
| q019 | Shared assumptions Paper1 and Paper2? | COMPARISON | COMPARISON | pass | fail | no | no | — |
| q020 | Do Paper1 and Paper2 address same problem? | COMPARISON | COMPARISON | pass | fail | no | no | — |
| q021 | Future research directions Paper2 suggests? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |
| q022 | Does Paper1 use reinforcement learning? | SPECIFIC | SPECIFIC | pass | fail | no | no | — |

### Summary Statistics

| Metric | top_k=20 | top_k=40 |
|--------|----------|----------|
| Total questions | 22 | 22 |
| Routed to `rag` | 22 (100%) | 22 (100%) |
| Tool routing errors | 0 | 0 |
| Fallback triggered | 0 | 0 |
| Rerank used / fallback | 22 / 0 | 22 / 0 |
| Distance gate passed | 22 (100%) | 22 (100%) |
| LLM relevance gate passed | 4 (18%) | 4 (18%) |
| **Final context sufficient** | **5 (23%)** | **5 (23%)** |
| Negative cases correct | 2/2 | 2/2 |
| Cross-paper retrieval confusion | ~15 | ~14 |
| Paper2-specific questions passed | 0 | **1** (q015) |

### Type Classifier Agreement

| Human Label | LLM top_k=20 | LLM top_k=40 | Count |
|-------------|-------------|-------------|-------|
| BROAD | BROAD | BROAD | 4 |
| BROAD | SPECIFIC | SPECIFIC | 1 (q016) |
| BROAD | BROAD | SPECIFIC | 1 (q004) — classifier flipped |
| BROAD | SPECIFIC | BROAD | 1 (q005) — classifier flipped |
| SPECIFIC | SPECIFIC | SPECIFIC | 10 |
| SPECIFIC | COMPARISON | SPECIFIC | 1 (q018) — classifier flipped |
| COMPARISON | COMPARISON | COMPARISON | 5 |

Agreement rate: top_k=20 **19/22 (86%)** → top_k=40 **19/22 (86%)**. Same rate, but three questions flipped classification between runs (q004, q005, q018), all due to different chunks being shown to the LLM type classifier.

---

## Key Observations

### 1. Quantitative summary: same pass count, better composition

The overall `context_sufficient` count stayed at 5/22, but the composition improved:

| Question | top_k=20 | top_k=40 | Quality change |
|----------|----------|----------|----------------|
| q004 | YES (BROAD soft warning, wrong-paper chunks) | no (SPECIFIC hard block) | **improvement** — false positive removed |
| q015 | no (Paper2 metadata only) | YES (Paper2 transformer chunk found) | **improvement** — genuine Paper2 hit |

top_k=40 traded one false positive (q004) for one genuine new hit (q015). Net count unchanged, net quality improved.

### 2. q015: the clearest win from top_k=40

With `top_k=20`, q015 ("Does Paper2 discuss transformer architectures?") failed because FAISS returned Paper2 chunks that only contained metadata (title, abstract header), not actual content about transformers.

With `top_k=40`, a chunk from Paper2 mentioning "Transformer architecture" in the context of the LStegT model entered the retrieval pool and survived reranking into the top-10. The LLM relevance gate correctly identified it and returned YES. The final answer accurately stated that Paper2 "mentions transformer architectures in relation to the LStegT model" but "does not discuss them in detail."

This demonstrates that `top_k=40` can surface relevant chunks that `top_k=20` misses — specifically for the less-dominant paper in the index.

### 3. q004: accidental improvement via type reclassification

With `top_k=20`, q004 was classified as BROAD and passed via soft warning despite chunks being from the wrong paper. With `top_k=40`, different chunks caused the type classifier to label it SPECIFIC, triggering a hard block. This was the correct outcome — the question cannot be answered from wrong-paper chunks — but it was achieved by a classifier flip rather than by better retrieval.

This is a fragile improvement: rerun with different random seeds and the classifier might flip back. It highlights that the type classifier's stability depends on which chunks happen to appear in the preview.

### 4. q003 regression: answer quality degraded

With `top_k=20`, q003's BROAD soft warning produced an answer that correctly described Paper2's DVSF framework (the chunks happened to include Paper2 content despite the LLM gate claiming otherwise). With `top_k=40`, the reranker made different choices from the larger candidate pool and the top-10 no longer included the Paper2 content chunks. The answer changed to "there is no information about Paper2."

This is a **rerank stability issue**: with 40 candidates instead of 20, the LLM reranker has a harder selection problem and can make different — sometimes worse — choices. A cross-encoder reranker or a more structured rerank prompt (including paper source metadata) could mitigate this.

### 5. Cross-paper retrieval confusion: persistent but slightly improved

The dominant failure mode remains FAISS returning chunks from the wrong paper. At `top_k=40`:

- Paper2 questions getting Paper1 chunks: ~11 questions
- Paper1 questions getting Paper2 chunks: ~3 questions
- Both papers are in the steganalysis domain, embeddings overlap heavily

The slight improvement (from ~15 to ~14) is negligible. Simply increasing `top_k` is not sufficient to solve cross-paper confusion. The retrieval layer needs metadata awareness.

### 6. COMPARISON questions: still 0/5 passed

All 5 COMPARISON questions continue to fail because the top-10 chunks (after rerank) are dominated by one paper. The LLM gate correctly blocks them all — without chunks from both papers, comparisons would be unreliable.

This is the correct behavior for the current retrieval quality, but it means the comparison feature is effectively disabled until cross-paper retrieval improves.

### 7. Rerank and routing remain rock-solid

Across both runs (44 total questions):
- **Tool routing**: 44/44 correct, 0 fallback triggers
- **Rerank parsing**: 44/44 success, 0 fallback events
- **Distance gate**: 44/44 passed (confirming distance alone is not discriminative)

### 8. Distance values comparison

| Metric | top_k=20 | top_k=40 |
|--------|----------|----------|
| Paper1 avg best_distance | 1.49 | 1.50 |
| Paper2 avg best_distance | 1.45 | 1.42 |
| Overall avg best_distance | 1.47 | 1.44 |

Distances are stable across `top_k` values, confirming that the distance gate thresholds (2.2 / 2.4) are appropriately set — they catch extremely poor matches but don't help discriminate between good and mediocre chunks within a domain.

---

## Comparison: Three Eval Runs

| Aspect | 2026-05-03 (4Q) | top_k=20 (22Q) | top_k=40 (22Q) |
|--------|-----------------|----------------|----------------|
| Relevance gate | Untyped LLM gate | Typed (BROAD/SPECIFIC/COMPARISON) | Same |
| LLM gate pass rate | 25% | 18% | 18% |
| Sufficient rate | 25% | 23% | 23% |
| Rerank fallback | 0% | 0% | 0% |
| BROAD over-rejection | 3 of 4 BROAD failed | 0 of 6 BROAD failed (soft warning) | 1 of 6 BROAD failed (classifier flip) |
| Paper2 questions passed | — | 0 | 1 |
| COMPARISON questions passed | — | 0 | 0 |
| Main bottleneck | Gate too conservative | Cross-paper retrieval | Cross-paper retrieval + rerank stability |

---

## Current Conclusion

`top_k=40` produced one clear win (q015: Paper2 transformer chunk surfaced) and removed one false positive (q004: type classifier flip), while introducing one answer quality regression (q003: different rerank ordering lost Paper2 content).

**The net effect is slightly positive but insufficient.** The core problems remain:

1. **Cross-paper retrieval confusion** — FAISS does not distinguish between two steganalysis papers. Simply increasing `top_k` cannot fix this. The system needs metadata-aware retrieval or per-paper index sharding.

2. **Rerank stability at larger candidate pools** — With 40 candidates, the LLM reranker's selection is less predictable. Feeding paper source metadata into the rerank prompt could help anchor the ranking.

3. **Type classifier stability** — The BROAD/SPECIFIC/COMPARISON classifier flips on 3/22 questions between runs because its input (the preview chunks) changes. This can change outcomes independently of actual retrieval quality.

**Recommended next steps:**

1. **Metadata injection in rerank prompt** — include each chunk's `source` paper name in the rerank text so the reranker can prioritize chunks from the requested paper
2. **Metadata-aware retrieval** — when the question names a specific paper (e.g., "Paper2"), pre-filter FAISS candidates by paper source or add a paper-name keyword boost
3. **Per-paper index sharding** — maintain separate FAISS indices per paper; for single-paper questions, search only the relevant index; for COMPARISON questions, merge results from both
4. **Reranker stability** — explore a cross-encoder (e.g., `bge-reranker`) for deterministic, distance-based reranking instead of LLM-based reranking
5. **Tune distance gate thresholds** — with 44 questions of data, the thresholds (2.2 / 2.4) could be calibrated to achieve non-trivial discrimination
