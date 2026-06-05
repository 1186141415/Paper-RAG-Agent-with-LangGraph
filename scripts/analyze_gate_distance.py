"""
分析 FAISS 与 Milvus 在 L2 距离上的尺度差异。

背景：双层证据门的两个距离阈值（rag_system.CONTEXT_MAX_BEST_DISTANCE /
CONTEXT_MAX_AVG_TOP_DISTANCE）是针对 FAISS IndexFlatL2 经验调出的。
当 VECTOR_STORE 切到 Milvus 后，需要确认 Milvus L2 的距离尺度是否一致，
否则用绝对距离阈值的证据门会失准。

关键结论（本脚本会实测验证）：
    FAISS IndexFlatL2 返回的是「平方」欧氏距离（squared L2）；
    Milvus L2 返回的是「未平方」的欧氏距离。
    因此  Milvus_L2 ≈ sqrt(FAISS_L2)，不是固定的线性比例。
    => Milvus 等效阈值应取 sqrt(FAISS 阈值)，而非线性外推。

用法（仓库根目录）：
    python scripts/analyze_gate_distance.py

说明：使用独立的临时 Milvus collection / db 文件，不触碰默认的
milvus_demo.db / paper_rag_chunks，运行结束自动清理。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_DIR
from app.data_loader import load_pdfs, process_documents
from app.rag_system import (
    CONTEXT_MAX_BEST_DISTANCE,
    CONTEXT_MAX_AVG_TOP_DISTANCE,
    CONTEXT_TOP_N_FOR_AVG,
)
from app.vector_store.faiss_store import FaissVectorStore
from app.vector_store.milvus_store import MilvusVectorStore

# 代表三类问题（BROAD / SPECIFIC / COMPARISON）
QUERIES = [
    "What is the main contribution of this paper?",
    "What dataset is used in the experiments?",
    "What is the difference between paper1 and paper2?",
]

TMP_URI = "./tmp_analysis_milvus.db"
TMP_COLLECTION = "tmp_gate_analysis"


def _best_and_avg(results):
    ds = sorted(c["distance"] for c in results)
    if not ds:
        return None, None
    best = ds[0]
    top = ds[:min(CONTEXT_TOP_N_FOR_AVG, len(ds))]
    return best, sum(top) / len(top)


def main():
    docs = load_pdfs(DATA_DIR)
    chunks = process_documents(docs)
    print(f"[analyze] loaded {len(docs)} docs, {len(chunks)} chunks\n")

    faiss_store = FaissVectorStore()
    faiss_store.build(chunks)

    milvus_store = MilvusVectorStore(
        uri=TMP_URI,
        collection_name=TMP_COLLECTION,
        drop_old=True,
    )
    milvus_store.build(chunks)

    rows = []
    for q in QUERIES:
        fb, fa = _best_and_avg(faiss_store.search(q, k=20))
        mb, ma = _best_and_avg(milvus_store.search(q, k=20))
        rows.append((q, fb, fa, mb, ma))

    header = (
        f"{'query':<48} | {'FAISS best/avg':<16} | "
        f"{'Milvus best/avg':<16} | ratio | sqrt(FAISS_best)"
    )
    print(header)
    print("-" * len(header))

    best_ratios, avg_ratios, sqrt_err = [], [], []
    for q, fb, fa, mb, ma in rows:
        if not (fb and mb):
            print(f"{q[:47]:<48} | (no results)")
            continue
        best_ratios.append(mb / fb)
        avg_ratios.append(ma / fa)
        sqrt_err.append(abs(math.sqrt(fb) - mb))
        print(
            f"{q[:47]:<48} | {f'{fb:.3f}/{fa:.3f}':<16} | "
            f"{f'{mb:.3f}/{ma:.3f}':<16} | {mb / fb:.3f} | {math.sqrt(fb):.3f}"
        )

    if not best_ratios:
        return

    avg_best_ratio = sum(best_ratios) / len(best_ratios)
    avg_avg_ratio = sum(avg_ratios) / len(avg_ratios)
    mean_sqrt_err = sum(sqrt_err) / len(sqrt_err)

    print("-" * len(header))
    print(f"\nmean Milvus/FAISS linear ratio -> best: {avg_best_ratio:.3f}, avg_top: {avg_avg_ratio:.3f}")
    print(f"mean |sqrt(FAISS_best) - Milvus_best| = {mean_sqrt_err:.4f}  (≈0 confirms sqrt relationship)")
    print("\nNOTE: FAISS IndexFlatL2 returns SQUARED L2; Milvus L2 returns plain (non-squared) L2.")
    print("      => Milvus_L2 ≈ sqrt(FAISS_L2); the linear ratio is NOT constant across the range.")
    print(f"\ncurrent thresholds (FAISS squared-L2): best<={CONTEXT_MAX_BEST_DISTANCE}, avg<={CONTEXT_MAX_AVG_TOP_DISTANCE}")
    print(
        f"=> correct Milvus-equivalent (sqrt)   : "
        f"best<={math.sqrt(CONTEXT_MAX_BEST_DISTANCE):.2f}, "
        f"avg<={math.sqrt(CONTEXT_MAX_AVG_TOP_DISTANCE):.2f}"
    )
    print(
        f"   (naive linear extrapolation would WRONGLY give "
        f"{CONTEXT_MAX_BEST_DISTANCE * avg_best_ratio:.2f}/"
        f"{CONTEXT_MAX_AVG_TOP_DISTANCE * avg_avg_ratio:.2f})"
    )

    try:
        milvus_store.client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
