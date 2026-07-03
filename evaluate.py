"""
evaluate.py
Evaluation script for Cost-Efficient RAG
"""
import json
import math
import pandas as pd
from rag import retrieve, answer

def recall_at_k(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / max(1, len(relevant))

def hit_rate(retrieved, relevant):
    return int(len(set(retrieved) & set(relevant)) > 0)

def mrr(retrieved, relevant):
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1 / i
    return 0

def ndcg_at_k(retrieved, relevant, k):
    dcg = 0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            dcg += 1 / math.log2(i + 1)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0

def evaluate_suite(path="evaluation/questions.json", k=5):
    with open(path, "r", encoding="utf-8") as f:
        suite = json.load(f)

    rows = []

    for case in suite:
        docs = retrieve(case["question"], k=k)
        retrieved = [d.metadata.get("chunk_id") for d in docs]
        relevant = case["relevant_chunks"]

        rag = answer(case["question"], k=k)

        rows.append({
            "question": case["question"],
            "recall@k": recall_at_k(retrieved, relevant),
            "hit_rate": hit_rate(retrieved, relevant),
            "mrr": mrr(retrieved, relevant),
            "ndcg@k": ndcg_at_k(retrieved, relevant, k),
            "retrieved_chunks": len(retrieved),
            "answer": rag["answer"]
        })

    df = pd.DataFrame(rows)
    os.makedirs("evaluation", exist_ok=True)
    out = "evaluation/results.csv"
    df.to_csv(out, index=False)

    print(df.mean(numeric_only=True))
    print(f"Saved results to {out}")

if __name__ == "__main__":
    evaluate_suite()
