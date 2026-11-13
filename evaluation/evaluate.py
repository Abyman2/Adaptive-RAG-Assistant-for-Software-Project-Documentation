"""Evaluation and benchmarking script.
Compares:
1. TF-IDF Keyword Baseline
2. Semantic MiniLM Retriever (Static top-k)
3. Adaptive MiniLM Retriever (Dynamic top-k + confidence gating)

Calculates Recall@1, Recall@3, Recall@5, MRR, Abstention Accuracy, and Latency
on both Development and Held-out evaluation sets.
Generates evaluation tables and visualization plots.
"""

import json
import time
import sys
import os
from pathlib import Path

# Use cached huggingface models offline to prevent network polling
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.document_loader import load_document
from src.chunker import process_documents
from src.baseline import TfidfBaseline
from src.retriever import SemanticRetriever
from src.adaptive_retrieval import AdaptiveRetriever


def is_chunk_relevant(chunk_text, expected_keywords):
    """Check if any expected keyword is contained in the chunk text (case-insensitive)."""
    text_lower = chunk_text.lower()
    for kw in expected_keywords:
        if kw.lower() in text_lower:
            return True
    return False


def evaluate_dataset(retriever, adaptive_retriever, baseline, dataset, chunks, dataset_name="Held-out"):
    """Evaluate retrievers against a labeled dataset."""
    results = []

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        answerable = item["answerable"]

        # 1. Baseline TF-IDF
        t0 = time.time()
        tfidf_results = baseline.search(question, top_k=5)
        tfidf_lat = (time.time() - t0) * 1000

        # 2. Semantic Standard (top_k=5)
        t0 = time.time()
        semantic_results = retriever.search(question, top_k=5)
        semantic_lat = (time.time() - t0) * 1000

        # 3. Adaptive Semantic
        t0 = time.time()
        adaptive_info = adaptive_retriever.retrieve(question)
        adaptive_lat = (time.time() - t0) * 1000

        row = {
            "id": qid,
            "dataset": dataset_name,
            "question": question,
            "answerable": answerable,
            "adaptive_mode": adaptive_info["mode"],
            "adaptive_confidence": adaptive_info["confidence"],
            "adaptive_top_k": adaptive_info["top_k"],
            "adaptive_sufficient_evidence": adaptive_info["sufficient_evidence"],
            "tfidf_lat_ms": tfidf_lat,
            "semantic_lat_ms": semantic_lat,
            "adaptive_lat_ms": adaptive_lat,
        }

        if answerable:
            # Check Recall@1, Recall@3, Recall@5 for TF-IDF
            tfidf_texts = [r["chunk"]["text"] for r in tfidf_results]
            row["tfidf_r1"] = 1 if is_chunk_relevant(tfidf_texts[0], expected_keywords) else 0
            row["tfidf_r3"] = 1 if any(is_chunk_relevant(t, expected_keywords) for t in tfidf_texts[:3]) else 0
            row["tfidf_r5"] = 1 if any(is_chunk_relevant(t, expected_keywords) for t in tfidf_texts[:5]) else 0

            # MRR TF-IDF
            mrr_tfidf = 0.0
            for rank, t in enumerate(tfidf_texts, start=1):
                if is_chunk_relevant(t, expected_keywords):
                    mrr_tfidf = 1.0 / rank
                    break
            row["tfidf_mrr"] = mrr_tfidf

            # Check Recall@1, Recall@3, Recall@5 for Semantic
            sem_texts = [r["chunk"]["text"] for r in semantic_results]
            row["sem_r1"] = 1 if is_chunk_relevant(sem_texts[0], expected_keywords) else 0
            row["sem_r3"] = 1 if any(is_chunk_relevant(t, expected_keywords) for t in sem_texts[:3]) else 0
            row["sem_r5"] = 1 if any(is_chunk_relevant(t, expected_keywords) for t in sem_texts[:5]) else 0

            # MRR Semantic
            mrr_sem = 0.0
            for rank, t in enumerate(sem_texts, start=1):
                if is_chunk_relevant(t, expected_keywords):
                    mrr_sem = 1.0 / rank
                    break
            row["sem_mrr"] = mrr_sem

            # Adaptive Recall: whether retrieved chunks contain evidence
            adaptive_texts = [r["chunk"]["text"] for r in adaptive_info["results"]]
            row["adaptive_hit"] = 1 if any(is_chunk_relevant(t, expected_keywords) for t in adaptive_texts) else 0
            row["adaptive_correct_abstention"] = None
        else:
            # Unanswerable question:
            # TF-IDF doesn't have confidence gating, so it retrieves irrelevant text
            row["tfidf_r1"] = 0
            row["tfidf_r3"] = 0
            row["tfidf_r5"] = 0
            row["tfidf_mrr"] = 0.0
            row["sem_r1"] = 0
            row["sem_r3"] = 0
            row["sem_r5"] = 0
            row["sem_mrr"] = 0.0
            row["adaptive_hit"] = 0
            # Correct abstention: adaptive system recognized insufficient evidence!
            row["adaptive_correct_abstention"] = 1 if not adaptive_info["sufficient_evidence"] else 0

        results.append(row)

    return results


def run_benchmark():
    """Execute complete benchmark on Dev and Held-out sets."""
    print("=" * 60)
    print("Starting Benchmark: Adaptive RAG Assistant for Software Docs")
    print("=" * 60)

    dataset_path = Path("evaluation/dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load and index documentation
    doc_path = Path(data["document"])
    doc = load_document(doc_path)
    chunks = process_documents([doc], chunk_size=100, overlap=25)
    print(f"Loaded document '{doc['source']}' ({len(doc['text'].split())} words) -> {len(chunks)} chunks.")

    print("\n[1/3] Indexing TF-IDF Baseline...")
    tfidf = TfidfBaseline()
    tfidf.index(chunks)

    print("[2/3] Loading & Indexing SentenceTransformer (all-MiniLM-L6-v2)...")
    semantic = SemanticRetriever()
    semantic.index(chunks)

    print("[3/3] Initializing Adaptive Retriever...")
    adaptive = AdaptiveRetriever(semantic)

    # Evaluate Development Set
    print("\n--- Evaluating Development Set (12 questions) ---")
    dev_results = evaluate_dataset(semantic, adaptive, tfidf, data["development_set"], chunks, dataset_name="Development")

    # Evaluate Held-out Set
    print("--- Evaluating Held-out Set (8 questions) ---")
    held_results = evaluate_dataset(semantic, adaptive, tfidf, data["held_out_set"], chunks, dataset_name="Held-out")

    all_results = dev_results + held_results
    df = pd.DataFrame(all_results)

    # Create output directory
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "comparison_table.csv", index=False)

    # Summary metrics computation
    summary = {}
    for subset in ["Development", "Held-out"]:
        sub_df = df[df["dataset"] == subset]
        answerable_df = sub_df[sub_df["answerable"] == True]
        unanswerable_df = sub_df[sub_df["answerable"] == False]

        summary[subset] = {
            "total_questions": len(sub_df),
            "answerable_count": len(answerable_df),
            "unanswerable_count": len(unanswerable_df),
            "tfidf_recall_at_1": float(answerable_df["tfidf_r1"].mean()),
            "tfidf_recall_at_3": float(answerable_df["tfidf_r3"].mean()),
            "tfidf_recall_at_5": float(answerable_df["tfidf_r5"].mean()),
            "tfidf_mrr": float(answerable_df["tfidf_mrr"].mean()),
            "tfidf_latency_ms": float(sub_df["tfidf_lat_ms"].mean()),
            "semantic_recall_at_1": float(answerable_df["sem_r1"].mean()),
            "semantic_recall_at_3": float(answerable_df["sem_r3"].mean()),
            "semantic_recall_at_5": float(answerable_df["sem_r5"].mean()),
            "semantic_mrr": float(answerable_df["sem_mrr"].mean()),
            "semantic_latency_ms": float(sub_df["semantic_lat_ms"].mean()),
            "adaptive_evidence_recall": float(answerable_df["adaptive_hit"].mean()),
            "adaptive_abstention_accuracy": float(unanswerable_df["adaptive_correct_abstention"].mean()) if len(unanswerable_df) > 0 else 1.0,
            "adaptive_latency_ms": float(sub_df["adaptive_lat_ms"].mean()),
        }

    with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print clean Markdown summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 60)
    print("\n### Held-Out Test Set Performance (N=8 questions):")
    print(f"| Metric | TF-IDF Baseline | MiniLM Semantic | Adaptive MiniLM |")
    print(f"| :--- | :---: | :---: | :---: |")
    print(f"| Recall@1 | {summary['Held-out']['tfidf_recall_at_1']*100:.1f}% | {summary['Held-out']['semantic_recall_at_1']*100:.1f}% | - |")
    print(f"| Recall@3 | {summary['Held-out']['tfidf_recall_at_3']*100:.1f}% | {summary['Held-out']['semantic_recall_at_3']*100:.1f}% | - |")
    print(f"| Recall@5 | {summary['Held-out']['tfidf_recall_at_5']*100:.1f}% | {summary['Held-out']['semantic_recall_at_5']*100:.1f}% | - |")
    print(f"| MRR (Mean Reciprocal Rank) | {summary['Held-out']['tfidf_mrr']:.3f} | {summary['Held-out']['semantic_mrr']:.3f} | - |")
    print(f"| Evidence Recall (Dynamic top-K) | - | - | {summary['Held-out']['adaptive_evidence_recall']*100:.1f}% |")
    print(f"| Abstention Accuracy (Unanswerable) | 0.0% (Always retrieves) | 0.0% (No gating) | {summary['Held-out']['adaptive_abstention_accuracy']*100:.1f}% |")
    print(f"| Mean Latency (ms) | {summary['Held-out']['tfidf_latency_ms']:.2f} ms | {summary['Held-out']['semantic_latency_ms']:.2f} ms | {summary['Held-out']['adaptive_latency_ms']:.2f} ms |")

    # Generate Visualization Plot
    plot_path = output_dir / "retrieval_comparison.png"
    plt.figure(figsize=(10, 5))
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    tfidf_vals = [
        summary["Held-out"]["tfidf_recall_at_1"],
        summary["Held-out"]["tfidf_recall_at_3"],
        summary["Held-out"]["tfidf_recall_at_5"],
        summary["Held-out"]["tfidf_mrr"]
    ]
    sem_vals = [
        summary["Held-out"]["semantic_recall_at_1"],
        summary["Held-out"]["semantic_recall_at_3"],
        summary["Held-out"]["semantic_recall_at_5"],
        summary["Held-out"]["semantic_mrr"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, [v * 100 for v in tfidf_vals], width, label="TF-IDF Baseline (Keyword)", color="#4A90E2")
    plt.bar(x + width/2, [v * 100 for v in sem_vals], width, label="MiniLM-L6-v2 (Dense Transformer)", color="#50E3C2")

    plt.ylabel("Score (%) / MRR x 100")
    plt.title("Held-out Retrieval Evaluation: TF-IDF vs Transformer MiniLM")
    plt.xticks(x, metrics)
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Failure Analysis Output
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS HIGHLIGHTS")
    print("=" * 60)
    # Find queries where TF-IDF failed but Semantic succeeded (Lexical gap)
    lexical_gap = df[(df["tfidf_r1"] == 0) & (df["sem_r1"] == 1)]
    print(f"\n1. Lexical Gap Failures in Baseline (TF-IDF Failed, Semantic Succeeded): {len(lexical_gap)} instances")
    for _, r in lexical_gap.head(2).iterrows():
        print(f"   - Query: '{r['question']}'")
        print(f"     Reason: Developer phrasing differed from exact terminology in doc.")

    # Unanswerable queries
    unans = df[df["answerable"] == False]
    print(f"\n2. Unanswerable & Out-of-Domain Queries: {len(unans)} instances")
    for _, r in unans.head(2).iterrows():
        status = "CORRECTLY ABSTAINED" if r["adaptive_correct_abstention"] == 1 else "FAILED TO ABSTAIN"
        print(f"   - Query: '{r['question']}' -> Confidence: {r['adaptive_confidence']:.2f} -> {status}")

    return summary


if __name__ == "__main__":
    run_benchmark()
