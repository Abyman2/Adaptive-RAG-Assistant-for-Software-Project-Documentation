"""Rigorous Benchmark & Evaluation Suite (V2).
Implements:
1. Strict Gold-Standard Relevance Ground Truth (chunk IDs, gold sections, evidence snippets).
2. Separate Retrieval Ranking Evaluation on Answerable Queries (Recall@1/3/5, MRR, Mean & Median Latency).
3. Separate Binary Classification Evaluation for Abstention (TP, TN, FP, FN, Precision, Recall, Specificity, F1, Balanced Accuracy).
4. Standard Classification F1 Threshold Calibration on Development Set.
5. Frozen Held-Out Evaluation (N=20 queries across technical, lexical gap, architectural, and unanswerable).
6. Separated Model Startup / Load Time vs Warm Query Latency.
7. Generated Visualizations (retrieval_comparison.png, abstention_confusion_matrix.png).
"""

import json
import time
import sys
import os
from pathlib import Path

# Use cached huggingface models offline to prevent network polling
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.document_loader import load_document
from src.chunker import process_documents
from src.baseline import TfidfBaseline
from src.retriever import SemanticRetriever
from src.adaptive_retrieval import AdaptiveRetriever
from src.generator import AnswerGenerator


def is_chunk_gold_relevant(chunk_dict, gold_chunk_ids, gold_sections, gold_evidence_snippets, score=1.0):
    """Consistent and strict ground-truth relevance check.
    
    A chunk is judged relevant iff ALL of the following criteria are met:
    1. Retrieval score > 0.0 (items with zero lexical/semantic similarity are non-matches).
    2. The chunk ID is in gold_chunk_ids.
    3. The chunk's section metadata matches at least one of gold_sections.
    4. All gold evidence snippets are contained within the chunk's text.
    """
    if score <= 0.0:
        return False

    chunk_id = chunk_dict.get("chunk_id")
    if chunk_id is None or chunk_id not in gold_chunk_ids:
        return False

    chunk_sections = chunk_dict.get("sections", [])
    if gold_sections and not any(sec in gold_sections for sec in chunk_sections):
        return False

    text_lower = chunk_dict.get("text", "").lower()
    if gold_evidence_snippets and not all(s.lower() in text_lower for s in gold_evidence_snippets):
        return False

    return True


def calibrate_thresholds(retriever, dev_questions):
    """Empirically sweep candidate abstention thresholds on the Development Set.
    
    Uses standard classification metrics: Precision, Recall, Specificity, F1, and Balanced Accuracy.
    """
    print("\n" + "=" * 65)
    print("STAGE 1: EMPIRICAL THRESHOLD CALIBRATION (Development Set N=12)")
    print("=" * 65)
    candidate_thresholds = [0.22, 0.25, 0.28, 0.31, 0.34]
    sweep_results = []

    for thresh in candidate_thresholds:
        adaptive = AdaptiveRetriever(retriever, min_threshold=thresh)
        tp = tn = fp = fn = 0

        for q in dev_questions:
            res = adaptive.retrieve(q["question"])
            accepted = res["sufficient_evidence"]
            answerable = q["answerable"]

            if answerable and accepted:
                tp += 1
            elif not answerable and not accepted:
                tn += 1
            elif not answerable and accepted:
                fp += 1
            elif answerable and not accepted:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        row = {
            "threshold": thresh,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_acc,
            "f1_score": f1
        }
        sweep_results.append(row)
        print(f"  Threshold {thresh:.2f} -> TP:{tp} TN:{tn} FP:{fp} FN:{fn} | Prec:{precision*100:.1f}% Rec:{recall*100:.1f}% Spec:{specificity*100:.1f}% | BalAcc:{balanced_acc*100:.1f}% F1:{f1:.3f}")

    # Select optimal threshold maximizing Balanced Accuracy and F1
    best = max(sweep_results, key=lambda x: (x["f1_score"], x["balanced_accuracy"]))
    print(f"\n==> Optimal Calibrated Threshold: {best['threshold']:.2f} (F1: {best['f1_score']:.3f}, BalAcc: {best['balanced_accuracy']*100:.1f}%)")
    return best["threshold"], sweep_results


def evaluate_dataset(retriever, adaptive_retriever, baseline, generator, dataset, dataset_name="Held-out"):
    """Evaluate retrieval and generation across answerable and unanswerable queries."""
    results = []

    # Warm-up run to eliminate cold-start cache anomalies
    baseline.search("warmup query", top_k=1)
    retriever.search("warmup query", top_k=1)

    for item in dataset:
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        answerable = item["answerable"]
        gold_chunk_ids = item.get("gold_chunk_ids", [])
        gold_sections = item.get("gold_sections", [])
        gold_snippets = item.get("gold_evidence_snippets", [])

        # 1. Baseline TF-IDF (Static top-5)
        t0 = time.perf_counter()
        tfidf_results = baseline.search(question, top_k=5)
        tfidf_lat = (time.perf_counter() - t0) * 1000

        # 2. Semantic Standard MiniLM (Static top-5)
        t0 = time.perf_counter()
        semantic_results = retriever.search(question, top_k=5)
        semantic_lat = (time.perf_counter() - t0) * 1000

        # 3. Adaptive Semantic
        t0 = time.perf_counter()
        adaptive_info = adaptive_retriever.retrieve(question)
        adaptive_lat = (time.perf_counter() - t0) * 1000

        # 4. Generate answer with generator
        gen_out = generator.generate(question, adaptive_info["results"], sufficient_evidence=adaptive_info["sufficient_evidence"])

        row = {
            "id": qid,
            "dataset": dataset_name,
            "category": category,
            "question": question,
            "answerable": answerable,
            "adaptive_mode": adaptive_info["mode"],
            "adaptive_confidence": adaptive_info["confidence"],
            "adaptive_top_k": adaptive_info["top_k"],
            "adaptive_accepted": adaptive_info["sufficient_evidence"],
            "generated_answer": gen_out["answer"],
            "grounded_flag": gen_out["grounded"],
            "tfidf_lat_ms": tfidf_lat,
            "semantic_lat_ms": semantic_lat,
            "adaptive_lat_ms": adaptive_lat,
        }

        # Ranking metrics calculated strictly on ANSWERABLE queries
        if answerable:
            # TF-IDF Ranking
            tfidf_hits = [is_chunk_gold_relevant(r["chunk"], gold_chunk_ids, gold_sections, gold_snippets, r["score"]) for r in tfidf_results]
            row["tfidf_r1"] = 1 if (len(tfidf_hits) > 0 and tfidf_hits[0]) else 0
            row["tfidf_r3"] = 1 if any(tfidf_hits[:3]) else 0
            row["tfidf_r5"] = 1 if any(tfidf_hits[:5]) else 0
            row["tfidf_mrr"] = next((1.0 / rank for rank, hit in enumerate(tfidf_hits, start=1) if hit), 0.0)

            # MiniLM Ranking
            sem_hits = [is_chunk_gold_relevant(r["chunk"], gold_chunk_ids, gold_sections, gold_snippets, r["score"]) for r in semantic_results]
            row["sem_r1"] = 1 if (len(sem_hits) > 0 and sem_hits[0]) else 0
            row["sem_r3"] = 1 if any(sem_hits[:3]) else 0
            row["sem_r5"] = 1 if any(sem_hits[:5]) else 0
            row["sem_mrr"] = next((1.0 / rank for rank, hit in enumerate(sem_hits, start=1) if hit), 0.0)

            # Adaptive RAG Ranking & Context Coverage
            adapt_hits = [is_chunk_gold_relevant(r["chunk"], gold_chunk_ids, gold_sections, gold_snippets, r["score"]) for r in adaptive_info["results"]]
            row["adapt_r1"] = 1 if (len(adapt_hits) > 0 and adapt_hits[0]) else 0
            row["adapt_r3"] = 1 if any(adapt_hits[:3]) else 0
            row["adapt_r5"] = 1 if any(adapt_hits[:5]) else 0
            row["adapt_mrr"] = next((1.0 / rank for rank, hit in enumerate(adapt_hits, start=1) if hit), 0.0)
            row["adapt_context_hit"] = 1 if any(adapt_hits) else 0

            # Binary Classification status for answerable queries
            row["classification_label"] = "TP" if adaptive_info["sufficient_evidence"] else "FN"
        else:
            # Unanswerable query
            row["tfidf_r1"] = None
            row["tfidf_r3"] = None
            row["tfidf_r5"] = None
            row["tfidf_mrr"] = None
            row["sem_r1"] = None
            row["sem_r3"] = None
            row["sem_r5"] = None
            row["sem_mrr"] = None
            row["adapt_r1"] = None
            row["adapt_r3"] = None
            row["adapt_r5"] = None
            row["adapt_mrr"] = None
            row["adapt_context_hit"] = None

            # Binary Classification status for unanswerable queries
            row["classification_label"] = "FP" if adaptive_info["sufficient_evidence"] else "TN"

        results.append(row)

    return results


def run_benchmark():
    """Execute complete benchmark following the approved redesign."""
    print("=" * 65)
    print("Adaptive RAG Assistant - Rigorous Evaluation Benchmark (V2)")
    print("=" * 65)

    # 1. Startup & Model Loading Profiling
    print("\n--- Model Startup / Ingestion Time ---")
    t0 = time.perf_counter()
    doc_path = Path("sample_docs/sample_project.md")
    doc = load_document(doc_path)
    chunks = process_documents([doc], chunk_size=100, overlap=25)
    doc_load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Document loading & chunking ({len(chunks)} chunks): {doc_load_ms:.2f} ms")

    t0 = time.perf_counter()
    tfidf = TfidfBaseline()
    tfidf.index(chunks)
    tfidf_init_ms = (time.perf_counter() - t0) * 1000
    print(f"  TF-IDF Baseline Indexing: {tfidf_init_ms:.2f} ms")

    t0 = time.perf_counter()
    semantic = SemanticRetriever()
    semantic.index(chunks)
    semantic_init_ms = (time.perf_counter() - t0) * 1000
    print(f"  MiniLM Embedding & Vector Indexing: {semantic_init_ms:.2f} ms")

    t0 = time.perf_counter()
    generator = AnswerGenerator()
    gen_init_ms = (time.perf_counter() - t0) * 1000
    print(f"  FLAN-T5 Small Generator Initialization: {gen_init_ms:.2f} ms")

    # 2. Load Evaluation Datasets
    with open("evaluation/dev_v2.json", "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open("evaluation/held_out_v2.json", "r", encoding="utf-8") as f:
        held_data = json.load(f)

    # 3. Calibrate Thresholds on Development Set
    optimal_threshold, calibration_sweep = calibrate_thresholds(semantic, dev_data["questions"])

    # 4. Initialize Adaptive Retriever with Calibrated Threshold
    adaptive = AdaptiveRetriever(semantic, min_threshold=optimal_threshold)

    # 5. Evaluate Development Set
    print("\n--- Evaluating Development Set (N=12) ---")
    dev_results = evaluate_dataset(semantic, adaptive, tfidf, generator, dev_data["questions"], dataset_name="Development")

    # 6. Evaluate Frozen Held-Out Set (N=20)
    print("--- Evaluating Frozen Held-Out Test Set (N=20) ---")
    held_results = evaluate_dataset(semantic, adaptive, tfidf, generator, held_data["questions"], dataset_name="Held-out")

    all_results = dev_results + held_results
    df = pd.DataFrame(all_results)

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "comparison_table_v2.csv", index=False)

    # 7. Compute Summary Metrics (Separating Retrieval Ranking from Classification)
    held_df = df[df["dataset"] == "Held-out"]
    held_ans_df = held_df[held_df["answerable"] == True]
    held_unans_df = held_df[held_df["answerable"] == False]

    # Binary Classification Metrics
    tp = sum(held_df["classification_label"] == "TP")
    tn = sum(held_df["classification_label"] == "TN")
    fp = sum(held_df["classification_label"] == "FP")
    fn = sum(held_df["classification_label"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Top-K distribution
    topk_dist = held_df["adaptive_top_k"].value_counts().to_dict()

    summary = {
        "model_startup_ms": {
            "doc_processing": doc_load_ms,
            "tfidf_indexing": tfidf_init_ms,
            "minilm_embedding": semantic_init_ms,
            "flan_t5_loading": gen_init_ms
        },
        "calibration_sweep": calibration_sweep,
        "calibrated_threshold": optimal_threshold,
        "held_out_dataset_size": {
            "total": len(held_df),
            "answerable": len(held_ans_df),
            "unanswerable": len(held_unans_df)
        },
        "retrieval_ranking_answerable": {
            "tfidf": {
                "recall_at_1": float(held_ans_df["tfidf_r1"].mean()),
                "recall_at_3": float(held_ans_df["tfidf_r3"].mean()),
                "recall_at_5": float(held_ans_df["tfidf_r5"].mean()),
                "mrr": float(held_ans_df["tfidf_mrr"].mean()),
                "mean_latency_ms": float(held_df["tfidf_lat_ms"].mean()),
                "median_latency_ms": float(held_df["tfidf_lat_ms"].median())
            },
            "minilm_semantic": {
                "recall_at_1": float(held_ans_df["sem_r1"].mean()),
                "recall_at_3": float(held_ans_df["sem_r3"].mean()),
                "recall_at_5": float(held_ans_df["sem_r5"].mean()),
                "mrr": float(held_ans_df["sem_mrr"].mean()),
                "mean_latency_ms": float(held_df["semantic_lat_ms"].mean()),
                "median_latency_ms": float(held_df["semantic_lat_ms"].median())
            },
            "adaptive_rag": {
                "recall_at_1": float(held_ans_df["adapt_r1"].mean()),
                "recall_at_3": float(held_ans_df["adapt_r3"].mean()),
                "recall_at_5": float(held_ans_df["adapt_r5"].mean()),
                "mrr": float(held_ans_df["adapt_mrr"].mean()),
                "evidence_coverage": float(held_ans_df["adapt_context_hit"].mean()),
                "mean_latency_ms": float(held_df["adaptive_lat_ms"].mean()),
                "median_latency_ms": float(held_df["adaptive_lat_ms"].median())
            }
        },
        "adaptive_topk_distribution": {str(k): int(v) for k, v in topk_dist.items()},
        "abstention_classification": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "f1_score": float(f1)
        }
    }

    with open(output_dir / "evaluation_metrics_v2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 8. Print Results Tables
    print("\n" + "=" * 65)
    print("HELD-OUT EVALUATION RESULTS (N=20 Questions)")
    print("=" * 65)

    print("\n### 1. Retrieval Ranking Performance (Answerable Queries N=12):")
    print("| Metric | TF-IDF Baseline | MiniLM Semantic (Static K=5) | Adaptive RAG (Dynamic K) |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| **Recall@1** | {summary['retrieval_ranking_answerable']['tfidf']['recall_at_1']*100:.1f}% | {summary['retrieval_ranking_answerable']['minilm_semantic']['recall_at_1']*100:.1f}% | {summary['retrieval_ranking_answerable']['adaptive_rag']['recall_at_1']*100:.1f}% |")
    print(f"| **Recall@3** | {summary['retrieval_ranking_answerable']['tfidf']['recall_at_3']*100:.1f}% | {summary['retrieval_ranking_answerable']['minilm_semantic']['recall_at_3']*100:.1f}% | {summary['retrieval_ranking_answerable']['adaptive_rag']['recall_at_3']*100:.1f}% |")
    print(f"| **Recall@5** | {summary['retrieval_ranking_answerable']['tfidf']['recall_at_5']*100:.1f}% | {summary['retrieval_ranking_answerable']['minilm_semantic']['recall_at_5']*100:.1f}% | {summary['retrieval_ranking_answerable']['adaptive_rag']['recall_at_5']*100:.1f}% |")
    print(f"| **MRR** | {summary['retrieval_ranking_answerable']['tfidf']['mrr']:.3f} | {summary['retrieval_ranking_answerable']['minilm_semantic']['mrr']:.3f} | {summary['retrieval_ranking_answerable']['adaptive_rag']['mrr']:.3f} |")
    print(f"| **Context Coverage** | - | - | {summary['retrieval_ranking_answerable']['adaptive_rag']['evidence_coverage']*100:.1f}% |")
    print(f"| **Mean Latency (ms)** | {summary['retrieval_ranking_answerable']['tfidf']['mean_latency_ms']:.2f} ms | {summary['retrieval_ranking_answerable']['minilm_semantic']['mean_latency_ms']:.2f} ms | {summary['retrieval_ranking_answerable']['adaptive_rag']['mean_latency_ms']:.2f} ms |")
    print(f"| **Median Latency (ms)** | {summary['retrieval_ranking_answerable']['tfidf']['median_latency_ms']:.2f} ms | {summary['retrieval_ranking_answerable']['minilm_semantic']['median_latency_ms']:.2f} ms | {summary['retrieval_ranking_answerable']['adaptive_rag']['median_latency_ms']:.2f} ms |")

    print("\n### 2. Answerability & Abstention Classification (N=20 Questions):")
    print(f"| Metric | Value | Meaning |")
    print(f"| :--- | :---: | :--- |")
    print(f"| **True Positives (TP)** | {tp} / 12 | Answerable questions accepted by system |")
    print(f"| **True Negatives (TN)** | {tn} / 8 | Unanswerable questions correctly abstained |")
    print(f"| **False Positives (FP)** | {fp} / 8 | Unanswerable questions falsely answered (Hallucination hazard) |")
    print(f"| **False Negatives (FN)** | {fn} / 12 | Answerable questions falsely rejected |")
    print(f"| **Answerability Precision** | {precision*100:.1f}% | Accuracy of accepted questions |")
    print(f"| **Answerability Recall** | {recall*100:.1f}% | Coverage of answerable questions |")
    print(f"| **Abstention Specificity** | {specificity*100:.1f}% | Rate of rejecting unanswerable queries |")
    print(f"| **Balanced Accuracy** | {balanced_accuracy*100:.1f}% | Macro average of Recall and Specificity |")
    print(f"| **Classification F1** | {f1:.3f} | Harmonic mean of Precision and Recall |")

    print("\n### 3. Dynamic Top-K Context Distribution:")
    for k_val, count in topk_dist.items():
        print(f"  - K = {k_val} chunks: {count} queries ({count/len(held_df)*100:.1f}%)")

    # 9. Plot 1: Retrieval Ranking Comparison Bar Chart
    plt.figure(figsize=(9, 4.5))
    metrics_names = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    tfidf_scores = [
        summary["retrieval_ranking_answerable"]["tfidf"]["recall_at_1"] * 100,
        summary["retrieval_ranking_answerable"]["tfidf"]["recall_at_3"] * 100,
        summary["retrieval_ranking_answerable"]["tfidf"]["recall_at_5"] * 100,
        summary["retrieval_ranking_answerable"]["tfidf"]["mrr"] * 100
    ]
    sem_scores = [
        summary["retrieval_ranking_answerable"]["minilm_semantic"]["recall_at_1"] * 100,
        summary["retrieval_ranking_answerable"]["minilm_semantic"]["recall_at_3"] * 100,
        summary["retrieval_ranking_answerable"]["minilm_semantic"]["recall_at_5"] * 100,
        summary["retrieval_ranking_answerable"]["minilm_semantic"]["mrr"] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, tfidf_scores, width, label="TF-IDF Baseline (Lexical)", color="#4A90E2")
    plt.bar(x + width/2, sem_scores, width, label="MiniLM-L6-v2 (Semantic 384d)", color="#2ECC71")
    plt.xticks(x, metrics_names)
    plt.ylabel("Score (%) / MRR x 100")
    plt.ylim(0, 115)
    plt.title("Held-Out Retrieval Ranking on Answerable Queries (N=12)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_comparison_v2.png", dpi=300)
    # Also overwrite retrieval_comparison.png for backward compatibility
    plt.savefig(output_dir / "retrieval_comparison.png", dpi=300)
    plt.close()

    # 10. Plot 2: Abstention Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm_matrix = np.array([[tp, fn], [fp, tn]])
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Adaptive Abstention Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Accepted", "Abstained"])
    plt.yticks(tick_marks, ["Answerable", "Unanswerable"])
    thresh_val = cm_matrix.max() / 2.
    labels_cm = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels_cm[i][j]}\n{cm_matrix[i, j]}",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_matrix[i, j] > thresh_val else "black",
                     fontsize=12, fontweight="bold")
    plt.ylabel("True Status")
    plt.xlabel("System Action")
    plt.tight_layout()
    plt.savefig(output_dir / "abstention_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\nPlots saved to:\n  - {output_dir / 'retrieval_comparison_v2.png'}\n  - {output_dir / 'abstention_confusion_matrix.png'}")

    return summary


if __name__ == "__main__":
    run_benchmark()
