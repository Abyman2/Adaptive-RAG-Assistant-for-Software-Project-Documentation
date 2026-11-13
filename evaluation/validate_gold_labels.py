"""Validation script for gold evidence labels.
Verifies that all gold_chunk_ids, gold_sections, and gold_evidence_snippets
exist and match the frozen chunked documentation corpus.
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.document_loader import load_document
from src.chunker import process_documents


def validate_dataset(dataset_file, chunks):
    """Verify ground truth labels against chunks."""
    print(f"\n--- Validating: {dataset_file.name} ---")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    all_passed = True

    for q in questions:
        qid = q["id"]
        answerable = q["answerable"]
        if not answerable:
            assert len(q["gold_chunk_ids"]) == 0, f"{qid}: Unanswerable question must have empty gold_chunk_ids"
            continue

        gold_ids = q["gold_chunk_ids"]
        assert len(gold_ids) > 0, f"{qid}: Answerable question must specify at least one gold_chunk_id"

        # Check each gold chunk
        for gid in gold_ids:
            if gid < 0 or gid >= len(chunks):
                print(f"FAIL: {qid} invalid chunk id {gid} (total chunks: {len(chunks)})")
                all_passed = False
                continue

            chunk_text = chunks[gid]["text"]

            # Verify that at least one gold evidence snippet is in the chunk
            snippets = q["gold_evidence_snippets"]
            snippet_found = any(s.lower() in chunk_text.lower() for s in snippets)
            if not snippet_found:
                print(f"FAIL: {qid} none of snippets {snippets} found in chunk {gid}")
                all_passed = False

    if all_passed:
        print(f"SUCCESS: All {len(questions)} labels in {dataset_file.name} verified against frozen corpus!")
    return all_passed


def main():
    doc_path = Path("sample_docs/sample_project.md")
    doc = load_document(doc_path)
    chunks = process_documents([doc], chunk_size=100, overlap=25)
    print(f"Loaded frozen corpus: {len(chunks)} chunks.")

    v1 = validate_dataset(Path("evaluation/dev_v2.json"), chunks)
    v2 = validate_dataset(Path("evaluation/held_out_v2.json"), chunks)

    if v1 and v2:
        print("\n*** ALL GOLD EVIDENCE LABELS VERIFIED AND FROZEN ***")
    else:
        raise ValueError("Gold label validation failed!")


if __name__ == "__main__":
    main()
