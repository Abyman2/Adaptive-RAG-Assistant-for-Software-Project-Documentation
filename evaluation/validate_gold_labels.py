"""Validation script for gold evidence labels.
Strictly verifies that:
1. Every answerable question has valid gold_chunk_ids.
2. Every gold chunk belongs to one of the question's gold_sections.
3. Every gold chunk contains the question's gold_evidence_snippets.
4. Unanswerable questions have empty gold lists.
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
            assert len(q["gold_sections"]) == 0, f"{qid}: Unanswerable question must have empty gold_sections"
            assert len(q["gold_evidence_snippets"]) == 0, f"{qid}: Unanswerable question must have empty snippets"
            continue

        gold_ids = q["gold_chunk_ids"]
        gold_sections = q["gold_sections"]
        snippets = q["gold_evidence_snippets"]

        assert len(gold_ids) > 0, f"{qid}: Answerable question must specify at least one gold_chunk_id"
        assert len(gold_sections) > 0, f"{qid}: Answerable question must specify at least one gold_section"
        assert len(snippets) > 0, f"{qid}: Answerable question must specify at least one gold_evidence_snippet"

        # Check each gold chunk
        for gid in gold_ids:
            if gid < 0 or gid >= len(chunks):
                print(f"FAIL: {qid} invalid chunk id {gid} (total chunks: {len(chunks)})")
                all_passed = False
                continue

            chunk = chunks[gid]
            chunk_text = chunk["text"]
            chunk_sections = chunk.get("sections", [])

            # 1. Section check: chunk must be associated with at least one gold section
            section_match = any(sec in gold_sections for sec in chunk_sections)
            if not section_match:
                print(f"FAIL: {qid} chunk {gid} sections {chunk_sections} do not match gold_sections {gold_sections}")
                all_passed = False

            # 2. Evidence snippet check: all snippets must be in chunk text
            snippet_found = all(s.lower() in chunk_text.lower() for s in snippets)
            if not snippet_found:
                print(f"FAIL: {qid} snippets {snippets} not all found in chunk {gid}")
                all_passed = False

    if all_passed:
        print(f"SUCCESS: All {len(questions)} questions in {dataset_file.name} passed section, chunk, and evidence verification!")
    return all_passed


def main():
    doc_path = Path("sample_docs/sample_project.md")
    doc = load_document(doc_path)
    chunks = process_documents([doc], chunk_size=100, overlap=25)
    print(f"Loaded frozen corpus: {len(chunks)} chunks.")

    v1 = validate_dataset(Path("evaluation/dev_v2.json"), chunks)
    v2 = validate_dataset(Path("evaluation/held_out_v2.json"), chunks)

    if v1 and v2:
        print("\n*** ALL GOLD EVIDENCE LABELS STRICTLY VALIDATED (SECTION + CHUNK ID + SNIPPET) ***")
    else:
        raise ValueError("Gold label validation failed!")


if __name__ == "__main__":
    main()
