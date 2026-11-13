"""Streamlit web interface for Adaptive RAG Assistant.
Provides:
1. Project Manager: Ingest project files, directories, or sample software docs.
2. AI Assistant: Natural language Q&A with adaptive confidence badges & grounded answers.
3. Retrieval Comparison: Side-by-side comparison of TF-IDF baseline vs MiniLM semantic search.
4. Evaluation Dashboard: Live benchmark metrics, calibration sweep, plots, and failure analysis.
"""

import os
import tempfile
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Configure offline mode for cached Hugging Face models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.document_loader import load_document, load_directory, SUPPORTED_EXTENSIONS
from src.chunker import process_documents
from src.retriever import SemanticRetriever
from src.adaptive_retrieval import AdaptiveRetriever
from src.generator import AnswerGenerator
from src.baseline import TfidfBaseline


st.set_page_config(
    page_title="Adaptive RAG Assistant for Software Docs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #4A90E2;
        margin-bottom: 12px;
    }
    .badge-focused {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-expanded {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-broad {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-insufficient {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Transformer Models (all-MiniLM-L6-v2 & flan-t5-small)...")
def load_models():
    """Load semantic retriever and generator models once and cache them."""
    semantic = SemanticRetriever()
    adaptive = AdaptiveRetriever(semantic, min_threshold=0.31)
    generator = AnswerGenerator()
    baseline = TfidfBaseline()
    return semantic, adaptive, generator, baseline


def main():
    st.title("📚 Adaptive RAG Assistant for Software Project Documentation")
    st.caption("A Transformer-powered engineering tool that dynamically adapts retrieval context and abstains on out-of-domain questions.")

    # Load models
    semantic, adaptive, generator, baseline = load_models()

    # Sidebar Configurations
    st.sidebar.header("⚙️ Adaptive Engine Calibration")
    focused_thresh = st.sidebar.slider("Focused Mode Threshold", 0.40, 0.80, 0.55, 0.05,
                                      help="Queries with top-1 score >= threshold retrieve top-3 chunks.")
    expanded_thresh = st.sidebar.slider("Expanded Mode Threshold", 0.25, 0.55, 0.35, 0.05,
                                       help="Queries in this range expand context to top-6 chunks.")
    min_thresh = st.sidebar.slider("Abstention Threshold (Calibrated)", 0.15, 0.45, 0.31, 0.01,
                                  help="Empirically calibrated cutoff on development set; queries below this are flagged as unanswerable.")

    adaptive.focused_threshold = focused_thresh
    adaptive.expanded_threshold = expanded_thresh
    adaptive.min_threshold = min_thresh

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Core Technology Stack:**
    - **Embeddings:** `all-MiniLM-L6-v2` (384d Dense)
    - **Generator:** `google/flan-t5-small` (Seq2Seq)
    - **Baseline:** Scikit-learn TF-IDF Vectorizer
    - **Related Precedent:** Calibrated abstention cutoff (Emirhan Yildirim)
    """)

    # Initialize Session State
    if "documents" not in st.session_state:
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.indexed = False

    # Main Navigation Tabs
    tab_proj, tab_chat, tab_comp, tab_eval = st.tabs([
        "📂 Project Manager",
        "💬 AI Documentation Assistant",
        "⚖️ Retrieval Comparison",
        "📊 Evaluation & Failure Analysis Dashboard"
    ])

    # -------------------------------------------------------------
    # 1. Project Manager
    # -------------------------------------------------------------
    with tab_proj:
        st.subheader("📁 Project Documentation Ingestion")
        st.write("Index an existing software project repository, upload documentation files, or load the pre-configured sample project.")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Option A: Load Built-in UniServe Project")
            st.info("UniServe is a full-stack Spring Boot 3 + React 18 microservice platform with PostgreSQL, Kafka, and Flyway documentation.")
            if st.button("🚀 Load UniServe Sample Documentation", type="primary", use_container_width=True):
                sample_path = Path("sample_docs/sample_project.md")
                if sample_path.exists():
                    doc = load_document(sample_path)
                    st.session_state.documents = [doc]
                    st.session_state.chunks = process_documents([doc], chunk_size=100, overlap=25)
                    semantic.index(st.session_state.chunks)
                    baseline.index(st.session_state.chunks)
                    st.session_state.indexed = True
                    st.success(f"Indexed UniServe documentation: {len(st.session_state.chunks)} chunks generated.")

        with col_b:
            st.markdown("#### Option B: Ingest Local Directory")
            dir_input = st.text_input("Enter local project path:", placeholder="e.g. C:/Projects/my-backend-app")
            if st.button("📁 Index Local Directory", use_container_width=True) and dir_input:
                target_dir = Path(dir_input)
                if target_dir.is_dir():
                    docs = load_directory(target_dir)
                    if docs:
                        st.session_state.documents = docs
                        st.session_state.chunks = process_documents(docs, chunk_size=100, overlap=25)
                        semantic.index(st.session_state.chunks)
                        baseline.index(st.session_state.chunks)
                        st.session_state.indexed = True
                        st.success(f"Successfully indexed {len(docs)} files ({len(st.session_state.chunks)} chunks) from {target_dir.name}")
                    else:
                        st.error("No supported documentation or code files found in directory.")
                else:
                    st.error("Invalid directory path.")

        st.markdown("---")
        st.markdown("#### Option C: Upload Project Files Directly")
        uploaded_files = st.file_uploader(
            "Upload documentation (PDF, DOCX, Markdown, Code, JSON, YAML):",
            type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("📥 Index Uploaded Files", use_container_width=True):
            docs = []
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    loaded = load_document(tmp_path)
                    loaded["source"] = uf.name
                    docs.append(loaded)
                except Exception as e:
                    st.error(f"Error loading {uf.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            if docs:
                st.session_state.documents = docs
                st.session_state.chunks = process_documents(docs, chunk_size=100, overlap=25)
                semantic.index(st.session_state.chunks)
                baseline.index(st.session_state.chunks)
                st.session_state.indexed = True
                st.success(f"Successfully indexed {len(docs)} uploaded files ({len(st.session_state.chunks)} chunks).")

        # Current status
        if st.session_state.indexed:
            st.markdown("---")
            st.markdown("### 📋 Currently Indexed Documents")
            stat_c1, stat_c2, stat_c3 = st.columns(3)
            stat_c1.metric("Indexed Documents", len(st.session_state.documents))
            stat_c2.metric("Total Context Chunks", len(st.session_state.chunks))
            stat_c3.metric("Vector Embeddings", f"{len(st.session_state.chunks)} x 384d")
            with st.expander("View indexed file list"):
                for d in st.session_state.documents:
                    st.write(f"- `{d['source']}` ({len(d['text'].split())} words)")

    # -------------------------------------------------------------
    # 2. AI Documentation Assistant
    # -------------------------------------------------------------
    with tab_chat:
        st.subheader("💬 Ask Questions About the Software Project")
        if not st.session_state.indexed:
            st.warning("👈 Please index project documents in the **Project Manager** tab first.")
        else:
            preset_questions = [
                "Select an example question...",
                "What terminal command launches the client-side user interface in development mode?",
                "What HTTP route delivers Prometheus performance metrics for application monitoring?",
                "How can an engineer manually trigger database structure updates from the CLI?",
                "What distinct access privilege tiers are enforced in the role-based security system?",
                "Which message queues or event streams receive technician routing and dispatch updates?",
                "What base endpoint does the web client target when making remote HTTP calls to the backend?",
                "What in-memory datastore caches geospatial technician positions and active session state?",
                "What Stripe API secret key is used for payment processing? (Unanswerable query)",
                "What is the Redis memory eviction policy when cache capacity exceeds 80%? (Unanswerable query)",
                "What is the team's cloud hosting monthly budget limit on AWS? (Unanswerable query)"
            ]
            selected_preset = st.selectbox("Quick-test example questions:", preset_questions)
            default_query = selected_preset.split(" (")[0] if selected_preset != preset_questions[0] else ""

            query = st.text_input("Enter your software engineering question:", value=default_query)

            if st.button("Submit Question", type="primary") and query.strip():
                with st.spinner("Analyzing documentation with Adaptive RAG..."):
                    adaptive_res = adaptive.retrieve(query)
                    mode = adaptive_res["mode"]
                    confidence = adaptive_res["confidence"]
                    top_k = adaptive_res["top_k"]
                    sufficient = adaptive_res["sufficient_evidence"]
                    results = adaptive_res["results"]

                    gen_res = generator.generate(query, results, sufficient_evidence=sufficient)

                st.markdown("---")
                st.markdown("### 🎯 Adaptive Assessment")

                if not sufficient:
                    badge_html = f'<span class="badge-insufficient">⚠️ INSUFFICIENT EVIDENCE (Confidence: {confidence:.2f} < {min_thresh:.2f})</span>'
                elif mode == "focused":
                    badge_html = f'<span class="badge-focused">🎯 FOCUSED MODE (Confidence: {confidence:.2f} >= {focused_thresh:.2f}) — Top {top_k} Chunks</span>'
                elif mode == "expanded":
                    badge_html = f'<span class="badge-expanded">🔍 EXPANDED MODE (Confidence: {confidence:.2f} >= {expanded_thresh:.2f}) — Top {top_k} Chunks</span>'
                else:
                    badge_html = f'<span class="badge-broad">🌐 BROAD MODE (Confidence: {confidence:.2f}) — Top {top_k} Chunks</span>'

                st.markdown(badge_html, unsafe_allow_html=True)
                st.caption(f"**Adaptive Strategy Rationale:** {adaptive_res['explanation']}")

                st.markdown("### 💡 Generated Grounded Answer")
                st.info(f"**Answer:** {gen_res['answer']}")

                if gen_res["citations"]:
                    st.markdown(f"**Source Document References:** `{', '.join(gen_res['citations'])}`")

                with st.expander(f"📄 Inspect Retrieved Context ({len(results)} Chunks)"):
                    for i, r in enumerate(results, start=1):
                        chunk = r["chunk"]
                        st.markdown(f"**Chunk #{i}** | Source: `{chunk['source']}` | Similarity Score: `{r['score']:.3f}`")
                        st.code(chunk["text"], language="markdown")

    # -------------------------------------------------------------
    # 3. Retrieval Comparison
    # -------------------------------------------------------------
    with tab_comp:
        st.subheader("⚖️ Retrieval Comparison: TF-IDF vs MiniLM vs Adaptive RAG")
        st.write("Compare how traditional keyword search evaluates relevance versus dense Transformer semantic embeddings.")

        if not st.session_state.indexed:
            st.warning("Please index project documents in the **Project Manager** tab first.")
        else:
            comp_query = st.text_input("Enter query for side-by-side comparison:",
                                       value="How can an engineer manually trigger database structure updates from the CLI?")
            if st.button("Run Side-by-Side Comparison", type="secondary") and comp_query.strip():
                tfidf_res = baseline.search(comp_query, top_k=3)
                dense_res = semantic.search(comp_query, top_k=3)
                adapt_res = adaptive.retrieve(comp_query)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 1. TF-IDF Keyword Baseline")
                    st.caption("Matches exact token occurrences and term frequency.")
                    if tfidf_res:
                        for r in tfidf_res:
                            st.metric(f"Rank {r['rank']} Score", f"{r['score']:.3f}")
                            st.caption("✅ Lexical overlap")
                            st.text_area(f"TF-IDF #{r['rank']} ({r['chunk']['source']})", r["chunk"]["text"][:220] + "...", height=100, disabled=True)
                    else:
                        st.warning("❌ No lexical matches found (zero term overlap).")

                with col2:
                    st.markdown("#### 2. MiniLM Dense Semantic")
                    st.caption("Matches contextual meaning in 384d embedding space.")
                    for r in dense_res:
                        st.metric(f"Rank {r['rank']} Score", f"{r['score']:.3f}")
                        st.caption("Dense vector cosine similarity")
                        st.text_area(f"MiniLM #{r['rank']} ({r['chunk']['source']})", r["chunk"]["text"][:220] + "...", height=100, disabled=True)

                with col3:
                    st.markdown("#### 3. Adaptive Dynamic Engine")
                    st.caption(f"Mode: **{adapt_res['mode'].upper()}** (Top-1: {adapt_res['confidence']:.2f})")
                    st.write(f"- **Top-K Selected:** `{adapt_res['top_k']}` chunks")
                    st.write(f"- **Evidence Sufficient:** `{adapt_res['sufficient_evidence']}`")
                    st.write(f"- **Action:** {adapt_res['explanation']}")
                    if not adapt_res['sufficient_evidence']:
                        st.error("System actively abstains to minimize hallucination risk!")
                    else:
                        st.success(f"Dispatched {len(adapt_res['results'])} evidence chunks to generator.")

    # -------------------------------------------------------------
    # 4. Evaluation & Failure Analysis Dashboard
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader("📊 Rigorous Evaluation & Failure Analysis Dashboard")
        st.write("Empirical benchmark results evaluating TF-IDF baseline, MiniLM semantic retriever, and Adaptive RAG across answerable ranking and abstention classification.")

        metrics_file = Path("evaluation/results/evaluation_metrics_v2.json")
        plot_retrieval = Path("evaluation/results/retrieval_comparison_v2.png")
        plot_cm = Path("evaluation/results/abstention_confusion_matrix.png")
        table_file = Path("evaluation/results/comparison_table_v2.csv")

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                m_data = json.load(f)

            st.markdown("### 1. Retrieval Ranking Performance on Answerable Queries (N=12)")
            st.caption("Strict ground truth check against verified gold chunk IDs and evidence snippets.")
            rank_data = m_data.get("retrieval_ranking_answerable", {})
            tf_data = rank_data.get("tfidf", {})
            sem_data = rank_data.get("minilm_semantic", {})
            adapt_data = rank_data.get("adaptive_rag", {})

            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
            r_col1.metric("TF-IDF Recall@1", f"{tf_data.get('recall_at_1', 0)*100:.1f}%")
            r_col2.metric("MiniLM Recall@1", f"{sem_data.get('recall_at_1', 0)*100:.1f}%")
            r_col3.metric("MiniLM Recall@3", f"{sem_data.get('recall_at_3', 0)*100:.1f}%")
            r_col4.metric("MiniLM MRR", f"{sem_data.get('mrr', 0):.3f}")
            r_col5.metric("Adaptive Context Coverage", f"{adapt_data.get('evidence_coverage', 0)*100:.1f}%")

            st.markdown("---")
            st.markdown("### 2. Answerability & Abstention Classification (N=20 Questions)")
            st.caption("Evaluating system capacity to accept answerable questions and reject out-of-domain questions.")
            cm = m_data.get("abstention_classification", {})

            c_c1, c_c2, c_c3, c_c4, c_c5 = st.columns(5)
            c_c1.metric("True Positives (TP)", f"{cm.get('tp', 0)} / 12")
            c_c2.metric("True Negatives (TN)", f"{cm.get('tn', 0)} / 8")
            c_c3.metric("Answerability Precision", f"{cm.get('precision', 0)*100:.1f}%")
            c_c4.metric("Abstention Specificity", f"{cm.get('specificity', 0)*100:.1f}%")
            c_c5.metric("Balanced Accuracy", f"{cm.get('balanced_accuracy', 0)*100:.1f}%")

            st.markdown("---")
            p_left, p_right = st.columns(2)

            with p_left:
                st.markdown("#### Retrieval Ranking Comparison")
                if plot_retrieval.exists():
                    st.image(str(plot_retrieval), use_container_width=True)

            with p_right:
                st.markdown("#### Abstention Decision Confusion Matrix")
                if plot_cm.exists():
                    st.image(str(plot_cm), use_container_width=True)

            st.markdown("---")
            st.markdown("### 3. Model Startup Time vs Warm Query Latency")
            su = m_data.get("model_startup_ms", {})
            st.write(f"- **Document Loading & Chunking:** `{su.get('doc_processing', 0):.2f} ms`")
            st.write(f"- **TF-IDF Index Construction:** `{su.get('tfidf_indexing', 0):.2f} ms`")
            st.write(f"- **MiniLM Embedding Generation:** `{su.get('minilm_embedding', 0):.2f} ms`")
            st.write(f"- **FLAN-T5 Model Loading:** `{su.get('flan_t5_loading', 0):.2f} ms`")
            st.write(f"- **Mean Warm Latencies:** TF-IDF: `{tf_data.get('mean_latency_ms', 0):.2f} ms` | MiniLM: `{sem_data.get('mean_latency_ms', 0):.2f} ms` | Adaptive RAG: `{adapt_data.get('mean_latency_ms', 0):.2f} ms`")

            st.markdown("---")
            st.markdown("### 4. Empirical Threshold Calibration Sweep (Development Set)")
            if "calibration_sweep" in m_data:
                sweep_df = pd.DataFrame(m_data["calibration_sweep"])
                st.dataframe(sweep_df.style.format({
                    "threshold": "{:.2f}",
                    "precision": "{:.1%}",
                    "recall": "{:.1%}",
                    "specificity": "{:.1%}",
                    "balanced_accuracy": "{:.1%}",
                    "f1_score": "{:.3f}"
                }), use_container_width=True)
                st.success(f"**Optimal Calibrated Cutoff Selected:** `{m_data.get('calibrated_threshold', 0.31):.2f}`")

            if table_file.exists():
                with st.expander("📄 View Detailed Per-Query Predictions Table (CSV)"):
                    df_preds = pd.read_csv(table_file)
                    st.dataframe(df_preds, use_container_width=True)
        else:
            st.info("Evaluation results file not found. Run `python evaluation/evaluate.py` to generate.")


if __name__ == "__main__":
    main()
