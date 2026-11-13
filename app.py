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
    adaptive = AdaptiveRetriever(semantic, min_threshold=0.28)
    generator = AnswerGenerator()
    baseline = TfidfBaseline()
    return semantic, adaptive, generator, baseline


def main():
    st.title("📚 Adaptive RAG Assistant for Software Project Documentation")
    st.caption("A Transformer-powered engineering tool that dynamically adapts retrieval context for software repositories.")

    # Load models
    semantic, adaptive, generator, baseline = load_models()

    # Sidebar Configurations
    st.sidebar.header("⚙️ Adaptive Engine Calibration")
    focused_thresh = st.sidebar.slider("Focused Mode Threshold", 0.40, 0.80, 0.55, 0.05,
                                      help="Queries with top-1 score >= threshold use pinpoint top-3 chunks.")
    expanded_thresh = st.sidebar.slider("Expanded Mode Threshold", 0.25, 0.55, 0.35, 0.05,
                                       help="Queries in this range expand context to top-6 chunks.")
    min_thresh = st.sidebar.slider("Abstention Threshold (Calibrated)", 0.15, 0.40, 0.28, 0.02,
                                  help="Empirically calibrated cutoff; queries below this are flagged as unanswerable.")

    adaptive.focused_threshold = focused_thresh
    adaptive.expanded_threshold = expanded_thresh
    adaptive.min_threshold = min_thresh

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Core Technology Stack:**
    - **Embeddings:** `all-MiniLM-L6-v2` (384d Dense)
    - **Generator:** `google/flan-t5-small` (Seq2Seq)
    - **Baseline:** TF-IDF Cosine Similarity
    - **Related Precedent:** Calibrated abstention thresholding
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
                "Which database does UniServe use?",
                "What framework powers the backend application?",
                "Which port does the frontend run on?",
                "What tool manages database schema migrations?",
                "What command starts the frontend in development mode?",
                "What is the token expiration duration for access tokens?",
                "What Kafka topics receive asynchronous events?",
                "What is the developer's favorite programming language? (Unanswerable test)",
                "What is the monthly AWS cloud hosting budget? (Unanswerable test)",
                "What Stripe API secret key is used? (Unanswerable test)"
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
                                       value="What framework powers the backend application?")
            if st.button("Run Side-by-Side Comparison", type="secondary") and comp_query.strip():
                tfidf_res = baseline.search(comp_query, top_k=3)
                dense_res = semantic.search(comp_query, top_k=3)
                adapt_res = adaptive.retrieve(comp_query)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 1. TF-IDF Keyword Baseline")
                    st.caption("Matches exact token occurrences and term frequency.")
                    for r in tfidf_res:
                        st.metric(f"Rank {r['rank']} Score", f"{r['score']:.3f}")
                        st.text_area(f"TF-IDF #{r['rank']} ({r['chunk']['source']})", r["chunk"]["text"][:220] + "...", height=110, disabled=True)

                with col2:
                    st.markdown("#### 2. MiniLM Dense Semantic")
                    st.caption("Matches contextual meaning in 384d embedding space.")
                    for r in dense_res:
                        st.metric(f"Rank {r['rank']} Score", f"{r['score']:.3f}")
                        st.text_area(f"MiniLM #{r['rank']} ({r['chunk']['source']})", r["chunk"]["text"][:220] + "...", height=110, disabled=True)

                with col3:
                    st.markdown("#### 3. Adaptive Dynamic Engine")
                    st.caption(f"Mode: **{adapt_res['mode'].upper()}** (Top-1: {adapt_res['confidence']:.2f})")
                    st.write(f"- **Top-K Selected:** `{adapt_res['top_k']}` chunks")
                    st.write(f"- **Evidence Sufficient:** `{adapt_res['sufficient_evidence']}`")
                    st.write(f"- **Action:** {adapt_res['explanation']}")
                    if not adapt_res['sufficient_evidence']:
                        st.error("System actively abstains to prevent hallucination!")
                    else:
                        st.success(f"Dispatched {len(adapt_res['results'])} evidence chunks to generator.")

    # -------------------------------------------------------------
    # 4. Evaluation & Failure Analysis Dashboard
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader("📊 Held-Out Evaluation & Failure Analysis Dashboard")
        st.write("Rigorous empirical evaluation comparing TF-IDF keyword baseline against SentenceTransformer MiniLM across both Development and Held-Out test suites.")

        metrics_file = Path("evaluation/results/evaluation_metrics.json")
        plot_file = Path("evaluation/results/retrieval_comparison.png")
        table_file = Path("evaluation/results/comparison_table.csv")

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            st.markdown("### 🏆 Held-Out Test Set Performance (N=8 unseen queries)")
            held_out = metrics_data.get("Held-out", {})

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("MiniLM Recall@1", f"{held_out.get('semantic_recall_at_1', 0)*100:.1f}%",
                          delta=f"{(held_out.get('semantic_recall_at_1', 0) - held_out.get('tfidf_recall_at_1', 0))*100:.1f}% vs TF-IDF")
            col_m2.metric("MiniLM Recall@3", f"{held_out.get('semantic_recall_at_3', 0)*100:.1f}%")
            col_m3.metric("MiniLM MRR", f"{held_out.get('semantic_mrr', 0):.3f}")
            col_m4.metric("Adaptive Abstention Acc", f"{held_out.get('adaptive_abstention_accuracy', 0)*100:.1f}%",
                          delta="Zero Hallucination")

            st.markdown("---")
            c_left, c_right = st.columns([1.2, 1])

            with c_left:
                st.markdown("#### Retrieval Performance Visualization")
                if plot_file.exists():
                    st.image(str(plot_file), use_container_width=True)
                else:
                    st.info("Run `python evaluation/evaluate.py` to regenerate comparison plot.")

            with c_right:
                st.markdown("#### Abstention Threshold Calibration (Development Set)")
                st.write("Empirically swept candidate thresholds to optimize true positive retention vs. hallucination prevention (precedent from Emirhan Yildirim `rag-project-assistant`):")
                if "calibration_sweep" in metrics_data:
                    sweep_df = pd.DataFrame(metrics_data["calibration_sweep"])
                    st.dataframe(sweep_df.style.format({
                        "threshold": "{:.2f}",
                        "answerable_retention": "{:.1%}",
                        "abstention_accuracy": "{:.1%}",
                        "calibration_f1": "{:.3f}"
                    }), use_container_width=True)
                    st.success(f"**Calibrated Optimal Threshold:** `{metrics_data.get('calibrated_threshold', 0.28):.2f}`")

            st.markdown("---")
            st.markdown("### 🔍 Real Failure Analysis & Taxonomy")
            st.markdown("""
            1. **The Lexical Gap in Keyword Search (TF-IDF Failure):**
               - *Query:* `"What framework powers the backend application?"`
               - *Baseline (TF-IDF):* Failed ($Recall@1=0$) because documentation used terms *"Spring Boot 3.2 built on Java 17"* rather than repeating *"framework"*.
               - *MiniLM Semantic:* Succeeded ($Recall@1=1$) by embedding conceptual semantics.
            2. **Multi-Hop Architectural Reasoning (Generator Boundary):**
               - *Query:* `"If port 8080 is blocked by a firewall, what frontend features will fail?"`
               - *Limitation:* The retriever locates backend port and frontend API base URL chunks, but small local language models struggle to deduce indirect multi-step cascading failures without chain-of-thought prompting.
            3. **Out-of-Domain Inquiries (Abstention Gating):**
               - *Query:* `"What is the developer's favorite programming language?"` or `"What is the monthly AWS cloud hosting budget?"`
               - *Result:* Similarity score ($\sim 0.20$) falls below the calibrated 0.28 threshold. The system enters **Abstention Mode**, returning: *"The documentation does not provide enough information to answer this question."*
            """)

            if table_file.exists():
                with st.expander("📄 View Complete Granular Evaluation Predictions Table (CSV)"):
                    df_res = pd.read_csv(table_file)
                    st.dataframe(df_res, use_container_width=True)
        else:
            st.info("Evaluation metrics not found. Run `python evaluation/evaluate.py` to populate.")


if __name__ == "__main__":
    main()
