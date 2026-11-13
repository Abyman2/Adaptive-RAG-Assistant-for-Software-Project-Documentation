"""Streamlit web interface for Adaptive RAG Assistant.
Provides document ingestion, query answering, confidence display,
adaptive mode explanation, and source citations.
"""

import os
import tempfile
from pathlib import Path
import streamlit as st

from src.document_loader import load_document, SUPPORTED_EXTENSIONS
from src.chunker import process_documents
from src.retriever import SemanticRetriever
from src.adaptive_retrieval import AdaptiveRetriever
from src.generator import AnswerGenerator
from src.baseline import TfidfBaseline


st.set_page_config(
    page_title="Adaptive RAG Assistant",
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
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-expanded {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-broad {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-insufficient {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Transformer Models (all-MiniLM-L6-v2 & flan-t5-small)...")
def load_models():
    """Load semantic retriever and generator models once and cache them."""
    semantic = SemanticRetriever()
    adaptive = AdaptiveRetriever(semantic)
    generator = AnswerGenerator()
    baseline = TfidfBaseline()
    return semantic, adaptive, generator, baseline


def main():
    st.title("📚 Adaptive RAG Assistant for Software Documentation")
    st.caption("A Transformer-powered engineering tool that dynamically adapts retrieval context for software repositories.")

    # Sidebar
    st.sidebar.header("📁 Document Ingestion")
    
    # Quick loader for default sample project
    sample_doc_path = Path("sample_docs/sample_project.md")
    use_sample = st.sidebar.button("📄 Load UniServe Sample Project Docs", use_container_width=True)

    uploaded_files = st.sidebar.file_uploader(
        "Or upload project files (PDF, DOCX, MD, Code):",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Adaptive Engine Thresholds")
    focused_thresh = st.sidebar.slider("Focused Mode Threshold", 0.40, 0.80, 0.55, 0.05)
    expanded_thresh = st.sidebar.slider("Expanded Mode Threshold", 0.25, 0.55, 0.35, 0.05)
    min_thresh = st.sidebar.slider("Abstention Threshold", 0.15, 0.35, 0.28, 0.02)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Architectural Components:**
    - **Embeddings:** `all-MiniLM-L6-v2` (384d)
    - **Generator:** `google/flan-t5-small` (Seq2Seq)
    - **Baseline:** TF-IDF Cosine Similarity
    - **Dynamic Strategy:** 3-tier Adaptive Top-K + Abstention
    """)

    # Initialize models
    semantic, adaptive, generator, baseline = load_models()
    adaptive.focused_threshold = focused_thresh
    adaptive.expanded_threshold = expanded_thresh
    adaptive.min_threshold = min_thresh

    # Document State
    if "documents" not in st.session_state:
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.indexed = False

    # Process loaded documents
    if use_sample and sample_doc_path.exists():
        doc = load_document(sample_doc_path)
        st.session_state.documents = [doc]
        st.session_state.chunks = process_documents([doc], chunk_size=100, overlap=25)
        semantic.index(st.session_state.chunks)
        baseline.index(st.session_state.chunks)
        st.session_state.indexed = True
        st.sidebar.success(f"Loaded '{doc['source']}' ({len(st.session_state.chunks)} chunks)")

    elif uploaded_files:
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
                st.sidebar.error(f"Error loading {uf.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        if docs:
            st.session_state.documents = docs
            st.session_state.chunks = process_documents(docs, chunk_size=100, overlap=25)
            semantic.index(st.session_state.chunks)
            baseline.index(st.session_state.chunks)
            st.session_state.indexed = True
            st.sidebar.success(f"Indexed {len(docs)} documents ({len(st.session_state.chunks)} chunks)")

    # Main Area
    if not st.session_state.indexed:
        st.info("👈 Please load the sample documentation or upload project files from the sidebar to begin.")
        st.markdown("""
        ### Why Adaptive RAG for Software Engineering?
        1. **Avoid Information Overload:** Simple queries don't need 10 chunks of context, which distract small language models.
        2. **Prevent Hallucinations:** When documentation doesn't contain the requested information, the system should **abstain** rather than hallucinate non-existent API routes or configs.
        3. **Bridge Lexical Gaps:** Traditional keyword search fails when developers ask questions with synonyms. Pretrained Transformer embeddings match concepts, not just words.
        """)
        return

    # Status overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed Documents", len(st.session_state.documents))
    col2.metric("Total Context Chunks", len(st.session_state.chunks))
    col3.metric("Adaptive Engine", "Active")

    # Developer Query Input
    st.markdown("### 💬 Ask a Technical Question")
    preset_questions = [
        "Select an example question...",
        "Which database does UniServe use?",
        "What framework powers the backend?",
        "What port does the frontend run on?",
        "What tool handles database schema migrations?",
        "What command starts the frontend in development mode?",
        "What is the developer's favorite programming language? (Unanswerable test)",
        "What Stripe API secret key is used? (Unanswerable test)"
    ]
    selected_preset = st.selectbox("Example Questions:", preset_questions)

    default_query = ""
    if selected_preset and selected_preset != preset_questions[0]:
        default_query = selected_preset.split(" (")[0]

    query = st.text_input("Enter your question about the project documentation:", value=default_query)

    if st.button("Submit Question", type="primary") and query.strip():
        with st.spinner("Analyzing query and evaluating documentation context..."):
            # 1. Run Adaptive Retrieval
            adaptive_res = adaptive.retrieve(query)
            mode = adaptive_res["mode"]
            confidence = adaptive_res["confidence"]
            top_k = adaptive_res["top_k"]
            sufficient = adaptive_res["sufficient_evidence"]
            results = adaptive_res["results"]

            # 2. Run Baseline for comparison
            tfidf_results = baseline.search(query, top_k=3)

            # 3. Generate Answer
            gen_res = generator.generate(query, results, sufficient_evidence=sufficient)

        # Display Results
        st.markdown("---")
        st.markdown("### 🎯 Adaptive Assessment & Generated Response")

        # Mode Badge
        if not sufficient:
            badge_html = f'<span class="badge-insufficient">⚠️ INSUFFICIENT EVIDENCE (Confidence: {confidence:.2f} < {min_thresh:.2f})</span>'
        elif mode == "focused":
            badge_html = f'<span class="badge-focused">🎯 FOCUSED MODE (Confidence: {confidence:.2f} >= {focused_thresh:.2f}) — Top {top_k} Chunks</span>'
        elif mode == "expanded":
            badge_html = f'<span class="badge-expanded">🔍 EXPANDED MODE (Confidence: {confidence:.2f} >= {expanded_thresh:.2f}) — Top {top_k} Chunks</span>'
        else:
            badge_html = f'<span class="badge-broad">🌐 BROAD SEARCH MODE (Confidence: {confidence:.2f}) — Top {top_k} Chunks</span>'

        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(f"Adaptive Decision: {adaptive_res['explanation']}")

        # Answer Box
        st.markdown(f"**Answer:**\n\n> {gen_res['answer']}")

        if gen_res["citations"]:
            st.markdown(f"**Source Citations:** `{', '.join(gen_res['citations'])}`")

        # Tabs for Evidence and Baseline Comparison
        tab1, tab2 = st.tabs(["📄 Retrieved Context Chunks", "⚖️ Baseline Comparison (TF-IDF vs Transformer)"])

        with tab1:
            st.markdown(f"**Retrieved {len(results)} Chunks via Semantic Transformer:**")
            for i, r in enumerate(results, start=1):
                chunk = r["chunk"]
                with st.expander(f"Chunk {i} | Source: {chunk['source']} | Score: {r['score']:.3f}"):
                    st.write(chunk["text"])

        with tab2:
            st.markdown("**Side-by-Side Retrieval Comparison:**")
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                st.markdown("**MiniLM Semantic Retriever (Proposed)**")
                for i, r in enumerate(results[:3], start=1):
                    st.info(f"**Rank {i} (Score: {r['score']:.3f})** - {r['chunk']['source']}\n\n{r['chunk']['text'][:200]}...")
            with comp_c2:
                st.markdown("**TF-IDF Keyword Baseline**")
                for i, r in enumerate(tfidf_results[:3], start=1):
                    st.warning(f"**Rank {i} (Score: {r['score']:.3f})** - {r['chunk']['source']}\n\n{r['chunk']['text'][:200]}...")


if __name__ == "__main__":
    main()
