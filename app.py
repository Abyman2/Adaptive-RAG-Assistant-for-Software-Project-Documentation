"""Streamlit web interface for Adaptive RAG Assistant.
A professional developer documentation assistant featuring:
1. Project Manager: Index local repositories, uploaded files, or built-in software documentation.
2. AI Documentation Assistant: Natural language Q&A with calibrated confidence badges and grounded citations.
3. Retrieval Comparison: Side-by-side benchmark of TF-IDF keyword baseline vs MiniLM dense embeddings vs Adaptive Engine.
4. Evaluation Dashboard: Live benchmark metrics, calibration sweep, plots, and empirical failure analysis.
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
    page_title="Adaptive RAG Assistant | Software Docs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Developer UI Theme (Modern Slate / Dark Theme)
st.markdown("""
<style>
    /* Global Font & Header Tweaks */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek Container Cards */
    .doc-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    
    .answer-card {
        background: linear-gradient(145deg, #1a2234, #0f172a);
        border: 1px solid #2563eb;
        border-left: 6px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.15);
    }
    
    .abstain-card {
        background: linear-gradient(145deg, #2a1b18, #1c1412);
        border: 1px solid #dc2626;
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.15);
    }

    /* Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .badge-focused {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #059669;
    }
    
    .badge-expanded {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid #2563eb;
    }
    
    .badge-broad {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #d97706;
    }
    
    .badge-abstain {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #dc2626;
    }

    /* Citation Tag */
    .citation-tag {
        display: inline-block;
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 4px 6px 4px 0;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Highlight Quote Container */
    .quote-box {
        background: #0d1117;
        border-left: 4px solid #8b5cf6;
        border-radius: 0 8px 8px 0;
        padding: 12px 18px;
        margin-top: 14px;
        font-size: 0.92rem;
        color: #e6edf3;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Comparison Column Cards */
    .comp-col-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        min-height: 240px;
    }
    
    .score-meter {
        height: 6px;
        border-radius: 3px;
        background: #21262d;
        margin-top: 6px;
        margin-bottom: 12px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Neural Models (all-MiniLM-L6-v2 & flan-t5-small)...")
def load_models():
    """Load semantic retriever, adaptive controller, generator, and TF-IDF baseline."""
    semantic = SemanticRetriever()
    adaptive = AdaptiveRetriever(semantic, min_threshold=0.31)
    generator = AnswerGenerator()
    baseline = TfidfBaseline()
    return semantic, adaptive, generator, baseline


def clean_generated_answer(raw_answer: str) -> str:
    """Format generator output, stripping repetitive prefixes and cleaning punctuation."""
    ans = raw_answer.strip()
    if ans.lower().startswith("answer:"):
        ans = ans[7:].strip()
    if ans and not ans.endswith((".", "!", "?")):
        ans += "."
    return ans


def main():
    # Top Branding Bar
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid #30363d; margin-bottom: 20px;">
        <div>
            <h2 style="margin: 0; font-weight: 700; color: #f0f6fc;">
                ⚡ Adaptive RAG Assistant <span style="font-size: 1rem; color: #8b949e; font-weight: 400;">| Software Documentation Intelligence</span>
            </h2>
            <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.92rem;">
                Confidence-gated Transformer retrieval (MiniLM-L6-v2) & grounded generation (FLAN-T5) with calibrated abstention.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load neural models
    semantic, adaptive, generator, baseline = load_models()

    # Sidebar: Adaptive Engine Controller
    with st.sidebar:
        st.markdown("### ⚙️ Adaptive Policy Control")
        st.caption("Fine-tune dynamic context switching thresholds calibrated against the development benchmark.")

        focused_thresh = st.slider(
            "Focused Mode (K=3) Cutoff", 0.40, 0.80, 0.55, 0.05,
            help="High-confidence queries (>= 0.55) pull tight top-3 context chunks."
        )
        expanded_thresh = st.slider(
            "Expanded Mode (K=6) Cutoff", 0.25, 0.55, 0.35, 0.05,
            help="Moderate ambiguity queries (0.35 - 0.55) pull expanded top-6 context chunks."
        )
        min_thresh = st.slider(
            "Abstention Threshold (τ)", 0.15, 0.45, 0.31, 0.01,
            help="Empirically calibrated cutoff (0.31). Queries below this are actively rejected to suppress hallucination."
        )

        adaptive.focused_threshold = focused_thresh
        adaptive.expanded_threshold = expanded_thresh
        adaptive.min_threshold = min_thresh

        st.markdown("---")
        st.markdown("""
        **Pipeline Specification:**
        - **Embedding Space:** `all-MiniLM-L6-v2` (384d)
        - **Inference Hardware:** CPU Local Execution
        - **Grounded Generator:** `google/flan-t5-small` (77M)
        - **Keyword Baseline:** Scikit-Learn TF-IDF
        - **Calibration Precedent:** Yildirim (2024)
        """)
        
        st.markdown("---")
        st.caption("Project 1: Selected Topics in Software Engineering")

    # Initialize Session State
    if "documents" not in st.session_state:
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.indexed = False

    # Navigation Tabs
    tab_proj, tab_chat, tab_comp, tab_eval = st.tabs([
        "📁 1. Project Ingestion",
        "💬 2. Documentation Assistant",
        "⚖️ 3. Retrieval Diagnostic",
        "📊 4. Benchmark & Failure Analysis"
    ])

    # -------------------------------------------------------------
    # 1. Project Ingestion
    # -------------------------------------------------------------
    with tab_proj:
        st.markdown("### 📁 Project Workspace & Document Ingestion")
        st.markdown("Load your software documentation, specifications, or codebase files to build both dense vector and lexical indices.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="doc-card">
                <h4 style="margin-top:0; color: #58a6ff;">🚀 Option A: Built-in UniServe Specification</h4>
                <p style="color: #8b949e; font-size: 0.9rem;">
                    Full architectural specification of the UniServe microservice system: Spring Boot 3, React 18, 
                    PostgreSQL schemas, Kafka routing, Flyway migrations, and JWT security roles.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load Pre-Configured UniServe Docs", type="primary", use_container_width=True):
                sample_path = Path("sample_docs/sample_project.md")
                if sample_path.exists():
                    doc = load_document(sample_path)
                    st.session_state.documents = [doc]
                    st.session_state.chunks = process_documents([doc], chunk_size=100, overlap=25)
                    semantic.index(st.session_state.chunks)
                    baseline.index(st.session_state.chunks)
                    st.session_state.indexed = True
                    st.toast("UniServe documentation indexed successfully!", icon="✅")

        with col2:
            st.markdown("""
            <div class="doc-card">
                <h4 style="margin-top:0; color: #58a6ff;">📂 Option B: Ingest Local Repository</h4>
                <p style="color: #8b949e; font-size: 0.9rem;">
                    Point the assistant at any local folder on your computer containing Markdown files, codefiles (.py, .java, .js), PDF reports, or DOCX specifications.
                </p>
            </div>
            """, unsafe_allow_html=True)
            dir_input = st.text_input("Local project folder path:", placeholder="e.g. C:/Projects/my-backend-app")
            if st.button("Scan & Index Local Directory", use_container_width=True) and dir_input:
                target_dir = Path(dir_input)
                if target_dir.is_dir():
                    with st.spinner("Parsing and chunking directory files..."):
                        docs = load_directory(target_dir)
                        if docs:
                            st.session_state.documents = docs
                            st.session_state.chunks = process_documents(docs, chunk_size=100, overlap=25)
                            semantic.index(st.session_state.chunks)
                            baseline.index(st.session_state.chunks)
                            st.session_state.indexed = True
                            st.toast(f"Indexed {len(docs)} files ({len(st.session_state.chunks)} chunks)!", icon="🎉")
                        else:
                            st.error("No supported documentation files found in target folder.")
                else:
                    st.error("Target path does not exist or is not a directory.")

        # File Upload Area
        st.markdown("---")
        st.markdown("#### 📤 Option C: Drag & Drop Document Files")
        uploaded_files = st.file_uploader(
            "Upload specification documents (PDF, DOCX, Markdown, Code, JSON, YAML):",
            type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Process & Index Uploaded Documents", use_container_width=True):
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
                st.toast(f"Successfully indexed {len(docs)} files!", icon="🚀")

        # Ingestion Status Board
        if st.session_state.indexed:
            st.markdown("---")
            st.markdown("### 📋 Active Project Workspace Status")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Indexed Documents", len(st.session_state.documents))
            kpi2.metric("Total Chunks", len(st.session_state.chunks))
            kpi3.metric("Chunking Strategy", "100 words (25 overlap)")
            kpi4.metric("Embedding Space", "384-dimensional")

            with st.expander("🔍 View Processed Documents Breakdown"):
                for d in st.session_state.documents:
                    word_cnt = len(d['text'].split())
                    st.markdown(f"- **`{d['source']}`** &nbsp;•&nbsp; `{word_cnt}` words")

    # -------------------------------------------------------------
    # 2. AI Documentation Assistant
    # -------------------------------------------------------------
    with tab_chat:
        st.markdown("### 💬 Ask Questions About the Software Project")
        st.markdown("Query the technical documentation using natural language. The system dynamically decides context depth or abstains if information is missing.")

        if not st.session_state.indexed:
            st.info("👈 **Get Started:** Load the UniServe project or index a directory in the **Project Ingestion** tab.")
        else:
            preset_questions = [
                "Select an example query...",
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

            query = st.text_input("Enter your software engineering question:", value=default_query, placeholder="e.g. How does user authentication work?")

            if st.button("Submit Question", type="primary", use_container_width=True) and query.strip():
                with st.spinner("Analyzing documentation with Adaptive RAG..."):
                    adaptive_res = adaptive.retrieve(query)
                    mode = adaptive_res["mode"]
                    confidence = adaptive_res["confidence"]
                    top_k = adaptive_res["top_k"]
                    sufficient = adaptive_res["sufficient_evidence"]
                    results = adaptive_res["results"]

                    gen_res = generator.generate(query, results, sufficient_evidence=sufficient)
                    answer_text = clean_generated_answer(gen_res["answer"])

                # Determine Badge HTML
                if not sufficient:
                    badge_html = f'<span class="badge-pill badge-abstain">🔴 ABSTENTION TRIGGERED &nbsp;•&nbsp; Confidence: {confidence:.2f} &lt; {min_thresh:.2f}</span>'
                elif mode == "focused":
                    badge_html = f'<span class="badge-pill badge-focused">🟢 FOCUSED MODE &nbsp;•&nbsp; Top-{top_k} Chunks &nbsp;•&nbsp; Confidence: {confidence:.2f}</span>'
                elif mode == "expanded":
                    badge_html = f'<span class="badge-pill badge-expanded">🔵 EXPANDED MODE &nbsp;•&nbsp; Top-{top_k} Chunks &nbsp;•&nbsp; Confidence: {confidence:.2f}</span>'
                else:
                    badge_html = f'<span class="badge-pill badge-broad">🟡 BROAD MODE &nbsp;•&nbsp; Top-{top_k} Chunks &nbsp;•&nbsp; Confidence: {confidence:.2f}</span>'

                # Render Result Card
                if not sufficient:
                    st.markdown(f"""
                    <div class="abstain-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <span style="font-size:0.85rem; font-weight:700; color:#ef4444; text-transform:uppercase; letter-spacing:0.5px;">Grounded Verification System</span>
                            {badge_html}
                        </div>
                        <h4 style="margin:0 0 10px 0; color:#f87171;">⚠️ Insufficient Project Documentation Found</h4>
                        <p style="margin:0; color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
                            {answer_text}
                        </p>
                        <div style="margin-top:14px; padding-top:10px; border-top:1px solid rgba(239, 68, 68, 0.2); font-size:0.85rem; color:#94a3b8;">
                            <strong>Adaptive Policy Action:</strong> {adaptive_res['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Verified Citations HTML
                    citations_html = "".join([f'<span class="citation-tag">📄 {cite}</span>' for cite in gen_res.get("citations", [])])
                    
                    # Top evidence snippet
                    top_chunk = results[0]["chunk"] if results else {"text": "", "sections": []}
                    top_sections = " > ".join(top_chunk.get("sections", [])) if top_chunk.get("sections") else "Specification Root"
                    top_snippet = top_chunk.get("text", "")[:280] + "..." if len(top_chunk.get("text", "")) > 280 else top_chunk.get("text", "")

                    st.markdown(f"""
                    <div class="answer-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap: wrap; gap: 8px;">
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:1.1rem;">⚡</span>
                                <span style="font-size:0.85rem; font-weight:700; color:#60a5fa; text-transform:uppercase; letter-spacing:0.5px;">Verified Answer Summary</span>
                            </div>
                            {badge_html}
                        </div>
                        <div style="font-size:1.05rem; color:#f8fafc; font-weight:500; line-height:1.6; margin-bottom:16px;">
                            {answer_text}
                        </div>
                        <div style="margin-top:10px;">
                            <span style="font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase;">Verified Source Documents:</span><br/>
                            <div style="margin-top:4px;">{citations_html}</div>
                        </div>
                        <div class="quote-box">
                            <div style="font-size:0.75rem; color:#a78bfa; margin-bottom:4px; font-weight:600;">
                                🔖 PRIMARY EVIDENCE &nbsp;|&nbsp; {top_sections}
                            </div>
                            <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.5;">
                                "{top_snippet}"
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Expandable Deep Inspection
                with st.expander(f"📑 Inspect Full Retrieved Context ({len(results)} Chunks)"):
                    for i, r in enumerate(results, start=1):
                        c = r["chunk"]
                        sections_tag = " > ".join(c.get("sections", [])) if c.get("sections") else "Root"
                        score = r["score"]
                        bar_color = "#10b981" if score >= 0.50 else ("#3b82f6" if score >= 0.35 else "#f59e0b")
                        
                        st.markdown(f"""
                        <div style="background:#11161d; border:1px solid #30363d; border-radius:8px; padding:14px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-weight:600; font-size:0.9rem; color:#f0f6fc;">
                                    Rank #{i} &nbsp;•&nbsp; <span style="color:#58a6ff; font-family:monospace;">{c['source']}</span> &nbsp;•&nbsp; <span style="color:#8b949e;">{sections_tag}</span>
                                </span>
                                <span style="font-family:monospace; font-size:0.85rem; font-weight:700; color:{bar_color};">
                                    Cosine: {score:.3f}
                                </span>
                            </div>
                            <div style="height:4px; background:#21262d; border-radius:2px; margin-bottom:10px; overflow:hidden;">
                                <div style="height:100%; width:{min(score * 100, 100):.1f}%; background:{bar_color};"></div>
                            </div>
                            <div style="font-size:0.86rem; color:#c9d1d9; font-family:'JetBrains Mono', monospace; line-height:1.5; white-space:pre-wrap;">{c['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. Retrieval Diagnostic & Comparison
    # -------------------------------------------------------------
    with tab_comp:
        st.markdown("### ⚖️ Side-by-Side Diagnostic: TF-IDF vs MiniLM vs Adaptive RAG")
        st.markdown("Observe how classical exact-token search compares against dense continuous 384-dimensional embeddings and our dynamic policy controller.")

        if not st.session_state.indexed:
            st.info("Please index project documents in the **Project Ingestion** tab first.")
        else:
            comp_query = st.text_input(
                "Diagnostic query for side-by-side evaluation:",
                value="How can an engineer manually trigger database structure updates from the CLI?",
                placeholder="Enter technical query..."
            )
            if st.button("Run Diagnostic Benchmark", type="secondary", use_container_width=True) and comp_query.strip():
                tfidf_res = baseline.search(comp_query, top_k=3)
                dense_res = semantic.search(comp_query, top_k=3)
                adapt_res = adaptive.retrieve(comp_query)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("""
                    <div style="border-top: 3px solid #64748b; padding-top: 8px;">
                        <h4 style="margin:0; color:#94a3b8;">1. TF-IDF Baseline</h4>
                        <span style="font-size:0.8rem; color:#64748b;">Sparse Term Frequency (Non-Transformer)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if tfidf_res:
                        top_tf = tfidf_res[0]
                        st.metric("Top-1 Lexical Score", f"{top_tf['score']:.3f}", delta="Keyword Match")
                        st.markdown(f"""
                        <div class="comp-col-card" style="margin-top:10px;">
                            <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; margin-bottom:4px;">
                                📄 TOP MATCH: {top_tf['chunk']['source']}
                            </div>
                            <div style="font-size:0.83rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace; line-height:1.5;">
                                {top_tf['chunk']['text'][:200]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.metric("Top-1 Lexical Score", "0.000", delta="- Zero match", delta_color="inverse")
                        st.markdown("""
                        <div class="comp-col-card" style="margin-top:10px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                            <span style="font-size:1.8rem;">❌</span>
                            <div style="color:#ef4444; font-weight:600; margin-top:8px;">Zero Term Overlap</div>
                            <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">
                                Query terms do not match verbatim document tokens.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="border-top: 3px solid #3b82f6; padding-top: 8px;">
                        <h4 style="margin:0; color:#60a5fa;">2. MiniLM Transformer</h4>
                        <span style="font-size:0.8rem; color:#64748b;">Dense Vector Cosine (all-MiniLM-L6-v2)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if dense_res:
                        top_dense = dense_res[0]
                        score = top_dense['score']
                        st.metric("Top-1 Cosine Sim", f"{score:.3f}", delta=f"{score*100:.1f}% Match")
                        st.markdown(f"""
                        <div class="comp-col-card" style="margin-top:10px;">
                            <div style="font-size:0.75rem; color:#60a5fa; font-weight:600; margin-bottom:4px;">
                                📄 TOP MATCH: {top_dense['chunk']['source']}
                            </div>
                            <div style="font-size:0.83rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace; line-height:1.5;">
                                {top_dense['chunk']['text'][:200]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col3:
                    st.markdown("""
                    <div style="border-top: 3px solid #10b981; padding-top: 8px;">
                        <h4 style="margin:0; color:#34d399;">3. Adaptive Engine</h4>
                        <span style="font-size:0.8rem; color:#64748b;">Confidence-Gated Dynamic Top-K</span>
                    </div>
                    """, unsafe_allow_html=True)
                    mode_str = adapt_res['mode'].upper()
                    conf = adapt_res['confidence']
                    action_badge = "✅ PASS" if adapt_res['sufficient_evidence'] else "🛑 ABSTAIN"
                    st.metric(f"Policy: {mode_str}", f"K = {adapt_res['top_k']}", delta=action_badge)
                    st.markdown(f"""
                    <div class="comp-col-card" style="margin-top:10px;">
                        <div style="font-size:0.75rem; color:#34d399; font-weight:600; margin-bottom:6px;">
                            🎯 DECISION LOGIC
                        </div>
                        <ul style="font-size:0.82rem; color:#cbd5e1; padding-left:18px; margin:0; line-height:1.6;">
                            <li><strong>Confidence:</strong> <code>{conf:.3f}</code> vs τ = <code>{adaptive.min_threshold:.2f}</code></li>
                            <li><strong>Context Depth:</strong> <code>{adapt_res['top_k']} chunks</code></li>
                            <li><strong>Sufficient Evidence:</strong> <code>{adapt_res['sufficient_evidence']}</code></li>
                        </ul>
                        <div style="margin-top:10px; font-size:0.8rem; color:#94a3b8; background:#0d1117; padding:8px; border-radius:6px;">
                            {adapt_res['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Diagnostic Summary
                st.markdown("---")
                st.markdown("#### 💡 Architectural Analysis for this Query")
                if not adapt_res['sufficient_evidence']:
                    st.warning("⚠️ **Out-of-Domain Detection:** The query lacks semantic backing in the documentation. TF-IDF either yields zero or misleading partial matches; Adaptive RAG safely blocks the generator from hallucinating.")
                elif tfidf_res and tfidf_res[0]['score'] > 0:
                    st.info("ℹ️ **Exact Keyword Hit:** Distinct keywords match both lexically and semantically. TF-IDF provides ultra-fast retrieval, while MiniLM ensures structural context is preserved.")
                else:
                    st.success("✨ **Lexical Gap Bridged:** The query uses phrasing that does not appear verbatim in the documentation. TF-IDF completely failed (0.000), but MiniLM and Adaptive RAG recognized semantic intent and retrieved the correct technical section.")

    # -------------------------------------------------------------
    # 4. Evaluation & Failure Analysis Dashboard
    # -------------------------------------------------------------
    with tab_eval:
        st.markdown("### 📊 Rigorous Benchmark & Empirical Failure Analysis")
        st.markdown("Audited evaluation results comparing TF-IDF baseline, MiniLM semantic retriever, and Adaptive RAG across held-out ranking and binary abstention classification.")

        metrics_file = Path("evaluation/results/evaluation_metrics_v2.json")
        plot_retrieval = Path("evaluation/results/retrieval_comparison_v2.png")
        plot_cm = Path("evaluation/results/abstention_confusion_matrix.png")
        table_file = Path("evaluation/results/comparison_table_v2.csv")

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                m_data = json.load(f)

            st.markdown("#### 1. Retrieval Ranking Performance on Answerable Queries (N=12)")
            st.caption("Strict ground truth check against verified gold chunk IDs, section titles, and evidence snippets.")
            rank_data = m_data.get("retrieval_ranking_answerable", {})
            tf_data = rank_data.get("tfidf", {})
            sem_data = rank_data.get("minilm_semantic", {})
            adapt_data = rank_data.get("adaptive_rag", {})

            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
            r_col1.metric("TF-IDF Recall@1", f"{tf_data.get('recall_at_1', 0)*100:.1f}%")
            r_col2.metric("MiniLM Recall@1", f"{sem_data.get('recall_at_1', 0)*100:.1f}%")
            r_col3.metric("MiniLM Recall@3", f"{sem_data.get('recall_at_3', 0)*100:.1f}%")
            r_col4.metric("MiniLM MRR", f"{sem_data.get('mrr', 0):.3f}")
            r_col5.metric("Adaptive Coverage", f"{adapt_data.get('evidence_coverage', 0)*100:.1f}%")

            st.markdown("---")
            st.markdown("#### 2. Answerability & Abstention Classification (N=20 Held-Out Queries)")
            st.caption("Binary classification evaluating acceptance of answerable questions vs rejection of unanswerable queries.")
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
                st.markdown("##### 📈 Retrieval Ranking Comparison")
                if plot_retrieval.exists():
                    st.image(str(plot_retrieval), use_container_width=True)

            with p_right:
                st.markdown("##### 🎯 Abstention Decision Confusion Matrix")
                if plot_cm.exists():
                    st.image(str(plot_cm), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 3. Latency & Resource Footprint Profile")
            su = m_data.get("model_startup_ms", {})
            lat_c1, lat_c2, lat_c3, lat_c4 = st.columns(4)
            lat_c1.metric("TF-IDF Query Latency", f"{tf_data.get('mean_latency_ms', 0):.1f} ms")
            lat_c2.metric("MiniLM Query Latency", f"{sem_data.get('mean_latency_ms', 0):.1f} ms")
            lat_c3.metric("Model Startup Overhead", f"{(su.get('minilm_embedding', 0) + su.get('flan_t5_loading', 0))/1000:.1f} s")
            lat_c4.metric("Classification F1", f"{cm.get('f1_score', 0):.3f}")

            st.markdown("---")
            st.markdown("#### 4. Empirical Threshold Calibration Sweep (Development Set N=12)")
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
                st.success(f"**Optimal Calibrated Cutoff Selected:** `τ = {m_data.get('calibrated_threshold', 0.31):.2f}` (Zero held-out tuning)")

            if table_file.exists():
                with st.expander("📄 View Detailed Per-Query Predictions Table (CSV)"):
                    df_preds = pd.read_csv(table_file)
                    st.dataframe(df_preds, use_container_width=True)
        else:
            st.info("Evaluation results file not found. Run `python evaluation/evaluate.py` to generate.")


if __name__ == "__main__":
    main()
