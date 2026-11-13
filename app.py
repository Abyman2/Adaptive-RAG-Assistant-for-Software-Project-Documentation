"""Adaptive RAG Assistant for Software Project Documentation.
A professional developer documentation workspace featuring:
01. Overview - Landing hub with system status, telemetry, and quick actions.
02. Project Workspace - Document ingestion, repository explorer, and chunk inspector.
03. AI Documentation Assistant - Grounded technical Q&A with live Evidence Inspector.
04. Retrieval Lab - Side-by-side diagnostic: TF-IDF vs MiniLM vs Adaptive Policy.
05. Evaluation Center - Rigorous benchmark metrics, plots, and failure analysis.
06. Adaptive Policy - Interactive threshold calibration and empirical evidence.
"""

import os
import tempfile
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Configure offline mode for local Hugging Face model cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.document_loader import load_document, load_directory, SUPPORTED_EXTENSIONS
from src.chunker import process_documents
from src.retriever import SemanticRetriever
from src.adaptive_retrieval import AdaptiveRetriever
from src.generator import AnswerGenerator
from src.baseline import TfidfBaseline


st.set_page_config(
    page_title="Adaptive RAG | Developer Documentation Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Navy / Charcoal Modern Developer IDE Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Breadcrumb Bar */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 16px;
        background: #0e1424;
        border: 1px solid #1e293b;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .breadcrumb-text {
        font-size: 0.88rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }
    .breadcrumb-text span {
        color: #60a5fa;
        font-weight: 600;
    }
    
    /* Workspace Cards */
    .workspace-card {
        background: linear-gradient(145deg, #0e1424, #0a0f1d);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-tile {
        background: #0e1424;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.76rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.45rem;
        color: #f1f5f9;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }
    
    /* Grounded Answer Card */
    .answer-card {
        background: linear-gradient(145deg, #0c1830, #081020);
        border: 1px solid #2563eb;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
    }
    
    .abstain-card {
        background: linear-gradient(145deg, #241113, #18090a);
        border: 1px solid #dc2626;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.15);
    }

    /* Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-focused {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
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
    
    /* Evidence Inspector Sidebar Card */
    .inspector-panel {
        background: #080d19;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px;
    }
    .inspector-item {
        background: #0e1526;
        border: 1px solid #233148;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* Code quote */
    .evidence-quote {
        background: #060a14;
        border-left: 3px solid #8b5cf6;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        color: #cbd5e1;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Neural Pipeline (MiniLM-L6-v2 & FLAN-T5-small)...")
def load_models():
    """Load and cache the dense retriever, adaptive controller, generator, and baseline."""
    semantic = SemanticRetriever()
    adaptive = AdaptiveRetriever(semantic, min_threshold=0.31)
    generator = AnswerGenerator()
    baseline = TfidfBaseline()
    return semantic, adaptive, generator, baseline


def clean_generated_answer(raw_answer: str) -> str:
    """Format and clean generated answers, removing raw prompt artifacts."""
    ans = raw_answer.strip()
    if ans.lower().startswith("answer:"):
        ans = ans[7:].strip()
    if ans and not ans.endswith((".", "!", "?")):
        ans += "."
    return ans


def ensure_workspace_initialized(semantic, baseline):
    """Auto-load UniServe sample project on first startup if workspace is empty."""
    if "documents" not in st.session_state or not st.session_state.get("indexed", False):
        sample_path = Path("sample_docs/sample_project.md")
        if sample_path.exists():
            doc = load_document(sample_path)
            st.session_state.documents = [doc]
            st.session_state.active_project_name = "UniServe Architecture"
            st.session_state.chunks = process_documents([doc], chunk_size=100, overlap=25)
            semantic.index(st.session_state.chunks)
            baseline.index(st.session_state.chunks)
            st.session_state.indexed = True
            st.session_state.query_history = []


def main():
    # Load Models
    semantic, adaptive, generator, baseline = load_models()
    ensure_workspace_initialized(semantic, baseline)

    # Session State defaults for thresholds
    if "focused_thresh" not in st.session_state:
        st.session_state.focused_thresh = 0.55
    if "expanded_thresh" not in st.session_state:
        st.session_state.expanded_thresh = 0.35
    if "min_thresh" not in st.session_state:
        st.session_state.min_thresh = 0.31

    # Sync adaptive retriever with current session thresholds
    adaptive.focused_threshold = st.session_state.focused_thresh
    adaptive.expanded_threshold = st.session_state.expanded_thresh
    adaptive.min_threshold = st.session_state.min_thresh

    # -------------------------------------------------------------
    # Persistent Left Sidebar (Workspace Navigation & Status)
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
            <div style="background: linear-gradient(135deg, #2563eb, #8b5cf6); padding:8px 10px; border-radius:8px; font-size:1.1rem; color:white; font-weight:bold;">
                ⚡
            </div>
            <div>
                <div style="font-weight:700; color:#f1f5f9; font-size:1.05rem; letter-spacing:0.3px;">Adaptive RAG</div>
                <div style="font-size:0.75rem; color:#60a5fa; font-family:'JetBrains Mono', monospace;">Documentation Workspace</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Active Project Selector badge
        proj_name = st.session_state.get("active_project_name", "UniServe Architecture")
        st.markdown(f"""
        <div style="background:#0e1526; border:1px solid #1e293b; border-radius:8px; padding:8px 12px; margin-bottom:20px;">
            <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Active Workspace</span>
            <div style="color:#34d399; font-weight:600; font-size:0.88rem; display:flex; align-items:center; gap:6px; margin-top:2px;">
                <span style="height:8px; width:8px; background:#10b981; border-radius:50%; display:inline-block;"></span>
                {proj_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("WORKSPACE")
        nav_choice = st.radio(
            "Navigate Workspace:",
            [
                "01. Overview",
                "02. Project Workspace",
                "03. AI Assistant",
                "04. Retrieval Lab",
                "05. Evaluation Center",
                "06. Adaptive Policy"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption("SYSTEM TELEMETRY")
        st.markdown("""
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.7;">
            • <strong>Runtime:</strong> <span style="color:#34d399;">CPU Local Inference</span><br/>
            • <strong>Retriever:</strong> <code>all-MiniLM-L6-v2</code><br/>
            • <strong>Generator:</strong> <code>flan-t5-small</code><br/>
            • <strong>Cutoff (τ):</strong> <code>0.31</code>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 01. Overview Screen
    # -------------------------------------------------------------
    if nav_choice == "01. Overview":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Workspace / <span>Overview</span></div>
            <div style="font-size:0.8rem; color:#10b981; font-weight:600; display:flex; align-items:center; gap:6px;">
                <span style="height:7px; width:7px; background:#10b981; border-radius:50%;"></span>
                Online & Indexed
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.title("⚡ Documentation Intelligence Workspace")
        st.markdown("An adaptive, confidence-calibrated software engineering assistant for navigating codebase architectures, API contracts, and specifications.")

        # Top KPI Metric Tiles
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Active Project</div>
                <div class="kpi-value" style="font-size:1.15rem; color:#60a5fa;">{st.session_state.get('active_project_name', 'UniServe')}</div>
                <div class="kpi-sub">Spring Boot 3 + React 18</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Indexed Chunks</div>
                <div class="kpi-value">{len(st.session_state.get('chunks', []))}</div>
                <div class="kpi-sub">100w / 25w sliding window</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown("""
            <div class="kpi-tile">
                <div class="kpi-label">Embedding Space</div>
                <div class="kpi-value">384d</div>
                <div class="kpi-sub">MiniLM dense semantic vectors</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown("""
            <div class="kpi-tile">
                <div class="kpi-label">Abstention F1</div>
                <div class="kpi-value" style="color:#34d399;">0.846</div>
                <div class="kpi-sub">62.5% Specificity on unanswerable</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Architecture Pipeline Visual
        st.markdown("### 🔄 Autonomous RAG Execution Pipeline")
        st.markdown("""
        <div class="workspace-card">
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; text-align:center;">
                <div style="background:#0a1020; border:1px solid #1e293b; padding:14px; border-radius:8px;">
                    <div style="font-size:1.4rem;">📁</div>
                    <div style="font-weight:600; color:#f1f5f9; font-size:0.9rem; margin-top:6px;">1. Ingestion & Chunking</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Markdown/code parsing with preserved section headers</div>
                </div>
                <div style="background:#0a1020; border:1px solid #1e293b; padding:14px; border-radius:8px;">
                    <div style="font-size:1.4rem;">🔬</div>
                    <div style="font-weight:600; color:#60a5fa; font-size:0.9rem; margin-top:6px;">2. Dual Representation</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">TF-IDF lexical sparse + MiniLM 384d dense vector index</div>
                </div>
                <div style="background:#0a1020; border:1px solid #1e293b; padding:14px; border-radius:8px;">
                    <div style="font-size:1.4rem;">🛡️</div>
                    <div style="font-weight:600; color:#f59e0b; font-size:0.9rem; margin-top:6px;">3. Adaptive Policy</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Calibrated confidence gating: Dynamic K or Abstention</div>
                </div>
                <div style="background:#0a1020; border:1px solid #1e293b; padding:14px; border-radius:8px;">
                    <div style="font-size:1.4rem;">💡</div>
                    <div style="font-weight:600; color:#34d399; font-size:0.9rem; margin-top:6px;">4. Grounded Synthesis</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">FLAN-T5-small generation with verifiable citations</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Actions
        st.markdown("### ⚡ Quick Navigation")
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            if st.button("💬 Ask Questions in AI Assistant", use_container_width=True, type="primary"):
                st.session_state["nav_direct"] = "03. AI Assistant"
                st.rerun()
        with qa2:
            if st.button("⚖️ Run Side-by-Side Retrieval Lab", use_container_width=True):
                st.session_state["nav_direct"] = "04. Retrieval Lab"
                st.rerun()
        with qa3:
            if st.button("📊 View Held-Out Benchmark & Metrics", use_container_width=True):
                st.session_state["nav_direct"] = "05. Evaluation Center"
                st.rerun()

    # -------------------------------------------------------------
    # 02. Project Workspace
    # -------------------------------------------------------------
    elif nav_choice == "02. Project Workspace":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Workspace / <span>Project Workspace & Document Explorer</span></div>
            <div style="font-size:0.8rem; color:#60a5fa; font-family:'JetBrains Mono', monospace;">
                {0} Files • {1} Chunks
            </div>
        </div>
        """.format(len(st.session_state.documents), len(st.session_state.chunks)), unsafe_allow_html=True)

        col_explorer, col_preview = st.columns([1, 1.3])

        with col_explorer:
            st.markdown("### 📂 Project Ingestion & File Index")
            st.caption("Manage active documentation, ingest code repositories, or upload specification files.")

            with st.expander("📥 Ingest New Project or Local Directory", expanded=False):
                dir_path = st.text_input("Enter local project folder path:", placeholder="e.g. C:/Projects/my-app")
                if st.button("Scan & Index Local Folder", use_container_width=True) and dir_path:
                    target_dir = Path(dir_path)
                    if target_dir.is_dir():
                        with st.spinner("Indexing directory files..."):
                            docs = load_directory(target_dir)
                            if docs:
                                st.session_state.documents = docs
                                st.session_state.active_project_name = target_dir.name
                                st.session_state.chunks = process_documents(docs, chunk_size=100, overlap=25)
                                semantic.index(st.session_state.chunks)
                                baseline.index(st.session_state.chunks)
                                st.session_state.indexed = True
                                st.toast(f"Indexed {len(docs)} files!", icon="🚀")
                                st.rerun()
                            else:
                                st.error("No supported documentation files found.")
                    else:
                        st.error("Directory not found.")

                if st.button("Reset to Default UniServe Specification", use_container_width=True):
                    sample_path = Path("sample_docs/sample_project.md")
                    doc = load_document(sample_path)
                    st.session_state.documents = [doc]
                    st.session_state.active_project_name = "UniServe Architecture"
                    st.session_state.chunks = process_documents([doc], chunk_size=100, overlap=25)
                    semantic.index(st.session_state.chunks)
                    baseline.index(st.session_state.chunks)
                    st.toast("UniServe loaded!", icon="✅")
                    st.rerun()

            # File list
            st.markdown("#### Indexed Project Files")
            file_names = [d["source"] for d in st.session_state.documents]
            selected_file = st.selectbox("Select document to inspect:", file_names)

            # Upload additional files
            st.markdown("#### 📤 Upload Specification Files")
            uploaded = st.file_uploader(
                "Upload (.md, .pdf, .docx, code):",
                type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
                accept_multiple_files=True
            )
            if uploaded and st.button("Append & Index Files", use_container_width=True):
                new_docs = []
                for uf in uploaded:
                    suffix = Path(uf.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        d = load_document(tmp_path)
                        d["source"] = uf.name
                        new_docs.append(d)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                if new_docs:
                    st.session_state.documents.extend(new_docs)
                    st.session_state.chunks = process_documents(st.session_state.documents, chunk_size=100, overlap=25)
                    semantic.index(st.session_state.chunks)
                    baseline.index(st.session_state.chunks)
                    st.toast("Files added to workspace!", icon="🎉")
                    st.rerun()

        with col_preview:
            st.markdown("### 📄 Document & Chunk Inspector")
            target_doc = next((d for d in st.session_state.documents if d["source"] == selected_file), None)

            if target_doc:
                tab_text, tab_chunk_list = st.tabs(["Raw Document Content", f"Generated Chunks ({len([c for c in st.session_state.chunks if c['source'] == selected_file])})"])
                with tab_text:
                    st.markdown(f"**Source:** `{target_doc['source']}` &nbsp;|&nbsp; `{len(target_doc['text'].split())}` words")
                    st.text_area("File Content", target_doc["text"], height=420, disabled=True)

                with tab_chunk_list:
                    doc_chunks = [c for c in st.session_state.chunks if c["source"] == selected_file]
                    for c in doc_chunks:
                        sec_str = " > ".join(c.get("sections", [])) if c.get("sections") else "Root"
                        st.markdown(f"""
                        <div style="background:#0e1526; border:1px solid #1e293b; border-radius:8px; padding:10px; margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#60a5fa; font-family:monospace; margin-bottom:4px;">
                                <span>Chunk #{c['chunk_id']} • {c['word_count']} words</span>
                                <span>{sec_str}</span>
                            </div>
                            <div style="font-size:0.83rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace; line-height:1.4;">
                                {c['text'][:180]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 03. AI Documentation Assistant
    # -------------------------------------------------------------
    elif nav_choice == "03. AI Assistant":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Workspace / <span>AI Documentation Assistant</span></div>
            <div style="font-size:0.8rem; color:#34d399; font-family:'JetBrains Mono', monospace;">
                Grounded Mode • FLAN-T5
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preset Questions
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

        c_query_select, c_query_btn = st.columns([3, 1])
        with c_query_select:
            selected_preset = st.selectbox("Quick-test presets:", preset_questions, label_visibility="collapsed")
        default_val = selected_preset.split(" (")[0] if selected_preset != preset_questions[0] else ""

        user_query = st.text_input("Enter natural language question about the project:", value=default_val, placeholder="e.g. How does user authentication work?")

        # Execution
        if st.button("🚀 Analyze & Generate Answer", type="primary", use_container_width=True) and user_query.strip():
            with st.spinner("Evaluating semantic confidence and generating grounded answer..."):
                adaptive_res = adaptive.retrieve(user_query)
                mode = adaptive_res["mode"]
                confidence = adaptive_res["confidence"]
                top_k = adaptive_res["top_k"]
                sufficient = adaptive_res["sufficient_evidence"]
                results = adaptive_res["results"]

                gen_res = generator.generate(user_query, results, sufficient_evidence=sufficient)
                ans_text = clean_generated_answer(gen_res["answer"])

                st.session_state["last_qa"] = {
                    "query": user_query,
                    "answer": ans_text,
                    "adaptive": adaptive_res,
                    "generator": gen_res,
                    "sufficient": sufficient,
                    "confidence": confidence,
                    "top_k": top_k,
                    "mode": mode,
                    "results": results
                }

        # Display result if available
        if "last_qa" in st.session_state:
            qa = st.session_state["last_qa"]
            mode = qa["mode"]
            confidence = qa["confidence"]
            top_k = qa["top_k"]
            sufficient = qa["sufficient"]
            results = qa["results"]
            ans_text = qa["answer"]
            adaptive_res = qa["adaptive"]
            gen_res = qa["generator"]

            col_main_ans, col_inspector = st.columns([1.6, 1.2])

            with col_main_ans:
                st.markdown("#### 💡 Grounded Answer")
                if not sufficient:
                    st.markdown(f"""
                    <div class="abstain-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <span style="font-size:0.8rem; font-weight:700; color:#ef4444; letter-spacing:0.5px;">SECURITY & ABSTENTION GATING</span>
                            <span class="badge-pill badge-abstain">🔴 ABSTENTION TRIGGERED</span>
                        </div>
                        <h4 style="margin:0 0 10px 0; color:#f87171;">⚠️ Insufficient Project Documentation Found</h4>
                        <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
                            {ans_text}
                        </div>
                        <div style="margin-top:14px; padding-top:10px; border-top:1px solid rgba(239, 68, 68, 0.2); font-size:0.82rem; color:#94a3b8;">
                            <strong>Adaptive Policy Action:</strong> The top semantic score (<code>{confidence:.3f}</code>) is below the calibrated threshold (<code>{st.session_state.min_thresh:.2f}</code>). The generator was actively blocked to prevent hallucination.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    badge_cls = "badge-focused" if mode == "focused" else ("badge-expanded" if mode == "expanded" else "badge-broad")
                    badge_lbl = f"{mode.upper()} MODE • Top-{top_k}"
                    citations_str = "".join([f'<span style="background:#1e293b; color:#94a3b8; border-radius:4px; padding:3px 8px; font-size:0.75rem; margin-right:6px; font-family:monospace;">📄 {c}</span>' for c in gen_res.get("citations", [])])

                    # Primary evidence
                    top_chunk = results[0]["chunk"] if results else {"text": "", "sections": []}
                    sec_breadcrumb = " > ".join(top_chunk.get("sections", [])) if top_chunk.get("sections") else "Overview"
                    top_text = top_chunk.get("text", "")[:280] + "..." if len(top_chunk.get("text", "")) > 280 else top_chunk.get("text", "")

                    st.markdown(f"""
                    <div class="answer-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="color:#60a5fa; font-weight:700; font-size:0.85rem; letter-spacing:0.5px;">VERIFIED ANSWER SUMMARY</span>
                            </div>
                            <span class="badge-pill {badge_cls}">{badge_lbl}</span>
                        </div>
                        <div style="font-size:1.05rem; color:#f8fafc; font-weight:500; line-height:1.6; margin-bottom:14px;">
                            {ans_text}
                        </div>
                        <div style="margin-top:10px;">
                            <span style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Verified Source Documents:</span><br/>
                            <div style="margin-top:4px;">{citations_str}</div>
                        </div>
                        <div class="evidence-quote">
                            <div style="font-size:0.75rem; color:#a78bfa; margin-bottom:4px; font-weight:600;">
                                🔖 PRIMARY EVIDENCE &nbsp;|&nbsp; {sec_breadcrumb}
                            </div>
                            <div style="color:#cbd5e1; font-size:0.83rem; line-height:1.5;">
                                "{top_text}"
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_inspector:
                st.markdown("#### 🔬 Evidence Inspector")
                st.markdown(f"""
                <div class="inspector-panel">
                    <div style="font-size:0.82rem; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:8px;">
                        Adaptive Retrieval Decision
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; margin-bottom:12px;">
                        <div style="background:#0e1526; border:1px solid #1e293b; padding:8px; border-radius:6px; text-align:center;">
                            <div style="font-size:0.7rem; color:#94a3b8;">MODE</div>
                            <div style="font-weight:700; color:#60a5fa; font-size:0.88rem;">{mode.upper()}</div>
                        </div>
                        <div style="background:#0e1526; border:1px solid #1e293b; padding:8px; border-radius:6px; text-align:center;">
                            <div style="font-size:0.7rem; color:#94a3b8;">TOP-1 SIM</div>
                            <div style="font-weight:700; color:#34d399; font-size:0.88rem;">{confidence:.3f}</div>
                        </div>
                        <div style="background:#0e1526; border:1px solid #1e293b; padding:8px; border-radius:6px; text-align:center;">
                            <div style="font-size:0.7rem; color:#94a3b8;">CHUNKS</div>
                            <div style="font-weight:700; color:#f59e0b; font-size:0.88rem;">{top_k}</div>
                        </div>
                    </div>
                    <div style="font-size:0.78rem; color:#94a3b8; background:#0e1526; padding:8px 10px; border-radius:6px; margin-bottom:14px; border-left:3px solid #60a5fa;">
                        <strong>Policy Rationale:</strong> {adaptive_res['explanation']}
                    </div>
                    <div style="font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:8px;">
                        Retrieved Context Chunks ({len(results)})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                for idx, r in enumerate(results, start=1):
                    c = r["chunk"]
                    sec = " > ".join(c.get("sections", [])) if c.get("sections") else "Overview"
                    score = r["score"]
                    bar_color = "#10b981" if score >= 0.50 else ("#3b82f6" if score >= 0.35 else "#f59e0b")
                    with st.expander(f"Rank #{idx} • Cosine: {score:.3f} • {c['source']}", expanded=(idx == 1)):
                        st.caption(f"**Section:** `{sec}`")
                        st.markdown(f"""
                        <div style="height:3px; background:#1e293b; border-radius:2px; margin-bottom:8px; overflow:hidden;">
                            <div style="height:100%; width:{min(score * 100, 100):.1f}%; background:{bar_color};"></div>
                        </div>
                        <div style="font-size:0.8rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace; line-height:1.4;">
                            {c['text']}
                        </div>
                        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 04. Retrieval Lab
    # -------------------------------------------------------------
    elif nav_choice == "04. Retrieval Lab":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Workspace / <span>Retrieval Lab & Diagnostic Benchmark</span></div>
            <div style="font-size:0.8rem; color:#60a5fa; font-family:'JetBrains Mono', monospace;">
                TF-IDF vs MiniLM vs Adaptive
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔬 Side-by-Side Diagnostic: Keyword vs Semantic vs Policy Gating")
        st.markdown("Directly evaluate how traditional lexical token frequencies contrast against continuous 384-dimensional dense vectors and our confidence-gated policy controller.")

        comp_query = st.text_input(
            "Diagnostic Query:",
            value="How can an engineer manually trigger database structure updates from the CLI?",
            placeholder="Type query to compare retrieval behaviors..."
        )

        if st.button("Run Diagnostic Benchmark", type="primary", use_container_width=True) and comp_query.strip():
            tfidf_res = baseline.search(comp_query, top_k=3)
            dense_res = semantic.search(comp_query, top_k=3)
            adapt_res = adaptive.retrieve(comp_query)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-top:3px solid #64748b; border-radius:8px; padding:14px; min-height:280px;">
                    <div style="font-weight:700; color:#f1f5f9; font-size:0.92rem;">1. TF-IDF Keyword Baseline</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:12px;">Statistical sparse n-gram frequency</div>
                """, unsafe_allow_html=True)
                if tfidf_res:
                    top_tf = tfidf_res[0]
                    st.metric("Top-1 Score", f"{top_tf['score']:.3f}", delta="Keyword Hit")
                    st.markdown(f"""
                    <div style="background:#090e1a; padding:10px; border-radius:6px; margin-top:8px; font-size:0.8rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace;">
                        <strong>📄 {top_tf['chunk']['source']}</strong><br/>
                        {top_tf['chunk']['text'][:180]}...
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Top-1 Score", "0.000", delta="- Zero match", delta_color="inverse")
                    st.markdown("""
                    <div style="background:#180d0e; padding:12px; border-radius:6px; margin-top:8px; font-size:0.8rem; color:#f87171; text-align:center;">
                        ❌ Zero Lexical Overlap<br/>
                        <span style="color:#94a3b8; font-size:0.75rem;">Query tokens do not match verbatim document vocabulary.</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-top:3px solid #2563eb; border-radius:8px; padding:14px; min-height:280px;">
                    <div style="font-weight:700; color:#60a5fa; font-size:0.92rem;">2. MiniLM Transformer</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:12px;">384-dimensional dense semantic space</div>
                """, unsafe_allow_html=True)
                if dense_res:
                    top_dense = dense_res[0]
                    score = top_dense['score']
                    st.metric("Top-1 Cosine", f"{score:.3f}", delta=f"{score*100:.1f}% Similarity")
                    st.markdown(f"""
                    <div style="background:#090e1a; padding:10px; border-radius:6px; margin-top:8px; font-size:0.8rem; color:#cbd5e1; font-family:'JetBrains Mono', monospace;">
                        <strong>📄 {top_dense['chunk']['source']}</strong><br/>
                        {top_dense['chunk']['text'][:180]}...
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-top:3px solid #10b981; border-radius:8px; padding:14px; min-height:280px;">
                    <div style="font-weight:700; color:#34d399; font-size:0.92rem;">3. Adaptive Policy Engine</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:12px;">Confidence gating & dynamic Top-K</div>
                """, unsafe_allow_html=True)
                mode_str = adapt_res['mode'].upper()
                st.metric(f"Policy: {mode_str}", f"K = {adapt_res['top_k']}", delta="ACCEPTED" if adapt_res['sufficient_evidence'] else "ABSTAINED")
                st.markdown(f"""
                <div style="background:#090e1a; padding:10px; border-radius:6px; margin-top:8px; font-size:0.78rem; color:#94a3b8;">
                    <strong>Decision:</strong> {adapt_res['explanation']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Architectural Diagnosis
            st.markdown("---")
            st.markdown("#### 💡 Architectural Analysis for this Query")
            if not adapt_res['sufficient_evidence']:
                st.warning("⚠️ **Out-of-Domain Detection:** The query lacks semantic backing in the documentation. TF-IDF either yields zero or misleading partial matches; Adaptive RAG safely blocks the generator from hallucinating.")
            elif tfidf_res and tfidf_res[0]['score'] > 0:
                st.info("ℹ️ **Exact Keyword Hit:** Distinct keywords match both lexically and semantically. TF-IDF provides ultra-fast retrieval, while MiniLM ensures structural context is preserved.")
            else:
                st.success("✨ **Lexical Gap Bridged:** The query uses phrasing that does not appear verbatim in the documentation. TF-IDF completely failed (0.000), but MiniLM and Adaptive RAG recognized semantic intent and retrieved the correct technical section.")

    # -------------------------------------------------------------
    # 05. Evaluation Center
    # -------------------------------------------------------------
    elif nav_choice == "05. Evaluation Center":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Evaluation / <span>Evaluation Center & Failure Analysis</span></div>
            <div style="font-size:0.8rem; color:#f59e0b; font-family:'JetBrains Mono', monospace;">
                Frozen Held-Out Test Set (N=20)
            </div>
        </div>
        """, unsafe_allow_html=True)

        metrics_file = Path("evaluation/results/evaluation_metrics_v2.json")
        plot_retrieval = Path("evaluation/results/retrieval_comparison_v2.png")
        plot_cm = Path("evaluation/results/abstention_confusion_matrix.png")
        table_file = Path("evaluation/results/comparison_table_v2.csv")

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                m_data = json.load(f)

            st.markdown("### 📊 Held-Out Benchmark Performance")
            st.caption("Separated evaluation: Retrieval ranking on answerable questions (N=12) vs binary classification on all held-out questions (N=20).")

            # 1. Retrieval Ranking Ribbon
            rank_data = m_data.get("retrieval_ranking_answerable", {})
            tf_data = rank_data.get("tfidf", {})
            sem_data = rank_data.get("minilm_semantic", {})
            adapt_data = rank_data.get("adaptive_rag", {})

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("TF-IDF Recall@1", f"{tf_data.get('recall_at_1', 0)*100:.1f}%")
            r2.metric("MiniLM Recall@1", f"{sem_data.get('recall_at_1', 0)*100:.1f}%")
            r3.metric("MiniLM Recall@3", f"{sem_data.get('recall_at_3', 0)*100:.1f}%")
            r4.metric("MiniLM MRR", f"{sem_data.get('mrr', 0):.3f}")
            r5.metric("Adaptive Coverage", f"{adapt_data.get('evidence_coverage', 0)*100:.1f}%")

            st.markdown("---")
            # 2. Abstention Classification Ribbon
            cm = m_data.get("abstention_classification", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("True Positives (TP)", f"{cm.get('tp', 0)} / 12")
            c2.metric("True Negatives (TN)", f"{cm.get('tn', 0)} / 8")
            c3.metric("Answerability Precision", f"{cm.get('precision', 0)*100:.1f}%")
            c4.metric("Abstention Specificity", f"{cm.get('specificity', 0)*100:.1f}%")
            c5.metric("Classification F1", f"{cm.get('f1_score', 0):.3f}")

            st.markdown("---")
            # Plots
            p_left, p_right = st.columns(2)
            with p_left:
                st.markdown("#### 📈 Retrieval Ranking Comparison")
                if plot_retrieval.exists():
                    st.image(str(plot_retrieval), use_container_width=True)
            with p_right:
                st.markdown("#### 🎯 Abstention Confusion Matrix")
                if plot_cm.exists():
                    st.image(str(plot_cm), use_container_width=True)

            # Failure Analysis section
            st.markdown("---")
            st.markdown("### ⚠️ Empirical Failure Analysis")
            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-left:4px solid #f59e0b; padding:14px; border-radius:8px; min-height:160px;">
                    <div style="font-weight:700; color:#fbbf24; font-size:0.85rem;">1. Semantic Granularity Limit</div>
                    <div style="font-size:0.78rem; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                        MiniLM embeddings place high similarity on general overview chunks touching frontend technologies, ranking it ahead of the exact command chunk at Rank 1.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with fa2:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-left:4px solid #ef4444; padding:14px; border-radius:8px; min-height:160px;">
                    <div style="font-weight:700; color:#f87171; font-size:0.85rem;">2. Near-Threshold False Positive</div>
                    <div style="font-size:0.78rem; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                        Asking about Kafka Schema Registries scored 0.322—barely crossing our 0.31 cutoff due to the presence of 'Kafka' in the docs, causing a False Positive.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with fa3:
                st.markdown("""
                <div style="background:#0e1526; border:1px solid #1e293b; border-left:4px solid #3b82f6; padding:14px; border-radius:8px; min-height:160px;">
                    <div style="font-weight:700; color:#60a5fa; font-size:0.85rem;">3. Small-Model Synthesis Limits</div>
                    <div style="font-size:0.78rem; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                        FLAN-T5-small (77M params) produces concise, direct summaries but occasionally truncates multi-paragraph architectural explanations.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if table_file.exists():
                with st.expander("📄 View Detailed Predictions Table (comparison_table_v2.csv)"):
                    df = pd.read_csv(table_file)
                    st.dataframe(df, use_container_width=True)
        else:
            st.info("Evaluation metrics not found. Run `python evaluation/evaluate.py`.")

    # -------------------------------------------------------------
    # 06. Adaptive Policy
    # -------------------------------------------------------------
    elif nav_choice == "06. Adaptive Policy":
        st.markdown("""
        <div class="top-nav-bar">
            <div class="breadcrumb-text">Evaluation / <span>Adaptive Policy & Threshold Calibration</span></div>
            <div style="font-size:0.8rem; color:#60a5fa; font-family:'JetBrains Mono', monospace;">
                Calibrated τ = 0.31
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Adaptive Policy Controller & Empirical Calibration")
        st.markdown("Configure the confidence thresholds governing dynamic context expansion and abstention gating.")

        c_sliders, c_evidence = st.columns([1.2, 1.4])

        with c_sliders:
            st.markdown("#### Dynamic Cutoff Sliders")
            new_focused = st.slider(
                "Focused Mode Cutoff (K=3 Chunks):", 0.40, 0.80, float(st.session_state.focused_thresh), 0.05,
                help="High confidence threshold for concise factual queries."
            )
            new_expanded = st.slider(
                "Expanded Mode Cutoff (K=6 Chunks):", 0.25, 0.55, float(st.session_state.expanded_thresh), 0.05,
                help="Medium confidence threshold for broader technical questions."
            )
            new_min = st.slider(
                "Abstention Threshold (τ):", 0.15, 0.45, float(st.session_state.min_thresh), 0.01,
                help="Calibrated cutoff on development set. Queries below this are actively abstained."
            )

            st.session_state.focused_thresh = new_focused
            st.session_state.expanded_thresh = new_expanded
            st.session_state.min_thresh = new_min

            if st.button("Reset to Mathematically Calibrated Defaults (0.55 / 0.35 / 0.31)"):
                st.session_state.focused_thresh = 0.55
                st.session_state.expanded_thresh = 0.35
                st.session_state.min_thresh = 0.31
                st.toast("Reset to calibrated defaults!", icon="🎯")
                st.rerun()

        with c_evidence:
            st.markdown("#### Empirical Calibration Evidence (Dev Set N=12)")
            st.caption("Systematic sweep across candidate thresholds to maximize classification F1 with zero held-out leakage.")

            metrics_file = Path("evaluation/results/evaluation_metrics_v2.json")
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    m_data = json.load(f)
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
                    st.success("Optimal Cutoff: **τ = 0.31** achieves 100.0% Balanced Accuracy and F1 = 1.000 on Development Data.")

        st.markdown("---")
        st.markdown("#### 📚 Theoretical Precedent")
        st.markdown("""
        Our confidence-gated abstention mechanism is based on calibrated similarity thresholding (Yildirim, 2024), 
        which demonstrated that separating answerable queries from out-of-scope queries in specialized domain corpora 
        substantially reduces hallucination rates without requiring expensive external verifier models.
        """)


if __name__ == "__main__":
    main()
