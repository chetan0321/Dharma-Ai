"""
Dharma.AI — Streamlit UI (premium build)

Run with:
    streamlit run app.py
"""

import os
import time
from dotenv import load_dotenv
import streamlit as st

from rag.pipeline import RAGPipeline
from rag.baseline import BaselineLLM

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dharma.AI — Indian Legal Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0f1923 100%);
    color: #e8eaf0;
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Hero Header ── */
.dharma-hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #162035 60%, #1c2540 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.dharma-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.dharma-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 150px;
    background: radial-gradient(ellipse, rgba(139,92,246,0.08) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #9f7aea 50%, #68d391 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(99, 179, 237, 0.12);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: #63b3ed;
    font-weight: 500;
}
.badge.green { background: rgba(104,211,145,0.1); border-color: rgba(104,211,145,0.3); color: #68d391; }
.badge.purple { background: rgba(159,122,234,0.1); border-color: rgba(159,122,234,0.3); color: #9f7aea; }

/* ── Question card ── */
.question-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(26,31,58,0.8) 0%, rgba(20,26,48,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(15px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    position: relative;
}
.answer-card.baseline {
    border-color: rgba(159,122,234,0.2);
}
.answer-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.answer-mode-badge {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 8px;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.3);
}
.answer-mode-badge.baseline {
    background: rgba(159,122,234,0.15);
    color: #9f7aea;
    border-color: rgba(159,122,234,0.3);
}
.answer-text {
    color: #e2e8f0;
    line-height: 1.75;
    font-size: 0.95rem;
}

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.35rem 0.85rem;
    font-size: 0.8rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.metric-chip span.val {
    color: #e2e8f0;
    font-weight: 600;
}

/* ── Source card ── */
.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #63b3ed;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #cbd5e0;
    transition: border-color 0.2s ease;
}
.source-card:hover { border-left-color: #9f7aea; }
.source-title {
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
}
.source-meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.source-excerpt {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.55;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 0.4rem;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #101824 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.5rem 0 0.75rem;
}

/* ── Streamlit overrides ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.75rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
}
.stExpander {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
}
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
.stMarkdown h3 {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

/* ── Comparison divider ── */
.vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #475569;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    padding: 0.5rem 0;
}

/* ── Sample queries ── */
.sample-btn {
    display: inline-block;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    color: #63b3ed;
    cursor: pointer;
    margin: 0.2rem;
    transition: all 0.15s ease;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: #92400e;
    color: #fbbf24;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached pipelines ───────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_pipeline(k: int, temp: float) -> RAGPipeline:
    return RAGPipeline(retriever_k=k, temperature=temp)


@st.cache_resource
def get_baseline(temp: float) -> BaselineLLM:
    return BaselineLLM(temperature=temp)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dharma-hero">
    <h1 class="hero-title">⚖️ Dharma.AI</h1>
    <p class="hero-subtitle">
        RAG-powered legal research over Indian law &mdash; Constitution, IPC, CrPC, IT Act, GST Acts.<br>
        Every answer is grounded in retrieved source documents and cited.
    </p>
    <div class="hero-badges">
        <span class="badge">Llama 3.3 70B · Groq</span>
        <span class="badge green">ChromaDB · MMR Retrieval</span>
        <span class="badge purple">LangChain · RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown('<p class="sidebar-section-title">Mode</p>', unsafe_allow_html=True)
    compare_mode = st.toggle("🔁 Compare RAG vs Baseline", value=False,
                              help="Show both RAG and no-RAG answers side-by-side")
    if not compare_mode:
        use_rag = st.toggle("🔍 Use RAG retrieval", value=True)
    else:
        use_rag = True

    st.markdown('<p class="sidebar-section-title">Retrieval</p>', unsafe_allow_html=True)
    k_chunks = st.slider("Retrieved chunks (k)", 1, 10, 5,
                          help="Number of document chunks retrieved from ChromaDB")
    show_sources = st.toggle("📄 Show source documents", value=True)

    st.markdown('<p class="sidebar-section-title">Generation</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                             help="Higher = more creative; lower = more factual")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.6;">
    <strong style="color:#64748b;">Data sources</strong><br>
    📜 Constitution of India<br>
    ⚖️ Indian Penal Code (IPC)<br>
    🏛️ Code of Criminal Procedure (CrPC)<br>
    💻 IT Act 2000<br>
    💰 CGST Act
    </div>
    """, unsafe_allow_html=True)


# ── Sample queries ─────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "What is the punishment for theft under IPC?",
    "Define fundamental rights in the Constitution",
    "What is anticipatory bail under CrPC?",
    "Explain Section 66 of IT Act",
]

st.markdown("**💡 Try a sample question:**")
cols = st.columns(len(SAMPLE_QUERIES))
selected_sample = None
for i, (col, q) in enumerate(zip(cols, SAMPLE_QUERIES)):
    with col:
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            selected_sample = q

# ── Input ──────────────────────────────────────────────────────────────────────
default_q = selected_sample or ""
question = st.text_input(
    "Ask a question about Indian law:",
    value=default_q,
    placeholder="e.g. What is the punishment for theft under IPC? (Section 379)",
    label_visibility="collapsed",
)

col_btn, col_clear, _ = st.columns([1, 1, 6])
with col_btn:
    submit = st.button("🔍 Get Answer", use_container_width=True)
with col_clear:
    if st.button("✕ Clear", use_container_width=True):
        st.rerun()


# ── Answer logic ───────────────────────────────────────────────────────────────
def run_rag(question: str, k: int, temp: float) -> dict:
    pipeline = get_rag_pipeline(k, temp)
    return pipeline.answer(question)


def run_baseline(question: str, temp: float) -> tuple[str, float]:
    baseline = get_baseline(temp)
    t0 = time.time()
    answer = baseline.answer(question)
    return answer, time.time() - t0


def render_sources(sources: list[dict]):
    if not sources:
        st.caption("_No sources retrieved._")
        return
    for i, src in enumerate(sources, 1):
        title = src.get("title", "Untitled")
        source = src.get("source", "unknown")
        section = src.get("section", "N/A")
        content = src.get("content", "")
        excerpt = content[:280] + ("…" if len(content) > 280 else "")
        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">[{i}] {title}</div>
            <div class="source-meta">📁 {source} &nbsp;|&nbsp; § {section}</div>
            <div class="source-excerpt">{excerpt}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main results ───────────────────────────────────────────────────────────────
if (submit or selected_sample) and question.strip():
    if compare_mode:
        # ── Side-by-side comparison ────────────────────────────────────────────
        with st.spinner("⚖️ Running RAG and Baseline in parallel…"):
            t0 = time.time()
            rag_result = run_rag(question, k_chunks, temperature)
            rag_elapsed = time.time() - t0
            rag_answer = rag_result["answer"]
            rag_sources = rag_result.get("sources", [])
            rag_tokens = rag_result.get("tokens", 0)

            base_answer, base_elapsed = run_baseline(question, temperature)

        col_rag, col_base = st.columns(2)

        with col_rag:
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-header">
                    <span class="answer-mode-badge">✦ With RAG</span>
                    <span style="font-size:0.78rem;color:#64748b;">Retrieval-augmented</span>
                </div>
                <div class="answer-text">{rag_answer.replace(chr(10), "<br>")}</div>
                <div class="metrics-row">
                    <div class="metric-chip">⏱ <span class="val">{rag_elapsed:.2f}s</span></div>
                    <div class="metric-chip">🔢 <span class="val">{rag_tokens}</span> tokens</div>
                    <div class="metric-chip">📄 <span class="val">{len(rag_sources)}</span> sources</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if show_sources and rag_sources:
                with st.expander("📚 Retrieved sources", expanded=False):
                    render_sources(rag_sources)

        with col_base:
            st.markdown(f"""
            <div class="answer-card baseline">
                <div class="answer-header">
                    <span class="answer-mode-badge baseline">◇ Without RAG</span>
                    <span style="font-size:0.78rem;color:#64748b;">Pure LLM, no grounding</span>
                </div>
                <div class="answer-text">{base_answer.replace(chr(10), "<br>")}</div>
                <div class="metrics-row">
                    <div class="metric-chip">⏱ <span class="val">{base_elapsed:.2f}s</span></div>
                    <div class="metric-chip" style="border-color:rgba(251,191,36,0.3);color:#fbbf24;">
                        ⚠ No citations
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Single mode ────────────────────────────────────────────────────────
        mode_label = "RAG" if use_rag else "Baseline (no RAG)"
        with st.spinner(f"⚖️ Searching Indian law corpus…"):
            t0 = time.time()
            if use_rag:
                result = run_rag(question, k_chunks, temperature)
                answer = result["answer"]
                sources = result.get("sources", [])
                tokens = result.get("tokens", 0)
            else:
                answer, _ = run_baseline(question, temperature)
                sources = []
                tokens = 0
            elapsed = time.time() - t0

        mode_badge_class = "" if use_rag else "baseline"
        mode_icon = "✦ With RAG" if use_rag else "◇ Without RAG"
        st.markdown(f"""
        <div class="answer-card {mode_badge_class}">
            <div class="answer-header">
                <span class="answer-mode-badge {mode_badge_class}">{mode_icon}</span>
                <span style="font-size:0.78rem;color:#64748b;">{mode_label}</span>
            </div>
            <div class="answer-text">{answer.replace(chr(10), "<br>")}</div>
            <div class="metrics-row">
                <div class="metric-chip">⏱ <span class="val">{elapsed:.2f}s</span></div>
                <div class="metric-chip">🔢 <span class="val">{tokens}</span> tokens</div>
                <div class="metric-chip">📄 <span class="val">{len(sources)}</span> sources</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_sources and sources:
            st.markdown("### 📚 Retrieved source documents")
            render_sources(sources)
        elif use_rag and not sources:
            st.info("No relevant documents found in the index for this query.")

    # ── Legal disclaimer ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Legal Disclaimer:</strong> Dharma.AI is a research tool, not legal advice.
        Always verify information with a qualified legal professional.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Idle state ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #475569;">
        <div style="font-size:3rem; margin-bottom:1rem;">⚖️</div>
        <div style="font-size:1.1rem; font-weight:500; color:#64748b; margin-bottom:0.5rem;">
            Ask any question about Indian law
        </div>
        <div style="font-size:0.85rem; color:#334155;">
            Grounded answers from Constitution · IPC · CrPC · IT Act · GST Act
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-size:0.75rem; color:#334155; padding: 0.25rem 0;">
    <span>⚖️ <strong style="color:#475569;">Dharma.AI</strong> · Built by Chetan Reddy</span>
    <span>Llama 3.3 70B via Groq · ChromaDB · LangChain · Streamlit</span>
    <span>Always verify with a qualified professional</span>
</div>
""", unsafe_allow_html=True)
