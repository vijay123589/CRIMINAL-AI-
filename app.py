# ─────────────────────────────────────────────
#  app.py – CRIMEINTEL AI | Streamlit Interface
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, GEMINI_API_KEY
from database import get_connection, execute_query, get_database_summary, ensure_database_exists
from rag_engine import build_vector_store, retrieve_context, search_semantic_case_files, get_store_stats
from llm_engine import generate_sql, generate_insights
from sql_validator import validate_sql, format_sql
from ml_engine import run_kmeans_clustering, train_random_forest_model, predict_offender_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CRIMEINTEL AI — Conversational AI for Criminal Database",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide sidebar ── */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── App background ── */
.stApp {
    background: #0e1117;
    color: #e2e8f0;
}

/* ── Main content padding ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ════ HERO BANNER ════ */
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #151c2c 60%, #1a2035 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-top: 3px solid #3b82f6;
    padding: 2rem 2.5rem;
    border-radius: 1rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon {
    font-size: 3rem;
    line-height: 1;
    flex-shrink: 0;
}
.hero-text {}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f0f4ff;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #7c9cc9;
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.8rem;
}
.hero-pill {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #93c5fd;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* ════ METRIC CARDS ════ */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #151c2c;
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 0.9rem;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0.9rem 0.9rem 0 0;
}
.metric-card.blue::before  { background: #3b82f6; }
.metric-card.green::before { background: #10b981; }
.metric-card.amber::before { background: #f59e0b; }
.metric-card.red::before   { background: #ef4444; }
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
    display: block;
}

/* ════ SECTION HEADERS ════ */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-desc {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0 0 1.2rem;
}
.section-divider {
    height: 1px;
    background: rgba(99, 179, 237, 0.1);
    margin: 1.5rem 0;
}

/* ════ SAMPLE QUERY BUTTONS ════ */
.query-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.query-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}

/* ════ SQL & RESULT BOXES ════ */
.sql-box {
    background: #0d1117;
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-left: 3px solid #3b82f6;
    border-radius: 0.6rem;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 0.85rem;
    color: #93c5fd;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.7;
    margin: 0.5rem 0 1rem;
}
.insight-box {
    background: #131d30;
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 3px solid #10b981;
    border-radius: 0.6rem;
    padding: 1.2rem 1.4rem;
    color: #d1fae5;
    line-height: 1.8;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* ════ ENGINE BADGES ════ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.badge-blue {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #93c5fd;
}
.badge-purple {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.35);
    color: #c084fc;
}
.badge-green {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
}
.badge-amber {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fcd34d;
}

/* ════ INFO / WARNING BOXES ════ */
.info-box {
    background: #131d30;
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 0.6rem;
    padding: 0.9rem 1.1rem;
    color: #93c5fd;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.warn-box {
    background: #1c1508;
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 0.6rem;
    padding: 0.9rem 1.1rem;
    color: #fcd34d;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ════ MATCH CARDS ════ */
.match-card {
    background: #151c2c;
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 0.7rem;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.match-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.match-caseid {
    font-weight: 700;
    font-size: 0.95rem;
    color: #e2e8f0;
}
.match-score {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
}
.match-doc {
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ════ ABOUT / METHODOLOGY ════ */
.about-card {
    background: #151c2c;
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 0.8rem;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.about-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.about-card-body {
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.7;
}

/* ════ STREAMLIT COMPONENT OVERRIDES ════ */
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: #151c2c;
    padding: 0.4rem;
    border-radius: 0.7rem;
    border: 1px solid rgba(99, 179, 237, 0.1);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 0.5rem;
    color: #64748b;
    font-size: 0.84rem;
    font-weight: 600;
    padding: 0.45rem 1rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border: 1px solid rgba(59, 130, 246, 0.35) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    border: none;
    border-radius: 0.5rem;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.55);
    transform: translateY(-1px);
}

/* Secondary button */
.stButton > button:not([kind="primary"]) {
    background: #1a2035;
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 0.5rem;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.45rem 0.9rem;
    transition: all 0.15s;
}
.stButton > button:not([kind="primary"]):hover {
    background: #1e2d47;
    border-color: rgba(59, 130, 246, 0.6);
    color: #bfdbfe;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #151c2c !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 0.5rem !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(59, 130, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 0.6rem;
    overflow: hidden;
    border: 1px solid rgba(99, 179, 237, 0.1);
}

/* Expander */
div[data-testid="stExpander"] {
    background: #151c2c;
    border: 1px solid rgba(99, 179, 237, 0.12) !important;
    border-radius: 0.7rem !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding: 0.5rem 0;
}

/* Multiselect */
.stMultiSelect [data-baseweb="select"] {
    background: #151c2c;
}

/* Metric */
[data-testid="stMetric"] {
    background: #151c2c;
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 0.7rem;
    padding: 0.9rem 1rem;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "db_ready": False,
        "rag_ready": False,
        "history": [],
        "api_key": GEMINI_API_KEY,
        "key_confirmed": bool(GEMINI_API_KEY),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.db_ready:
        st.session_state.db_ready = ensure_database_exists()
    if not st.session_state.rag_ready:
        st.session_state.rag_ready = build_vector_store()


# ════════════════════════════════════════════════════════════════════════════
#  API KEY DIALOG
# ════════════════════════════════════════════════════════════════════════════
@st.dialog("🔑 Gemini API Key Configuration")
def api_key_popup():
    st.markdown("""
    <div style='text-align:center;padding:.5rem 0 1rem'>
        <span style='font-size:2.5rem'>🛡️</span>
        <h3 style='color:#e2e8f0;margin:.5rem 0 .2rem'>CRIMEINTEL AI Key Setup</h3>
        <p style='color:#64748b;font-size:.85rem'>
            Enter your Google Gemini API key to enable natural-language SQL generation
        </p>
    </div>
    """, unsafe_allow_html=True)

    key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", key="popup_api_key")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Key", use_container_width=True, type="primary"):
            if key.strip():
                st.session_state.api_key = key.strip()
                st.session_state.key_confirmed = True
                st.rerun()
            else:
                st.error("Please enter a valid key")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def badge(text, kind="blue"):
    icons = {"blue": "⚡", "purple": "🔮", "green": "✅", "amber": "⚠️"}
    return f'<span class="badge badge-{kind}">{icons.get(kind, "")} {text}</span>'


def section_header(icon, title, desc=""):
    desc_html = f'<div class="section-desc">{desc}</div>' if desc else ""
    st.markdown(f"""
    <div class="section-title">{icon} {title}</div>
    {desc_html}
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    init_session()

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-icon">🛡️</div>
        <div class="hero-text">
            <div class="hero-title">CRIMEINTEL AI</div>
            <div class="hero-subtitle">
                Conversational AI for Criminal Database &nbsp;·&nbsp;
                Powered by Gemini AI, ChromaDB RAG, K-Means Clustering &amp; Random Forest Risk Classification
            </div>
            <div class="hero-pills">
                <span class="hero-pill">Gemini 3.8 Flash</span>
                <span class="hero-pill">ChromaDB RAG</span>
                <span class="hero-pill">Sentence Transformers</span>
                <span class="hero-pill">K-Means Clustering</span>
                <span class="hero-pill">Random Forest</span>
                <span class="hero-pill">SQLite</span>
                <span class="hero-pill">Streamlit</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API key warning ───────────────────────────────────────────────────────
    if not st.session_state.key_confirmed and not GEMINI_API_KEY:
        col_warn, col_btn = st.columns([4, 1])
        with col_warn:
            st.markdown("""
            <div class="info-box">
                💡 <strong>Gemini API Key Required</strong> — Chat Assistant needs a key to generate SQL.
                All other tabs (Search, Analytics, ML models) work without a key.
                Set <code>GEMINI_API_KEY</code> in your <code>.env</code> file or click the button.
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔑 Set API Key", use_container_width=True):
                api_key_popup()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_chat, tab_search, tab_criminals, tab_analytics, tab_kmeans, tab_rf, tab_about = st.tabs([
        "💬  Chat Assistant",
        "🔎  Case Search",
        "👤  Criminal Records",
        "📊  Crime Analytics",
        "📍  Hotspot Clustering",
        "🤖  Risk Classification",
        "📘  About / Methodology"
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — CHAT ASSISTANT
    # ────────────────────────────────────────────────────────────────────────
    with tab_chat:
        section_header("💬", "Natural Language Criminal Database Query",
                        "Ask questions in plain English — Gemini converts them to READ-ONLY SQL using RAG schema context.")

        st.markdown(badge("Engine: Gemini Text-to-SQL + ChromaDB RAG", "blue"), unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Sample query buttons
        st.markdown("<div style='font-size:0.82rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem'>Quick Demo Questions</div>", unsafe_allow_html=True)

        sample_q = ""
        q1, q2, q3, q4 = st.columns(4)
        if q1.button("📋  All Robbery Cases", use_container_width=True):
            sample_q = "Show all robbery cases."
        if q2.button("💻  Cybercrime Count", use_container_width=True):
            sample_q = "How many cybercrime cases were reported?"
        if q3.button("🔓  Unsolved in Chennai", use_container_width=True):
            sample_q = "Show unsolved cases in Chennai Central."
        if q4.button("⚡  High Severity 2024", use_container_width=True):
            sample_q = "Show high severity cases from 2024."

        q5, q6, q7 = st.columns(3)
        if q5.button("🔍  Case C102 Details", use_container_width=True):
            sample_q = "Give me details of case C102."
        if q6.button("📊  Most Common Crime", use_container_width=True):
            sample_q = "Which crime type is most common?"
        if q7.button("🏍️  Stolen Motorcycle", use_container_width=True):
            sample_q = "Find cases involving a stolen motorcycle."

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        query_input = st.text_area(
            "Your Question",
            value=sample_q,
            placeholder="e.g. Show unsolved robbery cases in T. Nagar with high severity",
            height=90,
            label_visibility="collapsed",
        )

        run_col, _ = st.columns([1, 4])
        with run_col:
            run_btn = st.button("🚀  Run Query", type="primary", use_container_width=True)

        if run_btn:
            if not query_input.strip():
                st.warning("Please enter a question before running.")
            else:
                with st.spinner("Retrieving RAG schema context..."):
                    context = retrieve_context(query_input)

                with st.spinner("Generating SQL with Gemini..."):
                    sql, gen_err = generate_sql(query_input, context, api_key=st.session_state.api_key)

                if gen_err:
                    st.error(f"**SQL Generation Error:** {gen_err}")
                else:
                    is_valid, val_msg = validate_sql(sql)
                    formatted_sql = format_sql(sql)

                    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-top:1rem;margin-bottom:0.3rem'>Generated Read-Only SQL</div>", unsafe_allow_html=True)
                    st.markdown(f'<div class="sql-box">{formatted_sql}</div>', unsafe_allow_html=True)

                    if not is_valid:
                        st.markdown(f"""
                        <div class="warn-box">
                            ⚠️ <strong>SQL Safety Blocked:</strong> {val_msg}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.spinner("Executing query on criminal database..."):
                            rows, columns, exec_err = execute_query(sql)

                        if exec_err:
                            st.error(f"**Query Execution Failed:** {exec_err}")
                        elif not rows:
                            st.info("No records found matching your query criteria.")
                        else:
                            df = pd.DataFrame(rows, columns=columns)
                            st.markdown(f"<div style='font-size:0.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem'>{len(df)} Records Returned</div>", unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True, height=min(350, 50 + len(df) * 35))

                            with st.spinner("Generating intelligence summary..."):
                                preview = df.head(10).to_string(index=False)
                                insights, ins_err = generate_insights(
                                    query_input, formatted_sql, preview,
                                    api_key=st.session_state.api_key
                                )

                            if not ins_err and insights:
                                st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-top:1rem;margin-bottom:0.3rem'>💡 Tactical Intelligence Brief</div>", unsafe_allow_html=True)
                                st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — SEMANTIC CASE SEARCH
    # ────────────────────────────────────────────────────────────────────────
    with tab_search:
        section_header("🔎", "Semantic Case & Modus-Operandi Search",
                        "Search unstructured case descriptions, modus-operandi, and forensic evidence logs using vector embedding similarity.")

        st.markdown(badge("Engine: ChromaDB Vector Store · all-MiniLM-L6-v2", "purple"), unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Demo semantic queries
        st.markdown("<div style='font-size:0.82rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem'>Example Semantic Searches</div>", unsafe_allow_html=True)

        sem_sample = ""
        s1, s2, s3 = st.columns(3)
        if s1.button("🔨  Crowbar night break-in", use_container_width=True):
            sem_sample = "Find cases involving a person using a crowbar at night"
        if s2.button("📹  CCTV evidence", use_container_width=True):
            sem_sample = "Find cases where CCTV evidence was important"
        if s3.button("🏍️  Stolen motorcycle", use_container_width=True):
            sem_sample = "Find cases involving a stolen motorcycle"

        s4, s5, _ = st.columns(3)
        if s4.button("💳  Phishing bank fraud", use_container_width=True):
            sem_sample = "Phishing attack targeting bank accounts"
        if s5.button("🔬  DNA evidence cases", use_container_width=True):
            sem_sample = "Cases where DNA samples were collected as evidence"

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        sem_q = st.text_input(
            "Semantic Query",
            value=sem_sample or "Find cases involving a person using a crowbar at night",
            placeholder="Describe the incident, MO, or evidence type...",
            label_visibility="collapsed",
        )

        scol1, scol2, scol3 = st.columns([2, 1, 2])
        with scol1:
            top_k = st.slider("Number of matches to return:", min_value=1, max_value=10, value=5)
        with scol2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍  Search Vector Store", type="primary", use_container_width=True)

        if search_btn:
            with st.spinner("Searching ChromaDB vector embeddings..."):
                matches = search_semantic_case_files(sem_q, top_k=top_k)

            if not matches:
                st.info("No matching vector embeddings found for this query.")
            else:
                st.markdown(f"<div style='font-size:0.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-top:0.8rem;margin-bottom:0.8rem'>Top {len(matches)} Semantic Matches</div>", unsafe_allow_html=True)
                for i, m in enumerate(matches, 1):
                    score_val = float(m["similarity_score"].replace("%", ""))
                    score_color = "#10b981" if score_val >= 50 else "#f59e0b" if score_val >= 30 else "#ef4444"
                    st.markdown(f"""
                    <div class="match-card">
                        <div class="match-header">
                            <div class="match-caseid">#{i} &nbsp; Case {m['case_id']}</div>
                            <div class="match-score" style="border-color:{score_color}33;color:{score_color};background:{score_color}18">
                                {m['similarity_score']} similarity
                            </div>
                        </div>
                        <div class="match-doc">{m['document']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — CRIMINAL RECORDS
    # ────────────────────────────────────────────────────────────────────────
    with tab_criminals:
        section_header("👤", "Criminal Records Directory",
                        "Registered offenders with status, affiliations, and prior conviction history.")

        conn = get_connection()
        c_df = pd.read_sql_query("SELECT * FROM criminal_records ORDER BY prior_convictions DESC", conn)
        conn.close()

        # Stats row
        total = len(c_df)
        at_large = len(c_df[c_df["status"] == "At Large"])
        in_custody = len(c_df[c_df["status"] == "In Custody"])
        on_bail = len(c_df[c_df["status"].isin(["On Bail", "Under Surveillance"])])

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="metric-card blue"><span class="metric-icon">👤</span><div class="metric-value">{total}</div><div class="metric-label">Total Registered</div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="metric-card red"><span class="metric-icon">🚨</span><div class="metric-value">{at_large}</div><div class="metric-label">At Large</div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="metric-card green"><span class="metric-icon">🔒</span><div class="metric-value">{in_custody}</div><div class="metric-label">In Custody</div></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="metric-card amber"><span class="metric-icon">⚖️</span><div class="metric-value">{on_bail}</div><div class="metric-label">On Bail / Monitored</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            status_filter = st.multiselect(
                "Filter by Status:",
                options=sorted(c_df["status"].unique().tolist()),
                default=c_df["status"].unique().tolist()
            )
        with fc2:
            gang_filter = st.multiselect(
                "Filter by Gang Affiliation:",
                options=sorted(c_df["gang_affiliation"].unique().tolist()),
                default=c_df["gang_affiliation"].unique().tolist()
            )

        filtered_c = c_df[
            c_df["status"].isin(status_filter) &
            c_df["gang_affiliation"].isin(gang_filter)
        ]

        st.markdown(f"<div style='font-size:0.82rem;color:#64748b;margin-bottom:0.4rem'>{len(filtered_c)} records shown</div>", unsafe_allow_html=True)
        st.dataframe(filtered_c, use_container_width=True, height=400)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4 — CRIME ANALYTICS
    # ────────────────────────────────────────────────────────────────────────
    with tab_analytics:
        section_header("📊", "Crime Analytics Dashboard",
                        "Distribution, trends, severity, and district-level crime intelligence.")

        summary = get_database_summary()

        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f'<div class="metric-card blue"><span class="metric-icon">📁</span><div class="metric-value">{summary["total_cases"]}</div><div class="metric-label">Total Incidents</div></div>', unsafe_allow_html=True)
        a2.markdown(f'<div class="metric-card amber"><span class="metric-icon">👤</span><div class="metric-value">{summary["total_criminals"]}</div><div class="metric-label">Registered Criminals</div></div>', unsafe_allow_html=True)
        a3.markdown(f'<div class="metric-card green"><span class="metric-icon">✅</span><div class="metric-value">{summary["solved_cases"]}</div><div class="metric-label">Solved Cases</div></div>', unsafe_allow_html=True)
        a4.markdown(f'<div class="metric-card red"><span class="metric-icon">🔍</span><div class="metric-value">{summary["unsolved_cases"]}</div><div class="metric-label">Unsolved Cases</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        conn = get_connection()
        type_df = pd.read_sql_query("SELECT crime_type, COUNT(*) AS count FROM crime_incidents GROUP BY crime_type ORDER BY count DESC", conn)
        sev_df  = pd.read_sql_query("SELECT severity, COUNT(*) AS count FROM crime_incidents GROUP BY severity", conn)
        dist_df = pd.read_sql_query("SELECT district, COUNT(*) AS count FROM crime_incidents GROUP BY district ORDER BY count DESC", conn)
        status_df = pd.read_sql_query("SELECT case_status, COUNT(*) AS count FROM crime_incidents GROUP BY case_status", conn)
        conn.close()

        COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"]
        CHART_LAYOUT = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(21,28,44,0.5)",
            font=dict(family="Inter", size=12, color="#94a3b8"),
            margin=dict(l=10, r=10, t=40, b=10),
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_type = px.bar(
                type_df, x="crime_type", y="count",
                title="Crime Type Distribution",
                color="crime_type",
                color_discrete_sequence=COLORS,
                text="count",
            )
            fig_type.update_traces(textposition="outside", textfont_color="#e2e8f0")
            fig_type.update_layout(**CHART_LAYOUT, showlegend=False)
            fig_type.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_type, use_container_width=True)

        with col_b:
            sev_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
            sev_df["_ord"] = sev_df["severity"].map(sev_order)
            sev_df = sev_df.sort_values("_ord")
            fig_sev = px.pie(
                sev_df, names="severity", values="count",
                title="Severity Level Breakdown",
                color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444", "#dc2626"],
                hole=0.5,
            )
            fig_sev.update_layout(**CHART_LAYOUT)
            fig_sev.update_traces(textfont_color="#e2e8f0", textinfo="label+percent")
            st.plotly_chart(fig_sev, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            fig_dist = px.bar(
                dist_df, x="count", y="district",
                orientation="h",
                title="Incidents by District",
                color="count",
                color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
                text="count",
            )
            fig_dist.update_traces(textposition="outside", textfont_color="#e2e8f0")
            fig_dist.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_d:
            fig_status = px.pie(
                status_df, names="case_status", values="count",
                title="Case Status Breakdown",
                color_discrete_sequence=COLORS,
                hole=0.45,
            )
            fig_status.update_layout(**CHART_LAYOUT)
            fig_status.update_traces(textfont_color="#e2e8f0", textinfo="label+percent")
            st.plotly_chart(fig_status, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5 — K-MEANS HOTSPOT CLUSTERING
    # ────────────────────────────────────────────────────────────────────────
    with tab_kmeans:
        section_header("📍", "Historical Crime Hotspot Clustering (K-Means)",
                        "Unsupervised spatial clustering of crime incident coordinates to identify historical density hotspots.")

        st.markdown(badge("Model: scikit-learn KMeans | Spatial Features: Latitude, Longitude", "amber"), unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Disclaimer:</strong> This module applies K-Means clustering to <em>historical</em> crime
            coordinates only. It identifies past concentration patterns. It does <strong>NOT</strong> predict future crime locations.
        </div>
        """, unsafe_allow_html=True)

        k_col1, k_col2 = st.columns([2, 3])
        with k_col1:
            k_val = st.slider("Number of spatial clusters (K):", min_value=2, max_value=6, value=4)

        cluster_res = run_kmeans_clustering(n_clusters=k_val)

        if "error" in cluster_res:
            st.error(cluster_res["error"])
        else:
            df_c = cluster_res["df"]
            centroids = cluster_res["centroids"]

            st.markdown(f"<div style='font-size:0.82rem;color:#64748b;margin-bottom:0.5rem'>{len(df_c)} incidents clustered into {k_val} spatial groups &nbsp;·&nbsp; K-Means Inertia: {cluster_res['inertia']}</div>", unsafe_allow_html=True)

            if hasattr(px, "scatter_map"):
                fig_map = px.scatter_map(
                    df_c, lat="latitude", lon="longitude",
                    color="cluster_label",
                    hover_name="case_id",
                    hover_data=["crime_type", "district", "severity"],
                    zoom=10, height=480,
                    title=f"Historical Crime Hotspot Clusters (K={k_val})",
                    color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"],
                )
                fig_map.update_layout(map_style="open-street-map")
            else:
                fig_map = px.scatter_mapbox(
                    df_c, lat="latitude", lon="longitude",
                    color="cluster_label",
                    hover_name="case_id",
                    hover_data=["crime_type", "district", "severity"],
                    zoom=10, height=480,
                    title=f"Historical Crime Hotspot Clusters (K={k_val})",
                    color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"],
                )
                fig_map.update_layout(mapbox_style="open-street-map")

            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem'>Cluster Centroid Coordinates</div>", unsafe_allow_html=True)
            st.dataframe(centroids, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 6 — RANDOM FOREST RISK CLASSIFICATION
    # ────────────────────────────────────────────────────────────────────────
    with tab_rf:
        section_header("🤖", "Offender Risk Classification Model (Random Forest)",
                        "Demonstration ML classifier evaluating suspect features against synthetic risk labels using scikit-learn.")

        st.markdown(badge("Model: RandomForestClassifier | Classes: Low / Medium / High Risk", "purple"), unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Disclaimer:</strong> This is an educational ML demonstration only. The model is trained
            on <em>synthetic</em> data and does <strong>NOT</strong> predict real criminal behavior or recidivism.
        </div>
        """, unsafe_allow_html=True)

        rf_res = train_random_forest_model()

        if "error" in rf_res:
            st.error(rf_res["error"])
        else:
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{rf_res['accuracy']}%")
            m2.metric("Precision", f"{rf_res['precision']}%")
            m3.metric("Recall",    f"{rf_res['recall']}%")
            m4.metric("F1-Score",  f"{rf_res['f1_score']}%")

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem'>🎯 Confusion Matrix</div>", unsafe_allow_html=True)
                st.dataframe(rf_res["confusion_matrix"], use_container_width=True)

            with col_rf2:
                feat_df = rf_res["feature_importance"]
                fig_fi = px.bar(
                    feat_df, x="Importance", y="Feature", orientation="h",
                    title="Feature Importance",
                    color="Importance",
                    color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
                    text=feat_df["Importance"].round(3),
                )
                fig_fi.update_traces(textposition="outside", textfont_color="#e2e8f0")
                fig_fi.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(21,28,44,0.5)",
                    font=dict(family="Inter", size=12, color="#94a3b8"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

            st.markdown("<div style='font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem'>🔮 Interactive Suspect Risk Predictor</div>", unsafe_allow_html=True)

            pi1, pi2, pi3 = st.columns(3)
            age_in    = pi1.number_input("Suspect Age", min_value=18, max_value=80, value=32)
            priors_in = pi2.number_input("Prior Convictions", min_value=0, max_value=15, value=3)
            gang_in   = pi3.selectbox("Gang Affiliation", ["None", "Syndicate-X", "Metro Cyber Net", "Shadow Cartel", "Bayfront Ring"])

            pi4, pi5, pi6 = st.columns(3)
            sev_in    = pi4.selectbox("Crime Severity", ["Low", "Medium", "High", "Critical"])
            cases_in  = pi5.number_input("Previous Cases Linked", min_value=0, max_value=20, value=2)
            cat_in    = pi6.selectbox("Crime Category", ["Robbery", "Cybercrime", "Homicide", "Narcotics", "Burglary", "Vehicle Theft", "Fraud", "Assault"])

            pred_col, _ = st.columns([1, 3])
            with pred_col:
                predict_btn = st.button("⚡  Predict Risk Category", type="primary", use_container_width=True)

            if predict_btn:
                with st.spinner("Running Random Forest prediction..."):
                    pred_res = predict_offender_risk(age_in, priors_in, gang_in, sev_in, cases_in, cat_in)

                if "error" in pred_res:
                    st.error(pred_res["error"])
                else:
                    risk = pred_res["predicted_risk"]
                    conf = pred_res["confidence"]
                    risk_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
                    rc = risk_colors.get(risk, "#94a3b8")
                    st.markdown(f"""
                    <div style='background:#151c2c;border:1px solid {rc}44;border-left:4px solid {rc};border-radius:0.7rem;padding:1.2rem 1.4rem;margin-top:0.8rem'>
                        <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{rc};margin-bottom:0.4rem'>Predicted Risk Category</div>
                        <div style='font-size:2.2rem;font-weight:800;color:{rc}'>{risk} Risk</div>
                        <div style='font-size:0.88rem;color:#94a3b8;margin-top:0.3rem'>Confidence: <strong style='color:{rc}'>{conf}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:0.8rem;font-size:0.82rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.3rem'>Class Probability Breakdown</div>", unsafe_allow_html=True)
                    prob_df = pd.DataFrame(list(pred_res["probabilities"].items()), columns=["Risk Class", "Probability (%)"])
                    fig_prob = px.bar(
                        prob_df, x="Risk Class", y="Probability (%)",
                        color="Risk Class",
                        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                        text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%"),
                    )
                    fig_prob.update_traces(textposition="outside", textfont_color="#e2e8f0")
                    fig_prob.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(21,28,44,0.5)",
                        font=dict(family="Inter", color="#94a3b8"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False,
                        height=220,
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 7 — ABOUT / METHODOLOGY
    # ────────────────────────────────────────────────────────────────────────
    with tab_about:
        section_header("📘", "Project Architecture & Methodology",
                        "Technical explanation for PBL viva — how every component works.")

        cards = [
            ("⚡", "Retrieval-Augmented Generation (RAG)",
             "User questions are converted to vector embeddings using <code>all-MiniLM-L6-v2</code>. ChromaDB retrieves the most relevant schema chunks and SQL examples, which are injected into the Gemini AI prompt. This prevents hallucinations by grounding the LLM in the actual database schema."),
            ("🧮", "Sentence Transformers (all-MiniLM-L6-v2)",
             "Maps text descriptions into a 384-dimensional dense vector space. Cosine similarity is computed between the query vector and stored case/evidence description vectors in ChromaDB to find semantically similar cases."),
            ("💬", "Text-to-SQL Engine (Gemini 3.8 Flash)",
             "Google Gemini AI receives the user question along with RAG-retrieved schema context and converts it into an executable SQLite SELECT query. The system prompt enforces table names, column types, and SQL syntax rules."),
            ("🔒", "SQL Safety Validator (sqlparse)",
             "Uses <code>sqlparse</code> AST token inspection to strictly enforce READ-ONLY access. Blocks all dangerous operations: <code>INSERT</code>, <code>UPDATE</code>, <code>DELETE</code>, <code>DROP</code>, <code>ALTER</code>, <code>CREATE</code>, <code>TRUNCATE</code>."),
            ("📍", "K-Means Hotspot Clustering (scikit-learn)",
             "Unsupervised machine learning algorithm grouping crime incident coordinates (latitude, longitude) into K spatial clusters. The algorithm iteratively assigns each incident to the nearest centroid and recomputes centroids until convergence."),
            ("🌲", "Random Forest Classification (scikit-learn)",
             "Ensemble of 100 decision trees trained on suspect features (age, prior convictions, gang affiliation, crime severity, previous cases). Majority vote across trees classifies risk as Low / Medium / High. Evaluated via 75/25 train/test split."),
            ("🗄️", "SQLite Relational Database",
             "Three relational tables: <code>crime_incidents</code> (60 records), <code>criminal_records</code> (40 offenders), and <code>evidence_records</code> (forensic logs). Linked via foreign keys. Auto-generated synthetic data using <code>generate_criminal_data.py</code>."),
            ("🖥️", "Streamlit UI Framework",
             "Interactive Python web application framework. Multi-tab layout renders real-time dataframes, Plotly charts, and ML model outputs. Session state manages API key persistence and vector store initialization status."),
        ]

        col1, col2 = st.columns(2)
        for i, (icon, title, body) in enumerate(cards):
            with (col1 if i % 2 == 0 else col2):
                st.markdown(f"""
                <div class="about-card">
                    <div class="about-card-title">{icon} {title}</div>
                    <div class="about-card-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;color:#334155;font-size:0.8rem;padding:0.5rem 0'>
            CRIMEINTEL AI &nbsp;·&nbsp; PBL Project Submission &nbsp;·&nbsp; All data is synthetic and fictional
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()