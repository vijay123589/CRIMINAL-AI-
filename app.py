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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"]         { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1527 100%);
    color: #e2e8f0;
}
.hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    padding: 1.8rem 2.2rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.hero-title {
    font-size: 2.2rem; font-weight: 700; color: #f8fafc;
    margin: 0 0 .3rem; display: flex; align-items: center; gap: 0.6rem;
}
.hero-subtitle {
    font-size: 1rem; color: #94a3b8; font-weight: 400; margin: 0;
}
.metric-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 0.8rem;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem; font-weight: 700; color: #60a5fa;
}
.metric-label { font-size: 0.82rem; color: #94a3b8; margin-top: 0.2rem; }

.sql-box {
    background: #090d16;
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-left: 4px solid #3b82f6;
    border-radius: 0.6rem;
    padding: 1rem 1.2rem;
    font-family: monospace;
    font-size: 0.88rem; color: #93c5fd;
    white-space: pre-wrap; word-break: break-all;
}
.insight-box {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 0.8rem;
    padding: 1.2rem 1.4rem; color: #e2e8f0; line-height: 1.7;
}
.rag-badge {
    display: inline-block; background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.4); color: #c084fc;
    padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
}
.sql-badge {
    display: inline-block; background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4); color: #60a5fa;
    padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
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
        <p style='color:#94a3b8;font-size:.85rem'>
            Enter your Google Gemini free API key to enable natural-language SQL generation
        </p>
    </div>
    """, unsafe_allow_html=True)

    key = st.text_input("Gemini API Key", type="password", key="popup_api_key")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Key", use_container_width=True, type="primary"):
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
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    init_session()

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ CRIMEINTEL AI</div>
        <div class="hero-subtitle">
            Conversational AI for Criminal Database — Text-to-SQL Agent, ChromaDB RAG, K-Means Clustering & Random Forest Risk Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API key prompt if missing
    if not st.session_state.key_confirmed and not GEMINI_API_KEY:
        st.info("💡 Note: You can configure a Gemini API Key using the button below or set GEMINI_API_KEY in your `.env` file.")
        if st.button("🔑 Set Gemini API Key"):
            api_key_popup()

    # Create Navigation Tabs
    tab_chat, tab_search, tab_criminals, tab_analytics, tab_kmeans, tab_rf, tab_about = st.tabs([
        "💬 Chat Assistant",
        "🔎 Case Search",
        "👤 Criminal Records",
        "📊 Crime Analytics",
        "📍 Hotspot Clustering",
        "🤖 Risk Classification",
        "📘 About / Methodology"
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: CHAT ASSISTANT (Text-to-SQL + RAG)
    # ────────────────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 Natural Language Criminal Database Query")
        st.markdown("Ask questions in plain English to query cases, counts, severity, or suspect details.")

        # Quick sample buttons
        st.markdown("**Sample Demo Questions:**")
        q_cols = st.columns(4)
        sample_q = ""
        if q_cols[0].button("📋 All Robbery Cases"):
            sample_q = "Show all robbery cases."
        if q_cols[1].button("💻 Cybercrime Count"):
            sample_q = "How many cybercrime cases were reported?"
        if q_cols[2].button("🔓 Unsolved Cases"):
            sample_q = "Show unsolved cases in Chennai Central."
        if q_cols[3].button("⚡ High Severity 2024"):
            sample_q = "Show high severity cases from 2024."

        q_cols2 = st.columns(3)
        if q_cols2[0].button("🔍 Details of Case C102"):
            sample_q = "Give me details of case C102."
        if q_cols2[1].button("📊 Most Common Crime"):
            sample_q = "Which crime type is most common?"
        if q_cols2[2].button("🏍️ Stolen Motorcycle"):
            sample_q = "Find cases involving a stolen motorcycle."

        query_input = st.text_area(
            "Enter your question:",
            value=sample_q,
            placeholder="e.g. Show unsolved robbery cases in T. Nagar",
            height=85,
        )

        if st.button("🚀 Run Conversational Query", type="primary"):
            if not query_input.strip():
                st.warning("Please enter a question.")
            else:
                st.markdown('<span class="sql-badge">Engine: Gemini Text-to-SQL + ChromaDB RAG</span>', unsafe_allow_html=True)
                
                with st.spinner("Retrieving RAG schema context..."):
                    context = retrieve_context(query_input)

                with st.spinner("Generating READ-ONLY SQL with Gemini..."):
                    sql, gen_err = generate_sql(query_input, context, api_key=st.session_state.api_key)

                if gen_err:
                    st.error(f"SQL Generation Error: {gen_err}")
                else:
                    is_valid, val_msg = validate_sql(sql)
                    formatted_sql = format_sql(sql)

                    st.markdown("#### 🧾 Generated Read-Only SQL")
                    st.markdown(f'<div class="sql-box">{formatted_sql}</div>', unsafe_allow_html=True)

                    if not is_valid:
                        st.error(f"SQL Safety Error: {val_msg}")
                    else:
                        with st.spinner("Executing query on criminal database..."):
                            rows, columns, exec_err = execute_query(sql)

                        if exec_err:
                            st.error(f"Query Execution Failed: {exec_err}")
                        elif not rows:
                            st.warning("No records found matching query criteria.")
                        else:
                            df = pd.DataFrame(rows, columns=columns)
                            st.markdown(f"#### 📊 Query Results ({len(df)} records returned)")
                            st.dataframe(df, use_container_width=True)

                            with st.spinner("Generating intelligence summary..."):
                                preview = df.head(10).to_string(index=False)
                                insights, ins_err = generate_insights(query_input, formatted_sql, preview, api_key=st.session_state.api_key)

                            if not ins_err and insights:
                                st.markdown("#### 💡 Tactical Intelligence Brief")
                                st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: CASE SEARCH (Semantic RAG Vector Search)
    # ────────────────────────────────────────────────────────────────────────
    with tab_search:
        st.markdown("### 🔎 Semantic Case & Modus-Operandi Search (ChromaDB Vector Store)")
        st.markdown("Search unstructured case descriptions, modus-operandi, and forensic evidence using vector embedding similarity (`all-MiniLM-L6-v2`).")

        st.markdown('<span class="rag-badge">Engine: ChromaDB Vector Embeddings</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sem_q = st.text_input(
            "Enter semantic query or case narrative:",
            value="Find cases involving a person using a crowbar at night",
            placeholder="e.g. CCTV evidence capturing suspect near ATM"
        )

        top_k = st.slider("Number of vector matches:", min_value=1, max_value=10, value=5)

        if st.button("🔍 Search Semantic Case Vectors", type="primary"):
            with st.spinner("Searching ChromaDB vector store..."):
                matches = search_semantic_case_files(sem_q, top_k=top_k)

            if not matches:
                st.info("No matching vector embeddings found.")
            else:
                st.markdown(f"#### 🎯 Top {len(matches)} Semantic Matches")
                for i, m in enumerate(matches, 1):
                    with st.expander(f"Match #{i} — Case {m['case_id']} (Similarity: {m['similarity_score']})", expanded=(i==1)):
                        st.markdown(f"**Similarity Score:** `{m['similarity_score']}`")
                        st.markdown(f"**Vector Document:**")
                        st.info(m["document"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: CRIMINAL RECORDS
    # ────────────────────────────────────────────────────────────────────────
    with tab_criminals:
        st.markdown("### 👤 Criminal Records Directory")
        conn = get_connection()
        c_df = pd.read_sql_query("SELECT * FROM criminal_records", conn)
        conn.close()

        st.markdown(f"**Total Records:** `{len(c_df)}` offenders registered")
        
        status_filter = st.multiselect("Filter by Status:", options=c_df["status"].unique().tolist(), default=c_df["status"].unique().tolist())
        filtered_c = c_df[c_df["status"].isin(status_filter)]

        st.dataframe(filtered_c, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: CRIME ANALYTICS
    # ────────────────────────────────────────────────────────────────────────
    with tab_analytics:
        st.markdown("### 📊 Crime Analytics & Dashboard")
        summary = get_database_summary()

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{summary["total_cases"]}</div><div class="metric-label">Total Crime Incidents</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{summary["total_criminals"]}</div><div class="metric-label">Registered Criminals</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{summary["solved_cases"]}</div><div class="metric-label">Solved Cases</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{summary["unsolved_cases"]}</div><div class="metric-label">Unsolved Cases</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        conn = get_connection()

        with col_a:
            type_df = pd.read_sql_query("SELECT crime_type, COUNT(*) AS count FROM crime_incidents GROUP BY crime_type", conn)
            fig_type = px.bar(type_df, x="crime_type", y="count", title="📊 Crime Type Distribution", color="crime_type", template="plotly_dark")
            st.plotly_chart(fig_type, use_container_width=True)

        with col_b:
            sev_df = pd.read_sql_query("SELECT severity, COUNT(*) AS count FROM crime_incidents GROUP BY severity", conn)
            fig_sev = px.pie(sev_df, names="severity", values="count", title="🥧 Severity Level Breakdown", template="plotly_dark", hole=0.4)
            st.plotly_chart(fig_sev, use_container_width=True)

        conn.close()

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5: HOTSPOT CLUSTERING (K-MEANS)
    # ────────────────────────────────────────────────────────────────────────
    with tab_kmeans:
        st.markdown("### 📍 Historical Crime Hotspot Clustering (K-Means)")
        st.info("⚠️ **Note:** This module applies `scikit-learn` K-Means clustering to historical crime location coordinates (latitude, longitude) to identify density concentration clusters. It does NOT predict future crime.")

        k_val = st.slider("Select number of spatial clusters (K):", min_value=2, max_value=6, value=4)

        cluster_res = run_kmeans_clustering(n_clusters=k_val)

        if "error" in cluster_res:
            st.error(cluster_res["error"])
        else:
            df_c = cluster_res["df"]
            centroids = cluster_res["centroids"]

            if hasattr(px, "scatter_map"):
                fig_map = px.scatter_map(
                    df_c,
                    lat="latitude",
                    lon="longitude",
                    color="cluster_label",
                    hover_name="case_id",
                    hover_data=["crime_type", "district", "severity"],
                    zoom=10,
                    height=500,
                    title=f"K-Means Spatial Crime Hotspot Clusters (K={k_val})",
                )
                fig_map.update_layout(map_style="open-street-map")
            else:
                fig_map = px.scatter_mapbox(
                    df_c,
                    lat="latitude",
                    lon="longitude",
                    color="cluster_label",
                    hover_name="case_id",
                    hover_data=["crime_type", "district", "severity"],
                    zoom=10,
                    height=500,
                    title=f"K-Means Spatial Crime Hotspot Clusters (K={k_val})",
                )
                fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("#### 📌 Calculated Cluster Centroid Coordinates")
            st.dataframe(centroids, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 6: RISK CLASSIFICATION (RANDOM FOREST)
    # ────────────────────────────────────────────────────────────────────────
    with tab_rf:
        st.markdown("### 🤖 Offender Risk Classification Model (Random Forest)")
        st.info("⚠️ **Note:** This is a demonstration machine learning classification model using `scikit-learn` Random Forest. It evaluates synthetic offender features for educational evaluation and does NOT predict real criminal behavior.")

        rf_res = train_random_forest_model()

        if "error" in rf_res:
            st.error(rf_res["error"])
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{rf_res['accuracy']}%")
            col_m2.metric("Precision", f"{rf_res['precision']}%")
            col_m3.metric("Recall", f"{rf_res['recall']}%")
            col_m4.metric("F1-Score", f"{rf_res['f1_score']}%")

            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                st.markdown("#### 🎯 Confusion Matrix")
                st.dataframe(rf_res["confusion_matrix"], use_container_width=True)

            with col_rf2:
                st.markdown("#### ⚡ Feature Importance Ranking")
                st.dataframe(rf_res["feature_importance"], use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔮 Interactive Suspect Risk Predictor Test")

            col_i1, col_i2, col_i3 = st.columns(3)
            age_in = col_i1.number_input("Suspect Age:", min_value=18, max_value=80, value=32)
            priors_in = col_i2.number_input("Prior Convictions:", min_value=0, max_value=15, value=3)
            gang_in = col_i3.selectbox("Gang Affiliation:", options=["None", "Syndicate-X", "Metro Cyber Net", "Shadow Cartel", "Bayfront Ring"])

            col_i4, col_i5, col_i6 = st.columns(3)
            sev_in = col_i4.selectbox("Crime Severity:", options=["Low", "Medium", "High", "Critical"])
            cases_in = col_i5.number_input("Previous Cases Linked:", min_value=0, max_value=20, value=2)
            cat_in = col_i6.selectbox("Crime Category:", options=["Robbery", "Cybercrime", "Homicide", "Narcotics", "Burglary", "Vehicle Theft", "Fraud", "Assault"])

            if st.button("⚡ Predict Offender Risk Category"):
                pred_res = predict_offender_risk(age_in, priors_in, gang_in, sev_in, cases_in, cat_in)
                if "error" in pred_res:
                    st.error(pred_res["error"])
                else:
                    st.success(f"Predicted Risk Category: **{pred_res['predicted_risk']}** (Confidence: {pred_res['confidence']}%)")
                    st.json(pred_res["probabilities"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 7: ABOUT / METHODOLOGY (COLLEGE VIVA PREPARATION)
    # ────────────────────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("### 📘 System Architecture & Project Methodology (Viva Guide)")

        st.markdown("""
        ### 🎓 Technical Stack & Concepts for PBL Viva Demonstration:

        1. **Retrieval-Augmented Generation (RAG)**:
           - **How it works:** User natural language questions are converted into embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`). ChromaDB vector store retrieves relevant database schema chunks and example SQL templates to construct the LLM prompt.
           - **Why it is used:** Eliminates LLM hallucinations by restricting schema knowledge to actual SQLite tables (`crime_incidents`, `criminal_records`, `evidence_records`).

        2. **Sentence Transformers (`all-MiniLM-L6-v2`)**:
           - **How it works:** Maps text descriptions into a 384-dimensional dense vector space. Cosine similarity is computed between query vectors and case description vectors.

        3. **Text-to-SQL Engine**:
           - **How it works:** Google Gemini API converts natural language into executable SQLite queries using context provided by RAG.

        4. **SQL Safety Validator (`sql_validator.py`)**:
           - **How it works:** Uses `sqlparse` to strictly enforce READ-ONLY access. Blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE` commands.

        5. **K-Means Hotspot Clustering (`scikit-learn`)**:
           - **How it works:** Groups spatial latitude and longitude coordinates of crime incidents into $K$ spatial clusters, computing centroid hotspot centers.

        6. **Random Forest Classification (`scikit-learn`)**:
           - **How it works:** An ensemble of decision trees trained on suspect demographics, prior convictions, and gang affiliations to classify offender risk categories (*Low, Medium, High*). Evaluated via Train/Test split metrics (Accuracy, Precision, Recall, F1).

        7. **SQLite Relational Database**:
           - Stores structured criminal records, incident logs, and evidence records with foreign key relationships.

        8. **Streamlit UI Framework**:
           - Interactive web app interface rendering real-time tables, Plotly visualizations, and model diagnostics.
        """)

if __name__ == "__main__":
    main()