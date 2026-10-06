# CRIMEINTEL AI — Conversational AI for Criminal Database

An AI & Machine Learning powered law enforcement intelligence platform combining **Text-to-SQL (Gemini AI)**, **Retrieval-Augmented Generation (ChromaDB + Sentence Transformers)**, **Scikit-Learn K-Means Spatial Hotspot Clustering**, and **Random Forest Offender Risk Classification**.

---

## 🚀 Key Features & Modules

1. **💬 Conversational AI & Text-to-SQL Engine**: Ask natural-language questions to query crime incidents, registered offenders, and forensic logs. Auto-generates READ-ONLY SQLite `SELECT` queries using Gemini AI.
2. **🔎 Semantic Case & MO Search (ChromaDB Vector Store)**: Perform similarity search over unstructured case narratives, modus-operandi descriptions, and evidence logs using `all-MiniLM-L6-v2` dense vector embeddings.
3. **📍 Historical Crime Hotspot Clustering (K-Means)**: Groups historical crime incident coordinates (latitude, longitude) into $K$ spatial concentration clusters.
4. **🤖 Offender Risk Classification Model (Random Forest)**: Evaluates suspect features (age, prior convictions, gang affiliation, crime severity) to predict threat levels (*Low, Medium, High*) with evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
5. **📊 Crime Analytics & Interactive Command Dashboard**: Interactive Plotly visual breakdowns of crime types, severity distribution, case status metrics, and criminal directories.
6. **🛡️ SQL Safety & READ-ONLY Validation**: Uses `sqlparse` to block dangerous database operations (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`).

---

## 📁 Application Structure

```
CRIMEINTEL_AI/
├── app.py                      ← Main Streamlit Multi-Tab Application
├── config.py                   ← Paths, Gemini API settings, RAG schema chunks
├── database.py                 ← SQLite connection & query executor
├── generate_criminal_data.py   ← Synthetic Criminal Database Generator
├── rag_engine.py               ← ChromaDB vector store & semantic search
├── llm_engine.py               ← Gemini AI SQL generation & intelligence summary
├── sql_validator.py            ← Read-Only SQL safety validator
├── ml_engine.py                ← K-Means Hotspot Clustering & Random Forest Classifier
├── run_system_tests.py         ← Full automated test checklist runner
├── requirements.txt            ← Project dependencies
└── criminal_database.db        ← SQLite database (auto-generated)
```

---

## 🛠️ Step-by-Step Setup & Run Guide

### Step 1 — Open Terminal in Project Folder

```powershell
cd "c:\Users\vijay\Desktop\RAG-based-SQL-Agent-Python-Streamlit-Google-Gemini-API-LangChain-ChromaDB-SQLite-RAG--main"
```

### Step 2 — Run Automated Verification Tests

```powershell
py -3.13 run_system_tests.py
```

### Step 3 — Launch the Application

```powershell
py -3.13 -m streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**

---

## 🎓 Viva Explanation Guide for College Submission

- **RAG (Retrieval-Augmented Generation)**: Uses `all-MiniLM-L6-v2` to convert schema knowledge and query examples into vectors. ChromaDB retrieves top-$k$ relevant context to construct the Gemini LLM prompt.
- **Sentence Transformers**: Converts text into a 384-dimensional dense vector space for cosine similarity matching.
- **Text-to-SQL**: Converts natural language into executable SQLite queries using Gemini AI.
- **SQL Validator**: Enforces read-only safety by parsing AST tokens and rejecting DDL/DML mutation keywords.
- **K-Means Clustering**: Unsupervised machine learning algorithm grouping geographical coordinates (latitude, longitude) into $K$ spatial clusters.
- **Random Forest Classifier**: Supervised ensemble of decision trees trained on offender features to classify risk categories.
