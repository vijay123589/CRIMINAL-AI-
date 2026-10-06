# ─────────────────────────────────────────────
#  llm_engine.py – Gemini AI SQL Generator & Tactical Summarizer
# ─────────────────────────────────────────────
from __future__ import annotations
import logging
import re
from openai import OpenAI
from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


def _get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API key is not configured.")
    return OpenAI(api_key=key, base_url=GEMINI_BASE_URL)


SQL_SYSTEM_PROMPT = """You are an expert SQLite Database Intelligence Analyst for Law Enforcement.
Your job is to convert a natural language question into a valid, executable, READ-ONLY SQLite SELECT query.

Database Tables:
1. crime_incidents (case_id, criminal_id, crime_type, description, incident_date, district, latitude, longitude, severity, case_status, investigating_officer)
2. criminal_records (criminal_id, name, alias, age, gender, occupation, district, prior_convictions, gang_affiliation, status)
3. evidence_records (evidence_id, case_id, evidence_type, description, confidence_score)

Rules:
1. Output ONLY the raw SQL query — no markdown fences, no explanations.
2. ONLY generate SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or DDL.
3. String matching rules:
   - For specific terms (e.g. 'crowbar', 'motorcycle', 'CCTV'), use LIKE '%term%' on description column.
   - Exact category matches use exact strings: crime_type IN ('Robbery', 'Cybercrime', 'Homicide', 'Narcotics', 'Burglary', 'Vehicle Theft', 'Fraud', 'Assault').
   - Case status values: 'Open', 'Solved', 'Under Investigation', 'Cold Case', 'Closed'.
   - Unsolved cases mean: case_status IN ('Open', 'Under Investigation', 'Cold Case').
4. Add LIMIT 100 unless an explicit aggregation (COUNT, SUM, GROUP BY) is requested.
"""


def generate_sql(
    question: str,
    context: str,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """
    Generate a SQLite query from a natural language question using Gemini.
    """
    try:
        client = _get_client(api_key)
        user_prompt = f"""
Relevant Criminal Database Schema Context:
{context}

User Question:
{question}

Generate the executable SQLite SELECT query:
"""
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": SQL_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        sql = response.choices[0].message.content.strip()
        # Clean markdown code blocks if returned
        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = sql.strip("`").strip()
        return sql, None
    except Exception as e:
        logger.error(f"❌ SQL generation failed: {e}")
        return "", str(e)


INSIGHT_SYSTEM_PROMPT = """You are a senior law enforcement intelligence officer.
Given a user query, executed SQL query, and query results,
provide a concise 3-4 bullet point tactical summary / investigative intelligence breakdown.
Highlight key crime stats, suspect risks, district trends, or evidence takeaways in professional natural language.
"""


def generate_insights(
    question: str,
    sql: str,
    results_preview: str,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """
    Generate investigative summary from query results using Gemini.
    """
    try:
        client = _get_client(api_key)
        user_prompt = f"""
Investigative Question: {question}
SQL Query Executed:
{sql}

Query Results Preview:
{results_preview}

Provide tactical intelligence summary (3-4 bullet points):
"""
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        insights = response.choices[0].message.content.strip()
        return insights, None
    except Exception as e:
        logger.error(f"❌ Insight generation failed: {e}")
        return "", str(e)
