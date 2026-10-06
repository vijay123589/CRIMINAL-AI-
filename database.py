# ─────────────────────────────────────────────
#  database.py – SQLite Query Engine for Criminal Database
# ─────────────────────────────────────────────
from __future__ import annotations
import sqlite3
import pandas as pd
from pathlib import Path
import logging
from config import DB_PATH, TABLE_CRIMES, TABLE_CRIMINALS, TABLE_EVIDENCE
from generate_criminal_data import create_and_seed_db

logger = logging.getLogger(__name__)


def ensure_database_exists() -> bool:
    """Ensure the SQLite database exists; create and seed if missing."""
    try:
        if not Path(DB_PATH).exists():
            logger.info("Database missing. Generating synthetic criminal database...")
            create_and_seed_db()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_connection() -> sqlite3.Connection:
    """Return a live SQLite connection."""
    ensure_database_exists()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(sql: str) -> tuple[list[dict], list[str], str | None]:
    """
    Execute a SELECT query against the criminal database.

    Returns:
        rows    – list of dicts
        columns – list of column names
        error   – error message string or None
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows    = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows, columns, None
    except Exception as e:
        return [], [], str(e)


def get_database_summary() -> dict:
    """Return key metrics and counts for the dashboard."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM crime_incidents")
        total_cases = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM criminal_records")
        total_criminals = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crime_incidents WHERE case_status = 'Solved'")
        solved_cases = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crime_incidents WHERE case_status IN ('Open', 'Under Investigation', 'Cold Case')")
        unsolved_cases = cursor.fetchone()[0]

        conn.close()
        return {
            "total_cases": total_cases,
            "total_criminals": total_criminals,
            "solved_cases": solved_cases,
            "unsolved_cases": unsolved_cases,
        }
    except Exception as e:
        logger.error(f"Error getting database summary: {e}")
        return {
            "total_cases": 0,
            "total_criminals": 0,
            "solved_cases": 0,
            "unsolved_cases": 0,
        }
