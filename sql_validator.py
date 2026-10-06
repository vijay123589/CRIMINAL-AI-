# ─────────────────────────────────────────────
#  sql_validator.py – SQL Safety & Read-Only Validator
# ─────────────────────────────────────────────
import re
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DML, DDL

VALID_TABLES = {"crime_incidents", "criminal_records", "evidence_records"}

BLOCKED_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE",
    "ALTER", "CREATE", "REPLACE", "ATTACH", "DETACH",
    "EXEC", "EXECUTE", "PRAGMA", "VACUUM", "GRANT", "REVOKE"
}


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate a generated SQL query for read-only safety and table validity.

    Returns:
        (is_valid: bool, message: str)
    """
    if not sql or not sql.strip():
        return False, "SQL query is empty."

    sql_upper = sql.upper()

    # 1. Block non-SELECT / dangerous operations
    for kw in BLOCKED_KEYWORDS:
        pattern = r"\b" + kw + r"\b"
        if re.search(pattern, sql_upper):
            return False, f"Blocked operation detected: '{kw}' is not allowed in Read-Only mode."

    # 2. Ensure statement parses as DML SELECT
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False, "Could not parse SQL syntax."

    stmt: Statement = parsed[0]
    first_token = next(
        (t for t in stmt.flatten() if t.ttype in (DML, DDL) or t.value.upper() in ("SELECT", "WITH")), None
    )

    if first_token is None or first_token.value.upper() not in ("SELECT", "WITH"):
        return False, "Only READ-ONLY SELECT queries are allowed."

    # 3. Must reference at least one valid table
    sql_lower = sql.lower()
    table_found = any(tbl in sql_lower for tbl in VALID_TABLES)
    if not table_found:
        return (
            False,
            f"Query must reference valid tables: {', '.join(VALID_TABLES)}.",
        )

    return True, "✅ Safe READ-ONLY SELECT query validated."


def format_sql(sql: str) -> str:
    """Format SQL query for clean display."""
    try:
        return sqlparse.format(
            sql,
            reindent=True,
            keyword_case="upper",
            identifier_case="lower",
            strip_comments=True,
        )
    except Exception:
        return sql
