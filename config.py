# ─────────────────────────────────────────────
#  config.py  – Central Configuration for CrimeIntel AI
# ─────────────────────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR
DB_PATH    = BASE_DIR / "criminal_database.db"
CHROMA_DIR = BASE_DIR / "chroma_store"

# ── Database Tables ─────────────────────────────
TABLE_CRIMES     = "crime_incidents"
TABLE_CRIMINALS  = "criminal_records"
TABLE_EVIDENCE   = "evidence_records"

# ── Gemini API Credentials & Model ─────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", os.getenv("GROQ_API_KEY", ""))
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TOKENS      = 1024
TEMPERATURE     = 0.1

# ── RAG / ChromaDB Configuration ───────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "criminal_intelligence_context"
TOP_K_RESULTS     = 5

# ── Schema Metadata for Prompt & RAG Context ──
SCHEMA_DESCRIPTION = """
Database: criminal_database.db (SQLite)

Tables & Schema:

1. crime_incidents
   - case_id               : TEXT PRIMARY KEY (e.g. 'C101', 'C102')
   - criminal_id           : TEXT (Foreign Key -> criminal_records.criminal_id)
   - crime_type            : TEXT ('Robbery', 'Cybercrime', 'Homicide', 'Narcotics', 'Burglary', 'Vehicle Theft', 'Fraud', 'Assault')
   - description           : TEXT (Case summary and modus-operandi details)
   - incident_date         : TEXT ('YYYY-MM-DD')
   - district              : TEXT ('Chennai Central', 'Chennai North', 'Chennai South', 'Anna Nagar', 'T. Nagar', 'Velachery', 'Tambaram', 'Adyar')
   - latitude              : REAL (e.g. 13.0827)
   - longitude             : REAL (e.g. 80.2707)
   - severity              : TEXT ('Low', 'Medium', 'High', 'Critical')
   - case_status           : TEXT ('Open', 'Solved', 'Under Investigation', 'Cold Case', 'Closed')
   - investigating_officer : TEXT (e.g. 'Insp. K. Ramesh')

2. criminal_records
   - criminal_id        : TEXT PRIMARY KEY (e.g. 'CRM-1001')
   - name               : TEXT (Full suspect name)
   - alias              : TEXT (Nickname or 'None')
   - age                : INTEGER
   - gender             : TEXT ('Male', 'Female')
   - occupation         : TEXT
   - district           : TEXT
   - prior_convictions  : INTEGER
   - gang_affiliation   : TEXT ('None', 'Syndicate-X', 'Metro Cyber Net', 'Shadow Cartel', 'Bayfront Ring')
   - status             : TEXT ('At Large', 'In Custody', 'On Bail', 'Under Surveillance')

3. evidence_records
   - evidence_id      : TEXT PRIMARY KEY (e.g. 'EVD-C101-1')
   - case_id          : TEXT (Foreign Key -> crime_incidents.case_id)
   - evidence_type    : TEXT ('CCTV Footage', 'DNA Sample', 'Fingerprint', 'Digital Logs', 'Ballistics', 'Tool Marks', 'Witness Statement')
   - description      : TEXT
   - confidence_score : REAL (0.0 to 1.0)
"""

# ── RAG Knowledge Chunks ─────────────────────────
RAG_CHUNKS = [
    {
        "id": "schema_overview",
        "text": SCHEMA_DESCRIPTION,
        "metadata": {"type": "schema", "topic": "criminal_database_schema"},
    },
    {
        "id": "crime_type_queries",
        "text": (
            "Common crime type queries:\n"
            "- Show all robbery cases: SELECT * FROM crime_incidents WHERE crime_type = 'Robbery';\n"
            "- Count cybercrime cases: SELECT COUNT(*) AS cybercrime_count FROM crime_incidents WHERE crime_type = 'Cybercrime';\n"
            "- Most common crime type: SELECT crime_type, COUNT(*) AS total FROM crime_incidents GROUP BY crime_type ORDER BY total DESC LIMIT 1;"
        ),
        "metadata": {"type": "example", "topic": "crime_type"},
    },
    {
        "id": "status_and_district_queries",
        "text": (
            "Filtering by status and district:\n"
            "- Show unsolved cases in Chennai Central: SELECT * FROM crime_incidents WHERE (case_status = 'Open' OR case_status = 'Under Investigation') AND district = 'Chennai Central';\n"
            "- Unsolved cases across all districts: SELECT district, COUNT(*) AS unsolved_count FROM crime_incidents WHERE case_status IN ('Open', 'Under Investigation') GROUP BY district;"
        ),
        "metadata": {"type": "example", "topic": "status_district"},
    },
    {
        "id": "severity_and_date_queries",
        "text": (
            "Filtering by severity and date:\n"
            "- High severity cases from 2024: SELECT * FROM crime_incidents WHERE severity IN ('High', 'Critical') AND incident_date LIKE '2024%';\n"
            "- Cases by severity level: SELECT severity, COUNT(*) FROM crime_incidents GROUP BY severity;"
        ),
        "metadata": {"type": "example", "topic": "severity_date"},
    },
    {
        "id": "keyword_search_queries",
        "text": (
            "Searching descriptions for specific items or modus operandi:\n"
            "- Find cases involving a stolen motorcycle: SELECT * FROM crime_incidents WHERE description LIKE '%motorcycle%' OR description LIKE '%vehicle%';\n"
            "- Find cases involving a crowbar: SELECT * FROM crime_incidents WHERE description LIKE '%crowbar%';"
        ),
        "metadata": {"type": "example", "topic": "keyword_search"},
    },
    {
        "id": "joins_and_suspects",
        "text": (
            "JOIN operations between crime_incidents, criminal_records, and evidence_records:\n"
            "- Case details with suspect info: SELECT c.case_id, c.crime_type, c.district, cr.name AS suspect_name, cr.status AS suspect_status FROM crime_incidents c LEFT JOIN criminal_records cr ON c.criminal_id = cr.criminal_id WHERE c.case_id = 'C102';\n"
            "- Cases with CCTV evidence: SELECT c.*, e.evidence_type, e.description AS evidence_desc FROM crime_incidents c JOIN evidence_records e ON c.case_id = e.case_id WHERE e.evidence_type = 'CCTV Footage';"
        ),
        "metadata": {"type": "example", "topic": "joins"},
    },
    {
        "id": "sql_rules",
        "text": (
            "CRITICAL SQL RULES:\n"
            "1. Only READ-ONLY SELECT queries are allowed.\n"
            "2. Table names: crime_incidents, criminal_records, evidence_records.\n"
            "3. Case statuses: 'Open', 'Solved', 'Under Investigation', 'Cold Case', 'Closed'.\n"
            "4. Crime types: 'Robbery', 'Cybercrime', 'Homicide', 'Narcotics', 'Burglary', 'Vehicle Theft', 'Fraud', 'Assault'.\n"
            "5. Always use LIKE '%keyword%' for keyword search questions."
        ),
        "metadata": {"type": "rules", "topic": "sql_rules"},
    },
]