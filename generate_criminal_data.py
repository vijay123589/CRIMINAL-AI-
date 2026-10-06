# ─────────────────────────────────────────────
#  generate_criminal_data.py – Synthetic Criminal Database Generator
# ─────────────────────────────────────────────
import sqlite3
import random
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "criminal_database.db"

# Seed for reproducibility
random.seed(42)

DISTRICTS = [
    "Chennai Central", "Chennai North", "Chennai South", 
    "Anna Nagar", "T. Nagar", "Velachery", "Tambaram", "Adyar"
]

CRIME_TYPES = [
    "Robbery", "Cybercrime", "Homicide", "Narcotics", 
    "Burglary", "Vehicle Theft", "Fraud", "Assault"
]

SEVERITIES = ["Low", "Medium", "High", "Critical"]
CASE_STATUSES = ["Open", "Solved", "Under Investigation", "Cold Case", "Closed"]
GANG_AFFILIATIONS = ["None", "Syndicate-X", "Metro Cyber Net", "Shadow Cartel", "Bayfront Ring"]
OFFICERS = ["Insp. K. Ramesh", "Det. S. Vijay", "Insp. M. Lakshmi", "Det. R. Ananth", "Insp. P. Suresh"]

FIRST_NAMES = ["Rajesh", "Vikram", "Anita", "Arun", "Priya", "David", "Suresh", "Kavita", "Manoj", "Dinesh",
               "Santhosh", "Deepak", "Rohan", "Meena", "Karthik", "Ganesh", "Sanjay", "Preeti", "Bala", "Ramesh"]
LAST_NAMES = ["Kumar", "Singh", "Desai", "Varma", "Sharma", "D'Souza", "Menon", "Nair", "Patel", "Reddy",
              "Rao", "Joshi", "Iyer", "Pillai", "Chawla", "Babu", "Gupta", "Hegde", "Fernandez", "Das"]

ALIASES = ["Shadow", "Viper", "Ghost", "Hammer", "Techie", "Blade", "Phantom", "Slick", "None", "None", "None"]
OCCUPATIONS = ["Unemployed", "Mechanic", "IT Consultant", "Driver", "Contractor", "Accountant", "Laborer", "Trader"]

# Specific realistic case descriptions for RAG testing
SPECIAL_CASES = [
    {
        "case_id": "C101",
        "crime_type": "Robbery",
        "description": "Suspect used a crowbar at night to force open the rear entrance of a jewelry shop and fled with gold ornaments.",
        "incident_date": "2024-03-15",
        "district": "T. Nagar",
        "latitude": 13.0418,
        "longitude": 80.2341,
        "severity": "High",
        "case_status": "Under Investigation",
        "investigating_officer": "Insp. K. Ramesh"
    },
    {
        "case_id": "C102",
        "crime_type": "Cybercrime",
        "description": "Phishing attack targeting bank customers using a duplicate banking portal to steal OTPs and transfer funds.",
        "incident_date": "2024-05-20",
        "district": "Anna Nagar",
        "latitude": 13.0850,
        "longitude": 80.2101,
        "severity": "Critical",
        "case_status": "Open",
        "investigating_officer": "Det. S. Vijay"
    },
    {
        "case_id": "C103",
        "crime_type": "Vehicle Theft",
        "description": "Stolen motorcycle (TN-01-AB-1234) hijacked near metro station parking lot during peak evening hours.",
        "incident_date": "2024-01-10",
        "district": "Velachery",
        "latitude": 12.9759,
        "longitude": 80.2212,
        "severity": "Medium",
        "case_status": "Solved",
        "investigating_officer": "Insp. M. Lakshmi"
    },
    {
        "case_id": "C104",
        "crime_type": "Homicide",
        "description": "Fatal altercation outside commercial building; CCTV evidence was vital in identifying the primary suspect.",
        "incident_date": "2024-08-11",
        "district": "Chennai Central",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "severity": "Critical",
        "case_status": "Under Investigation",
        "investigating_officer": "Det. R. Ananth"
    },
    {
        "case_id": "C105",
        "crime_type": "Burglary",
        "description": "Night break-in residential house using a crowbar to pry window grilles while occupants were away.",
        "incident_date": "2024-02-28",
        "district": "Adyar",
        "latitude": 13.0012,
        "longitude": 80.2565,
        "severity": "High",
        "case_status": "Open",
        "investigating_officer": "Insp. P. Suresh"
    }
]


def create_and_seed_db():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 1. criminal_records table
    cursor.execute("""
    CREATE TABLE criminal_records (
        criminal_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        alias TEXT,
        age INTEGER,
        gender TEXT,
        occupation TEXT,
        district TEXT,
        prior_convictions INTEGER,
        gang_affiliation TEXT,
        status TEXT
    );
    """)

    # 2. crime_incidents table
    cursor.execute("""
    CREATE TABLE crime_incidents (
        case_id TEXT PRIMARY KEY,
        criminal_id TEXT,
        crime_type TEXT,
        description TEXT,
        incident_date TEXT,
        district TEXT,
        latitude REAL,
        longitude REAL,
        severity TEXT,
        case_status TEXT,
        investigating_officer TEXT,
        FOREIGN KEY (criminal_id) REFERENCES criminal_records(criminal_id)
    );
    """)

    # 3. evidence_records table
    cursor.execute("""
    CREATE TABLE evidence_records (
        evidence_id TEXT PRIMARY KEY,
        case_id TEXT,
        evidence_type TEXT,
        description TEXT,
        confidence_score REAL,
        FOREIGN KEY (case_id) REFERENCES crime_incidents(case_id)
    );
    """)

    # Generate 40 Criminal Records
    criminals = []
    for i in range(1, 41):
        cid = f"CRM-{1000 + i}"
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        alias = random.choice(ALIASES)
        age = random.randint(19, 62)
        gender = random.choice(["Male", "Male", "Male", "Female"])
        occupation = random.choice(OCCUPATIONS)
        district = random.choice(DISTRICTS)
        priors = random.randint(0, 7)
        gang = random.choice(GANG_AFFILIATIONS)
        status = random.choice(["At Large", "In Custody", "On Bail", "Under Surveillance"])
        
        criminals.append((cid, name, alias, age, gender, occupation, district, priors, gang, status))

    cursor.executemany("""
        INSERT INTO criminal_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, criminals)

    # Generate 60 Crime Incidents
    incidents = []
    
    # First insert special pre-defined cases
    for i, sc in enumerate(SPECIAL_CASES):
        cid = f"CRM-{1001 + (i % 40)}"
        incidents.append((
            sc["case_id"],
            cid,
            sc["crime_type"],
            sc["description"],
            sc["incident_date"],
            sc["district"],
            sc["latitude"],
            sc["longitude"],
            sc["severity"],
            sc["case_status"],
            sc["investigating_officer"]
        ))

    # Generate remaining cases (C106 to C160)
    for i in range(6, 61):
        case_id = f"C{100 + i}"
        criminal_id = f"CRM-{1000 + random.randint(1, 40)}"
        crime_type = random.choice(CRIME_TYPES)
        district = random.choice(DISTRICTS)
        
        # Base lat/long around Chennai district centers
        dist_coords = {
            "Chennai Central": (13.0827, 80.2707),
            "Chennai North": (13.1200, 80.2800),
            "Chennai South": (13.0000, 80.2200),
            "Anna Nagar": (13.0850, 80.2101),
            "T. Nagar": (13.0418, 80.2341),
            "Velachery": (12.9759, 80.2212),
            "Tambaram": (12.9249, 80.1000),
            "Adyar": (13.0012, 80.2565),
        }
        base_lat, base_lng = dist_coords.get(district, (13.0827, 80.2707))
        lat = round(base_lat + random.uniform(-0.02, 0.02), 4)
        lng = round(base_lng + random.uniform(-0.02, 0.02), 4)

        year = random.choice([2023, 2024])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        incident_date = f"{year}-{month:02d}-{day:02d}"

        severity = random.choice(SEVERITIES)
        case_status = random.choice(CASE_STATUSES)
        officer = random.choice(OFFICERS)

        if crime_type == "Cybercrime":
            desc = f"Unauthorized access to database server in {district}. CCTV and digital logs captured IP anomaly."
        elif crime_type == "Robbery":
            desc = f"Armed robbery reported at retail establishment in {district}. Suspect fled scene with cash."
        elif crime_type == "Homicide":
            desc = f"Fatal assault under investigation in {district}. CCTV evidence was crucial for suspect tracking."
        elif crime_type == "Narcotics":
            desc = f"Illegal contraband possession seized during patrol in {district}."
        elif crime_type == "Vehicle Theft":
            desc = f"Stolen vehicle incident reported in {district}. Vehicle tracking initiated."
        elif crime_type == "Burglary":
            desc = f"Residential burglary reported in {district}. Entry gained via forced lock picking."
        elif crime_type == "Fraud":
            desc = f"Financial document forgery and unauthorized transfer recorded in {district}."
        else:
            desc = f"Physical altercation incident reported in {district} involving multiple individuals."

        incidents.append((
            case_id, criminal_id, crime_type, desc, incident_date, 
            district, lat, lng, severity, case_status, officer
        ))

    cursor.executemany("""
        INSERT INTO crime_incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, incidents)

    # Generate Evidence Records for Cases
    evidences = []
    ev_types = ["CCTV Footage", "DNA Sample", "Fingerprint", "Digital Logs", "Ballistics", "Tool Marks", "Witness Statement"]
    
    for idx, inc in enumerate(incidents):
        case_id = inc[0]
        crime_type = inc[2]
        
        # Add 1-2 evidence records per case
        for e_num in range(1, random.randint(2, 3)):
            ev_id = f"EVD-{case_id}-{e_num}"
            if "CCTV" in inc[3] or crime_type in ["Homicide", "Robbery"]:
                etype = "CCTV Footage"
                edesc = f"CCTV footage recorded at scene for case {case_id} revealing suspect appearance."
            elif crime_type in ["Burglary", "Vehicle Theft"]:
                etype = random.choice(["Tool Marks", "Fingerprint"])
                edesc = f"Forensic fingerprints recovered from scene matching database profile."
            elif crime_type == "Cybercrime":
                etype = "Digital Logs"
                edesc = f"Encrypted web server access logs and IP trace for case {case_id}."
            else:
                etype = random.choice(ev_types)
                edesc = f"Forensic evidence collected from crime scene for case {case_id}."

            conf = round(random.uniform(0.65, 0.98), 2)
            evidences.append((ev_id, case_id, etype, edesc, conf))

    cursor.executemany("""
        INSERT INTO evidence_records VALUES (?, ?, ?, ?, ?)
    """, evidences)

    conn.commit()
    conn.close()
    print(f"[OK] Synthetic Criminal Database created at: {DB_PATH}")

if __name__ == "__main__":
    create_and_seed_db()
