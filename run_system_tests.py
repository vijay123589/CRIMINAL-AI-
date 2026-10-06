# ─────────────────────────────────────────────
#  run_system_tests.py – Verification & Test Suite
# ─────────────────────────────────────────────
import sqlite3
import pandas as pd
from pathlib import Path
from database import get_connection, execute_query, get_database_summary
from sql_validator import validate_sql
from rag_engine import build_vector_store, retrieve_context, search_semantic_case_files
from ml_engine import run_kmeans_clustering, train_random_forest_model, predict_offender_risk

def run_all_tests():
    print("==================================================")
    print("   CRIMEINTEL AI — SYSTEM VERIFICATION TESTS")
    print("==================================================")
    
    # 1. Database connection & counts check (Dashboard Check)
    print("\n--- 1. DASHBOARD & DATABASE CHECKS (5 Checks) ---")
    summary = get_database_summary()
    print(f"[Check 1] Total Cases: {summary['total_cases']} (Pass: {summary['total_cases'] > 0})")
    print(f"[Check 2] Registered Criminals: {summary['total_criminals']} (Pass: {summary['total_criminals'] > 0})")
    print(f"[Check 3] Solved Cases: {summary['solved_cases']}")
    print(f"[Check 4] Unsolved Cases: {summary['unsolved_cases']}")
    print(f"[Check 5] DB Path Exists: {Path('criminal_database.db').exists()}")

    # 2. SQL Query Execution Tests (5 SQL Queries)
    print("\n--- 2. DIRECT SQL EXECUTION TESTS (5 Queries) ---")
    sql_tests = [
        "SELECT * FROM crime_incidents WHERE crime_type = 'Robbery'",
        "SELECT COUNT(*) AS cyber_count FROM crime_incidents WHERE crime_type = 'Cybercrime'",
        "SELECT * FROM crime_incidents WHERE case_status = 'Open' AND district = 'Chennai Central'",
        "SELECT * FROM crime_incidents WHERE severity IN ('High', 'Critical') AND incident_date LIKE '2024%'",
        "SELECT c.case_id, c.crime_type, cr.name AS suspect FROM crime_incidents c LEFT JOIN criminal_records cr ON c.criminal_id = cr.criminal_id WHERE c.case_id = 'C102'",
    ]

    for i, sql in enumerate(sql_tests, 1):
        rows, cols, err = execute_query(sql)
        status = "FAIL" if err else f"PASS ({len(rows)} rows)"
        print(f"[SQL Query {i}] {sql[:50]}... -> {status}")

    # 3. Case Detail Queries (5 Case-Detail Queries)
    print("\n--- 3. CASE DETAIL QUERIES (5 Queries) ---")
    case_ids = ["C101", "C102", "C103", "C104", "C105"]
    for cid in case_ids:
        rows, cols, err = execute_query(f"SELECT * FROM crime_incidents WHERE case_id = '{cid}'")
        print(f"[Case Detail {cid}] Found: {len(rows) == 1} | Type: {rows[0]['crime_type'] if rows else 'N/A'}")

    # 4. SQL Injection / Safety Validator Tests
    print("\n--- 4. SQL SAFETY & INJECTION TESTS ---")
    safety_tests = [
        ("SELECT * FROM crime_incidents", True),
        ("DROP TABLE crime_incidents", False),
        ("DELETE FROM criminal_records", False),
        ("INSERT INTO crime_incidents VALUES ('C999', 'Robbery')", False),
        ("UPDATE criminal_records SET status = 'At Large'", False),
    ]

    for sql, expected in safety_tests:
        valid, msg = validate_sql(sql)
        passed = (valid == expected)
        print(f"[Safety Check] '{sql}' -> Allowed: {valid} (Expected: {expected}) | Result: {'PASS' if passed else 'FAIL'}")

    # 5. Semantic / RAG Search Tests (5 Vector Queries)
    print("\n--- 5. SEMANTIC / RAG SEARCH TESTS (5 Queries) ---")
    build_vector_store()
    rag_queries = [
        "Find cases involving a person using a crowbar at night",
        "Find cases where CCTV evidence was important",
        "Find cases involving a stolen motorcycle",
        "Phishing attack targeting bank accounts",
        "Drug trafficking shipment near harbor",
    ]

    for i, rq in enumerate(rag_queries, 1):
        matches = search_semantic_case_files(rq, top_k=2)
        top_match = matches[0] if matches else None
        cid = top_match['case_id'] if top_match else 'None'
        score = top_match['similarity_score'] if top_match else '0%'
        print(f"[RAG Query {i}] '{rq}' -> Top Match Case: {cid} ({score})")

    # 6. K-Means Hotspot Test
    print("\n--- 6. K-MEANS HOTSPOT CLUSTERING TEST ---")
    km_res = run_kmeans_clustering(n_clusters=4)
    if "error" in km_res:
        print(f"[K-Means Test] FAIL: {km_res['error']}")
    else:
        print(f"[K-Means Test] PASS | Inertia: {km_res['inertia']} | Rows Clustered: {len(km_res['df'])}")

    # 7. Random Forest Risk Classifier Test
    print("\n--- 7. RANDOM FOREST RISK CLASSIFIER TEST ---")
    rf_res = train_random_forest_model()
    if "error" in rf_res:
        print(f"[Random Forest Test] FAIL: {rf_res['error']}")
    else:
        print(f"[Random Forest Test] PASS | Accuracy: {rf_res['accuracy']}% | F1: {rf_res['f1_score']}%")
        pred = predict_offender_risk(age=35, prior_convictions=4, gang_affiliation="Syndicate-X", crime_severity="High", previous_case_count=3, crime_category="Robbery")
        print(f"[Interactive Risk Predictor Test] Predicted: {pred['predicted_risk']} (Confidence: {pred['confidence']}%)")

    print("\n==================================================")
    print("   ALL CRIMEINTEL AI SYSTEM TESTS COMPLETED!")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
