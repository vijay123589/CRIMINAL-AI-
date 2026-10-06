# ─────────────────────────────────────────────
#  ml_engine.py – Scikit-Learn Machine Learning Models
#  1. K-Means Historical Crime Hotspot Clustering
#  2. Random Forest Offender Risk Category Classification
# ─────────────────────────────────────────────
from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from config import DB_PATH


# ─────────────────────────────────────────────
#  1. K-MEANS CRIME HOTSPOT CLUSTERING
# ─────────────────────────────────────────────
def run_kmeans_clustering(n_clusters: int = 4) -> dict:
    """
    Perform K-Means clustering on historical incident coordinates (latitude, longitude).
    Returns DataFrame with cluster labels and cluster centroid coordinates.
    """
    if not Path(DB_PATH).exists():
        return {"error": "Database file not found."}

    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT case_id, crime_type, district, latitude, longitude, severity, case_status
        FROM crime_incidents
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty or len(df) < n_clusters:
        return {"error": "Insufficient data for clustering."}

    X = df[["latitude", "longitude"]].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)
    df["cluster_label"] = df["cluster"].apply(lambda c: f"Cluster {c+1}")

    centroids = kmeans.cluster_centers_
    centroids_df = pd.DataFrame(centroids, columns=["latitude", "longitude"])
    centroids_df["cluster_name"] = [f"Hotspot Center {i+1}" for i in range(n_clusters)]

    return {
        "df": df,
        "centroids": centroids_df,
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 4),
    }


# ─────────────────────────────────────────────
#  2. RANDOM FOREST RISK CLASSIFICATION
# ─────────────────────────────────────────────
class RiskClassifierModel:
    """Random Forest Risk Classification Model for Demonstration Purposes."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoders = {}

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT 
                r.age,
                r.prior_convictions,
                r.gang_affiliation,
                i.severity AS crime_severity,
                i.crime_type AS crime_category,
                (SELECT COUNT(*) FROM crime_incidents sub WHERE sub.criminal_id = r.criminal_id) AS previous_case_count
            FROM criminal_records r
            LEFT JOIN crime_incidents i ON r.criminal_id = i.criminal_id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        def assign_risk(row):
            score = (row["prior_convictions"] * 12) + (row["previous_case_count"] * 8)
            if row["gang_affiliation"] != "None":
                score += 20
            if row["crime_severity"] in [2, 3]:  # High or Critical
                score += 15
            if score >= 35:
                return "High"
            elif score >= 18:
                return "Medium"
            else:
                return "Low"

        df["risk_category"] = df.apply(assign_risk, axis=1)

        cat_cols = ["gang_affiliation", "crime_severity", "crime_category"]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        X = df[["age", "prior_convictions", "crime_severity", "previous_case_count", "gang_affiliation", "crime_category"]]
        y = df["risk_category"]

        return X, y


def train_random_forest_model() -> dict:
    """Train Random Forest model and compute evaluation metrics."""
    if not Path(DB_PATH).exists():
        return {"error": "Database not found."}

    classifier = RiskClassifierModel()
    X, y = classifier.prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

    labels = ["Low", "Medium", "High"]
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])

    feat_names = X.columns.tolist()
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    return {
        "model": clf,
        "encoders": classifier.encoders,
        "accuracy": round(float(acc * 100), 2),
        "precision": round(float(prec * 100), 2),
        "recall": round(float(rec * 100), 2),
        "f1_score": round(float(f1 * 100), 2),
        "confusion_matrix": cm_df,
        "feature_importance": feat_df,
        "classes": labels,
    }


def predict_offender_risk(
    age: int,
    prior_convictions: int,
    gang_affiliation: str,
    crime_severity: str,
    previous_case_count: int,
    crime_category: str,
) -> dict:
    """Predict risk category for an interactive user test suspect."""
    try:
        model_info = train_random_forest_model()
        if "error" in model_info:
            return model_info

        clf = model_info["model"]
        encoders = model_info["encoders"]

        def safe_transform(col_name, val):
            le = encoders[col_name]
            if val in le.classes_:
                return le.transform([val])[0]
            else:
                return 0

        gang_enc = safe_transform("gang_affiliation", gang_affiliation)
        sev_enc = safe_transform("crime_severity", crime_severity)
        cat_enc = safe_transform("crime_category", crime_category)

        input_data = pd.DataFrame([{
            "age": age,
            "prior_convictions": prior_convictions,
            "crime_severity": sev_enc,
            "previous_case_count": previous_case_count,
            "gang_affiliation": gang_enc,
            "crime_category": cat_enc,
        }])

        pred_class = clf.predict(input_data)[0]
        probs = clf.predict_proba(input_data)[0]
        prob_dict = {cls: round(float(prob * 100), 1) for cls, prob in zip(clf.classes_, probs)}

        return {
            "predicted_risk": pred_class,
            "probabilities": prob_dict,
            "confidence": prob_dict.get(pred_class, 0.0),
        }
    except Exception as e:
        return {"error": str(e)}
