import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
CSV_PATH = os.path.join(DATA_DIR, "student_demo.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "previous_percentage", "attendance", "study_hours", "projects",
    "internships", "certifications", "coding_skill", "technical_skill",
    "communication_skill", "aptitude_score"
]

def make_demo_data(n=2500, seed=42):
    """Generate synthetic demo rows only; this is not real student/placement data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "previous_percentage": rng.uniform(35, 100, n).round(1),
        "attendance": rng.uniform(35, 100, n).round(1),
        "study_hours": rng.uniform(0, 10, n).round(1),
        "projects": rng.integers(0, 8, n),
        "internships": rng.integers(0, 4, n),
        "certifications": rng.integers(0, 8, n),
        "coding_skill": rng.uniform(1, 10, n).round(1),
        "technical_skill": rng.uniform(1, 10, n).round(1),
        "communication_skill": rng.uniform(1, 10, n).round(1),
        "aptitude_score": rng.uniform(10, 100, n).round(1),
    })
    score = (
        df.previous_percentage * .22 + df.attendance * .12 +
        np.minimum(df.study_hours * 10, 100) * .08 +
        np.minimum(df.projects * 12, 100) * .10 +
        np.minimum(df.internships * 25, 100) * .12 +
        np.minimum(df.certifications * 12, 100) * .05 +
        df.coding_skill * 1.0 + df.technical_skill * 1.0 +
        df.communication_skill * .5 + df.aptitude_score * .06 +
        rng.normal(0, 7, n)
    ).clip(0, 100)
    df["performance_label"] = np.where(
        score >= 70, "GOOD",
        np.where(score >= 50, "AVERAGE", "NEEDS IMPROVEMENT")
    )
    # Synthetic label for demonstrating the complete workflow only.
    prob = 1 / (1 + np.exp(-(score - 58) / 11))
    df["placed"] = rng.binomial(1, prob)
    return df

def main():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        required = FEATURES + ["performance_label", "placed"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("CSV missing required columns: " + ", ".join(missing))
        print("Using CSV:", CSV_PATH)
    else:
        df = make_demo_data()
        df.to_csv(CSV_PATH, index=False)
        print("Created synthetic demo CSV:", CSV_PATH)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_perf = df["performance_label"].astype(str)
    y_place = pd.to_numeric(df["placed"], errors="coerce").fillna(0).astype(int)
    X_train, X_test, yp_train, yp_test, yl_train, yl_test = train_test_split(
        X, y_perf, y_place, test_size=0.2, random_state=42, stratify=y_perf
    )

    def make_model():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150, random_state=42, class_weight="balanced"
            ))
        ])

    performance_model = make_model()
    placement_model = make_model()
    performance_model.fit(X_train, yp_train)
    placement_model.fit(X_train, yl_train)

    metrics = {
        "rows": int(len(df)),
        "performance_accuracy": round(float(accuracy_score(yp_test, performance_model.predict(X_test))), 4),
        "placement_accuracy": round(float(accuracy_score(yl_test, placement_model.predict(X_test))), 4),
        "warning": "Synthetic demo data metrics are not real-world model evidence."
    }
    joblib.dump(performance_model, os.path.join(MODEL_DIR, "performance_model.joblib"))
    joblib.dump(placement_model, os.path.join(MODEL_DIR, "placement_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Training complete. Metrics:", metrics)

if __name__ == "__main__":
    main()
