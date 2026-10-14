from flask import Flask, render_template, request, redirect, url_for, flash
import os, sys, sqlite3, subprocess
from datetime import datetime
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
DB_PATH = os.path.join(ROOT, "student_app.db")
PERF_PATH = os.path.join(MODEL_DIR, "performance_model.joblib")
PLACE_PATH = os.path.join(MODEL_DIR, "placement_model.joblib")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-before-public-deployment")
FEATURES = [
    "previous_percentage", "attendance", "study_hours", "projects",
    "internships", "certifications", "coding_skill", "technical_skill",
    "communication_skill", "aptitude_score"
]

def load_models():
    if not (os.path.exists(PERF_PATH) and os.path.exists(PLACE_PATH)):
        run = subprocess.run([sys.executable, os.path.join(ROOT, "train_models.py")],
                             cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError("Model training failed. Run python train_models.py. " + run.stderr)
    return joblib.load(PERF_PATH), joblib.load(PLACE_PATH)

performance_model, placement_model = load_models()

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_name TEXT NOT NULL,
            previous_percentage REAL NOT NULL, attendance REAL NOT NULL,
            study_hours REAL NOT NULL, projects INTEGER NOT NULL,
            internships INTEGER NOT NULL, certifications INTEGER NOT NULL,
            coding_skill REAL NOT NULL, technical_skill REAL NOT NULL,
            communication_skill REAL NOT NULL, aptitude_score REAL NOT NULL,
            created_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER NOT NULL,
            performance_label TEXT NOT NULL, placement_probability REAL NOT NULL,
            placement_status TEXT NOT NULL, strong_areas TEXT NOT NULL,
            weak_areas TEXT NOT NULL, recommendations TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id))""")

def read_form(form):
    name = form.get("student_name", "").strip()
    if not name:
        raise ValueError("Please enter the student's name.")
    rules = {
        "previous_percentage": (0, 100, float), "attendance": (0, 100, float),
        "study_hours": (0, 24, float), "projects": (0, 100, int),
        "internships": (0, 50, int), "certifications": (0, 100, int),
        "coding_skill": (0, 10, float), "technical_skill": (0, 10, float),
        "communication_skill": (0, 10, float), "aptitude_score": (0, 100, float)
    }
    values = {}
    for key, (low, high, cast) in rules.items():
        try:
            value = cast(form.get(key, "").strip())
        except (ValueError, TypeError):
            raise ValueError("Enter a valid number for " + key.replace("_", " ") + ".")
        if not low <= value <= high:
            raise ValueError(key.replace("_", " ").title() + f" must be between {low} and {high}.")
        values[key] = value
    return name, values

def get_areas(v):
    checks = {
        "Academics": v["previous_percentage"] >= 70,
        "Attendance": v["attendance"] >= 80,
        "Study routine": v["study_hours"] >= 2,
        "Projects": v["projects"] >= 2,
        "Internships": v["internships"] >= 1,
        "Certifications": v["certifications"] >= 1,
        "Coding skills": v["coding_skill"] >= 7,
        "Technical skills": v["technical_skill"] >= 7,
        "Communication": v["communication_skill"] >= 7,
        "Aptitude": v["aptitude_score"] >= 70
    }
    strong = [k for k, val in checks.items() if val]
    weak = [k for k, val in checks.items() if not val]
    tips = []
    if v["projects"] < 2: tips.append("Build at least two practical projects.")
    if v["internships"] < 1: tips.append("Look for an internship or practical experience.")
    if v["communication_skill"] < 7: tips.append("Practise speaking and interview answers.")
    if v["technical_skill"] < 7: tips.append("Follow a focused technical learning plan.")
    if v["coding_skill"] < 7: tips.append("Solve coding problems and revise programming basics.")
    if v["aptitude_score"] < 70: tips.append("Practise quantitative aptitude and reasoning.")
    if v["attendance"] < 80: tips.append("Improve attendance and study routine.")
    if v["previous_percentage"] < 70: tips.append("Revise core subjects to improve academic results.")
    if not tips: tips.append("Keep practising, building projects, and preparing for interviews.")
    return strong, weak, tips

def predict(v):
    # pandas DataFrame keeps model feature names and order consistent.
    row = pd.DataFrame([{feature: v[feature] for feature in FEATURES}], columns=FEATURES)
    performance = str(performance_model.predict(row)[0])
    probs = placement_model.predict_proba(row)[0]
    classes = list(placement_model.named_steps["model"].classes_)
    probability = round(float(probs[classes.index(1)]) * 100, 1) if 1 in classes else 0.0
    status = "LIKELY" if probability >= 65 else ("POSSIBLE" if probability >= 45 else "NEEDS PREPARATION")
    strong, weak, tips = get_areas(v)
    return performance, probability, status, strong, weak, tips

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def make_prediction():
    try:
        name, values = read_form(request.form)
        performance, probability, status, strong, weak, tips = predict(values)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db() as conn:
            cur = conn.execute("""INSERT INTO students
                (student_name, previous_percentage, attendance, study_hours, projects,
                 internships, certifications, coding_skill, technical_skill,
                 communication_skill, aptitude_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, *(values[f] for f in FEATURES), now))
            student_id = cur.lastrowid
            conn.execute("""INSERT INTO predictions
                (student_id, performance_label, placement_probability, placement_status,
                 strong_areas, weak_areas, recommendations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (student_id, performance, probability, status, ", ".join(strong) or "None",
                 ", ".join(weak) or "None", "\n".join(tips), now))
        return render_template("result.html", name=name, values=values,
            performance=performance, probability=probability, status=status,
            strong=strong, weak=weak, suggestions=tips)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

@app.route("/history")
def history():
    with db() as conn:
        rows = conn.execute("""SELECT s.student_name, s.created_at,
            p.performance_label, p.placement_probability, p.placement_status
            FROM students s JOIN predictions p ON p.student_id=s.id
            ORDER BY s.id DESC LIMIT 100""").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/health")
def health():
    return {"status": "ok", "database": "SQLite", "ml": "scikit-learn", "data": "pandas"}

init_db()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
