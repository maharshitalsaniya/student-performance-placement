# Student Performance & Placement Prediction

Responsive Python Flask app using pandas, scikit-learn, and SQL.

## Windows
Install Python 3.10 or newer. Extract the ZIP and open Command Prompt in the project folder:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_models.py
python app.py
```

Open `http://127.0.0.1:5000`.

## macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
python app.py
```

## Model data warning
No real dataset was included in the supplied team guide. The training script creates synthetic sample data so you can run and demonstrate the complete app. Predictions and accuracy scores from synthetic data are not reliable real-world placement evidence. For a real project, replace `data/student_demo.csv` with your approved dataset containing the ten feature columns plus `performance_label` and `placed`, and confirm the targets with the team.

SQLite is used automatically by the app. `schema.sql` gives equivalent MySQL tables, but the current app connection uses SQLite.

Features: `previous_percentage`, `attendance`, `study_hours`, `projects`, `internships`, `certifications`, `coding_skill`, `technical_skill`, `communication_skill`, `aptitude_score`.

The app listens on port 5000 and may be reached by other devices on the same Wi-Fi using the host computer's local IP, if firewall rules permit. Do not expose this development MVP directly to the public internet.
