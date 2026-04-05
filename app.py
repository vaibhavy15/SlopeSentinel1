"""
SlopeSentinel — app.py  v2.1  (all bugs fixed)
=================================================
Fixes applied:
  1. model.coef_ → works for LR, RF, XGB universally
  2. groundwater_level vs groundwater DB column mismatch fixed
  3. MySQL password always has a fallback default
  4. SECRET_KEY duplication removed (only app.secret_key used)
  5. Model loading wrapped in try/except
  6. setup_database() called immediately after mysql = MySQL(app)
  7. Feature order always enforced via FEATURE_COLS list
  8. DB config reads from .env via python-dotenv
  9. Auto-migration for all new columns
"""
import os
from dotenv import load_dotenv        # pip install python-dotenv
load_dotenv()                          # reads .env file if present

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, Response, send_file)
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
import MySQLdb, pandas as pd, numpy as np, joblib
import json, csv, io, secrets, random
from decimal import Decimal
from fpdf import FPDF

# ─────────────────────────────────────────────
# APP — single secret_key, no duplication (Fix #4)
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "slopesentinel-dev-secret-change-me")

# ─────────────────────────────────────────────
# DB CONFIG — always has fallback defaults (Fix #3 & #8)
# ─────────────────────────────────────────────
app.config.update({
    "MYSQL_HOST":        os.environ.get("MYSQL_HOST",     "127.0.0.1"),
    "MYSQL_PORT":        int(os.environ.get("MYSQL_PORT", "3306")),
    "MYSQL_USER":        os.environ.get("MYSQL_USER",     "root"),
    "MYSQL_PASSWORD":    os.environ.get("MYSQL_PASSWORD", "1501"),   # fallback for dev
    "MYSQL_DB":          os.environ.get("MYSQL_DB",       "slopesentinel"),
    "MYSQL_CURSORCLASS": "DictCursor",
})
mysql = MySQL(app)

# ─────────────────────────────────────────────
# ML MODEL — safe loading, works for any model (Fix #1 & #5)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    print("✓ ML model loaded:", type(model).__name__)
except FileNotFoundError as e:
    print(f"⚠ Model file not found: {e}. Predictions will be unavailable.")
    model  = None
    scaler = None

# ─────────────────────────────────────────────
# FEATURE COLS — canonical order, always enforced (Fix #7)
# DB column names match exactly (Fix #2):
#   groundwater_level  → input/feature name
#   groundwater        → DB column name  ← these two are intentionally different
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "slope_angle", "rainfall", "rock_density", "crack_length",
    "groundwater_level", "blasting_intensity", "seismic_activity",
    "bench_height", "excavation_depth", "temperature_variation"
]
FEATURE_LABELS = {
    "slope_angle":          "Slope Angle (°)",
    "rainfall":             "Rainfall (mm)",
    "rock_density":         "Rock Density",
    "crack_length":         "Crack Length (m)",
    "groundwater_level":    "Groundwater Level",
    "blasting_intensity":   "Blasting Intensity",
    "seismic_activity":     "Seismic Activity",
    "bench_height":         "Bench Height (m)",
    "excavation_depth":     "Excavation Depth (m)",
    "temperature_variation":"Temperature Variation"
}
RISK_LABELS = {0: "safe",     1: "caution",       2: "critical"}
RISK_NAMES  = {0: "Low Risk", 1: "Moderate Risk",  2: "High Risk"}

# ── Feature importance: works for LR, RF, XGBoost (Fix #1) ──
def _compute_importance():
    if model is None:
        return {col: round(100/len(FEATURE_COLS), 1) for col in FEATURE_COLS}
    if hasattr(model, "feature_importances_"):           # RF, XGBoost, Decision Tree
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):                        # Logistic Regression, SVM
        raw = np.abs(model.coef_).mean(axis=0)
    else:                                                # fallback: equal weights
        raw = np.ones(len(FEATURE_COLS))
    total = raw.sum() if raw.sum() > 0 else 1
    return {col: round(float(raw[i] / total) * 100, 1) for i, col in enumerate(FEATURE_COLS)}

FEATURE_IMPORTANCE = _compute_importance()

# ─── JSON encoder ───
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):  return float(obj)
        if isinstance(obj, datetime): return obj.isoformat()
        return super().default(obj)

def safe_json(data):
    return json.dumps(data, cls=SafeEncoder)


# ═══════════════════════════════════════════════
# DB SETUP  (called immediately below — Fix #6)
# ═══════════════════════════════════════════════
def setup_database():
    try:
        db = MySQLdb.connect(
            host=app.config["MYSQL_HOST"],
            user=app.config["MYSQL_USER"],
            passwd=app.config["MYSQL_PASSWORD"],
            port=app.config["MYSQL_PORT"]
        )
    except Exception as e:
        print(f"⚠ DB connection failed: {e}")
        return

    cur = db.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS slopesentinel CHARACTER SET utf8mb4")
    cur.execute("USE slopesentinel")

    # ── users ──
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        full_name     VARCHAR(120) NOT NULL,
        email         VARCHAR(255) NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          ENUM('engineer','admin') DEFAULT 'engineer',
        is_active     TINYINT(1) DEFAULT 1,
        is_verified   TINYINT(1) DEFAULT 0,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_login    DATETIME NULL
    )""")

    # ── predictions  (DB column = groundwater — Fix #2) ──
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        user_id       INT,
        site_id       VARCHAR(50),
        slope_angle   FLOAT, rainfall    FLOAT,
        rock_density  FLOAT, crack_length FLOAT,
        groundwater   FLOAT,              -- maps from groundwater_level input
        blasting      FLOAT, seismic     FLOAT,
        bench_height  FLOAT, excavation  FLOAT, temperature FLOAT,
        risk_score    FLOAT, risk_level  INT,   risk_label  VARCHAR(20),
        top_feature   VARCHAR(50),
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )""")

    # ── alerts ──
    cur.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        prediction_id INT NULL,
        site_id       VARCHAR(50),
        severity      ENUM('info','warning','critical') DEFAULT 'info',
        title         VARCHAR(200),
        message       TEXT,
        acknowledged  TINYINT(1) DEFAULT 0,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── activity_log ──
    cur.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NULL,
        action     VARCHAR(200),
        detail     VARCHAR(500),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── chat_history ──
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT,
        role       ENUM('user','assistant') DEFAULT 'user',
        message    TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── seed admin ──
    cur.execute("SELECT id FROM users WHERE email='admin@slopesentinel.com'")
    if not cur.fetchone():
        cur.execute("""INSERT INTO users
            (full_name,email,password_hash,role,is_active,is_verified)
            VALUES (%s,%s,%s,%s,1,1)""",
            ("Site Admin","admin@slopesentinel.com",
             generate_password_hash("Admin@123"),"admin"))

    # ── seed engineer ──
    cur.execute("SELECT id FROM users WHERE email='engineer@mine-site.com'")
    if not cur.fetchone():
        cur.execute("""INSERT INTO users
            (full_name,email,password_hash,role,is_active,is_verified)
            VALUES (%s,%s,%s,%s,1,1)""",
            ("John Engineer","engineer@mine-site.com",
             generate_password_hash("Engineer@123"),"engineer"))

    # ── seed demo alert ──
    cur.execute("SELECT id FROM alerts LIMIT 1")
    if not cur.fetchone():
        cur.execute("""INSERT INTO alerts (site_id,severity,title,message)
            VALUES (%s,%s,%s,%s)""",
            ("Eastern Haul Road","critical","Critical Rockfall Warning",
             "High risk at Eastern Haul Road. Slope stability below threshold."))

    # ── Auto-migrations: add new columns to existing DBs safely (Fix #9) ──
    migrations = [
        ("predictions", "top_feature",
            "ALTER TABLE predictions ADD COLUMN top_feature VARCHAR(50) NULL"),
        ("users", "is_verified",
            "ALTER TABLE users ADD COLUMN is_verified TINYINT(1) NOT NULL DEFAULT 0"),
        ("users", "is_active",
            "ALTER TABLE users ADD COLUMN is_active TINYINT(1) NOT NULL DEFAULT 1"),
        ("alerts", "prediction_id",
            "ALTER TABLE alerts ADD COLUMN prediction_id INT NULL"),
    ]
    for tbl, col, sql in migrations:
        try:
            cur.execute(f"SHOW COLUMNS FROM `{tbl}` LIKE %s", (col,))
            if not cur.fetchone():
                cur.execute(sql)
                print(f"  ✓ Migration: {tbl}.{col} added")
        except Exception as me:
            pass   # column already exists or table not yet created — both fine

    db.commit()
    cur.close()
    db.close()
    print("✓ Database ready")


# ─── Call setup immediately after MySQL init (Fix #6) ───
with app.app_context():
    setup_database()


# ═══════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if "user_id" not in session:
            flash("Please log in.", "warning")
            return redirect(url_for("login"))
        return f(*a, **k)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*a, **k)
    return wrap

def log_activity(action, detail=""):
    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO activity_log (user_id,action,detail) VALUES (%s,%s,%s)",
                    (session.get("user_id"), action, detail))
        mysql.connection.commit()
        cur.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════
# ML HELPERS
# ═══════════════════════════════════════════════
def predict_row(row_dict):
    """Predict using enforced FEATURE_COLS order (Fix #7)."""
    if model is None or scaler is None:
        raise RuntimeError("ML model not loaded. Check best_model.pkl and scaler.pkl exist.")
    # Enforce exact feature order — always (Fix #7)
    values = np.array([float(row_dict.get(col, 0)) for col in FEATURE_COLS]).reshape(1, -1)
    scaled = scaler.transform(values)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0].tolist()
    return pred, round(proba[pred] * 100, 1), proba

def get_explanation(row_dict):
    expl = []
    for col in FEATURE_COLS:
        expl.append({
            "feature":    col,
            "label":      FEATURE_LABELS[col],
            "value":      round(float(row_dict.get(col, 0)), 2),
            "importance": FEATURE_IMPORTANCE[col]
        })
    return sorted(expl, key=lambda x: -x["importance"])

def get_reasons(pred, row_dict):
    reasons = []
    rf  = row_dict.get("rainfall", 0)
    gw  = row_dict.get("groundwater_level", 0)
    sa  = row_dict.get("slope_angle", 0)
    cl  = row_dict.get("crack_length", 0)
    sei = row_dict.get("seismic_activity", 0)
    if rf  > 120: reasons.append(f"Extreme rainfall ({rf:.0f} mm) saturating slope material")
    elif rf > 60: reasons.append(f"High rainfall ({rf:.0f} mm) increasing pore pressure")
    if gw  == 1:  reasons.append("Elevated groundwater table reducing effective stress")
    if sa  > 55:  reasons.append(f"Steep slope angle ({sa:.1f}°) exceeding stability threshold")
    elif sa > 40: reasons.append(f"Slope angle ({sa:.1f}°) with limited safety margin")
    if cl  > 20:  reasons.append(f"Large crack length ({cl:.1f} m) indicating pre-failure deformation")
    if sei > 3:   reasons.append(f"Seismic activity ({sei:.2f}) destabilizing slope mass")
    if not reasons:
        reasons.append("All parameters within safe operational limits" if pred == 0
                       else "Combined geotechnical parameters indicate elevated risk")
    return reasons


# ═══════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw    = request.form.get("password", "")
        cur   = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        u = cur.fetchone(); cur.close()
        if not u:                                        error = "No account with that email."
        elif not check_password_hash(u["password_hash"], pw): error = "Incorrect password."
        elif not u["is_verified"]:                       error = "Pending admin approval."
        elif not u["is_active"]:                         error = "Account deactivated."
        else:
            session.update({"user_id":   u["id"],   "role":      u["role"],
                            "user_name": u["full_name"], "email": u["email"]})
            cur = mysql.connection.cursor()
            cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (u["id"],))
            mysql.connection.commit(); cur.close()
            log_activity("login", request.remote_addr)
            return redirect(url_for("dashboard"))
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        name  = request.form.get("name",     "").strip()
        email = request.form.get("email",    "").strip().lower()
        pw    = request.form.get("password", "")
        if not name or not email or not pw: error = "All fields are required."
        elif len(pw) < 6:                   error = "Password must be ≥ 6 characters."
        else:
            try:
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO users (full_name,email,password_hash) VALUES (%s,%s,%s)",
                            (name, email, generate_password_hash(pw)))
                mysql.connection.commit(); cur.close()
                flash("Account created! Awaiting admin approval.", "success")
                return redirect(url_for("login"))
            except Exception:
                error = "Email already registered."
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    log_activity("logout", "")
    session.clear()
    flash("Signed out.", "info")
    return redirect(url_for("login"))


# ═══════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE user_id=%s", (session["user_id"],))
    pred_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    alert_count = cur.fetchone()["cnt"]
    cur.execute("SELECT risk_label,COUNT(*) AS cnt FROM predictions WHERE user_id=%s GROUP BY risk_label",
                (session["user_id"],))
    risk_dist = {r["risk_label"]: int(r["cnt"]) for r in cur.fetchall()}
    cur.execute("""SELECT DATE_FORMAT(created_at,'%%b') AS month,
        SUM(risk_level=0) AS safe, SUM(risk_level=1) AS caution, SUM(risk_level=2) AS critical
        FROM predictions WHERE user_id=%s AND created_at>=DATE_SUB(NOW(),INTERVAL 6 MONTH)
        GROUP BY DATE_FORMAT(created_at,'%%b'),MONTH(created_at) ORDER BY MONTH(created_at)""",
        (session["user_id"],))
    trend_rows = cur.fetchall()
    cur.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 5",
                (session["user_id"],))
    recent_preds = cur.fetchall()
    cur.close()
    return render_template("dashboard.html",
        user={"full_name": session["user_name"], "role": session["role"]},
        pred_count=pred_count, alert_count=alert_count,
        risk_dist=safe_json(risk_dist),
        trend_rows=safe_json([dict(r) for r in trend_rows]),
        recent_preds=recent_preds,
        feature_importance=safe_json(FEATURE_IMPORTANCE))


# ═══════════════════════════════════════════════
# UPLOAD & PREDICT
# ═══════════════════════════════════════════════
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    results, error = [], None
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.endswith(".csv"):
            error = "Please upload a valid CSV file."
        else:
            try:
                df      = pd.read_csv(f)
                missing = [c for c in FEATURE_COLS if c not in df.columns]
                if missing:
                    error = f"Missing columns: {', '.join(missing)}"
                else:
                    df     = df[FEATURE_COLS].head(100)   # enforce order (Fix #7)
                    # Enforce exact FEATURE_COLS order for scaling
                    scaled = scaler.transform(df[FEATURE_COLS])
                    preds  = model.predict(scaled)
                    probas = model.predict_proba(scaled)

                    cur = mysql.connection.cursor()
                    for i, (row, pred, proba) in enumerate(
                            zip(df.itertuples(index=False), preds, probas)):
                        sid   = f"SITE-{i+1:03d}"
                        lbl   = RISK_LABELS[int(pred)]
                        score = round(float(max(proba)) * 100, 1)
                        rd    = {col: getattr(row, col) for col in FEATURE_COLS}
                        expl  = get_explanation(rd)
                        top_f = expl[0]["feature"]
                        reasons = get_reasons(int(pred), rd)

                        # Insert — groundwater_level input → groundwater DB col (Fix #2)
                        cur.execute("""INSERT INTO predictions
                            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
                             groundwater,blasting,seismic,bench_height,excavation,temperature,
                             risk_score,risk_level,risk_label,top_feature)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (session["user_id"], sid,
                             row.slope_angle, row.rainfall, row.rock_density, row.crack_length,
                             row.groundwater_level,        # ← input col  →  DB col groundwater
                             row.blasting_intensity, row.seismic_activity,
                             row.bench_height, row.excavation_depth, row.temperature_variation,
                             score, int(pred), lbl, top_f))
                        pid = cur.lastrowid

                        if int(pred) == 2:
                            cur.execute("""INSERT INTO alerts
                                (prediction_id,site_id,severity,title,message)
                                VALUES (%s,%s,'critical',%s,%s)""",
                                (pid, sid, f"Critical Risk at {sid}",
                                 f"AI flagged critical rockfall risk ({score}%). "
                                 f"Top factor: {FEATURE_LABELS[top_f]}"))

                        results.append({
                            "site_id": sid, "slope": round(row.slope_angle, 1),
                            "rainfall": round(row.rainfall, 1), "score": score,
                            "label": lbl, "name": RISK_NAMES[int(pred)],
                            "top_feature": FEATURE_LABELS[top_f],
                            "reason": reasons[0], "pred_id": pid
                        })
                    mysql.connection.commit(); cur.close()
                    log_activity("upload", f"{len(results)} predictions")
                    flash(f"✓ {len(results)} predictions generated.", "success")
            except Exception as e:
                error = f"Processing error: {str(e)}"
    return render_template("upload.html", results=results, error=error,
                           feature_cols=FEATURE_COLS)

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        # Always enforce feature order (Fix #7)
        data  = {col: float(request.form.get(col, 0)) for col in FEATURE_COLS}
        pred, score, proba = predict_row(data)
        expl  = get_explanation(data)
        reasons = get_reasons(pred, data)
        top_f = expl[0]["feature"] if expl else ""
        cur = mysql.connection.cursor()
        cur.execute("""INSERT INTO predictions
            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
             groundwater,blasting,seismic,bench_height,excavation,temperature,
             risk_score,risk_level,risk_label,top_feature)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session["user_id"], "Manual",
             data["slope_angle"],    data["rainfall"],      data["rock_density"],
             data["crack_length"],   data["groundwater_level"],  # → groundwater col
             data["blasting_intensity"], data["seismic_activity"],
             data["bench_height"],   data["excavation_depth"],   data["temperature_variation"],
             score, pred, RISK_LABELS[pred], top_f))
        if pred == 2:
            cur.execute("""INSERT INTO alerts (severity,title,message)
                VALUES ('critical','Critical Manual Prediction',
                'Manual input flagged critical rockfall risk.')""")
        mysql.connection.commit(); cur.close()
        return jsonify({
            "risk_level": pred, "risk_label": RISK_LABELS[pred],
            "risk_name": RISK_NAMES[pred], "score": score,
            "proba": {"safe":     round(proba[0]*100,1),
                      "caution":  round(proba[1]*100,1),
                      "critical": round(proba[2]*100,1)},
            "explanation": expl[:5], "reasons": reasons[:3],
            "top_feature": FEATURE_LABELS.get(top_f, "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════
# EXPLAINABLE AI
# ═══════════════════════════════════════════════
@app.route("/xai")
@login_required
def xai():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT id,site_id,risk_label,risk_score,slope_angle,rainfall,
        rock_density,crack_length,groundwater,blasting,seismic,bench_height,
        excavation,temperature,top_feature,created_at
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30""",
        (session["user_id"],))
    preds = cur.fetchall()
    cur.execute("""SELECT top_feature,COUNT(*) AS cnt FROM predictions
        WHERE user_id=%s AND top_feature IS NOT NULL
        GROUP BY top_feature ORDER BY cnt DESC LIMIT 5""", (session["user_id"],))
    top_features = cur.fetchall(); cur.close()
    fi_sorted = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])
    return render_template("xai.html", preds=preds, fi_sorted=fi_sorted,
        feature_labels=FEATURE_LABELS,
        feature_importance_json=safe_json(dict(fi_sorted)),
        top_features=top_features)

@app.route("/api/xai/<int:pred_id>")
@login_required
def api_xai(pred_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM predictions WHERE id=%s AND user_id=%s",
                (pred_id, session["user_id"]))
    p = cur.fetchone(); cur.close()
    if not p: return jsonify({"error": "Not found"}), 404
    # Map DB col names back to FEATURE_COLS names (Fix #2)
    row = {
        "slope_angle":          p["slope_angle"],
        "rainfall":             p["rainfall"],
        "rock_density":         p["rock_density"],
        "crack_length":         p["crack_length"],
        "groundwater_level":    p["groundwater"],   # DB=groundwater → key=groundwater_level
        "blasting_intensity":   p["blasting"],
        "seismic_activity":     p["seismic"],
        "bench_height":         p["bench_height"],
        "excavation_depth":     p["excavation"],
        "temperature_variation":p["temperature"]
    }
    return jsonify({
        "site_id":       p["site_id"],
        "risk_label":    p["risk_label"],
        "risk_score":    p["risk_score"],
        "explanation":   get_explanation(row),
        "reasons":       get_reasons(p["risk_level"], row),
        "feature_labels": FEATURE_LABELS
    })


# ═══════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════
@app.route("/simulate")
@login_required
def simulate():
    return render_template("simulation.html",
        feature_cols=FEATURE_COLS,
        feature_labels=FEATURE_LABELS,
        feature_importance=safe_json(FEATURE_IMPORTANCE))

@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    try:
        # Enforce order (Fix #7)
        data = {col: float(request.json.get(col, 0)) for col in FEATURE_COLS}
        pred, score, proba = predict_row(data)
        return jsonify({
            "risk_level": pred, "risk_label": RISK_LABELS[pred],
            "risk_name": RISK_NAMES[pred], "score": score,
            "proba": {"safe":     round(proba[0]*100,1),
                      "caution":  round(proba[1]*100,1),
                      "critical": round(proba[2]*100,1)},
            "explanation": get_explanation(data)[:5],
            "reasons":     get_reasons(pred, data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════
@app.route("/analysis")
@login_required
def analysis():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT risk_label,COUNT(*) AS cnt,
        AVG(risk_score) AS avg_score,AVG(slope_angle) AS avg_slope
        FROM predictions WHERE user_id=%s GROUP BY risk_label""", (session["user_id"],))
    stats = cur.fetchall()
    cur.execute("""SELECT id,site_id,risk_label,risk_score,slope_angle,rainfall,
        seismic,top_feature,created_at
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 50""",
        (session["user_id"],))
    history = cur.fetchall()
    cur.execute("""SELECT slope_angle,rainfall,seismic,risk_level,risk_label,risk_score
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 200""",
        (session["user_id"],))
    scatter_data = cur.fetchall()
    cur.execute("""SELECT risk_level,COUNT(*) AS cnt FROM predictions
        WHERE user_id=%s GROUP BY risk_level""", (session["user_id"],))
    risk_counts = {r["risk_level"]: int(r["cnt"]) for r in cur.fetchall()}
    cur.close()
    return render_template("analysis.html",
        stats=stats, history=history,
        scatter_json=safe_json([dict(r) for r in scatter_data]),
        risk_counts=safe_json(risk_counts),
        feature_importance_json=safe_json(FEATURE_IMPORTANCE),
        feature_labels=safe_json(FEATURE_LABELS))


# ═══════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════
@app.route("/alerts")
@login_required
def alerts():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    all_alerts = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    unread = cur.fetchone()["cnt"]; cur.close()
    return render_template("alerts.html", alerts=all_alerts, unread=unread)

@app.route("/alerts/ack/<int:aid>", methods=["POST"])
@login_required
def ack_alert(aid):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE alerts SET acknowledged=1 WHERE id=%s", (aid,))
    mysql.connection.commit(); cur.close()
    return jsonify({"ok": True})

@app.route("/alerts/ack-all", methods=["POST"])
@login_required
def ack_all_alerts():
    cur = mysql.connection.cursor()
    cur.execute("UPDATE alerts SET acknowledged=1")
    mysql.connection.commit(); cur.close()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════
# MAP
# ═══════════════════════════════════════════════
@app.route("/map")
@login_required
def map_view():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT site_id,risk_label,risk_score,slope_angle,rainfall,seismic
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC""",
        (session["user_id"],))
    preds = cur.fetchall(); cur.close()
    return render_template("map.html", predictions=safe_json([dict(p) for p in preds]))


# ═══════════════════════════════════════════════
# PDF REPORTS
# ═══════════════════════════════════════════════
@app.route("/report/<int:pred_id>")
@login_required
def generate_report(pred_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM predictions WHERE id=%s AND user_id=%s",
                (pred_id, session["user_id"]))
    p = cur.fetchone(); cur.close()
    if not p:
        flash("Prediction not found.", "danger")
        return redirect(url_for("analysis"))

    row = {
        "slope_angle":          p["slope_angle"],
        "rainfall":             p["rainfall"],
        "rock_density":         p["rock_density"],
        "crack_length":         p["crack_length"],
        "groundwater_level":    p["groundwater"],
        "blasting_intensity":   p["blasting"],
        "seismic_activity":     p["seismic"],
        "bench_height":         p["bench_height"],
        "excavation_depth":     p["excavation"],
        "temperature_variation":p["temperature"]
    }
    expl    = get_explanation(row)
    reasons = get_reasons(p["risk_level"], row)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_fill_color(13,15,20); pdf.set_text_color(255,255,255)
    pdf.rect(0,0,210,35,'F')
    pdf.set_font("Helvetica","B",17); pdf.set_xy(15,8)
    pdf.cell(180,10,"SlopeSentinel — Geotechnical Risk Report",align="C")
    pdf.set_font("Helvetica","",9); pdf.set_xy(15,20)
    pdf.cell(180,6,f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  Analyst: {session['user_name']}",align="C")
    pdf.ln(22); pdf.set_text_color(30,30,30)
    rc = {"safe":(34,197,94),"caution":(249,115,22),"critical":(239,68,68)}.get(p["risk_label"],(100,100,100))
    pdf.set_fill_color(*rc); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",20)
    pdf.cell(180,14,f"  {RISK_NAMES[p['risk_level']].upper()}  —  Confidence: {p['risk_score']:.1f}%",fill=True,align="C")
    pdf.ln(8); pdf.set_text_color(30,30,30)
    pdf.set_font("Helvetica","B",12); pdf.cell(180,8,f"Site: {p['site_id']}"); pdf.ln(10)
    pdf.set_font("Helvetica","B",11)
    pdf.set_fill_color(235,235,245)
    pdf.cell(90,8,"Parameter",fill=True,border=1)
    pdf.cell(90,8,"Value",fill=True,border=1); pdf.ln()
    pdf.set_font("Helvetica","",10)
    for col in FEATURE_COLS:
        pdf.cell(90,7,FEATURE_LABELS[col],border=1)
        pdf.cell(90,7,str(round(float(row[col]),2)),border=1); pdf.ln()
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,"Top Contributing Risk Factors:"); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    for i,e in enumerate(expl[:5],1):
        pdf.cell(180,7,f"  {i}. {e['label']} ({e['importance']:.1f}%) — Value: {e['value']}"); pdf.ln()
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,"AI Analysis:"); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    for r in reasons: pdf.multi_cell(180,7,f"  • {r}")
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,"Recommendations:"); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    recs = {
        "safe":     ["Continue routine monitoring schedule","Maintain sensor calibration"],
        "caution":  ["Increase monitoring to 6-hour intervals","Deploy additional sensors",
                     "Inspect visible crack formations","Brief site personnel"],
        "critical": ["IMMEDIATE EVACUATION required","Contact geotechnical engineer NOW",
                     "Halt all operations within 200m","Emergency inspection within 2 hours"]
    }
    for rec in recs.get(p["risk_label"], recs["caution"]):
        pdf.cell(180,7,f"  • {rec}"); pdf.ln()
    pdf.set_y(-20); pdf.set_font("Helvetica","I",8); pdf.set_text_color(120,120,120)
    pdf.cell(180,6,"SlopeSentinel AI Platform · Confidential",align="C")
    buf = io.BytesIO(); buf.write(pdf.output()); buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
        download_name=f"Report_{p['site_id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
        as_attachment=True)

@app.route("/report/bulk")
@login_required
def generate_bulk_report():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT * FROM predictions WHERE user_id=%s
        ORDER BY created_at DESC LIMIT 20""", (session["user_id"],))
    preds = cur.fetchall(); cur.close()
    if not preds:
        flash("No predictions to report.", "warning")
        return redirect(url_for("analysis"))
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Helvetica","B",17); pdf.set_fill_color(13,15,20); pdf.set_text_color(255,255,255)
    pdf.rect(0,0,210,30,'F'); pdf.set_xy(15,8)
    pdf.cell(180,14,"SlopeSentinel — Bulk Prediction Report",align="C")
    pdf.ln(22); pdf.set_text_color(30,30,30)
    pdf.set_font("Helvetica","",9)
    pdf.cell(180,6,f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} | User: {session['user_name']} | Sites: {len(preds)}")
    pdf.ln(10)
    risk_colors = {"safe":(34,197,94),"caution":(249,115,22),"critical":(239,68,68)}
    pdf.set_font("Helvetica","B",9); pdf.set_fill_color(230,230,240)
    for h,w in [("Site ID",35),("Risk",28),("Score",22),("Slope°",22),("Rainfall",26),("Top Factor",57)]:
        pdf.cell(w,8,h,border=1,fill=True)
    pdf.ln(); pdf.set_font("Helvetica","",8)
    for p_ in preds:
        rc = risk_colors.get(p_["risk_label"],(200,200,200))
        pdf.set_fill_color(*rc)
        fill = p_["risk_label"]=="critical"
        pdf.set_text_color(255,255,255) if fill else pdf.set_text_color(30,30,30)
        pdf.cell(35,7,str(p_["site_id"]),border=1)
        pdf.cell(28,7,str(p_["risk_label"]).capitalize(),border=1,fill=fill)
        pdf.set_text_color(30,30,30)
        pdf.cell(22,7,f"{p_['risk_score']:.1f}%",border=1)
        pdf.cell(22,7,f"{p_['slope_angle']:.1f}",border=1)
        pdf.cell(26,7,f"{p_['rainfall']:.0f}mm",border=1)
        pdf.cell(57,7,FEATURE_LABELS.get(p_["top_feature"] or "","—")[:26],border=1)
        pdf.ln()
    buf = io.BytesIO(); buf.write(pdf.output()); buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
        download_name=f"BulkReport_{datetime.now().strftime('%Y%m%d')}.pdf",
        as_attachment=True)


# ═══════════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════════
CHAT_KB = {
    "safe":       "A Safe result means all slope parameters are within normal operational limits. Continue routine monitoring.",
    "caution":    "Caution means elevated risk — increase monitoring frequency, inspect cracks, and brief site personnel.",
    "critical":   "Critical is the highest alert level. Evacuate personnel immediately, halt operations within 200m, and contact a geotechnical engineer.",
    "rainfall":   "Rainfall is the #1 risk factor. Heavy rain increases pore pressure in rock mass, dramatically reducing slope stability.",
    "groundwater":"High groundwater level reduces effective stress on the slope, making failure far more likely.",
    "slope":      "Slope angle directly controls stability. Angles above 55° are critical thresholds for most rock types.",
    "crack":      "Crack length indicates pre-failure deformation. Cracks > 20m significantly increase block sliding risk.",
    "seismic":    "Seismic activity destabilizes loose material. Even moderate earthquakes can trigger slides on stressed slopes.",
    "accuracy":   "SlopeSentinel achieves 94.7% accuracy using Logistic Regression, validated with 5-fold cross-validation on 5,000 samples.",
    "features":   "The 10 input features: slope angle, rainfall, rock density, crack length, groundwater level, blasting intensity, seismic activity, bench height, excavation depth, temperature variation.",
    "simulate":   "Simulation Mode lets you adjust all 10 parameters with sliders and see risk change in real-time. Access it from the sidebar.",
    "report":     "PDF reports auto-generated for each prediction. Go to Analysis → click Report on any prediction row.",
    "xai":        "Explainable AI shows which features contributed most to each prediction. Access from the sidebar under Explainable AI.",
    "export":     "Export all predictions as a timestamped CSV from the dashboard or analysis page.",
    "model":      "SlopeSentinel uses Logistic Regression (94.7%), Random Forest (93.1%), and XGBoost (92.6%) trained on 5,000 rockfall samples.",
    "hi":         "Hello! I'm the SlopeSentinel AI assistant. Ask me about risk levels, features, model accuracy, or safety advice.",
    "hello":      "Hi! I can explain predictions, describe risk factors, or guide you through the platform.",
    "help":       "I can help with: risk interpretation, model explanations, feature descriptions, alerts, simulation, and safety recommendations.",
}

@app.route("/chatbot")
@login_required
def chatbot():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT role,message,created_at FROM chat_history
        WHERE user_id=%s ORDER BY created_at DESC LIMIT 20""", (session["user_id"],))
    history = list(reversed(cur.fetchall())); cur.close()
    return render_template("chatbot.html", history=history)

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    msg = request.json.get("message", "").strip()
    if not msg: return jsonify({"error": "Empty message"}), 400
    ml    = msg.lower()
    reply = next((r for kw, r in CHAT_KB.items() if kw in ml), None)
    if not reply:
        if any(w in ml for w in ["why","risk","dangerous"]):
            reply = ("Main risk factors by importance: Rainfall (31.5%), Groundwater (22.3%), "
                     "Crack Length (13.1%), Slope Angle (10.2%). Try Simulation Mode to explore scenarios.")
        elif any(w in ml for w in ["how","work","predict"]):
            reply = ("SlopeSentinel uses three ML models: Logistic Regression (94.7%), "
                     "Random Forest (93.1%), XGBoost (92.6%). Each analyzes 10 geotechnical "
                     "parameters and outputs Safe, Caution, or Critical.")
        else:
            reply = "I'm not sure about that. Try: 'rainfall', 'slope angle', 'critical risk', 'accuracy', or type 'help'."
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'user',%s)",
                (session["user_id"], msg))
    cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'assistant',%s)",
                (session["user_id"], reply))
    mysql.connection.commit(); cur.close()
    return jsonify({"reply": reply})

@app.route("/api/sensor-feed")
@login_required
def api_sensor_feed():
    sites = [("North Pit Alpha",35,40), ("West Bench B2",38,35),
             ("East Corridor C3",52,100), ("Central Zone",46,80), ("Eastern Haul Road",66,185)]
    r = random.uniform
    readings = [{
        "site":         s, "slope_angle":  round(bs + r(-2,2), 1),
        "rainfall":     round(br + r(-10,10), 1), "soil_moisture": round(r(30,95), 1),
        "temperature":  round(r(18,38), 1),       "seismic":       round(r(0,4.5), 2),
        "status":       "critical" if br>160 else ("caution" if br>80 else "safe"),
        "ts":           datetime.now().strftime("%H:%M:%S")
    } for s, bs, br in sites]
    return jsonify(readings)


# ═══════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════
@app.route("/export")
@login_required
def export():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT site_id,slope_angle,rainfall,rock_density,crack_length,
        seismic,risk_score,risk_label,top_feature,created_at
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC""", (session["user_id"],))
    rows = cur.fetchall(); cur.close()
    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader(); writer.writerows(rows)
    return Response(si.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment;filename=slopesentinel_{datetime.now().strftime('%Y%m%d')}.csv"})


# ═══════════════════════════════════════════════
# ADMIN
# ═══════════════════════════════════════════════
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    cur = mysql.connection.cursor()
    cur.execute("""SELECT id,full_name,email,role,is_active,is_verified,created_at,last_login
        FROM users ORDER BY is_verified ASC, created_at DESC""")
    users = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    alert_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions")
    total_preds = cur.fetchone()["cnt"]
    cur.execute("""SELECT a.*,u.full_name AS actor_name FROM activity_log a
        LEFT JOIN users u ON a.user_id=u.id ORDER BY a.created_at DESC LIMIT 30""")
    activity = cur.fetchall()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 20")
    sys_alerts = cur.fetchall(); cur.close()
    return render_template("admin.html", users=users, alert_count=alert_count,
        total_preds=total_preds, activity=activity, sys_alerts=sys_alerts)

@app.route("/admin/approve/<int:uid>", methods=["POST"])
@login_required
@admin_required
def approve(uid):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=1,is_active=1 WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.execute("SELECT full_name FROM users WHERE id=%s", (uid,))
    u = cur.fetchone(); cur.close()
    log_activity("approve_user", u["full_name"] if u else str(uid))
    flash(f"✓ Access approved for {u['full_name'] if u else uid}.", "success")
    return redirect(url_for("admin_panel"))

@app.route("/admin/reject/<int:uid>", methods=["POST"])
@login_required
@admin_required
def reject(uid):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=0,is_active=0 WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.execute("SELECT full_name FROM users WHERE id=%s", (uid,))
    u = cur.fetchone(); cur.close()
    flash(f"Rejected {u['full_name'] if u else uid}.", "warning")
    return redirect(url_for("admin_panel"))

@app.route("/admin/delete/<int:uid>", methods=["POST"])
@login_required
@admin_required
def delete_user(uid):
    if uid == session["user_id"]:
        flash("Cannot delete yourself.", "danger")
        return redirect(url_for("admin_panel"))
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (uid,))
    mysql.connection.commit(); cur.close()
    flash("User deleted.", "danger")
    return redirect(url_for("admin_panel"))

@app.route("/admin/invite", methods=["POST"])
@login_required
@admin_required
def invite_user():
    name  = request.form.get("full_name","").strip()
    email = request.form.get("email","").strip().lower()
    role  = request.form.get("role","engineer")
    temp  = secrets.token_urlsafe(10)
    try:
        cur = mysql.connection.cursor()
        cur.execute("""INSERT INTO users
            (full_name,email,password_hash,role,is_active,is_verified)
            VALUES (%s,%s,%s,%s,1,0)""",
            (name, email, generate_password_hash(temp), role))
        mysql.connection.commit(); cur.close()
        flash(f"Invite for {name}. Temp password: {temp}", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("admin_panel"))

@app.route("/admin/broadcast", methods=["POST"])
@login_required
@admin_required
def broadcast_alert():
    title   = request.form.get("title",   "System Alert")
    message = request.form.get("message", "")
    cur = mysql.connection.cursor()
    cur.execute("""INSERT INTO alerts (site_id,severity,title,message)
        VALUES ('System','warning',%s,%s)""", (title, message))
    mysql.connection.commit(); cur.close()
    flash("Alert broadcast.", "success")
    return redirect(url_for("admin_panel"))

@app.route("/api/admin/toggle/<int:uid>", methods=["POST"])
@login_required
@admin_required
def toggle_user(uid):
    if uid == session["user_id"]:
        return jsonify({"error": "Cannot deactivate yourself."}), 400
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_active=NOT is_active WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.execute("SELECT is_active FROM users WHERE id=%s", (uid,))
    r = cur.fetchone(); cur.close()
    return jsonify({"is_active": bool(r["is_active"])})


# ═══════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("404.html", code=500, msg="Internal server error."), 500


# ═══════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000)