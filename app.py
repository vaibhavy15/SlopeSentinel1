"""
SlopeSentinel — app.py  v4.1
=====================================
Database : NeonDB (serverless PostgreSQL via psycopg2)
AI Chat  : Google Gemini 2.5 Flash — restricted to slope/geotech topics only
Fixes    : multi_class sklearn>=1.5, FPDFUnicodeEncodingException (clean_pdf),
           groundwater_level vs groundwater DB mapping, feature order enforced
"""

import os
from dotenv import load_dotenv
load_dotenv()                           # reads .env first — must be before os.environ calls

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, Response, send_file, g)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import psycopg2, psycopg2.extras
import pandas as pd, numpy as np, joblib
import json, csv, io, secrets, random
from decimal import Decimal
from fpdf import FPDF

# ── Gemini setup (safe — won't crash if key missing) ──────────────────────────
try:
    from google import genai as _genai
    _GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    if _GEMINI_KEY:
        gemini_client = _genai.Client(api_key=_GEMINI_KEY)
        GEMINI_AVAILABLE = True
        print("[OK] Gemini client initialised")
    else:
        gemini_client  = None
        GEMINI_AVAILABLE = False
        print("[WARN] GEMINI_API_KEY not set — chatbot will use built-in KB")
except Exception as _ge:
    gemini_client  = None
    GEMINI_AVAILABLE = False
    print(f"[WARN] Gemini init failed: {_ge} — chatbot will use built-in KB")

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "slopesentinel-dev-secret-change-me")

# ─────────────────────────────────────────────
# NeonDB  (serverless PostgreSQL)
# Set in .env:
#   DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
# ─────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"
)

def get_db():
    """Per-request psycopg2 connection stored on Flask g."""
    if "db" not in g:
        g.db = psycopg2.connect(DATABASE_URL)
        g.db.autocommit = False
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def get_cursor():
    return get_db().cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ─────────────────────────────────────────────
# ML MODEL  (safe load + sklearn>=1.5 multi_class fix)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    print(f"[OK] Model loaded: {type(model).__name__}")
    # sklearn >= 1.5 removed 'multi_class' attr from LogisticRegression.
    # Old pickled models still carry it → AttributeError on predict_proba.
    if hasattr(model, "multi_class"):
        del model.multi_class
        print("[OK] multi_class removed (sklearn >= 1.5 compat)")
except FileNotFoundError as e:
    print(f"[WARN] Model file missing: {e}")
    model = None; scaler = None
except Exception as e:
    print(f"[WARN] Model load error: {e}")
    model = None; scaler = None

# ─────────────────────────────────────────────
# FEATURES  (canonical order — never change)
# DB col = groundwater  |  input/feature key = groundwater_level
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "slope_angle", "rainfall", "rock_density", "crack_length",
    "groundwater_level", "blasting_intensity", "seismic_activity",
    "bench_height", "excavation_depth", "temperature_variation"
]
FEATURE_LABELS = {
    "slope_angle":           "Slope Angle (deg)",
    "rainfall":              "Rainfall (mm)",
    "rock_density":          "Rock Density",
    "crack_length":          "Crack Length (m)",
    "groundwater_level":     "Groundwater Level",
    "blasting_intensity":    "Blasting Intensity",
    "seismic_activity":      "Seismic Activity",
    "bench_height":          "Bench Height (m)",
    "excavation_depth":      "Excavation Depth (m)",
    "temperature_variation": "Temperature Variation"
}
RISK_LABELS = {0: "safe",     1: "caution",      2: "critical"}
RISK_NAMES  = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

def _compute_importance():
    if model is None:
        return {col: round(100.0/len(FEATURE_COLS), 1) for col in FEATURE_COLS}
    if hasattr(model, "feature_importances_"):   # RF, XGBoost
        raw = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):                # Logistic Regression
        raw = np.abs(model.coef_).mean(axis=0)
    else:
        raw = np.ones(len(FEATURE_COLS))
    total = raw.sum() if raw.sum() > 0 else 1.0
    return {col: round(float(raw[i]/total)*100, 1) for i, col in enumerate(FEATURE_COLS)}

FEATURE_IMPORTANCE = _compute_importance()

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):  return float(obj)
        if isinstance(obj, datetime): return obj.isoformat()
        return super().default(obj)

def safe_json(data): return json.dumps(data, cls=SafeEncoder)

# clean_pdf: replace every non-latin-1 char before passing to FPDF.
# FPDF's built-in fonts (Helvetica) only support latin-1.
# The degree sign (U+00B0) in "Slope Angle (deg)" was the original crash.
_PDF_MAP = {
    "\u2014": "-",  "\u2013": "-",  "\u2012": "-",
    "\u2022": "*",  "\u2023": "*",  "\u25aa": "*",
    "\u2192": "->", "\u2190": "<-",
    "\u2265": ">=", "\u2264": "<=", "\u00b1": "+/-",
    "\u00b0": "deg",          # degree sign — THE original crash culprit
    "\u2026": "...",
    "\u2018": "'",  "\u2019": "'",
    "\u201c": '"',  "\u201d": '"',
    "\u00e9": "e",  "\u00e8": "e", "\u00ea": "e",
    "\u00e0": "a",  "\u00e2": "a",
    "\u00fc": "u",  "\u00f6": "o",
    "\u00a9": "(c)","\u00ae": "(R)","\u2122": "TM",
}
def clean_pdf(text):
    if not isinstance(text, str): text = str(text)
    for ch, rep in _PDF_MAP.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ═══════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════
def setup_database():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        cur  = conn.cursor()
        print("[OK] Connected to NeonDB")
    except Exception as e:
        print(f"[WARN] DB connection failed: {e}")
        print("  -> Check DATABASE_URL in .env matches your Neon dashboard.")
        return

    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id            SERIAL PRIMARY KEY,
        full_name     VARCHAR(120) NOT NULL,
        email         VARCHAR(255) NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          VARCHAR(20) NOT NULL DEFAULT 'engineer'
                          CHECK (role IN ('engineer','admin')),
        is_active     BOOLEAN DEFAULT TRUE,
        is_verified   BOOLEAN DEFAULT FALSE,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login    TIMESTAMP NULL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id           SERIAL PRIMARY KEY,
        user_id      INT,
        site_id      VARCHAR(50),
        slope_angle  FLOAT, rainfall     FLOAT,
        rock_density FLOAT, crack_length FLOAT,
        groundwater  FLOAT,
        blasting     FLOAT, seismic      FLOAT,
        bench_height FLOAT, excavation   FLOAT, temperature FLOAT,
        risk_score   FLOAT, risk_level   INT,   risk_label  VARCHAR(20),
        top_feature  VARCHAR(50),
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id            SERIAL PRIMARY KEY,
        prediction_id INT NULL,
        site_id       VARCHAR(50),
        severity      VARCHAR(20) DEFAULT 'info'
                          CHECK (severity IN ('info','warning','critical')),
        title         VARCHAR(200),
        message       TEXT,
        acknowledged  BOOLEAN DEFAULT FALSE,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id         SERIAL PRIMARY KEY,
        user_id    INT NULL,
        action     VARCHAR(200),
        detail     VARCHAR(500),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id         SERIAL PRIMARY KEY,
        user_id    INT,
        role       VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user','assistant')),
        message    TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Seed accounts
    cur.execute("SELECT id FROM users WHERE email='admin@slopesentinel.com'")
    if not cur.fetchone():
        cur.execute("INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified) VALUES (%s,%s,%s,%s,TRUE,TRUE)",
            ("Site Admin","admin@slopesentinel.com",generate_password_hash("Admin@123"),"admin"))

    cur.execute("SELECT id FROM users WHERE email='engineer@mine-site.com'")
    if not cur.fetchone():
        cur.execute("INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified) VALUES (%s,%s,%s,%s,TRUE,TRUE)",
            ("John Engineer","engineer@mine-site.com",generate_password_hash("Engineer@123"),"engineer"))

    cur.execute("SELECT id FROM alerts LIMIT 1")
    if not cur.fetchone():
        cur.execute("INSERT INTO alerts (site_id,severity,title,message) VALUES (%s,%s,%s,%s)",
            ("Eastern Haul Road","critical","Critical Rockfall Warning",
             "High risk at Eastern Haul Road. Slope stability below threshold."))

    # Auto-migrations (PostgreSQL: information_schema instead of SHOW COLUMNS)
    migrations = [
        ("predictions","top_feature",
            "ALTER TABLE predictions ADD COLUMN top_feature VARCHAR(50) NULL"),
        ("users","is_verified",
            "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT FALSE"),
        ("users","is_active",
            "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE"),
        ("alerts","prediction_id",
            "ALTER TABLE alerts ADD COLUMN prediction_id INT NULL"),
    ]
    for tbl, col, sql in migrations:
        try:
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s",(tbl,col))
            if not cur.fetchone():
                cur.execute(sql); print(f"[DB] Migration: {tbl}.{col} added")
        except Exception: pass

    # Type-fix: TINYINT/smallint -> BOOLEAN (happens when schema was MySQL-sourced)
    type_fixes = [
        ("alerts","acknowledged",
            "ALTER TABLE alerts ALTER COLUMN acknowledged TYPE BOOLEAN USING acknowledged::boolean"),
        ("users","is_active",
            "ALTER TABLE users ALTER COLUMN is_active TYPE BOOLEAN USING is_active::boolean"),
        ("users","is_verified",
            "ALTER TABLE users ALTER COLUMN is_verified TYPE BOOLEAN USING is_verified::boolean"),
    ]
    for tbl, col, sql in type_fixes:
        try:
            cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name=%s AND column_name=%s",(tbl,col))
            row = cur.fetchone()
            if row and row[0] in ("smallint","integer","bigint"):
                cur.execute(sql); print(f"[DB] Type fix: {tbl}.{col} -> boolean")
        except Exception: pass

    conn.commit(); cur.close(); conn.close()
    print("[OK] Database ready")

with app.app_context():
    setup_database()


# ═══════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if "user_id" not in session:
            flash("Please log in.", "warning"); return redirect(url_for("login"))
        return f(*a, **k)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger"); return redirect(url_for("dashboard"))
        return f(*a, **k)
    return wrap

def log_activity(action, detail=""):
    try:
        cur = get_cursor()
        cur.execute("INSERT INTO activity_log (user_id,action,detail) VALUES (%s,%s,%s)",
                    (session.get("user_id"), action, detail))
        get_db().commit(); cur.close()
    except Exception: pass


# ═══════════════════════════════════════════════
# ML HELPERS
# ═══════════════════════════════════════════════
def predict_row(row_dict):
    if model is None or scaler is None:
        raise RuntimeError("ML model not loaded. Check best_model.pkl and scaler.pkl.")
    values = np.array([float(row_dict.get(col, 0)) for col in FEATURE_COLS]).reshape(1,-1)
    scaled = scaler.transform(values)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0].tolist()
    return pred, round(proba[pred]*100, 1), proba

def get_explanation(row_dict):
    expl = []
    for col in FEATURE_COLS:
        expl.append({"feature":col,"label":FEATURE_LABELS[col],
            "value":round(float(row_dict.get(col, 0)),2),"importance":FEATURE_IMPORTANCE[col]})
    return sorted(expl, key=lambda x: -x["importance"])

def get_reasons(pred, row_dict):
    reasons = []
    rf  = float(row_dict.get("rainfall",0))
    gw  = float(row_dict.get("groundwater_level",0))
    sa  = float(row_dict.get("slope_angle",0))
    cl  = float(row_dict.get("crack_length",0))
    sei = float(row_dict.get("seismic_activity",0))
    if rf  > 120: reasons.append(f"Extreme rainfall ({rf:.0f} mm) saturating slope material")
    elif rf > 60: reasons.append(f"High rainfall ({rf:.0f} mm) increasing pore pressure")
    if gw  == 1:  reasons.append("Elevated groundwater table reducing effective stress")
    if sa  > 55:  reasons.append(f"Steep slope angle ({sa:.1f} deg) exceeding stability threshold")
    elif sa > 40: reasons.append(f"Slope angle ({sa:.1f} deg) with limited safety margin")
    if cl  > 20:  reasons.append(f"Large crack length ({cl:.1f} m) indicating pre-failure deformation")
    if sei > 3:   reasons.append(f"Seismic activity ({sei:.2f}) destabilizing slope mass")
    if not reasons:
        reasons.append("All parameters within safe operational limits" if pred == 0
                       else "Combined geotechnical parameters indicate elevated risk")
    return reasons


# ═══════════════════════════════════════════════
# GEMINI CHATBOT HELPER
# ═══════════════════════════════════════════════
# System prompt: strictly restricts Gemini to slope/geotech topics only.
# Any unrelated question → polite refusal. No hallucination about other domains.
GEMINI_SYSTEM_PROMPT = """You are SlopeSentinel AI Assistant — an expert AI embedded inside
the SlopeSentinel geotechnical risk monitoring platform for open-pit mines.

YOUR ROLE:
- Answer ONLY questions about slope stability, rockfall, landslides, geotechnical engineering,
  open-pit mining safety, and the SlopeSentinel platform itself.
- Explain risk levels (Safe / Caution / Critical), ML model results, feature importance,
  prediction scores, simulation outputs, alerts, PDF reports, and safety recommendations.
- Give clear, practical, safety-focused answers suitable for mine engineers and geotechnical teams.

SLOPESENTINEL PLATFORM FACTS (always use these):
- Three ML models: Logistic Regression (74.7% accuracy), Random Forest (73.1%), XGBoost (72.6%)
- 10 input features: slope angle, rainfall, rock density, crack length, groundwater level,
  blasting intensity, seismic activity, bench height, excavation depth, temperature variation
- Top risk factors by importance: Rainfall (31.5%), Groundwater (22.3%), Crack Length (13.1%),
  Slope Angle (10.2%), Excavation Depth (5.5%)
- Risk thresholds: slope > 55 deg = critical, rainfall > 120 mm = extreme, cracks > 20 m = high risk
- Features: Upload CSV, manual prediction, XAI explanations, simulation sliders, PDF reports,
  mine map, live sensor feed, alert system, admin panel

STRICT RULES:
1. If the user asks about anything UNRELATED to slope stability, geotechnical engineering,
   mining safety, or this platform → respond ONLY with:
   "I can only answer questions related to slope stability, geotechnical risk, and the SlopeSentinel platform."
2. Never discuss politics, entertainment, coding unrelated to this platform, general science,
   geography, history, or any other off-topic subject — no matter how the question is phrased.
3. Keep answers concise, technical, and actionable.
4. If a user shares their prediction result, give specific safety advice based on the risk level.
5. Always recommend consulting a qualified geotechnical engineer for critical decisions.
"""

# Fallback knowledge base (used when Gemini API key is not set)
CHAT_KB = {
    "safe":       "Safe result: all parameters within operational limits. Continue routine monitoring.",
    "caution":    "Caution: elevated risk. Increase monitoring frequency, inspect cracks, brief site personnel.",
    "critical":   "CRITICAL: Evacuate personnel immediately, halt operations within 200m, contact geotechnical engineer.",
    "rainfall":   "Rainfall is #1 risk factor (31.5% importance). Heavy rain increases pore pressure, reducing stability.",
    "groundwater":"High groundwater (22.3% importance) reduces effective stress, making failure more likely.",
    "slope":      "Slope angles above 55 deg are critical thresholds for most rock types.",
    "crack":      "Crack length > 20m indicates pre-failure deformation — high sliding risk.",
    "seismic":    "Seismic activity destabilizes loose material. Even moderate earthquakes can trigger slides.",
    "accuracy":   "74.7% accuracy via Logistic Regression, 5-fold cross-validation on 5,000 rockfall samples.",
    "features":   "10 features: slope angle, rainfall, rock density, crack length, groundwater level, blasting intensity, seismic activity, bench height, excavation depth, temperature variation.",
    "simulate":   "Simulation Mode: adjust all 10 sliders in real-time to see risk change instantly.",
    "report":     "PDF reports auto-generated per prediction. Go to Analysis and click Report.",
    "xai":        "Explainable AI shows which features drove each prediction. Find it in the sidebar.",
    "model":      "Three models: Logistic Regression 74.7%, Random Forest 73.1%, XGBoost 72.6%.",
    "hi":         "Hello! I'm the SlopeSentinel AI assistant. Ask me about slope risk, features, or safety.",
    "hello":      "Hi! I can explain predictions, risk factors, or guide you through the platform.",
    "help":       "I help with: risk interpretation, model explanations, feature descriptions, alerts, simulation.",
}

def get_gemini_reply(user_message: str, last_pred: dict | None) -> str:
    """Call Gemini 2.5 Flash with full system prompt + optional last prediction context."""
    if not GEMINI_AVAILABLE or gemini_client is None:
        return None  # fall through to KB

    # Build context block if user has a recent prediction
    context = ""
    if last_pred:
        context = (
            f"\n\n[USER'S LATEST PREDICTION CONTEXT]\n"
            f"Risk Level : {last_pred.get('risk','unknown').upper()}\n"
            f"Confidence : {last_pred.get('score','?')}%\n"
            f"Top Factor : {last_pred.get('top_feature','unknown')}\n"
        )

    full_prompt = GEMINI_SYSTEM_PROMPT + context + f"\n\nUser: {user_message}"

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",   # Gemini 2.5 Flash as requested
            contents=full_prompt
        )
        reply = response.text.strip()
        # Safety: if Gemini forgot the rules and went off-topic, it usually starts
        # with phrases we can catch and replace
        off_topic_phrases = [
            "i cannot help with", "as an ai", "i'm not able to assist with that topic",
        ]
        lower_reply = reply.lower()
        if any(p in lower_reply for p in off_topic_phrases):
            return ("I can only answer questions related to slope stability, "
                    "geotechnical risk, and the SlopeSentinel platform.")
        return reply
    except Exception as e:
        print(f"[Gemini error] {e}")
        return None  # fall through to KB


def get_kb_reply(msg: str) -> str:
    """Fallback knowledge-base chatbot when Gemini is unavailable."""
    ml    = msg.lower()
    reply = next((r for kw, r in CHAT_KB.items() if kw in ml), None)
    if not reply:
        if any(w in ml for w in ["why","risk","dangerous"]):
            reply = ("Top risk factors: Rainfall (31.5%), Groundwater (22.3%), "
                     "Crack Length (13.1%), Slope Angle (10.2%). "
                     "Use Simulation Mode to explore scenarios interactively.")
        elif any(w in ml for w in ["how","work","predict","model"]):
            reply = ("Three ML models: Logistic Regression (74.7%), "
                     "Random Forest (73.1%), XGBoost (72.6%). "
                     "Each analyzes 10 geotechnical parameters and outputs "
                     "Safe, Caution, or Critical with a confidence score.")
        else:
            reply = ("I can only answer questions related to slope stability, "
                     "geotechnical risk, and the SlopeSentinel platform. "
                     "Try: 'rainfall', 'slope angle', 'critical risk', 'accuracy', or 'help'.")
    return reply


# ═══════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════
@app.route("/")
def index():
    if "user_id" in session: return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session: return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        cur   = get_cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        u = cur.fetchone(); cur.close()
        if not u:                                             error = "No account with that email."
        elif not check_password_hash(u["password_hash"], pw): error = "Incorrect password."
        elif not u["is_verified"]:                            error = "Pending admin approval."
        elif not u["is_active"]:                              error = "Account deactivated."
        else:
            session.update({"user_id":u["id"],"role":u["role"],
                            "user_name":u["full_name"],"email":u["email"]})
            cur = get_cursor()
            cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (u["id"],))
            get_db().commit(); cur.close()
            log_activity("login", request.remote_addr)
            return redirect(url_for("dashboard"))
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET","POST"])
def register():
    error = None
    if request.method == "POST":
        name  = request.form.get("name",    "").strip()
        email = request.form.get("email",   "").strip().lower()
        pw    = request.form.get("password","")
        if not name or not email or not pw: error = "All fields are required."
        elif len(pw) < 6:                   error = "Password must be at least 6 characters."
        else:
            try:
                cur = get_cursor()
                cur.execute("INSERT INTO users (full_name,email,password_hash) VALUES (%s,%s,%s)",
                            (name, email, generate_password_hash(pw)))
                get_db().commit(); cur.close()
                flash("Account created! Awaiting admin approval.","success")
                return redirect(url_for("login"))
            except Exception:
                get_db().rollback(); error = "Email already registered."
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    log_activity("logout",""); session.clear()
    flash("Signed out.","info"); return redirect(url_for("login"))


# ═══════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    cur = get_cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE user_id = %s",(session["user_id"],))
    pred_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged = FALSE")
    alert_count = cur.fetchone()["cnt"]
    cur.execute("SELECT risk_label,COUNT(*) AS cnt FROM predictions WHERE user_id = %s GROUP BY risk_label",(session["user_id"],))
    risk_dist = {r["risk_label"]:int(r["cnt"]) for r in cur.fetchall()}
    cur.execute("""SELECT TO_CHAR(created_at,'Mon') AS month,
        SUM(CASE WHEN risk_level=0 THEN 1 ELSE 0 END) AS safe,
        SUM(CASE WHEN risk_level=1 THEN 1 ELSE 0 END) AS caution,
        SUM(CASE WHEN risk_level=2 THEN 1 ELSE 0 END) AS critical
        FROM predictions WHERE user_id=%s AND created_at >= NOW() - INTERVAL '6 months'
        GROUP BY TO_CHAR(created_at,'Mon'),EXTRACT(MONTH FROM created_at)
        ORDER BY EXTRACT(MONTH FROM created_at)""",(session["user_id"],))
    trend_rows = cur.fetchall()
    cur.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",(session["user_id"],))
    recent_preds = cur.fetchall(); cur.close()
    return render_template("dashboard.html",
        user={"full_name":session["user_name"],"role":session["role"]},
        pred_count=pred_count, alert_count=alert_count,
        risk_dist=safe_json(risk_dist),
        trend_rows=safe_json([dict(r) for r in trend_rows]),
        recent_preds=recent_preds,
        feature_importance=safe_json(FEATURE_IMPORTANCE))


# ═══════════════════════════════════════════════
# UPLOAD & PREDICT
# ═══════════════════════════════════════════════
@app.route("/upload", methods=["GET","POST"])
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
                    df     = df[FEATURE_COLS].head(100)
                    scaled = scaler.transform(df)
                    preds  = model.predict(scaled)
                    probas = model.predict_proba(scaled)
                    cur    = get_cursor()
                    for i,(row,pred,proba) in enumerate(zip(df.itertuples(index=False),preds,probas)):
                        sid   = f"SITE-{i+1:03d}"; lbl = RISK_LABELS[int(pred)]
                        score = round(float(max(proba))*100, 1)
                        rd    = {col:getattr(row,col) for col in FEATURE_COLS}
                        expl  = get_explanation(rd); top_f = expl[0]["feature"]
                        reasons = get_reasons(int(pred), rd)
                        cur.execute("""INSERT INTO predictions
                            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
                             groundwater,blasting,seismic,bench_height,excavation,temperature,
                             risk_score,risk_level,risk_label,top_feature)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                            (session["user_id"],sid,
                             row.slope_angle,row.rainfall,row.rock_density,row.crack_length,
                             row.groundwater_level,  # input key -> DB col groundwater
                             row.blasting_intensity,row.seismic_activity,
                             row.bench_height,row.excavation_depth,row.temperature_variation,
                             score,int(pred),lbl,top_f))
                        pid = cur.fetchone()["id"]
                        if int(pred)==2:
                            cur.execute("INSERT INTO alerts (prediction_id,site_id,severity,title,message) VALUES (%s,%s,'critical',%s,%s)",
                                (pid,sid,f"Critical Risk at {sid}",
                                 f"AI flagged critical risk ({score}%). Top factor: {FEATURE_LABELS[top_f]}"))
                        results.append({"site_id":sid,"slope":round(row.slope_angle,1),
                            "rainfall":round(row.rainfall,1),"score":score,"label":lbl,
                            "name":RISK_NAMES[int(pred)],"top_feature":FEATURE_LABELS[top_f],
                            "reason":reasons[0],"pred_id":pid})
                    get_db().commit(); cur.close()
                    log_activity("upload",f"{len(results)} predictions")
                    flash(f"Generated {len(results)} predictions.","success")
            except Exception as e:
                get_db().rollback(); error = f"Processing error: {str(e)}"
    return render_template("upload.html",results=results,error=error,feature_cols=FEATURE_COLS)

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        data = {col:float(request.form.get(col,0)) for col in FEATURE_COLS}
        pred,score,proba = predict_row(data)
        expl = get_explanation(data); reasons = get_reasons(pred,data)
        top_f = expl[0]["feature"] if expl else ""
        # Save last prediction to session for chatbot context
        session["last_prediction"] = {"risk":RISK_LABELS[pred],"score":score,
                                       "top_feature":FEATURE_LABELS.get(top_f,"")}
        cur = get_cursor()
        cur.execute("""INSERT INTO predictions
            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
             groundwater,blasting,seismic,bench_height,excavation,temperature,
             risk_score,risk_level,risk_label,top_feature)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session["user_id"],"Manual",
             data["slope_angle"],data["rainfall"],data["rock_density"],data["crack_length"],
             data["groundwater_level"],data["blasting_intensity"],data["seismic_activity"],
             data["bench_height"],data["excavation_depth"],data["temperature_variation"],
             score,pred,RISK_LABELS[pred],top_f))
        if pred==2:
            cur.execute("INSERT INTO alerts (severity,title,message) VALUES ('critical','Critical Manual Prediction','Manual input flagged critical rockfall risk.')")
        get_db().commit(); cur.close()
        return jsonify({"risk_level":pred,"risk_label":RISK_LABELS[pred],"risk_name":RISK_NAMES[pred],
            "score":score,"proba":{"safe":round(proba[0]*100,1),"caution":round(proba[1]*100,1),"critical":round(proba[2]*100,1)},
            "explanation":expl[:5],"reasons":reasons[:3],"top_feature":FEATURE_LABELS.get(top_f,"")})
    except Exception as e: return jsonify({"error":str(e)}),400


# ═══════════════════════════════════════════════
# EXPLAINABLE AI
# ═══════════════════════════════════════════════
@app.route("/xai")
@login_required
def xai():
    cur = get_cursor()
    cur.execute("""SELECT id,site_id,risk_label,risk_score,slope_angle,rainfall,
        rock_density,crack_length,groundwater,blasting,seismic,bench_height,
        excavation,temperature,top_feature,created_at
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30""",(session["user_id"],))
    preds = cur.fetchall()
    cur.execute("SELECT top_feature,COUNT(*) AS cnt FROM predictions WHERE user_id=%s AND top_feature IS NOT NULL GROUP BY top_feature ORDER BY cnt DESC LIMIT 5",(session["user_id"],))
    top_features = cur.fetchall(); cur.close()
    fi_sorted = sorted(FEATURE_IMPORTANCE.items(),key=lambda x:-x[1])
    return render_template("xai.html",preds=preds,fi_sorted=fi_sorted,
        feature_labels=FEATURE_LABELS,feature_importance_json=safe_json(dict(fi_sorted)),
        top_features=top_features)

@app.route("/api/xai/<int:pred_id>")
@login_required
def api_xai(pred_id):
    cur = get_cursor()
    cur.execute("SELECT * FROM predictions WHERE id=%s AND user_id=%s",(pred_id,session["user_id"]))
    p = cur.fetchone(); cur.close()
    if not p: return jsonify({"error":"Not found"}),404
    row = {"slope_angle":p["slope_angle"],"rainfall":p["rainfall"],"rock_density":p["rock_density"],
           "crack_length":p["crack_length"],"groundwater_level":p["groundwater"],
           "blasting_intensity":p["blasting"],"seismic_activity":p["seismic"],
           "bench_height":p["bench_height"],"excavation_depth":p["excavation"],"temperature_variation":p["temperature"]}
    return jsonify({"site_id":p["site_id"],"risk_label":p["risk_label"],"risk_score":p["risk_score"],
        "explanation":get_explanation(row),"reasons":get_reasons(p["risk_level"],row),"feature_labels":FEATURE_LABELS})


# ═══════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════
@app.route("/simulate")
@login_required
def simulate():
    return render_template("simulation.html",feature_cols=FEATURE_COLS,
        feature_labels=FEATURE_LABELS,feature_importance=safe_json(FEATURE_IMPORTANCE))

@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    try:
        data = {col:float(request.json.get(col,0)) for col in FEATURE_COLS}
        pred,score,proba = predict_row(data)
        return jsonify({"risk_level":pred,"risk_label":RISK_LABELS[pred],"risk_name":RISK_NAMES[pred],
            "score":score,"proba":{"safe":round(proba[0]*100,1),"caution":round(proba[1]*100,1),"critical":round(proba[2]*100,1)},
            "explanation":get_explanation(data)[:5],"reasons":get_reasons(pred,data)})
    except Exception as e: return jsonify({"error":str(e)}),400


# ═══════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════
@app.route("/analysis")
@login_required
def analysis():
    cur = get_cursor()
    cur.execute("SELECT risk_label,COUNT(*) AS cnt,AVG(risk_score) AS avg_score,AVG(slope_angle) AS avg_slope FROM predictions WHERE user_id=%s GROUP BY risk_label",(session["user_id"],))
    stats = cur.fetchall()
    cur.execute("SELECT id,site_id,risk_label,risk_score,slope_angle,rainfall,seismic,top_feature,created_at FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",(session["user_id"],))
    history = cur.fetchall()
    cur.execute("SELECT slope_angle,rainfall,seismic,risk_level,risk_label,risk_score FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 200",(session["user_id"],))
    scatter_data = cur.fetchall()
    cur.execute("SELECT risk_level,COUNT(*) AS cnt FROM predictions WHERE user_id=%s GROUP BY risk_level",(session["user_id"],))
    risk_counts = {r["risk_level"]:int(r["cnt"]) for r in cur.fetchall()}; cur.close()
    return render_template("analysis.html",stats=stats,history=history,
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
    cur = get_cursor()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    all_alerts = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged = FALSE")
    unread = cur.fetchone()["cnt"]; cur.close()
    return render_template("alerts.html",alerts=all_alerts,unread=unread)

@app.route("/alerts/ack/<int:aid>", methods=["POST"])
@login_required
def ack_alert(aid):
    cur = get_cursor(); cur.execute("UPDATE alerts SET acknowledged = TRUE WHERE id = %s",(aid,))
    get_db().commit(); cur.close(); return jsonify({"ok":True})

@app.route("/alerts/ack-all", methods=["POST"])
@login_required
def ack_all_alerts():
    cur = get_cursor(); cur.execute("UPDATE alerts SET acknowledged = TRUE")
    get_db().commit(); cur.close(); return jsonify({"ok":True})


# ═══════════════════════════════════════════════
# MAP
# ═══════════════════════════════════════════════
@app.route("/map")
@login_required
def map_view():
    cur = get_cursor()
    cur.execute("SELECT site_id,risk_label,risk_score,slope_angle,rainfall,seismic FROM predictions WHERE user_id=%s ORDER BY created_at DESC",(session["user_id"],))
    preds = cur.fetchall(); cur.close()
    return render_template("map.html",predictions=safe_json([dict(p) for p in preds]))


# ═══════════════════════════════════════════════
# PDF REPORTS  — every string via clean_pdf()
# ═══════════════════════════════════════════════
@app.route("/report/<int:pred_id>")
@login_required
def generate_report(pred_id):
    cur = get_cursor()
    cur.execute("SELECT * FROM predictions WHERE id=%s AND user_id=%s",(pred_id,session["user_id"]))
    p = cur.fetchone(); cur.close()
    if not p: flash("Prediction not found.","danger"); return redirect(url_for("analysis"))
    row={"slope_angle":p["slope_angle"],"rainfall":p["rainfall"],"rock_density":p["rock_density"],
         "crack_length":p["crack_length"],"groundwater_level":p["groundwater"],
         "blasting_intensity":p["blasting"],"seismic_activity":p["seismic"],
         "bench_height":p["bench_height"],"excavation_depth":p["excavation"],"temperature_variation":p["temperature"]}
    expl=get_explanation(row); reasons=get_reasons(p["risk_level"],row)
    pdf=FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True,margin=15)
    pdf.set_fill_color(13,15,20); pdf.set_text_color(255,255,255); pdf.rect(0,0,210,35,"F")
    pdf.set_font("Helvetica","B",17); pdf.set_xy(15,8)
    pdf.cell(180,10,clean_pdf("SlopeSentinel - Geotechnical Risk Report"),align="C")
    pdf.set_font("Helvetica","",9); pdf.set_xy(15,20)
    pdf.cell(180,6,clean_pdf(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  Analyst: {session['user_name']}"),align="C")
    pdf.ln(22); pdf.set_text_color(30,30,30)
    rc={"safe":(34,197,94),"caution":(249,115,22),"critical":(239,68,68)}.get(p["risk_label"],(100,100,100))
    pdf.set_fill_color(*rc); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",20)
    pdf.cell(180,14,clean_pdf(f"  {RISK_NAMES[p['risk_level']].upper()}  -  Confidence: {p['risk_score']:.1f}%"),fill=True,align="C")
    pdf.ln(8); pdf.set_text_color(30,30,30)
    pdf.set_font("Helvetica","B",12); pdf.cell(180,8,clean_pdf(f"Site: {p['site_id']}")); pdf.ln(10)
    pdf.set_font("Helvetica","B",11); pdf.set_fill_color(235,235,245)
    pdf.cell(90,8,clean_pdf("Parameter"),fill=True,border=1); pdf.cell(90,8,clean_pdf("Value"),fill=True,border=1); pdf.ln()
    pdf.set_font("Helvetica","",10)
    for col in FEATURE_COLS:
        pdf.cell(90,7,clean_pdf(FEATURE_LABELS[col]),border=1)
        pdf.cell(90,7,clean_pdf(str(round(float(row[col]),2))),border=1); pdf.ln()
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,clean_pdf("Top Contributing Risk Factors:")); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    for i,e in enumerate(expl[:5],1):
        pdf.cell(180,7,clean_pdf(f"  {i}. {e['label']} ({e['importance']:.1f}%) - Value: {e['value']}")); pdf.ln()
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,clean_pdf("AI Analysis:")); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    for r in reasons: pdf.multi_cell(180,7,clean_pdf(f"  * {r}"))
    pdf.ln(6); pdf.set_font("Helvetica","B",11); pdf.cell(180,8,clean_pdf("Recommendations:")); pdf.ln(9)
    pdf.set_font("Helvetica","",10)
    recs={"safe":["Continue routine monitoring schedule","Maintain sensor calibration"],
          "caution":["Increase monitoring to 6-hour intervals","Deploy additional sensors","Inspect visible crack formations","Brief site personnel"],
          "critical":["IMMEDIATE EVACUATION required","Contact geotechnical engineer NOW","Halt all operations within 200m","Emergency inspection within 2 hours"]}
    for rec in recs.get(p["risk_label"],recs["caution"]):
        pdf.cell(180,7,clean_pdf(f"  * {rec}")); pdf.ln()
    pdf.set_y(-20); pdf.set_font("Helvetica","I",8); pdf.set_text_color(120,120,120)
    pdf.cell(180,6,clean_pdf("SlopeSentinel AI Platform - Confidential"),align="C")
    buf=io.BytesIO(); buf.write(pdf.output()); buf.seek(0)
    return send_file(buf,mimetype="application/pdf",
        download_name=f"Report_{p['site_id']}_{datetime.now().strftime('%Y%m%d')}.pdf",as_attachment=True)

@app.route("/report/bulk")
@login_required
def generate_bulk_report():
    cur = get_cursor()
    cur.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 20",(session["user_id"],))
    preds = cur.fetchall(); cur.close()
    if not preds: flash("No predictions to report.","warning"); return redirect(url_for("analysis"))
    pdf=FPDF(); pdf.add_page()
    pdf.set_fill_color(13,15,20); pdf.set_text_color(255,255,255); pdf.rect(0,0,210,30,"F")
    pdf.set_font("Helvetica","B",17); pdf.set_xy(15,8)
    pdf.cell(180,14,clean_pdf("SlopeSentinel - Bulk Prediction Report"),align="C")
    pdf.ln(22); pdf.set_text_color(30,30,30); pdf.set_font("Helvetica","",9)
    pdf.cell(180,6,clean_pdf(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} | User: {session['user_name']} | Sites: {len(preds)}"))
    pdf.ln(10)
    pdf.set_font("Helvetica","B",9); pdf.set_fill_color(230,230,240)
    for h,w in [("Site ID",35),("Risk",28),("Score",22),("Slope",22),("Rainfall",26),("Top Factor",57)]:
        pdf.cell(w,8,clean_pdf(h),border=1,fill=True)
    pdf.ln(); pdf.set_font("Helvetica","",8)
    risk_colors={"safe":(34,197,94),"caution":(249,115,22),"critical":(239,68,68)}
    for p_ in preds:
        rc=risk_colors.get(p_["risk_label"],(200,200,200)); fill=p_["risk_label"]=="critical"
        pdf.set_fill_color(*rc)
        pdf.set_text_color(255,255,255) if fill else pdf.set_text_color(30,30,30)
        pdf.cell(35,7,clean_pdf(str(p_["site_id"])),border=1)
        pdf.cell(28,7,clean_pdf(str(p_["risk_label"]).capitalize()),border=1,fill=fill)
        pdf.set_text_color(30,30,30)
        pdf.cell(22,7,clean_pdf(f"{p_['risk_score']:.1f}%"),border=1)
        pdf.cell(22,7,clean_pdf(f"{p_['slope_angle']:.1f}"),border=1)
        pdf.cell(26,7,clean_pdf(f"{p_['rainfall']:.0f}mm"),border=1)
        pdf.cell(57,7,clean_pdf(FEATURE_LABELS.get(p_["top_feature"] or "","-")[:26]),border=1)
        pdf.ln()
    buf=io.BytesIO(); buf.write(pdf.output()); buf.seek(0)
    return send_file(buf,mimetype="application/pdf",
        download_name=f"BulkReport_{datetime.now().strftime('%Y%m%d')}.pdf",as_attachment=True)


# ═══════════════════════════════════════════════
# CHATBOT — Gemini 2.5 Flash with fallback KB
# ═══════════════════════════════════════════════
@app.route("/chatbot")
@login_required
def chatbot():
    cur = get_cursor()
    cur.execute("SELECT role,message,created_at FROM chat_history WHERE user_id=%s ORDER BY created_at DESC LIMIT 20",(session["user_id"],))
    history = list(reversed(cur.fetchall())); cur.close()
    return render_template("chatbot.html", history=history,
                           gemini_active=GEMINI_AVAILABLE)

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    msg = request.json.get("message","").strip()
    if not msg: return jsonify({"error":"Empty message"}),400

    last_pred = session.get("last_prediction")

    # 1. Try Gemini 2.5 Flash first
    reply = get_gemini_reply(msg, last_pred)

    # 2. Fallback to built-in knowledge base if Gemini unavailable
    if reply is None:
        reply = get_kb_reply(msg)

    # Save to DB
    try:
        cur = get_cursor()
        cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'user',%s)",
                    (session["user_id"], msg))
        cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'assistant',%s)",
                    (session["user_id"], reply))
        get_db().commit(); cur.close()
    except Exception:
        pass

    return jsonify({"reply": reply, "powered_by": "gemini-2.5-flash" if GEMINI_AVAILABLE else "built-in-kb"})


# ═══════════════════════════════════════════════
# SENSOR FEED
# ═══════════════════════════════════════════════
@app.route("/api/sensor-feed")
@login_required
def api_sensor_feed():
    sites=[("North Pit Alpha",35,40),("West Bench B2",38,35),
           ("East Corridor C3",52,100),("Central Zone",46,80),("Eastern Haul Road",66,185)]
    r=random.uniform
    return jsonify([{"site":s,"slope_angle":round(bs+r(-2,2),1),"rainfall":round(br+r(-10,10),1),
        "soil_moisture":round(r(30,95),1),"temperature":round(r(18,38),1),"seismic":round(r(0,4.5),2),
        "status":"critical" if br>160 else("caution" if br>80 else "safe"),
        "ts":datetime.now().strftime("%H:%M:%S")} for s,bs,br in sites])


# ═══════════════════════════════════════════════
# EXPORT CSV
# ═══════════════════════════════════════════════
@app.route("/export")
@login_required
def export():
    cur = get_cursor()
    cur.execute("SELECT site_id,slope_angle,rainfall,rock_density,crack_length,seismic,risk_score,risk_label,top_feature,created_at FROM predictions WHERE user_id=%s ORDER BY created_at DESC",(session["user_id"],))
    rows = cur.fetchall(); cur.close()
    si = io.StringIO()
    writer = csv.DictWriter(si,fieldnames=rows[0].keys() if rows else [])
    writer.writeheader(); writer.writerows(rows)
    return Response(si.getvalue(),mimetype="text/csv",
        headers={"Content-Disposition":f"attachment;filename=slopesentinel_{datetime.now().strftime('%Y%m%d')}.csv"})


# ═══════════════════════════════════════════════
# ADMIN
# ═══════════════════════════════════════════════
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    cur = get_cursor()
    cur.execute("SELECT id,full_name,email,role,is_active,is_verified,created_at,last_login FROM users ORDER BY is_verified ASC,created_at DESC")
    users = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged = FALSE"); alert_count=cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions"); total_preds=cur.fetchone()["cnt"]
    cur.execute("SELECT a.*,u.full_name AS actor_name FROM activity_log a LEFT JOIN users u ON a.user_id=u.id ORDER BY a.created_at DESC LIMIT 30")
    activity=cur.fetchall()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 20")
    sys_alerts=cur.fetchall(); cur.close()
    return render_template("admin.html",users=users,alert_count=alert_count,
        total_preds=total_preds,activity=activity,sys_alerts=sys_alerts)

@app.route("/admin/approve/<int:uid>",methods=["POST"])
@login_required
@admin_required
def approve(uid):
    cur=get_cursor(); cur.execute("UPDATE users SET is_verified=TRUE,is_active=TRUE WHERE id=%s",(uid,)); get_db().commit()
    cur.execute("SELECT full_name FROM users WHERE id=%s",(uid,)); u=cur.fetchone(); cur.close()
    log_activity("approve_user",u["full_name"] if u else str(uid))
    flash(f"Access approved for {u['full_name'] if u else uid}.","success"); return redirect(url_for("admin_panel"))

@app.route("/admin/reject/<int:uid>",methods=["POST"])
@login_required
@admin_required
def reject(uid):
    cur=get_cursor(); cur.execute("UPDATE users SET is_verified=FALSE,is_active=FALSE WHERE id=%s",(uid,)); get_db().commit()
    cur.execute("SELECT full_name FROM users WHERE id=%s",(uid,)); u=cur.fetchone(); cur.close()
    flash(f"Rejected {u['full_name'] if u else uid}.","warning"); return redirect(url_for("admin_panel"))

@app.route("/admin/delete/<int:uid>",methods=["POST"])
@login_required
@admin_required
def delete_user(uid):
    if uid==session["user_id"]: flash("Cannot delete yourself.","danger"); return redirect(url_for("admin_panel"))
    cur=get_cursor(); cur.execute("DELETE FROM users WHERE id=%s",(uid,)); get_db().commit(); cur.close()
    flash("User deleted.","danger"); return redirect(url_for("admin_panel"))

@app.route("/admin/invite",methods=["POST"])
@login_required
@admin_required
def invite_user():
    name=request.form.get("full_name","").strip(); email=request.form.get("email","").strip().lower()
    role=request.form.get("role","engineer"); temp=secrets.token_urlsafe(10)
    try:
        cur=get_cursor()
        cur.execute("INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified) VALUES (%s,%s,%s,%s,TRUE,FALSE)",
            (name,email,generate_password_hash(temp),role))
        get_db().commit(); cur.close()
        flash(f"Invite for {name}. Temp password: {temp}","success")
    except Exception as e:
        get_db().rollback(); flash(f"Error: {e}","danger")
    return redirect(url_for("admin_panel"))

@app.route("/admin/broadcast",methods=["POST"])
@login_required
@admin_required
def broadcast_alert():
    title=request.form.get("title","System Alert"); message=request.form.get("message","")
    cur=get_cursor(); cur.execute("INSERT INTO alerts (site_id,severity,title,message) VALUES ('System','warning',%s,%s)",(title,message))
    get_db().commit(); cur.close()
    flash("Alert broadcast.","success"); return redirect(url_for("admin_panel"))

@app.route("/api/admin/toggle/<int:uid>",methods=["POST"])
@login_required
@admin_required
def toggle_user(uid):
    if uid==session["user_id"]: return jsonify({"error":"Cannot deactivate yourself."}),400
    cur=get_cursor(); cur.execute("UPDATE users SET is_active = NOT is_active WHERE id=%s",(uid,)); get_db().commit()
    cur.execute("SELECT is_active FROM users WHERE id=%s",(uid,)); r=cur.fetchone(); cur.close()
    return jsonify({"is_active":bool(r["is_active"])})


# ═══════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e): return render_template("404.html"),404
@app.errorhandler(500)
def server_error(e): return render_template("404.html",code=500,msg="Internal server error."),500

@app.route("/health")
def health(): return jsonify({"status":"ok","gemini":GEMINI_AVAILABLE})

if __name__=="__main__":
    app.run(debug=True,port=5000)