"""
SlopeSentinel — app.py
Full Flask backend with ML prediction, auth, admin, alerts, export
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, Response)
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import MySQLdb
import pandas as pd
import numpy as np
import joblib
import json
import csv
import io
import os
import secrets
import folium
from decimal import Decimal

# ── Custom JSON encoder — handles MySQL Decimal & datetime types ──
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def safe_json(data):
    """Serialize data containing MySQL Decimal/datetime safely."""
    return json.dumps(data, cls=SafeEncoder)

# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "slopesentinel-dev-secret-change-in-prod")

# ─── MySQL config ───
app.config["MYSQL_HOST"]        = os.environ.get("MYSQL_HOST",     "127.0.0.1")
app.config["MYSQL_PORT"]        = int(os.environ.get("MYSQL_PORT", 3306))
app.config["MYSQL_USER"]        = os.environ.get("MYSQL_USER",     "root")
app.config["MYSQL_PASSWORD"]    = os.environ.get("MYSQL_PASSWORD", "1501")
app.config["MYSQL_DB"]          = os.environ.get("MYSQL_DB",       "slopesentinel")
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

# ─── Load ML model ───
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_COLS = [
    "slope_angle", "rainfall", "rock_density", "crack_length",
    "groundwater_level", "blasting_intensity", "seismic_activity",
    "bench_height", "excavation_depth", "temperature_variation"
]

RISK_LABELS = {0: "safe", 1: "caution", 2: "critical"}
RISK_NAMES  = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def setup_database():
    db = MySQLdb.connect(
        host=app.config["MYSQL_HOST"],
        user=app.config["MYSQL_USER"],
        passwd=app.config["MYSQL_PASSWORD"],
        port=app.config["MYSQL_PORT"]
    )
    cur = db.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS slopesentinel CHARACTER SET utf8mb4")
    cur.execute("USE slopesentinel")

    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
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

    # Predictions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        user_id         INT,
        site_id         VARCHAR(50),
        slope_angle     FLOAT, rainfall          FLOAT,
        rock_density    FLOAT, crack_length      FLOAT,
        groundwater     FLOAT, blasting          FLOAT,
        seismic         FLOAT, bench_height      FLOAT,
        excavation      FLOAT, temperature       FLOAT,
        risk_score      FLOAT,
        risk_level      INT,
        risk_label      VARCHAR(20),
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )""")

    # Alerts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id           INT AUTO_INCREMENT PRIMARY KEY,
        prediction_id INT NULL,
        site_id      VARCHAR(50),
        severity     ENUM('info','warning','critical') DEFAULT 'info',
        title        VARCHAR(200),
        message      TEXT,
        acknowledged TINYINT(1) DEFAULT 0,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # Activity log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS activity_log (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NULL,
        action     VARCHAR(200),
        detail     VARCHAR(500),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # Seed admin
    cur.execute("SELECT id FROM users WHERE email='admin@slopesentinel.com'")
    if not cur.fetchone():
        cur.execute("""
        INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified)
        VALUES (%s,%s,%s,%s,1,1)
        """, ("Site Admin", "admin@slopesentinel.com",
              generate_password_hash("Admin@123"), "admin"))

    # Seed demo engineer
    cur.execute("SELECT id FROM users WHERE email='engineer@mine-site.com'")
    if not cur.fetchone():
        cur.execute("""
        INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified)
        VALUES (%s,%s,%s,%s,1,1)
        """, ("John Engineer", "engineer@mine-site.com",
              generate_password_hash("Engineer@123"), "engineer"))

    # Seed demo alert
    cur.execute("SELECT id FROM alerts LIMIT 1")
    if not cur.fetchone():
        cur.execute("""
        INSERT INTO alerts (site_id,severity,title,message)
        VALUES ('Eastern Haul Road','critical',
                'Critical Rockfall Warning',
                'High risk detected at Eastern Haul Road. Immediate inspection required.')
        """)

    db.commit()
    cur.close()
    db.close()


# ─────────────────────────────────────────────
# DECORATORS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrap

def log_activity(action, detail=""):
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO activity_log (user_id,action,detail) VALUES (%s,%s,%s)",
            (session.get("user_id"), action, detail)
        )
        mysql.connection.commit()
        cur.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def predict_row(row_dict):
    """Run ML prediction on a dict of feature values."""
    df = pd.DataFrame([row_dict], columns=FEATURE_COLS)
    scaled = scaler.transform(df)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0].tolist()
    score  = round(proba[pred] * 100, 1)
    return pred, score, proba


# ─────────────────────────────────────────────
# PUBLIC ROUTES
# ─────────────────────────────────────────────
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
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if not user:
            error = "No account found with that email."
        elif not check_password_hash(user["password_hash"], password):
            error = "Incorrect password."
        elif not user["is_verified"]:
            error = "Account pending approval. Contact your administrator."
        elif not user["is_active"]:
            error = "Account has been deactivated."
        else:
            session["user_id"]   = user["id"]
            session["role"]      = user["role"]
            session["user_name"] = user["full_name"]
            session["email"]     = user["email"]

            cur = mysql.connection.cursor()
            cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (user["id"],))
            mysql.connection.commit()
            cur.close()
            log_activity("login", f"Logged in from {request.remote_addr}")
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO users (full_name,email,password_hash) VALUES (%s,%s,%s)",
                    (name, email, generate_password_hash(password))
                )
                mysql.connection.commit()
                cur.close()
                flash("Account created! Waiting for admin approval.", "success")
                return redirect(url_for("login"))
            except Exception:
                error = "Email already registered."

    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    log_activity("logout", "")
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ─────────────────────────────────────────────
# PROTECTED ROUTES
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    cur = mysql.connection.cursor()

    # Stats
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE user_id=%s",
                (session["user_id"],))
    pred_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    alert_count = cur.fetchone()["cnt"]

    cur.execute("""
        SELECT risk_label, COUNT(*) AS cnt FROM predictions
        WHERE user_id=%s GROUP BY risk_label
    """, (session["user_id"],))
    risk_dist = {r["risk_label"]: r["cnt"] for r in cur.fetchall()}

    # Recent predictions (last 6 months)
    cur.execute("""
        SELECT DATE_FORMAT(created_at,'%%b') AS month,
               SUM(risk_level=0) AS safe,
               SUM(risk_level=1) AS caution,
               SUM(risk_level=2) AS critical
        FROM predictions
        WHERE user_id=%s AND created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
        GROUP BY DATE_FORMAT(created_at,'%%b'), MONTH(created_at)
        ORDER BY MONTH(created_at)
    """, (session["user_id"],))
    trend_rows = cur.fetchall()

    cur.execute("""
        SELECT * FROM predictions WHERE user_id=%s
        ORDER BY created_at DESC LIMIT 5
    """, (session["user_id"],))
    recent_preds = cur.fetchall()

    cur.close()

    user = {"full_name": session["user_name"], "role": session["role"]}
    return render_template("dashboard.html",
        user=user,
        pred_count=pred_count,
        alert_count=alert_count,
        risk_dist=safe_json({k: int(v) for k, v in risk_dist.items()}),
        trend_rows=safe_json([dict(r) for r in trend_rows]),
        recent_preds=recent_preds
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    results = []
    error   = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.endswith(".csv"):
            error = "Please upload a valid CSV file."
        else:
            try:
                df = pd.read_csv(f)
                missing = [c for c in FEATURE_COLS if c not in df.columns]
                if missing:
                    error = f"Missing columns: {', '.join(missing)}"
                else:
                    df = df[FEATURE_COLS].head(100)
                    scaled = scaler.transform(df)
                    preds  = model.predict(scaled)
                    probas = model.predict_proba(scaled)

                    cur = mysql.connection.cursor()
                    for i, (row, pred, proba) in enumerate(zip(df.itertuples(index=False), preds, probas)):
                        site_id   = f"SITE-{i+1:03d}"
                        risk_lbl  = RISK_LABELS[int(pred)]
                        risk_score = round(float(max(proba)) * 100, 1)

                        cur.execute("""
                            INSERT INTO predictions
                            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
                             groundwater,blasting,seismic,bench_height,excavation,temperature,
                             risk_score,risk_level,risk_label)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (
                            session["user_id"], site_id,
                            row.slope_angle, row.rainfall, row.rock_density, row.crack_length,
                            row.groundwater_level, row.blasting_intensity, row.seismic_activity,
                            row.bench_height, row.excavation_depth, row.temperature_variation,
                            risk_score, int(pred), risk_lbl
                        ))
                        pred_id = cur.lastrowid

                        # Auto-create alert for critical predictions
                        if int(pred) == 2:
                            cur.execute("""
                                INSERT INTO alerts (prediction_id,site_id,severity,title,message)
                                VALUES (%s,%s,'critical',%s,%s)
                            """, (pred_id, site_id,
                                  f"Critical Risk at {site_id}",
                                  f"AI model flagged critical rockfall risk (score: {risk_score}%) at {site_id}."))

                        results.append({
                            "site_id":   site_id,
                            "slope":     round(row.slope_angle, 1),
                            "rainfall":  round(row.rainfall, 1),
                            "score":     risk_score,
                            "label":     risk_lbl,
                            "name":      RISK_NAMES[int(pred)],
                        })

                    mysql.connection.commit()
                    cur.close()
                    log_activity("upload", f"Uploaded {len(results)} predictions")
                    flash(f"✓ {len(results)} predictions generated successfully.", "success")

            except Exception as e:
                error = f"Processing error: {str(e)}"

    return render_template("upload.html", results=results, error=error,
                           feature_cols=FEATURE_COLS)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON API for single-row prediction from manual input form."""
    try:
        data = {col: float(request.form.get(col, 0)) for col in FEATURE_COLS}
        pred, score, proba = predict_row(data)

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO predictions
            (user_id,site_id,slope_angle,rainfall,rock_density,crack_length,
             groundwater,blasting,seismic,bench_height,excavation,temperature,
             risk_score,risk_level,risk_label)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            session["user_id"], data.get("site_id", "Manual"),
            data["slope_angle"], data["rainfall"], data["rock_density"],
            data["crack_length"], data["groundwater_level"],
            data["blasting_intensity"], data["seismic_activity"],
            data["bench_height"], data["excavation_depth"],
            data["temperature_variation"],
            score, pred, RISK_LABELS[pred]
        ))
        mysql.connection.commit()
        cur.close()

        return jsonify({
            "risk_level": pred,
            "risk_label": RISK_LABELS[pred],
            "risk_name":  RISK_NAMES[pred],
            "score":      score,
            "proba":      {"safe": round(proba[0]*100,1),
                           "caution": round(proba[1]*100,1),
                           "critical": round(proba[2]*100,1)}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/analysis")
@login_required
def analysis():
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT risk_label, COUNT(*) AS cnt,
               AVG(risk_score) AS avg_score,
               AVG(slope_angle) AS avg_slope
        FROM predictions WHERE user_id=%s
        GROUP BY risk_label
    """, (session["user_id"],))
    stats = cur.fetchall()

    cur.execute("""
        SELECT site_id, risk_label, risk_score, slope_angle,
               rainfall, seismic, created_at
        FROM predictions WHERE user_id=%s
        ORDER BY created_at DESC LIMIT 50
    """, (session["user_id"],))
    history = cur.fetchall()

    cur.execute("""
        SELECT slope_angle, rainfall, seismic,
               risk_level, risk_label
        FROM predictions WHERE user_id=%s
        ORDER BY created_at DESC LIMIT 200
    """, (session["user_id"],))
    scatter_data = cur.fetchall()
    cur.close()

    return render_template("analysis.html",
        stats=stats,
        history=history,
        scatter_json=safe_json([dict(r) for r in scatter_data])
    )


@app.route("/alerts")
@login_required
def alerts():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    all_alerts = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    unread = cur.fetchone()["cnt"]
    cur.close()
    return render_template("alerts.html", alerts=all_alerts, unread=unread)


@app.route("/alerts/ack/<int:alert_id>", methods=["POST"])
@login_required
def ack_alert(alert_id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE alerts SET acknowledged=1 WHERE id=%s", (alert_id,))
    mysql.connection.commit()
    cur.close()
    return jsonify({"ok": True})


@app.route("/alerts/ack-all", methods=["POST"])
@login_required
def ack_all_alerts():
    cur = mysql.connection.cursor()
    cur.execute("UPDATE alerts SET acknowledged=1")
    mysql.connection.commit()
    cur.close()
    return jsonify({"ok": True})


@app.route("/map")
@login_required
def map_view():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT site_id, risk_label, risk_score,
               slope_angle, rainfall, seismic
        FROM predictions WHERE user_id=%s
        ORDER BY created_at DESC
    """, (session["user_id"],))
    preds = cur.fetchall()
    cur.close()
    return render_template("map.html", predictions=safe_json([dict(p) for p in preds]))


@app.route("/export")
@login_required
def export():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT site_id, slope_angle, rainfall, rock_density,
               crack_length, seismic,
               risk_score, risk_label, created_at
        FROM predictions WHERE user_id=%s ORDER BY created_at DESC
    """, (session["user_id"],))
    rows = cur.fetchall()
    cur.close()

    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader()
    writer.writerows(rows)

    filename = f"slopesentinel_export_{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# ─────────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────────
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT id, full_name, email, role, is_active, is_verified,
               created_at, last_login
        FROM users ORDER BY is_verified ASC, created_at DESC
    """)
    users = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS cnt FROM alerts WHERE acknowledged=0")
    alert_count = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM predictions")
    total_preds = cur.fetchone()["cnt"]

    cur.execute("""
        SELECT a.*, u.full_name AS actor_name
        FROM activity_log a
        LEFT JOIN users u ON a.user_id=u.id
        ORDER BY a.created_at DESC LIMIT 20
    """)
    activity = cur.fetchall()

    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 20")
    sys_alerts = cur.fetchall()

    cur.close()
    return render_template("admin.html",
        users=users, alert_count=alert_count,
        total_preds=total_preds, activity=activity,
        sys_alerts=sys_alerts
    )


@app.route("/admin/approve/<int:uid>", methods=["POST"])
@login_required
@admin_required
def approve(uid):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=1,is_active=1 WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.execute("SELECT full_name FROM users WHERE id=%s", (uid,))
    u = cur.fetchone()
    cur.close()
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
    u = cur.fetchone()
    cur.close()
    log_activity("reject_user", u["full_name"] if u else str(uid))
    flash(f"Access rejected for {u['full_name'] if u else uid}.", "warning")
    return redirect(url_for("admin_panel"))


@app.route("/admin/delete/<int:uid>", methods=["POST"])
@login_required
@admin_required
def delete_user(uid):
    if uid == session["user_id"]:
        flash("Cannot delete your own account.", "danger")
        return redirect(url_for("admin_panel"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT full_name FROM users WHERE id=%s", (uid,))
    u = cur.fetchone()
    cur.execute("DELETE FROM users WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.close()
    log_activity("delete_user", u["full_name"] if u else str(uid))
    flash(f"User deleted.", "danger")
    return redirect(url_for("admin_panel"))


@app.route("/admin/invite", methods=["POST"])
@login_required
@admin_required
def invite_user():
    name  = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    role  = request.form.get("role", "engineer")
    temp  = secrets.token_urlsafe(10)
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified)
            VALUES (%s,%s,%s,%s,1,0)
        """, (name, email, generate_password_hash(temp), role))
        mysql.connection.commit()
        cur.close()
        log_activity("invite_user", f"{email} ({role})")
        flash(f"Invite created for {name}. Temp password: {temp}", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("admin_panel"))


@app.route("/api/admin/toggle/<int:uid>", methods=["POST"])
@login_required
@admin_required
def toggle_user(uid):
    if uid == session["user_id"]:
        return jsonify({"error": "Cannot deactivate yourself."}), 400
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_active = NOT is_active WHERE id=%s", (uid,))
    mysql.connection.commit()
    cur.execute("SELECT is_active FROM users WHERE id=%s", (uid,))
    r = cur.fetchone()
    cur.close()
    return jsonify({"is_active": bool(r["is_active"])})


@app.route("/admin/broadcast", methods=["POST"])
@login_required
@admin_required
def broadcast_alert():
    title   = request.form.get("title", "System Alert")
    message = request.form.get("message", "")
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO alerts (site_id,severity,title,message)
        VALUES ('System','warning',%s,%s)
    """, (title, message))
    mysql.connection.commit()
    cur.close()
    log_activity("broadcast_alert", title)
    flash("Alert broadcast to all users.", "success")
    return redirect(url_for("admin_panel"))


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("404.html", code=500,
                           msg="Internal server error."), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    setup_database()
    app.run(debug=True, port=5000)