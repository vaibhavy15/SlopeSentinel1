from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import MySQLdb
import os
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret"

# ---------------- DB CONFIG ----------------
app.config["MYSQL_HOST"] = "127.0.0.1"
app.config["MYSQL_PORT"] = 3306
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "1501"   # 🔴 CHANGE THIS
app.config["MYSQL_DB"] = "slopesentinel"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

# ---------------- DB SETUP ----------------
def setup_database():
    db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="1501",   # 🔴 SAME PASSWORD
        port=3306
    )
    cursor = db.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS slopesentinel")
    cursor.execute("USE slopesentinel")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(100),
        email VARCHAR(100) UNIQUE,
        password_hash TEXT,
        role ENUM('engineer','admin') DEFAULT 'engineer',
        is_active BOOLEAN DEFAULT TRUE,
        is_verified BOOLEAN DEFAULT FALSE
    )
    """)

    cursor.execute("SELECT * FROM users WHERE email='admin@test.com'")
    if not cursor.fetchone():
        hashed = generate_password_hash("123456")
        cursor.execute("""
        INSERT INTO users (full_name,email,password_hash,role,is_verified)
        VALUES (%s,%s,%s,%s,%s)
        """, ("Admin","admin@test.com",hashed,"admin",True))

    db.commit()
    db.close()

# ---------------- DECORATORS ----------------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrap

# ---------------- ROUTES ----------------

# ✅ FIXED (was home → now index)
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = cur.fetchone()

        if not user:
            return "User not found"
        if not user["is_verified"]:
            return "Account not approved"
        if not check_password_hash(user["password_hash"],password):
            return "Wrong password"

        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["name"] = user["full_name"]

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (full_name,email,password_hash) VALUES (%s,%s,%s)",
                    (name,email,password))
        mysql.connection.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user={
        "full_name": session.get("name"),
        "role": session.get("role")
    })


@app.route("/admin")
@login_required
@admin_required
def admin():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    return render_template("admin.html", users=users)


# ---------------- ADMIN ACTIONS ----------------



@app.route("/toggle/<int:id>")
@login_required
@admin_required
def toggle(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_active = NOT is_active WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for("admin"))



# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

import folium
from flask import send_file
import csv
from io import StringIO

# ---------------- MINE MAP ----------------
@app.route("/map")
@login_required
def map_view():
    m = folium.Map(location=[28.6, 77.2], zoom_start=5)

    folium.Marker([28.6, 77.2], tooltip="Mine 1", popup="Safe").add_to(m)
    folium.Marker([22.5, 88.3], tooltip="Mine 2", popup="Critical").add_to(m)

    map_html = m._repr_html_()
    return render_template("map.html", map=map_html)


# ---------------- UPLOAD + PREDICT ----------------
@app.route("/upload", methods=["GET","POST"])
@login_required
def upload_page():
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)

        df_scaled = scaler.transform(df)
        preds = model.predict(df_scaled)

        results = []
        for i, p in enumerate(preds):
            label = ["Safe","Caution","Critical"][int(p)]
            results.append({"id": i+1, "result": label})

        return render_template("upload.html", results=results)

    return render_template("upload.html")


# ---------------- ALERTS ----------------
@app.route("/alerts")
@login_required
def alerts():
    alerts_data = [
        {"msg": "Slope instability detected", "level": "Critical"},
        {"msg": "Water level rising", "level": "Caution"}
    ]
    return render_template("alerts.html", alerts=alerts_data)


# ---------------- EXPORT REPORT ----------------
@app.route("/export")
@login_required
def export():
    si = StringIO()
    writer = csv.writer(si)

    writer.writerow(["Site","Status"])
    writer.writerow(["Mine 1","Safe"])
    writer.writerow(["Mine 2","Critical"])

    output = si.getvalue()

    return app.response_class(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=report.csv"}
    )


# ---------------- NEW ANALYSIS ----------------
@app.route("/analysis")
@login_required
def analysis():
    return render_template("analysis.html")
from flask import jsonify

# ---------------- MAP DATA ----------------
@app.route("/api/map")
@login_required
def get_map():
    data = [
        {"name": "North Pit Alpha", "status": "Safe"},
        {"name": "Eastern Haul Road", "status": "Critical"}
    ]
    return jsonify(data)


# ---------------- ALERTS DATA ----------------
@app.route("/api/alerts")
@login_required
def get_alerts():
    alerts = [
        {
            "title": "Critical Rockfall Warning",
            "desc": "High risk detected at Eastern Haul Road!",
            "time": "22:00"
        }
    ]
    return jsonify(alerts)


# ---------------- PREDICTION ----------------
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)

    result = []
    for i, p in enumerate(preds):
        label = ["Safe","Caution","Critical"][int(p)]
        result.append({
            "id": i+1,
            "risk": label
        })

    return jsonify(result)
@app.route("/approve/<int:id>", methods=["POST"])
def approve(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=1 WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for("admin"))

@app.route("/reject/<int:id>", methods=["POST"])
def reject(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=0 WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for("admin"))

@app.route("/delete/<int:id>", methods=["POST"])
def delete_user(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (id,))
    mysql.connection.commit()
    return redirect(url_for("admin"))
# ---------------- RUN ----------------
if __name__ == "__main__":
    setup_database()
    app.run(debug=True)