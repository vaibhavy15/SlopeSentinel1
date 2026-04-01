
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import MySQLdb
import pandas as pd
import folium
import csv
from io import StringIO
import secrets

app = Flask(__name__)
app.secret_key = "secret"

# ---------------- DB CONFIG ----------------
app.config["MYSQL_HOST"] = "127.0.0.1"
app.config["MYSQL_PORT"] = 3306
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "1501"
app.config["MYSQL_DB"] = "slopesentinel"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

# ---------------- DB SETUP ----------------
def setup_database():
    db = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="1501",
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
        is_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP NULL
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

        cur.execute("UPDATE users SET last_login = NOW() WHERE id=%s", (user["id"],))
        mysql.connection.commit()

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users (full_name,email,password_hash) VALUES (%s,%s,%s)",
            (name,email,password)
        )
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


# ---------------- ADMIN PANEL ----------------
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, full_name, email, role, is_active, is_verified,
               created_at, last_login
        FROM users
        ORDER BY is_verified ASC, created_at DESC
    """)
    users = cur.fetchall()

    alert_count = 1  # demo

    cur.close()
    return render_template("admin.html", users=users, alert_count=alert_count)


# ---------------- ADMIN ACTIONS ----------------
@app.route("/admin/approve/<int:id>", methods=["POST"])
@login_required
@admin_required
def approve(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=1, is_active=1 WHERE id=%s", (id,))
    mysql.connection.commit()
    flash("User approved", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/reject/<int:id>", methods=["POST"])
@login_required
@admin_required
def reject(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET is_verified=0, is_active=0 WHERE id=%s", (id,))
    mysql.connection.commit()
    flash("User rejected", "warning")
    return redirect(url_for("admin_panel"))


@app.route("/admin/delete/<int:id>", methods=["POST"])
@login_required
@admin_required
def delete_user(id):
    if id == session["user_id"]:
        flash("Cannot delete yourself", "danger")
        return redirect(url_for("admin_panel"))

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (id,))
    mysql.connection.commit()
    flash("User deleted", "danger")
    return redirect(url_for("admin_panel"))


@app.route("/admin/invite", methods=["POST"])
@login_required
@admin_required
def invite_user():
    name = request.form.get("full_name")
    email = request.form.get("email")
    role = request.form.get("role")

    temp_password = secrets.token_urlsafe(8)
    hashed = generate_password_hash(temp_password)

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO users (full_name,email,password_hash,role,is_active,is_verified)
        VALUES (%s,%s,%s,%s,1,0)
    """, (name,email,hashed,role))
    mysql.connection.commit()

    flash(f"Invite created. Temp password: {temp_password}", "success")
    return redirect(url_for("admin_panel"))


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- MAP ----------------
@app.route("/map")
@login_required
def map_view():
    m = folium.Map(location=[28.6, 77.2], zoom_start=5)
    folium.Marker([28.6, 77.2], tooltip="Mine 1").add_to(m)
    return render_template("map.html", map=m._repr_html_())


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET","POST"])
@login_required
def upload_page():
    return render_template("upload.html")


# ---------------- ALERTS ----------------
@app.route("/alerts")
@login_required
def alerts():
    return render_template("alerts.html")


# ---------------- EXPORT ----------------
@app.route("/export")
@login_required
def export():
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(["Site","Status"])
    writer.writerow(["Mine 1","Safe"])
    return app.response_class(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=report.csv"}
    )


# ---------------- ANALYSIS ----------------
@app.route("/analysis")
@login_required
def analysis():
    return render_template("analysis.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    setup_database()
    app.run(debug=True)

