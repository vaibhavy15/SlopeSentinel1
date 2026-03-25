from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import timedelta, datetime
import os
import joblib
import pandas as pd
import uuid
import MySQLdb

app = Flask(__name__)

# ---------------- CONFIG ----------------
app.secret_key = "super-secret-key"
app.permanent_session_lifetime = timedelta(days=7)

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "1501"  # 🔴 CHANGE THIS
app.config["MYSQL_DB"] = "slopesentinel"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

# ---------------- AUTO DB SETUP ----------------
def setup_database():
    try:
        db = MySQLdb.connect(
            host="localhost",
            user="root",
            passwd="1501"  # 🔴 CHANGE THIS
        )
        cursor = db.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS slopesentinel")
        cursor.execute("USE slopesentinel")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role ENUM('engineer','admin') DEFAULT 'engineer',
            is_active BOOLEAN DEFAULT TRUE,
            is_verified BOOLEAN DEFAULT FALSE,
            reset_token VARCHAR(255),
            reset_token_expiry DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL
        )
        """)

        cursor.execute("SELECT * FROM users WHERE email='admin@test.com'")
        if not cursor.fetchone():
            hashed = generate_password_hash("123456")
            cursor.execute("""
            INSERT INTO users (full_name, email, password_hash, role, is_verified)
            VALUES (%s, %s, %s, %s, %s)
            """, ("Admin", "admin@test.com", hashed, "admin", True))

            print("✅ Admin user created")

        db.commit()
        db.close()
        print("✅ Database ready")

    except Exception as e:
        print("❌ DB Error:", e)

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- DECORATORS ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def role_required(role):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get("role") != role:
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated
    return wrapper

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    error = None

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if not user:
            error = "No account found"
        elif not check_password_hash(user["password_hash"], password):
            error = "Wrong password"
        elif not user["is_verified"]:
            error = "Account not approved"
        elif not user["is_active"]:
            error = "Account disabled"
        else:
            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["email"] = user["email"]
            session["role"] = user["role"]

            cur.execute("UPDATE users SET last_login = NOW() WHERE id=%s", (user["id"],))
            mysql.connection.commit()

            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    error = None

    if request.method == "POST":
        name = request.form.get("full_name")
        email = request.form.get("email")
        password = request.form.get("password")

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            error = "Email already exists"
        else:
            hashed = generate_password_hash(password)
            cur.execute("""
                INSERT INTO users (full_name, email, password_hash, role, is_verified)
                VALUES (%s,%s,%s,%s,%s)
            """, (name, email, hashed, "engineer", False))
            mysql.connection.commit()

            flash("Account created! Wait for admin approval.", "info")
            return redirect(url_for("login"))

    return render_template("register.html", error=error)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user={
        "full_name": session["user_name"],
        "email": session["email"],
        "role": session["role"]
    })

# ---------------- ADMIN PANEL ----------------
@app.route("/admin")
@login_required
@role_required("admin")
def admin_panel():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    return render_template("admin.html", users=users)

# ---------------- USER APPROVAL ----------------
@app.route("/api/users/<int:user_id>/toggle", methods=["POST"])
@login_required
def toggle_user(user_id):
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    cur = mysql.connection.cursor()
    cur.execute("SELECT is_verified FROM users WHERE id=%s", (user_id,))
    user = cur.fetchone()

    new_status = not user["is_verified"]
    cur.execute("UPDATE users SET is_verified=%s WHERE id=%s", (new_status, user_id))
    mysql.connection.commit()

    return jsonify({"success": True})

# ---------------- AI PREDICTION ----------------
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    file = request.files["file"]

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    path = os.path.join("uploads", file.filename)
    file.save(path)

    df = pd.read_csv(path)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)

    results = []
    for i, p in enumerate(preds):
        label = ["safe","caution","critical"][int(p)]
        results.append({
            "id": f"Site-{i+1}",
            "prediction": label
        })

    return jsonify({"results": results})

# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot", methods=["GET","POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")

        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(minutes=15)

        cur = mysql.connection.cursor()
        cur.execute("""
            UPDATE users SET reset_token=%s, reset_token_expiry=%s WHERE email=%s
        """, (token, expiry, email))
        mysql.connection.commit()

        return f"Reset link: http://localhost:5000/reset/{token}"

    return render_template("forgot.html")

# ---------------- RESET PASSWORD ----------------
@app.route("/reset/<token>", methods=["GET","POST"])
def reset_password(token):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT * FROM users WHERE reset_token=%s AND reset_token_expiry > NOW()
    """, (token,))
    user = cur.fetchone()

    if not user:
        return "Invalid or expired token"

    if request.method == "POST":
        new_password = request.form.get("password")
        hashed = generate_password_hash(new_password)

        cur.execute("""
            UPDATE users SET password_hash=%s, reset_token=NULL WHERE id=%s
        """, (hashed, user["id"]))
        mysql.connection.commit()

        return redirect(url_for("login"))

    return render_template("reset.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    setup_database()
    app.run(debug=True)