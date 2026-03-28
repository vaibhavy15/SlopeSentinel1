# SlopeSentinel — Setup Guide

## Project Structure

```
slopesentinel/
├── app.py                  ← Flask application (routes, auth logic)
├── seed_db.py              ← One-time DB setup + demo user seeding
├── schema.sql              ← Raw SQL schema (alternative to seed_db.py)
├── requirements.txt        ← Python dependencies
│
├── templates/
│   ├── base.html           ← Base layout (inherited by all pages)
│   ├── index.html          ← Landing page
│   ├── login.html          ← Login page with tabs + form validation
│   ├── dashboard.html      ← Main dashboard (protected)
│   └── admin.html          ← Admin user management (admin-only)
│
└── static/
    ├── css/styles.css      ← All styles
    └── js/app.js           ← Frontend interactions (charts, map, upload)
```

---

## Step 1 — Install dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

> **Note:** `Flask-MySQLdb` requires the MySQL client library.
> - Ubuntu/Debian: `sudo apt-get install libmysqlclient-dev`
> - macOS: `brew install mysql-client`
> - Windows: Install [MySQL Connector C](https://dev.mysql.com/downloads/connector/c/)

---

## Step 2 — Create the MySQL database

```bash
# Log into MySQL
mysql -u root -p

# Inside MySQL shell:
CREATE DATABASE slopesentinel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

---

## Step 3 — Configure your credentials

Edit `app.py` and update these lines (or set environment variables):

```python
app.config["MYSQL_HOST"]     = "localhost"         # your MySQL host
app.config["MYSQL_USER"]     = "root"              # your MySQL username
app.config["MYSQL_PASSWORD"] = "your_password"    # your MySQL password
app.config["MYSQL_DB"]       = "slopesentinel"    # database name
```

Or use environment variables:
```bash
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DB=slopesentinel
export SECRET_KEY=change-this-to-something-random
```

---

## Step 4 — Seed the database

```bash
python seed_db.py
```

This will:
- Create the `users` table
- Insert two demo accounts with hashed passwords

---

## Step 5 — Run the app

```bash
python app.py
```

Open: **http://localhost:5000**

---

## Demo Credentials

| Role       | Email                    | Password       |
|------------|--------------------------|----------------|
| Engineer   | engineer@mine-site.com   | Engineer@123   |
| Admin      | admin@mine-site.com      | Admin@123      |

---

## Features

### Auth System
- **Email + Password** login with server-side validation
- **Bcrypt password hashing** (Werkzeug PBKDF2-SHA256)
- **Role-based access control** — `engineer` and `admin` roles
- **Remember Me** — persistent sessions for 7 days
- **Login tab switching** — Engineer/Miner vs Administrator
- **Client-side validation** — email format, empty fields, spinner
- **Server-side error messages** — wrong password, wrong role, inactive account
- **Session protection** — `@login_required` and `@role_required` decorators

### Routes
| Route          | Access      | Description                    |
|----------------|-------------|--------------------------------|
| `/`            | Public      | Landing page                   |
| `/login`       | Public      | Login form (GET + POST)        |
| `/logout`      | Auth        | Clears session, redirects      |
| `/dashboard`   | All users   | Main dashboard                 |
| `/admin`       | Admin only  | User management table          |
| `/api/me`      | All users   | Returns current user JSON      |
| `/api/users`   | Admin only  | List all users JSON            |
| `/api/users/<id>/toggle` | Admin | Toggle user active status |

---

## Security Notes

- Always change `SECRET_KEY` in production
- Use HTTPS in production (set `SESSION_COOKIE_SECURE=True`)
- Never commit credentials — use environment variables
- For production, use Gunicorn: `gunicorn -w 4 app:app`
