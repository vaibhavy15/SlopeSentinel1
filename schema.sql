-- ================================================================
--  SlopeSentinel — Complete MySQL Schema
--  Run:  mysql -u root -p < schema.sql
-- ================================================================

CREATE DATABASE IF NOT EXISTS slopesentinel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE slopesentinel;

-- ── Users ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    full_name     VARCHAR(120) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          ENUM('engineer','admin') NOT NULL DEFAULT 'engineer',
    is_active     TINYINT(1) NOT NULL DEFAULT 1,
    is_verified   TINYINT(1) NOT NULL DEFAULT 0,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ── Predictions ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    user_id             INT,
    site_id             VARCHAR(50),
    slope_angle         FLOAT,
    rainfall            FLOAT,
    rock_density        FLOAT,
    crack_length        FLOAT,
    groundwater         FLOAT,
    blasting            FLOAT,
    seismic             FLOAT,
    bench_height        FLOAT,
    excavation          FLOAT,
    temperature         FLOAT,
    risk_score          FLOAT,
    risk_level          INT,
    risk_label          VARCHAR(20),
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- ── Alerts ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id  INT NULL,
    site_id        VARCHAR(50),
    severity       ENUM('info','warning','critical') DEFAULT 'info',
    title          VARCHAR(200),
    message        TEXT,
    acknowledged   TINYINT(1) DEFAULT 0,
    created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ── Activity log ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS activity_log (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT NULL,
    action     VARCHAR(200),
    detail     VARCHAR(500),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ── Seed demo accounts ─────────────────────────────────────────
-- Admin:    admin@slopesentinel.com  / Admin@123
-- Engineer: engineer@mine-site.com  / Engineer@123
-- (Run seed_db.py instead for correct hashes, OR run app.py which auto-seeds)

INSERT IGNORE INTO alerts (site_id, severity, title, message)
VALUES (
    'Eastern Haul Road', 'critical',
    'Critical Rockfall Warning',
    'High risk detected at Eastern Haul Road. Slope stability index dropped below threshold. Immediate inspection required.'
);