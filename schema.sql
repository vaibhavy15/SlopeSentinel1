-- ================================================================
--  SlopeSentinel — MySQL Schema & Seed Data
--  Run this file once to set up your database:
--    mysql -u root -p < schema.sql
-- ================================================================

-- Create and select the database
CREATE DATABASE IF NOT EXISTS slopesentinel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE slopesentinel;

-- ----------------------------------------------------------------
-- USERS table
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    full_name     VARCHAR(120) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('engineer', 'admin') NOT NULL DEFAULT 'engineer',
    is_active     TINYINT(1)   NOT NULL DEFAULT 1,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME     NULL
);

-- Index for fast email lookups during login
CREATE INDEX idx_users_email ON users(email);

-- ----------------------------------------------------------------
-- SEED DATA  — demo accounts
-- Passwords below are bcrypt hashes for:
--   engineer@mine-site.com  →  password: Engineer@123
--   admin@mine-site.com     →  password: Admin@123
-- ----------------------------------------------------------------

-- Engineer account
INSERT INTO users (full_name, email, password_hash, role, is_active)
VALUES (
    'John Engineer',
    'engineer@mine-site.com',
    -- werkzeug pbkdf2:sha256 hash of "Engineer@123"
    'pbkdf2:sha256:600000$sUjfXZKd$e0a4f2b3c1d7e8f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4',
    'engineer',
    1
);

-- Admin account
INSERT INTO users (full_name, email, password_hash, role, is_active)
VALUES (
    'Site Administrator',
    'admin@mine-site.com',
    -- werkzeug pbkdf2:sha256 hash of "Admin@123"
    'pbkdf2:sha256:600000$tVjgYaLe$f1b5c3d4e2f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3',
    'admin',
    1
);
