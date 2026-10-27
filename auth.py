"""Local staff accounts and sessions for the Streamlit workspace."""

from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time

ITERATIONS = 600_000
SESSION_SECONDS = 8 * 60 * 60
MAX_ATTEMPTS = 5
COOLDOWN_SECONDS = 60


class AuthError(ValueError):
    """A message safe to display in the account form."""


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, path=None):
        default = Path(__file__).resolve().parent / ".local" / "accounts.sqlite3"
        self.path = Path(path or os.environ.get("STUDENT_AUTH_DB", default))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            # Upgrade old installations atomically, including active sessions.
            # The write lock also serializes simultaneous first requests.
            db.execute("BEGIN IMMEDIATE")
            db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL
                )
            """)
            legacy = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'owner'").fetchone()
            if legacy:
                db.execute("INSERT INTO users SELECT * FROM owner")
                db.execute("ALTER TABLE sessions RENAME TO legacy_sessions")
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS account_login_guard (
                    username TEXT PRIMARY KEY,
                    failures INTEGER NOT NULL DEFAULT 0,
                    blocked_until REAL NOT NULL DEFAULT 0
                )
            """)
            if legacy:
                db.execute("INSERT INTO sessions SELECT token_hash, owner_id, expires_at FROM legacy_sessions")
                db.execute("""INSERT INTO account_login_guard
                              SELECT owner.username, login_guard.failures, login_guard.blocked_until
                              FROM owner CROSS JOIN login_guard""")
                db.execute("DROP TABLE legacy_sessions")
                db.execute("DROP TABLE owner")
                db.execute("DROP TABLE login_guard")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def needs_setup(self):
        with self.connection() as db:
            return db.execute("SELECT 1 FROM users").fetchone() is None

    def create_owner(self, display_name, username, password):
        """Compatibility entry point for existing setup scripts."""
        return self.create_account(display_name, username, password)

    def create_account(self, display_name, username, password):
        display_name = display_name.strip()
        username = username.strip().lower()
        if not 1 <= len(display_name) <= 80:
            raise AuthError("Enter your name (up to 80 characters).")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,39}", username):
            raise AuthError("Use 3–40 letters, numbers, dots, underscores or hyphens for your username. Start with a letter or number.")
        if not 15 <= len(password) <= 128 or not password.strip():
            raise AuthError("Use a password or passphrase with 15–128 characters.")
        salt = secrets.token_bytes(16)
        digest = _password_hash(password, salt)
        try:
            with self.connection() as db:
                # UNIQUE enforces normalized usernames even for concurrent requests.
                db.execute("INSERT INTO users (username, display_name, salt, password_hash) VALUES (?, ?, ?, ?)",
                           (username, display_name, salt, digest))
        except sqlite3.IntegrityError as exc:
            raise AuthError("That username is already taken. Choose another username or sign in.") from exc

    def login(self, username, password):
        now = time.time()
        token = None
        normalized_username = username.strip().lower()[:40]
        message = "The username or password is incorrect. Please try again."
        with self.connection() as db:
            # Serialize attempts across browser sessions; commit failures before raising.
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO account_login_guard(username) VALUES (?)", (normalized_username,))
            guard = db.execute("SELECT * FROM account_login_guard WHERE username = ?", (normalized_username,)).fetchone()
            if guard["blocked_until"] > now:
                raise AuthError("Too many sign-in attempts. Wait one minute, then try again.")
            owner = db.execute("SELECT * FROM users WHERE username = ?", (normalized_username,)).fetchone()
            # Unknown accounts still do the expensive hash; no plaintext credentials are saved.
            salt = owner["salt"] if owner else b"\x00" * 16
            digest = _password_hash(password[:128], salt)
            expected = owner["password_hash"] if owner else b"\x00" * 32
            matched = hmac.compare_digest(digest, expected)
            if owner and len(username.strip()) <= 40 and len(password) <= 128 and matched:
                token = secrets.token_urlsafe(32)
                db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
                db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                           (_token_hash(token), owner["id"], now + SESSION_SECONDS))
                db.execute("DELETE FROM account_login_guard WHERE username = ?", (normalized_username,))
            else:
                failures = (0 if guard["blocked_until"] else guard["failures"]) + 1
                blocked_until = now + COOLDOWN_SECONDS if failures >= MAX_ATTEMPTS else 0
                db.execute("UPDATE account_login_guard SET failures = ?, blocked_until = ? WHERE username = ?",
                           (failures, blocked_until, normalized_username))
                if blocked_until:
                    message = "Too many sign-in attempts. Wait one minute, then try again."
        if token is None:
            raise AuthError(message)
        return token

    def current_user(self, token):
        if not isinstance(token, str) or not token:
            return None
        with self.connection() as db:
            row = db.execute("""
                SELECT users.username, users.display_name FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """, (_token_hash(token), time.time())).fetchone()
        return dict(row) if row else None

    def logout(self, token):
        if token:
            with self.connection() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
