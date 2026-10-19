"""Local, single-owner authentication for the Streamlit workspace."""

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
            db.executescript("""
                CREATE TABLE IF NOT EXISTS owner (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL REFERENCES owner(id),
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS login_guard (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    failures INTEGER NOT NULL DEFAULT 0,
                    blocked_until REAL NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO login_guard(id) VALUES (1);
            """)

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
            return db.execute("SELECT 1 FROM owner").fetchone() is None

    def create_owner(self, display_name, username, password):
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
                # The singleton primary key also prevents two concurrent setup requests.
                db.execute("INSERT INTO owner VALUES (1, ?, ?, ?, ?)",
                           (username, display_name, salt, digest))
        except sqlite3.IntegrityError as exc:
            raise AuthError("This workspace already has an account. Refresh the page to sign in.") from exc

    def login(self, username, password):
        now = time.time()
        token = None
        message = "The username or password is incorrect. Please try again."
        with self.connection() as db:
            # Serialize attempts across browser sessions; commit failures before raising.
            db.execute("BEGIN IMMEDIATE")
            guard = db.execute("SELECT * FROM login_guard WHERE id = 1").fetchone()
            if guard["blocked_until"] > now:
                raise AuthError("Too many sign-in attempts. Wait one minute, then try again.")
            owner = db.execute("SELECT * FROM owner WHERE username = ?",
                               (username.strip().lower()[:40],)).fetchone()
            # Unknown accounts still do the expensive hash; no plaintext credentials are saved.
            salt = owner["salt"] if owner else b"\x00" * 16
            digest = _password_hash(password[:128], salt)
            expected = owner["password_hash"] if owner else b"\x00" * 32
            matched = hmac.compare_digest(digest, expected)
            if owner and len(username.strip()) <= 40 and len(password) <= 128 and matched:
                token = secrets.token_urlsafe(32)
                db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
                db.execute("INSERT INTO sessions VALUES (?, 1, ?)",
                           (_token_hash(token), now + SESSION_SECONDS))
                db.execute("UPDATE login_guard SET failures = 0, blocked_until = 0 WHERE id = 1")
            else:
                failures = (0 if guard["blocked_until"] else guard["failures"]) + 1
                blocked_until = now + COOLDOWN_SECONDS if failures >= MAX_ATTEMPTS else 0
                db.execute("UPDATE login_guard SET failures = ?, blocked_until = ? WHERE id = 1",
                           (failures, blocked_until))
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
                SELECT owner.username, owner.display_name FROM sessions
                JOIN owner ON owner.id = sessions.owner_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """, (_token_hash(token), time.time())).fetchone()
        return dict(row) if row else None

    def logout(self, token):
        if token:
            with self.connection() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
