"""Authentication checks use isolated stores, never the real workspace account."""

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from auth import AuthError, AuthStore, COOLDOWN_SECONDS, MAX_ATTEMPTS, SESSION_SECONDS

PASSWORD = "a private learning workspace"


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "accounts.sqlite3"
        self.store = AuthStore(self.path)

    def create_owner(self):
        self.store.create_owner("Alex Morgan", "alex", PASSWORD)

    def test_registration_validation_and_unique_usernames(self):
        self.assertTrue(self.store.needs_setup())
        for name, username, password in [("", "alex", PASSWORD), ("Alex", "x", PASSWORD),
                                         ("Alex", "alex", "short"), ("Alex", "alex", " " * 16)]:
            with self.assertRaises(AuthError):
                self.store.create_owner(name, username, password)
        self.create_owner()
        self.assertFalse(self.store.needs_setup())
        self.store.create_account("Another teacher", "other", PASSWORD)
        with self.assertRaisesRegex(AuthError, "already taken"):
            self.store.create_account("Duplicate", " ALEX ", PASSWORD)
        first = self.store.login("alex", PASSWORD)
        second = self.store.login("other", PASSWORD)
        self.assertEqual(self.store.current_user(first)["display_name"], "Alex Morgan")
        self.assertEqual(self.store.current_user(second)["display_name"], "Another teacher")
        self.store.logout(first)
        self.assertIsNotNone(self.store.current_user(second))

    def test_hash_storage_persistence_and_revocation(self):
        self.create_owner()
        store = AuthStore(self.path)
        token = store.login(" ALEX ", PASSWORD)
        self.assertEqual(store.current_user(token)["display_name"], "Alex Morgan")
        with closing(sqlite3.connect(self.path)) as db:
            salt, digest = db.execute("SELECT salt, password_hash FROM users").fetchone()
            saved_token = db.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertEqual(len(salt), 16)
        self.assertEqual(len(digest), 32)
        self.assertNotIn(PASSWORD.encode(), self.path.read_bytes())
        self.assertNotEqual(saved_token, token)
        self.assertIsNone(store.current_user("forged-token"))
        store.logout(token)
        self.assertIsNone(store.current_user(token))

    def test_generic_errors_and_per_account_rate_limit(self):
        self.create_owner()
        self.store.create_account("Other teacher", "other", PASSWORD)
        with patch("auth.time.time", return_value=1000):
            messages = []
            for username in ("alex", "unknown"):
                with self.assertRaises(AuthError) as caught:
                    self.store.login(username, "incorrect")
                messages.append(str(caught.exception))
            self.assertEqual(messages[0], messages[1])
            for _ in range(MAX_ATTEMPTS - 1):
                with self.assertRaises(AuthError):
                    AuthStore(self.path).login("alex", "incorrect")
            with self.assertRaisesRegex(AuthError, "Wait one minute"):
                AuthStore(self.path).login("alex", PASSWORD)
            self.assertIsNotNone(self.store.login("other", PASSWORD))
        with patch("auth.time.time", return_value=1000 + COOLDOWN_SECONDS + 1):
            self.assertIsNotNone(self.store.login("alex", PASSWORD))

    def test_session_expires(self):
        self.create_owner()
        with patch("auth.time.time", return_value=1000):
            token = self.store.login("alex", PASSWORD)
            self.assertIsNotNone(self.store.current_user(token))
        with patch("auth.time.time", return_value=1000 + SESSION_SECONDS):
            self.assertIsNone(self.store.current_user(token))

    def test_legacy_account_and_session_migration_is_repeatable(self):
        path = Path(self.directory.name) / "legacy.sqlite3"
        salt = b"test salt value!"
        digest = hashlib.pbkdf2_hmac("sha256", PASSWORD.encode(), salt, 600_000)
        token = "legacy-session-test-token"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE owner (id INTEGER PRIMARY KEY CHECK (id=1),
                    username TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
                    salt BLOB NOT NULL, password_hash BLOB NOT NULL);
                CREATE TABLE sessions (token_hash TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL REFERENCES owner(id), expires_at REAL NOT NULL);
                CREATE TABLE login_guard (id INTEGER PRIMARY KEY CHECK (id=1),
                    failures INTEGER NOT NULL DEFAULT 0, blocked_until REAL NOT NULL DEFAULT 0);
                INSERT INTO login_guard VALUES (1, 2, 0);
            """)
            db.execute("INSERT INTO owner VALUES (1, 'alex', 'Alex Morgan', ?, ?)", (salt, digest))
            db.execute("INSERT INTO sessions VALUES (?, 1, ?)", (token_hash, 4_000_000_000))
            db.commit()
        for _ in range(2):
            store = AuthStore(path)
            self.assertEqual(store.current_user(token)["username"], "alex")
        with store.connection() as db:
            row = db.execute("SELECT salt, password_hash FROM users WHERE id=1").fetchone()
            self.assertEqual(tuple(row), (salt, digest))
            self.assertEqual(db.execute("SELECT failures FROM account_login_guard WHERE username='alex'").fetchone()[0], 2)
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertIsNotNone(store.login("alex", PASSWORD))
        store.create_account("New teacher", "newteacher", PASSWORD)
        new_token = store.login("newteacher", PASSWORD)
        self.assertEqual(store.current_user(new_token)["username"], "newteacher")
        self.assertEqual(store.current_user(token)["username"], "alex")


if __name__ == "__main__":
    unittest.main()
