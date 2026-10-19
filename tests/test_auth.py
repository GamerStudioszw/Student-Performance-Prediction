"""Authentication checks use isolated stores, never the real workspace account."""

from contextlib import closing
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

    def test_setup_validation_and_single_owner(self):
        self.assertTrue(self.store.needs_setup())
        for name, username, password in [("", "alex", PASSWORD), ("Alex", "x", PASSWORD),
                                         ("Alex", "alex", "short"), ("Alex", "alex", " " * 16)]:
            with self.assertRaises(AuthError):
                self.store.create_owner(name, username, password)
        self.create_owner()
        self.assertFalse(self.store.needs_setup())
        with self.assertRaises(AuthError):
            self.store.create_owner("Another owner", "other", PASSWORD)

    def test_hash_storage_persistence_and_revocation(self):
        self.create_owner()
        store = AuthStore(self.path)
        token = store.login(" ALEX ", PASSWORD)
        self.assertEqual(store.current_user(token)["display_name"], "Alex Morgan")
        with closing(sqlite3.connect(self.path)) as db:
            salt, digest = db.execute("SELECT salt, password_hash FROM owner").fetchone()
            saved_token = db.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertEqual(len(salt), 16)
        self.assertEqual(len(digest), 32)
        self.assertNotIn(PASSWORD.encode(), self.path.read_bytes())
        self.assertNotEqual(saved_token, token)
        self.assertIsNone(store.current_user("forged-token"))
        store.logout(token)
        self.assertIsNone(store.current_user(token))

    def test_generic_errors_and_shared_rate_limit(self):
        self.create_owner()
        with patch("auth.time.time", return_value=1000):
            messages = []
            for username in ("alex", "unknown"):
                with self.assertRaises(AuthError) as caught:
                    self.store.login(username, "incorrect")
                messages.append(str(caught.exception))
            self.assertEqual(messages[0], messages[1])
            for _ in range(MAX_ATTEMPTS - 2):
                with self.assertRaises(AuthError):
                    AuthStore(self.path).login("alex", "incorrect")
            with self.assertRaisesRegex(AuthError, "Wait one minute"):
                AuthStore(self.path).login("alex", PASSWORD)
        with patch("auth.time.time", return_value=1000 + COOLDOWN_SECONDS + 1):
            self.assertIsNotNone(self.store.login("alex", PASSWORD))

    def test_session_expires(self):
        self.create_owner()
        with patch("auth.time.time", return_value=1000):
            token = self.store.login("alex", PASSWORD)
            self.assertIsNotNone(self.store.current_user(token))
        with patch("auth.time.time", return_value=1000 + SESSION_SECONDS):
            self.assertIsNone(self.store.current_user(token))


if __name__ == "__main__":
    unittest.main()
