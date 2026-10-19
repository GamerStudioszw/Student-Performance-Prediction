"""Exercise the real login, setup, prediction and logout flows."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

from model import ROOT, predict_grade
from auth import AuthStore

PASSWORD = "a private learning workspace"


def fill(app, label, value):
    next(field for field in app.text_input if field.label == label).set_value(value)


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()


@unittest.skipUnless((ROOT / "artifacts" / "model.joblib").exists(), "Run train.py first")
class AppTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db_path = Path(directory.name) / "accounts.sqlite3"
        env = patch.dict(os.environ, {"STUDENT_AUTH_DB": str(self.db_path)})
        env.start()
        self.addCleanup(env.stop)

    def login(self, app):
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        click(app, "Sign in →")
        self.assertEqual(len(app.exception), 0)

    def test_first_time_setup(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual([field.label for field in app.text_input], ["Username", "Password"])
        click(app, "Create account")
        self.assertEqual(app.title[0].value, "Make it yours.")
        click(app, "Sign in")
        self.assertEqual(app.title[0].value, "Welcome back.")
        click(app, "Create account")
        fill(app, "Your name", "Alex Morgan")
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", "a different password")
        click(app, "Create account →")
        self.assertTrue(AuthStore().needs_setup())
        self.assertIn("do not match", app.error[0].value)
        fill(app, "Your name", "Alex Morgan")
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", PASSWORD)
        click(app, "Create account →")
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(AuthStore().needs_setup())
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual(len(app.tabs), 0)
        self.assertNotIn("Create account", [button.label for button in app.button])
        self.login(app)
        self.assertEqual(len(app.tabs), 3)

    def test_render_and_predict(self):
        store = AuthStore()
        store.create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(len(app.number_input), 0)
        fill(app, "Username", "alex")
        fill(app, "Password", "wrong password")
        click(app, "Sign in →")
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("incorrect", app.error[0].value)
        self.login(app)
        self.assertEqual(len(app.tabs), 3)
        app.number_input[0].set_value(10)
        app.number_input[1].set_value(11)
        click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        pipeline = joblib.load(ROOT / "artifacts" / "model.joblib")
        profile = pd.DataFrame([dict(G1=10, G2=11, studytime=2, failures=0, schoolsup="no", famsup="yes",
                                     traveltime=1, paid="no", activities="yes", internet="yes", higher="yes")])
        expected = float(predict_grade(pipeline, profile)[0])
        self.assertEqual(app.metric[0].label, "Estimated final grade")
        self.assertEqual(app.metric[0].value, f"{expected:.1f} / 20")
        self.assertEqual(app.metric[1].label, "Percentage equivalent")
        self.assertEqual(app.metric[1].value, f"{expected * 5:.1f}%")
        report = app.session_state["grade_report"]["report"]
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]["subject"], "Mathematics")
        app.run()
        self.assertEqual(app.metric[0].label, "Estimated final grade")
        app.selectbox(key="subject").set_value("por").run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotEqual(app.metric[0].label, "Estimated final grade")
        click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        portuguese = joblib.load(ROOT / "artifacts" / "por_model.joblib")
        expected_por = float(predict_grade(portuguese, profile)[0])
        self.assertEqual(app.metric[0].value, f"{expected_por:.1f} / 20")
        self.assertEqual(app.session_state["grade_report"]["report"].iloc[0]["subject"], "Portuguese language")
        token = app.session_state["auth_token"]
        click(app, "Sign out")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertIsNone(store.current_user(token))
        app.run()
        self.assertEqual(len(app.tabs), 0)

    def test_revoked_session_returns_to_login(self):
        store = AuthStore()
        store.create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        store.logout(app.session_state["auth_token"])
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("session has ended", app.info[0].value)

    def test_empty_workspace_sign_in_explains_missing_account(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        fill(app, "Username", "existing_elsewhere")
        fill(app, "Password", PASSWORD)
        click(app, "Sign in →")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("No account is registered on this app yet", app.error[0].value)
        self.assertTrue(AuthStore().needs_setup())

    def test_stale_setup_view_cannot_reopen_registration(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        click(app, "Create account")
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertNotIn("Your name", [field.label for field in app.text_input])
        self.login(app)
        self.assertEqual(len(app.tabs), 3)


if __name__ == "__main__":
    unittest.main()
