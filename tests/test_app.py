"""Exercise the real login, setup, prediction and logout flows."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from io import BytesIO
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from model import ROOT, predict_grade
from auth import AuthStore
from cohort import csv_template
from model_service import load_bundle
from test_subject_training import history_fixture

PASSWORD = "a private learning workspace"


def fill(app, label, value):
    next(field for field in app.text_input if field.label == label).set_value(value)


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()


def metric(app, label):
    return next(item for item in app.metric if item.label == label)


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
        self.assertEqual(len(app.tabs), 4)

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
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(app.tabs[0].label, "Class dashboard")
        self.assertEqual(metric(app, "Students in view").value, "79")
        app.number_input(key="profile_G1").set_value(10)
        app.number_input(key="profile_G2").set_value(11)
        click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        pipeline = load_bundle("mat")[0]
        profile = pd.DataFrame([dict(G1=10, G2=11, studytime=2, failures=0, schoolsup="no", famsup="yes",
                                     traveltime=1, paid="no", activities="yes", internet="yes", higher="yes")])
        expected = float(predict_grade(pipeline, profile)[0])
        self.assertEqual(metric(app, "Estimated final grade").value, f"{expected:.1f} / 20")
        self.assertEqual(metric(app, "Percentage equivalent").value, f"{expected * 5:.1f}%")
        report = app.session_state["grade_report"]["report"]
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]["subject"], "Mathematics")
        app.run()
        self.assertIsNotNone(metric(app, "Estimated final grade"))
        app.selectbox(key="subject").set_value("por").run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Estimated final grade", [item.label for item in app.metric])
        self.assertEqual(metric(app, "Students in view").value, "130")
        click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        portuguese = load_bundle("por")[0]
        expected_por = float(predict_grade(portuguese, profile)[0])
        self.assertEqual(metric(app, "Estimated final grade").value, f"{expected_por:.1f} / 20")
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
        self.assertEqual(len(app.tabs), 4)

    def test_dashboard_filters_and_upload_empty_state(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        groups = next(widget for widget in app.multiselect if widget.label == "Classes / groups")
        groups.set_value([]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Students in view", [item.label for item in app.metric])
        self.assertTrue(any("Select at least one" in item.value for item in app.info))
        groups = next(widget for widget in app.multiselect if widget.label == "Classes / groups")
        groups.set_value([groups.options[0]]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(int(metric(app, "Students in view").value), 0)
        app.text_input(key="cohort_search").set_value("nonexistent-student-code").run()
        self.assertTrue(any("No student records match" in item.value for item in app.info))
        app.radio(key="cohort_source").set_value("Upload class CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Upload a class CSV" in item.value for item in app.info))
        self.assertNotIn("Students in view", [item.label for item in app.metric])

    def test_class_upload_renders_and_bad_upload_clears_dashboard(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        with patch("dashboard.st.file_uploader", return_value=BytesIO(csv_template())):
            app.radio(key="cohort_source").set_value("Upload class CSV").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(metric(app, "Students in view").value, "1")
        with patch("dashboard.st.file_uploader", return_value=BytesIO(b"bad,columns\n1,2\n")):
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Missing columns", app.error[0].value)
            self.assertNotIn("Students in view", [item.label for item in app.metric])

    def test_fresh_host_without_models_opens_dashboard(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory:
            with patch("model_service.load_bundle", side_effect=lambda subject: load_bundle(subject, artifact_dir=Path(directory))):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                self.login(app)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.tabs[0].label, "Class dashboard")
            self.assertEqual(metric(app, "Students in view").value, "79")
            self.assertEqual(len(app.code), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_new_subject_training_switching_and_session_isolation(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.login(app)
        for subject in ["eng", "phy", "chem", "bio", "geo", "hist", "cs"]:
            app.selectbox(key="subject").set_value(subject).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 0)
            self.assertTrue(any("No reference dataset" in item.value for item in app.info))
        app.selectbox(key="subject").set_value("eng").run()
        content = history_fixture().to_csv(index=False).encode()
        with patch("streamlit.file_uploader", return_value=BytesIO(content)):
            app.run()
            click(app, "Train English model")
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(metric(app, "Students in view").value, "12")
            saved = app.session_state["trained_subject_eng"]
            self.assertEqual(saved["bundle"][1]["subject"], "eng")
            held_out = set(saved["bundle"][1]["split_rows"]["test"])
            fitting = set(saved["bundle"][1]["split_rows"]["train"])
            self.assertFalse(held_out & fitting)
            click(app, "Predict final grade")
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["grade_report"]["report"].iloc[0]["subject"], "English")
        with patch("streamlit.file_uploader", return_value=BytesIO(b"bad,columns\n1,2\n")):
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(len(app.metric), 0)
        app.selectbox(key="subject").set_value("phy").run()
        self.assertEqual(len(app.tabs), 0)
        app.selectbox(key="subject").set_value("eng").run()
        self.assertEqual(metric(app, "Students in view").value, "12")
        other = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.login(other)
        other.selectbox(key="subject").set_value("eng").run()
        self.assertEqual(len(other.tabs), 0)
        click(app, "Clear this subject model")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        with self.assertRaises(KeyError):
            app.session_state["trained_subject_eng"]
        click(app, "Sign out")
        self.assertEqual(app.title[0].value, "Welcome back.")


if __name__ == "__main__":
    unittest.main()
