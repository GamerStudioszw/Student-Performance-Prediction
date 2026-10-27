"""Exercise the real login, setup, prediction and logout flows."""

import os
import sqlite3
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
        click(app, "Sign up")
        self.assertEqual(app.title[0].value, "Create your staff account")
        click(app, "Sign in")
        self.assertEqual(app.title[0].value, "Welcome back.")
        click(app, "Sign up")
        fill(app, "Your name", "Alex Morgan")
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", "a different password")
        click(app, "Create account →")
        self.assertTrue(AuthStore().needs_setup())
        self.assertIn("do not match", app.error[0].value)
        self.assertEqual(app.text_input(key="auth_name").value, "Alex Morgan")
        self.assertEqual(app.text_input(key="auth_username").value, "alex")
        self.assertEqual(app.text_input(key="auth_password").value, "")
        self.assertEqual(app.text_input(key="auth_confirmation").value, "")
        fill(app, "Your name", "Alex Morgan")
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", PASSWORD)
        click(app, "Create account →")
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(AuthStore().needs_setup())
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("Sign up", [button.label for button in app.button])
        self.assertEqual(app.text_input(key="auth_username").value, "alex")
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
        self.assertEqual(app.text_input(key="auth_username").value, "alex")
        self.assertEqual(app.text_input(key="auth_password").value, "")
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

    def test_additional_staff_can_register_and_sign_in(self):
        store = AuthStore()
        store.create_account("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertIn("Sign up", [button.label for button in app.button])
        captions = " ".join(item.value for item in app.caption)
        self.assertNotIn("registered with this app", captions)
        self.assertNotIn("registration is closed", captions)
        click(app, "Sign up")
        fill(app, "Your name", "New Teacher")
        fill(app, "Username", " ALEX ")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", PASSWORD)
        click(app, "Create account →")
        self.assertIn("already taken", app.error[0].value)
        self.assertEqual(app.text_input(key="auth_password").value, "")
        fill(app, "Username", "teacher2")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", PASSWORD)
        click(app, "Create account →")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual(app.text_input(key="auth_username").value, "teacher2")
        fill(app, "Password", PASSWORD)
        click(app, "Sign in →")
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(store.current_user(app.session_state["auth_token"])["username"], "teacher2")
        app.session_state["private_marker"] = "teacher2-only"
        click(app, "Sign out")
        self.assertNotIn("private_marker", app.session_state)
        self.login(app)
        self.assertEqual(store.current_user(app.session_state["auth_token"])["username"], "alex")
        self.assertEqual(len(app.tabs), 4)

    def test_blank_login_does_not_consume_attempts(self):
        store = AuthStore()
        store.create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        for _ in range(6):
            click(app, "Sign in →")
            self.assertIn("Enter your username and password", app.error[0].value)
        self.login(app)
        self.assertEqual(len(app.tabs), 4)

    def test_auth_navigation_clears_passwords(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        click(app, "Sign up")
        fill(app, "Your name", "Alex Morgan")
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        fill(app, "Confirm password", "mismatch")
        click(app, "Create account →")
        click(app, "Sign in")
        self.assertEqual(app.title[0].value, "Welcome back.")
        self.assertEqual(app.text_input(key="auth_password").value, "")
        self.assertEqual(len(app.error), 0)
        self.assertTrue(AuthStore().needs_setup())

    def test_database_failure_does_not_expose_details_or_open_dashboard(self):
        AuthStore().create_account("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        fill(app, "Username", "alex")
        fill(app, "Password", PASSWORD)
        with patch("auth.AuthStore.login", side_effect=sqlite3.OperationalError("private/database/path")):
            click(app, "Sign in →")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertIn("account store is unavailable", app.error[0].value)
        self.assertNotIn("private/database/path", app.error[0].value)
        self.assertEqual(app.text_input(key="auth_password").value, "")

    def test_invalid_profile_clears_previous_prediction(self):
        AuthStore().create_account("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        click(app, "Predict final grade")
        self.assertIn("Estimated final grade", [item.label for item in app.metric])
        with patch("input_ui.profile_fields", return_value=pd.DataFrame({"G2": [True]})):
            click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(len(app.error))
        self.assertNotIn("grade_report", app.session_state)
        self.assertNotIn("Estimated final grade", [item.label for item in app.metric])

    def test_named_single_student_report_and_identity_validation(self):
        AuthStore().create_account("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        self.assertEqual(app.tabs[1].label, "Single-student prediction")
        fill(app, "Student name (optional)", "Tariro Moyo")
        fill(app, "Class / group (optional)", "Form 3A")
        app.number_input(key="profile_G2").set_value(7)
        with patch("streamlit.download_button", wraps=st.download_button) as downloads:
            click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Student: Tariro Moyo", [item.value for item in app.text])
        report = app.session_state["grade_report"]["report"]
        self.assertEqual(report.student_name.iloc[0], "Tariro Moyo")
        self.assertEqual(report.class_name.iloc[0], "Form 3A")
        self.assertEqual(report.student_id.iloc[0], "")
        call = next(call for call in downloads.call_args_list if call.args[0] == "Download this prediction report")
        exported = pd.read_csv(BytesIO(call.args[1]))
        self.assertEqual(exported.student_name.iloc[0], "Tariro Moyo")
        fill(app, "Student name (optional)", "Another Student")
        click(app, "Predict final grade")
        self.assertEqual(app.session_state["grade_report"]["report"].predicted_grade.iloc[0], report.predicted_grade.iloc[0])
        fill(app, "Student name (optional)", "Bad\u202ename")
        click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(len(app.error))
        self.assertNotIn("grade_report", app.session_state)

    def test_names_only_class_upload_search_and_download(self):
        AuthStore().create_account("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.login(app)
        content = b"student_name,class_name,G2\nTariro Moyo,Form 3A,12\nAlex Morgan,Form 3A,8\n"
        with patch("streamlit.file_uploader", return_value=BytesIO(content)):
            app.radio(key="cohort_source").set_value("Upload class CSV").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(metric(app, "Students in view").value, "2")
            table = next(item.value for item in app.dataframe if "Student name" in item.value.columns)
            self.assertEqual(set(table["Student name"]), {"Tariro Moyo", "Alex Morgan"})
            self.assertNotIn("Student ID", table.columns)
            with patch("streamlit.download_button", wraps=st.download_button) as downloads:
                app.text_input(key="cohort_search").set_value("tariro").run()
            self.assertEqual(len(app.exception), 0)
            call = next(call for call in downloads.call_args_list if call.args[0] == "Download filtered student report")
            exported = pd.read_csv(BytesIO(call.args[1]))
            self.assertEqual(exported.student_name.tolist(), ["Tariro Moyo"])

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

    def test_minimal_inputs_and_downloads_match_displayed_forecast(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.login(app)
        self.assertNotIn("Weekly study time", [widget.label for widget in app.selectbox])
        self.assertIsNone(app.number_input(key="profile_G1").value)
        app.number_input(key="profile_G2").set_value(7)
        with patch("streamlit.download_button", wraps=st.download_button) as downloads:
            click(app, "Predict final grade")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(metric(app, "Estimated final grade").value, "7.0 / 20")
        self.assertEqual(metric(app, "Observed change: G1 to G2").value, "Unavailable")
        saved = app.session_state["grade_report"]
        call = next(call for call in downloads.call_args_list if call.args[0] == "Download this prediction report")
        exported = pd.read_csv(BytesIO(call.args[1]))
        self.assertEqual(exported.predicted_grade.iloc[0], 7)
        self.assertEqual(exported.interval_lower.iloc[0], 6)
        self.assertEqual(exported.bundle_id.iloc[0], saved["bundle_id"])
        self.assertTrue(pd.isna(exported.G1.iloc[0]))
        self.assertNotIn("studytime", exported)

    def test_minimal_class_upload_and_alert_exports_are_separate_from_model_test(self):
        AuthStore().create_owner("Alex Morgan", "alex", PASSWORD)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.login(app)
        payload = b"student_id,class_name,G2,G3\na,Class A,9,9\nb,Class A,10,10\n"
        with patch("streamlit.file_uploader", return_value=BytesIO(payload)):
            with patch("streamlit.download_button", wraps=st.download_button) as downloads:
                app.radio(key="cohort_source").set_value("Upload class CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(metric(app, "Students in view").value, "2")
        alert_calls = {call.kwargs.get("key"): call for call in downloads.call_args_list
                       if call.args[0] == "Download alert evaluation"}
        cohort = pd.read_csv(BytesIO(alert_calls["cohort_alerts_mat"].args[1])).iloc[0]
        model = pd.read_csv(BytesIO(alert_calls["model_alerts_mat"].args[1])).iloc[0]
        self.assertEqual(cohort.evaluated, 2)
        self.assertEqual(model.evaluated, 79)
        self.assertEqual(cohort.precision, 1)
        self.assertEqual(cohort.tp, 1)
        self.assertEqual(cohort.tn, 1)
        self.assertIn("Uploaded class", cohort.source)
        self.assertEqual(cohort.bundle_id, model.bundle_id)

    def test_each_feature_schema_has_matching_form_and_csv_requirements(self):
        from model import FEATURE_SETS
        from cohort import read_cohort_csv
        for schema, features in FEATURE_SETS.items():
            script = ("import streamlit as st\nfrom input_ui import profile_fields\n"
                      f"with st.form('input'):\n    profile_fields({features!r})\n    st.form_submit_button('Submit')")
            app = AppTest.from_string(script).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual("Weekly study time" in [widget.label for widget in app.selectbox], schema == "expanded_v1")
            self.assertEqual(app.number_input(key="profile_G1").value is None, "G1" not in features)
            self.assertTrue(set(features).issubset(read_cohort_csv(csv_template(features), features)))


if __name__ == "__main__":
    unittest.main()
