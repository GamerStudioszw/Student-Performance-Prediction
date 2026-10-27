"""Adversarial tests use temporary stores and synthetic software-test inputs only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from auth import (AuthError, AuthStore, MAX_ATTEMPTS, MAX_REGISTRATIONS_PER_MINUTE,
                  MAX_SESSIONS_PER_ACCOUNT)
from cohort import MAX_ROWS, parse_csv, read_cohort_csv, validate_cohort
from download_data import download, MAX_DOWNLOAD_BYTES
from model import predict_grade, prediction_report, validate_frame
from model_service import _active_directory, _read_json, save_bundle
from subject_training import train_subject

PASSWORD = "a private learning workspace"


class AccountSecurityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "accounts.sqlite3"
        self.store = AuthStore(self.path)

    def test_invalid_types_sizes_controls_and_injection_fail_closed(self):
        for value in (None, [], {}, 123, "", "x" * 10000, "bad\x00name", "bad\ud800name", "a' OR 1=1 --"):
            with self.subTest(value=repr(value)[:60]):
                with self.assertRaises(AuthError):
                    self.store.create_account("Teacher", value, PASSWORD)
                with self.assertRaises(AuthError):
                    self.store.login(value, PASSWORD)
        for value in (None, [], {}, 123, "x" * 10000, "bad\ud800password here", "bad\x00password here"):
            with self.subTest(password_type=type(value)):
                with self.assertRaises(AuthError):
                    self.store.create_account("Teacher", "teacher", value)
                with self.assertRaises(AuthError):
                    self.store.login("teacher", value)
        for value in ("Mr\nTeacher", "Mr\u202eTeacher", "Mr\u200bTeacher"):
            with self.assertRaises(AuthError):
                self.store.create_account(value, "teacher", PASSWORD)
        self.assertTrue(self.store.needs_setup())

    def test_unknown_user_attempts_do_not_grow_database(self):
        for i in range(MAX_ATTEMPTS + 10):
            with self.assertRaises(AuthError):
                self.store.login(f"unknown{i}", PASSWORD)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM account_login_guard").fetchone()[0], 1)

    def test_registration_budget_is_shared_persistent_and_expires(self):
        with patch("auth.time.time", return_value=1000):
            self.store.create_account("Teacher", "teacher", PASSWORD)
            for _ in range(MAX_REGISTRATIONS_PER_MINUTE - 1):
                with self.assertRaisesRegex(AuthError, "already taken"):
                    AuthStore(self.path).create_account("Teacher", "teacher", PASSWORD)
            with patch("auth._password_hash", side_effect=AssertionError("Blocked attempts must not hash")):
                with self.assertRaisesRegex(AuthError, "Wait one minute"):
                    AuthStore(self.path).create_account("Teacher", "another", PASSWORD)
        with patch("auth.time.time", return_value=1061):
            self.store.create_account("Teacher", "another", PASSWORD)

    def test_concurrent_duplicate_registration_creates_exactly_one_account(self):
        def register(name):
            try:
                AuthStore(self.path).create_account("Teacher", name, PASSWORD)
                return True
            except AuthError:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(register, ["Teacher", " teacher ", "TEACHER", "teacher"]))
        self.assertEqual(sum(results), 1)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
        self.assertIsNotNone(self.store.login("teacher", PASSWORD))

    def test_session_cap_revokes_oldest_and_rejects_malformed_tokens(self):
        self.store.create_account("Teacher", "teacher", PASSWORD)
        with patch("auth.time.time", return_value=1000):
            tokens = [self.store.login("teacher", PASSWORD) for _ in range(MAX_SESSIONS_PER_ACCOUNT + 1)]
            self.assertIsNone(self.store.current_user(tokens[0]))
            self.assertIsNotNone(self.store.current_user(tokens[-1]))
            for token in (None, [], {}, 42, "x" * 10000, "\ud800"):
                self.assertIsNone(self.store.current_user(token))
                self.store.logout(token)
            self.store.logout(tokens[-1])
            self.assertIsNone(self.store.current_user(tokens[-1]))

    def test_failed_legacy_migration_rolls_back_without_losing_owner(self):
        path = self.path.parent / "broken-legacy.sqlite3"
        with closing(sqlite3.connect(path)) as db:
            # Simulate an incomplete legacy DB: no sessions table.
            db.execute("CREATE TABLE owner (id, username, display_name, salt, password_hash)")
            db.execute("INSERT INTO owner VALUES (1, 'teacher', 'Teacher', x'01', x'02')")
            db.commit()
        with self.assertRaises(sqlite3.Error):
            AuthStore(path)
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT username FROM owner").fetchone()[0], "teacher")
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='users'").fetchone())


class InputSecurityTests(unittest.TestCase):
    def test_ambiguous_or_malformed_csv_is_rejected(self):
        cases = [
            b"student_id,class_name,G2,G2\na,A,10,20\n",
            b"student_id,class_name,G2,g2\na,A,10,20\n",
            b"student_id,class_name,G2\na,A,10,20\n",
            b"student_id,class_name,G2\na,A\n",
            b"student_id,class_name,G2\na,A,10\n\n",
            b"student_id,,G2\na,A,10\n",
            b" student_id,class_name,G2\na,A,10\n",
            b'student_id,class_name,G2\n"a,A,10\n',
            b"student_id,class_name,G2\na\x00,A,10\n",
            b"student_id,class_name,G2\na,A,10\xff\n",
            b"student_id,class_name,G2\n",
            b"student_id,class_name,G2\na," + b"A" * 1001 + b",10\n",
        ]
        for content in cases:
            with self.subTest(content=content[:70]), self.assertRaises(ValueError):
                read_cohort_csv(content, ["G2"])

    def test_limits_rows_columns_and_identifier_preservation(self):
        content = "student_id,class_name,G2\n" + "".join(f"{i},A,10\n" for i in range(MAX_ROWS))
        self.assertEqual(len(read_cohort_csv(content.encode(), ["G2"])), MAX_ROWS)
        with self.assertRaises(ValueError):
            read_cohort_csv((content + "extra,A,10\n").encode(), ["G2"])
        with self.assertRaises(ValueError):
            parse_csv((",".join(f"field{i}" for i in range(65)) + "\n").encode())
        for delimiter in (",", ";", "\t"):
            content = delimiter.join(["student_id", "class_name", "G2"]) + "\n" + delimiter.join(["NA", "A", "10"])
            self.assertEqual(read_cohort_csv(content.encode(), ["G2"]).student_id.iloc[0], "NA")

    def test_grade_boundaries_and_duplicate_columns(self):
        for bad in (True, np.bool_(False), complex(1), None, np.nan, np.inf, -np.inf, -1, 21, 10.5, "yes", "0x10", [], {}):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                validate_frame(pd.DataFrame({"G2": [bad]}), features=["G2"])
        for good in (0, 20, "0", "20", 10.0):
            self.assertEqual(validate_frame(pd.DataFrame({"G2": [good]}), features=["G2"]).G2.iloc[0], int(good))
        duplicated = pd.DataFrame([[10, 20]], columns=["G2", "G2"])
        for validator in (validate_frame, validate_cohort):
            with self.assertRaises(ValueError):
                validator(duplicated, features=["G2"])

    def test_seeded_malformed_csv_fuzz_never_escapes_validation(self):
        rng = random.Random(42)
        alphabet = 'abc012,;\t\n\r"\x00' + "\u202e"
        for _ in range(250):
            content = "".join(rng.choice(alphabet) for _ in range(rng.randrange(300))).encode()
            try:
                result = read_cohort_csv(content, ["G2"])
            except ValueError:
                continue
            self.assertFalse(result.empty)
            self.assertTrue(result.G2.between(0, 20).all())

    def test_invalid_source_is_rejected_before_training(self):
        for source in (None, "", "x" * 201, "School\nspoofed log"):
            with patch("subject_training.train_and_evaluate", side_effect=AssertionError("Must validate first")):
                with self.assertRaises(ValueError):
                    train_subject(b"", "eng", source=source)

    def test_invalid_predictor_outputs_and_report_parameters_fail_closed(self):
        frame = pd.DataFrame({"G2": [10]})
        fitted = Mock(input_features_=["G2"])
        for prediction in ([np.nan], [np.inf], [10, 11], [[10]], {"grade": 10}, ["invalid"]):
            fitted.predict.return_value = prediction
            with self.assertRaises(ValueError):
                predict_grade(fitted, frame)
        fitted.predict.return_value = [10]
        for threshold in (None, [], True, "10", np.nan, np.inf, -1, 21, 10 ** 1000):
            with self.assertRaises(ValueError):
                prediction_report(fitted, frame, {"interval": {"radius": 1}}, threshold)
        for radius in (None, [], True, "1", np.nan, np.inf, -1):
            with self.assertRaises(ValueError):
                prediction_report(fitted, frame, {"interval": {"radius": radius}})


class ArtifactSecurityTests(unittest.TestCase):
    def test_pointer_traversal_and_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mat").mkdir()
            pointer = root / "mat" / "latest.json"
            for identifier in ("../outside", "a" * 33, "Z" * 32, None, [], 42):
                pointer.write_text(json.dumps({"bundle_id": identifier}))
                with self.assertRaises(ValueError):
                    _active_directory("mat", root)
            for invalid in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]', '[' * 2000 + '0' + ']' * 2000,
                            'x' * (1024 * 1024 + 1)):
                pointer.write_text(invalid)
                with self.assertRaises(ValueError):
                    _read_json(pointer)

    def test_save_rejects_untrusted_identity_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            for subject, identity in (("../outside", "a" * 32), ("mat", "../outside")):
                with self.assertRaises(ValueError):
                    save_bundle((None, {"subject": subject, "bundle_id": identity}, None), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_limits_network_and_decompressed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "student-mat.csv"
            destination.write_bytes(b"existing dataset")
            with patch("download_data.urlopen", return_value=BytesIO(b"x" * (MAX_DOWNLOAD_BYTES + 1))):
                with self.assertRaises(ValueError):
                    download(destination)
            payload = BytesIO()
            with ZipFile(payload, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("student-mat.csv", b"x" * (MAX_DOWNLOAD_BYTES + 1))
            with patch("download_data.urlopen", return_value=BytesIO(payload.getvalue())):
                with self.assertRaises(ValueError):
                    download(destination)
            self.assertEqual(destination.read_bytes(), b"existing dataset")


if __name__ == "__main__":
    unittest.main()
