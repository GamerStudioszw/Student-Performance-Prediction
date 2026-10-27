"""Training validation and subject isolation; fixtures are not deployed subject datasets."""

import unittest
from io import BytesIO
from unittest.mock import patch

import pandas as pd

from model import FEATURES, ROOT, SUBJECTS
from model_service import bundle_signature, load_bundle
from subject_training import read_training_csv, training_template, train_subject


def history_fixture(subject="eng", rows=60):
    # Reuse public values solely to exercise the schema in tests. This is not
    # evidence about English accuracy and is never installed as an app dataset.
    frame = pd.read_csv(ROOT / "data" / "student-mat.csv", sep=";").head(rows)[[*FEATURES, "G3"]]
    return frame.assign(subject=subject, student_id=[f"S{i:04}" for i in range(rows)], class_name="Group A")


class SubjectTrainingTests(unittest.TestCase):
    def test_names_only_history_stays_out_of_training_features(self):
        frame = history_fixture().drop(columns="student_id")
        frame["student_name"] = [f"Student {i}" for i in range(len(frame))]
        content = frame.to_csv(index=False).encode()
        clean = read_training_csv(content, "eng")
        self.assertTrue(clean.student_id.eq("").all())
        self.assertEqual(clean.student_name.tolist(), frame.student_name.tolist())
        from types import SimpleNamespace
        with patch("subject_training.train_and_evaluate", return_value=(SimpleNamespace(), {"feature_set": "expanded_v1", "selected_model": "fixture", "prediction_timing": "after period two"}, pd.DataFrame())) as train:
            train_subject(content, "eng")
        self.assertEqual(list(train.call_args.args[0]), [*FEATURES, "G3"])
        self.assertNotIn("student_name", train.call_args.args[0])
        self.assertIn("student_name", pd.read_csv(BytesIO(training_template(identifier="student_name"))))

    def test_subject_registry_and_public_model_boundary(self):
        self.assertEqual(len(SUBJECTS), 9)
        for subject in ["eng", "phy", "chem", "bio", "geo", "hist", "cs"]:
            with self.subTest(subject=subject):
                content = history_fixture(subject).to_csv(index=False).encode()
                self.assertEqual(len(read_training_csv(content, subject)), 60)
                with self.assertRaises(ValueError):
                    load_bundle(subject)
                with self.assertRaises(ValueError):
                    bundle_signature(subject)

    def test_history_requires_complete_consistent_subject_data(self):
        frame = history_fixture()
        cases = [frame.drop(columns="subject"), frame.assign(subject="phy"), frame.head(49),
                 frame.drop(columns="G3"), frame.assign(G3=10), frame.assign(student_id="same")]
        missing = frame.copy()
        missing.loc[0, "G3"] = float("nan")
        cases.append(missing)
        mixed = frame.copy()
        mixed.loc[0, "subject"] = "phy"
        cases.append(mixed)
        for invalid in cases:
            with self.subTest(columns=list(invalid), rows=len(invalid)):
                with self.assertRaises(ValueError):
                    read_training_csv(invalid.to_csv(index=False).encode(), "eng")
        self.assertEqual(len(read_training_csv(frame.assign(subject=" English ").to_csv(index=False).encode(), "eng")), 60)

    def test_template_is_header_only_and_model_uses_selected_history(self):
        template = pd.read_csv(BytesIO(training_template()))
        self.assertTrue(template.empty)
        self.assertIn("subject", template)
        self.assertIn("G3", template)
        content = history_fixture("phy").to_csv(index=False).encode()
        from types import SimpleNamespace
        with patch("subject_training.train_and_evaluate", return_value=(SimpleNamespace(), {"feature_set": "expanded_v1", "selected_model": "fixture", "prediction_timing": "after period two"}, pd.DataFrame())) as train:
            saved = train_subject(content, "phy")
        used = train.call_args.args[0]
        self.assertEqual(list(used), [*FEATURES, "G3"])
        self.assertEqual(len(used), 60)
        self.assertEqual(saved["bundle"][1]["subject"], "phy")
        self.assertEqual(saved["bundle"][1]["subject_label"], "Physics")
        self.assertIn("dataset_sha256", saved["bundle"][1])
        with patch("subject_training.train_and_evaluate") as train:
            with self.assertRaises(ValueError):
                train_subject(content, "eng")
            train.assert_not_called()

    def test_grade_only_history_needs_no_invented_support_values(self):
        frame = history_fixture()[["subject", "student_id", "class_name", "G1", "G2", "G3"]]
        clean = read_training_csv(frame.to_csv(index=False).encode(), "eng", "grades_v1")
        self.assertNotIn("studytime", clean)
        self.assertNotIn("famsup", clean)
        with self.assertRaises(ValueError):
            read_training_csv(frame.to_csv(index=False).encode(), "eng", "expanded_v1")
        baseline = frame.drop(columns="G1")
        self.assertTrue(read_training_csv(baseline.to_csv(index=False).encode(), "eng", "persistence_v1").G1.isna().all())
