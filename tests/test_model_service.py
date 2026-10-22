"""Artifact verification rejects stale or mismatched bundles without overwriting evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from model import ROOT, predict_grade, prediction_report
from model_service import load_bundle, save_bundle, file_hash, _load_verified


class ModelServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle("mat")

    def test_saved_bundle_is_consistent_with_inference_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            run = save_bundle(self.bundle, Path(directory))
            with patch("model_service.train_and_evaluate", side_effect=AssertionError("Compatible bundle must load")):
                fitted, metrics, results = load_bundle("mat", artifact_dir=Path(directory))
            from model import load_data
            data = load_data().loc[metrics["split_rows"]["test"]]
            report = prediction_report(fitted, data, metrics)
            np.testing.assert_allclose(report.predicted_grade, results.predicted_grade)
            np.testing.assert_allclose(report.interval_lower, results.interval_lower)
            self.assertTrue(results.bundle_id.eq(report.bundle_id.iloc[0]).all())
            self.assertEqual(fitted.input_features_, metrics["features"])
            self.assertIn("Verified", metrics["load_notice"])
            self.assertTrue((run / "manifest.json").exists())

    def test_incompatible_metadata_is_rejected_before_loading_binary(self):
        for key, invalid in [("schema_version", 1), ("sklearn_version", "0.0.0"),
                             ("features", ["G1"]), ("dataset_sha256", "wrong"), ("code_sha256", "old")]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                run = save_bundle(self.bundle, Path(directory))
                metrics_path = run / "metrics.json"
                metrics = json.loads(metrics_path.read_text())
                metrics[key] = invalid
                metrics_path.write_text(json.dumps(metrics))
                manifest_path = run / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["files"]["metrics.json"] = file_hash(metrics_path)
                manifest_path.write_text(json.dumps(manifest))
                from model import load_data
                with patch("model_service.joblib.load", side_effect=AssertionError("Do not load incompatible binary")):
                    with self.assertRaises(ValueError):
                        _load_verified("mat", run, load_data(), file_hash(ROOT / "data" / "student-mat.csv"))

    def test_corrupt_artifact_recovers_as_whole_bundle_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            run = save_bundle(self.bundle, Path(directory))
            (run / "test_predictions.csv").write_text("corrupt data")
            before = {str(path): file_hash(path) for path in Path(directory).rglob("*") if path.is_file()}
            core = copy.deepcopy(self.bundle)
            with patch("model_service.joblib.load", side_effect=AssertionError("Integrity check must happen first")):
                with patch("model_service.train_and_evaluate", return_value=core) as rebuild:
                    fitted, metrics, results = load_bundle("mat", artifact_dir=Path(directory))
            rebuild.assert_called_once()
            self.assertNotEqual(metrics["bundle_id"], self.bundle[1]["bundle_id"])
            self.assertEqual(fitted.bundle_id_, metrics["bundle_id"])
            self.assertTrue(results.bundle_id.eq(metrics["bundle_id"]).all())
            self.assertIn("recomputed", metrics["load_notice"])
            self.assertEqual(before, {str(path): file_hash(path) for path in Path(directory).rglob("*") if path.is_file()})

    def test_matching_file_hashes_do_not_hide_wrong_interval_or_model_pair(self):
        for target in ("interval", "model"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                run = save_bundle(self.bundle, Path(directory))
                if target == "interval":
                    path = run / "metrics.json"
                    metrics = json.loads(path.read_text())
                    metrics["interval"]["radius"] += 1
                    path.write_text(json.dumps(metrics))
                else:
                    path = run / "model.joblib"
                    fitted = copy.deepcopy(self.bundle[0])
                    fitted.bundle_id_ = "wrong"
                    joblib.dump(fitted, path)
                manifest_path = run / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["files"][path.name] = file_hash(path)
                manifest_path.write_text(json.dumps(manifest))
                from model import load_data
                with self.assertRaises(ValueError):
                    _load_verified("mat", run, load_data(), file_hash(ROOT / "data" / "student-mat.csv"))

    def test_report_rejects_predictor_from_a_different_evaluation_bundle(self):
        fitted, metrics, _ = self.bundle
        wrong = dict(metrics, bundle_id="different")
        import pandas as pd
        with self.assertRaisesRegex(ValueError, "do not match"):
            prediction_report(fitted, pd.DataFrame({"G2": [12]}), wrong)

    def test_unknown_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            load_bundle("unrecognised")
