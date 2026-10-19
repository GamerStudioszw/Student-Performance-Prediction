"""Deployment recovery uses only bundled public data and never requires shell access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model import FEATURES, ROOT
from model_service import load_bundle


class ModelServiceTests(unittest.TestCase):
    def test_mismatched_artifacts_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
            metrics["sklearn_version"] = "0.0.0"
            (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            with patch("model_service.joblib.load", side_effect=AssertionError("Outdated binary must not be loaded")):
                fitted, regenerated, results = load_bundle("mat", artifact_dir=output)
            self.assertEqual(list(fitted.feature_names_in_), FEATURES)
            self.assertNotEqual(regenerated["sklearn_version"], "0.0.0")
            self.assertEqual(len(results), 79)
            self.assertEqual([path.name for path in output.iterdir()], ["metrics.json"])

    def test_unknown_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            load_bundle("unrecognised")


if __name__ == "__main__":
    unittest.main()
