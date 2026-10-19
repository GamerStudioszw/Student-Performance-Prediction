"""Load trusted model artifacts, or build a complete bundle on a fresh deployment."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import joblib
import pandas as pd
import sklearn

from model import ROOT, FEATURES, SCHEMA_VERSION, SUBJECTS, load_data, train_and_evaluate


def bundle_signature(subject, artifact_dir=ROOT / "artifacts", data_dir=ROOT / "data"):
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject.")
    prefix = "" if subject == "mat" else "por_"
    paths = [Path(data_dir) / f"student-{subject}.csv", Path(__file__), ROOT / "model.py"]
    paths += [Path(artifact_dir) / f"{prefix}{name}" for name in ("model.joblib", "metrics.json", "test_predictions.csv")]
    return (SCHEMA_VERSION, sklearn.__version__, tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (str(path), None, None)
        for path in paths))


def load_bundle(subject, artifact_dir=ROOT / "artifacts", data_dir=ROOT / "data"):
    """A bundle is returned together; recovered models do not require writable artifact storage."""
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject.")
    data_path = Path(data_dir) / f"student-{subject}.csv"
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    prefix = "" if subject == "mat" else "por_"
    artifact_dir = Path(artifact_dir)
    try:
        metrics = json.loads((artifact_dir / f"{prefix}metrics.json").read_text(encoding="utf-8"))
        if (metrics["schema_version"] != SCHEMA_VERSION or metrics["features"] != FEATURES
                or metrics["dataset_sha256"] != fingerprint or metrics["subject"] != subject
                or metrics["sklearn_version"] != sklearn.__version__):
            raise ValueError("Stored model metadata is outdated.")
        fitted = joblib.load(artifact_dir / f"{prefix}model.joblib")
        results = pd.read_csv(artifact_dir / f"{prefix}test_predictions.csv")
        if list(fitted.feature_names_in_) != FEATURES or len(results) != metrics["test_rows"]:
            raise ValueError("Stored model and evaluation do not match.")
        if not {"source_row", "actual_grade", "predicted_grade", "interval_lower", "interval_upper"}.issubset(results):
            raise ValueError("Stored evaluation is incomplete.")
        return fitted, metrics, results
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, ImportError, pickle.UnpicklingError):
        # Git deliberately excludes binary model artifacts. A new hosted instance
        # can fit from the bundled public CSV and cache this bundle in memory.
        fitted, metrics, results = train_and_evaluate(load_data(data_path))
        metrics.update(subject=subject, subject_label=SUBJECTS[subject], dataset_sha256=fingerprint,
                       sklearn_version=sklearn.__version__, trained_at=datetime.now(timezone.utc).isoformat())
        return fitted, metrics, results
