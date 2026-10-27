"""Consistent versioned bundles; old evaluations are never overwritten on recovery."""
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
import re
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn

from model import (ROOT, SCHEMA_VERSION, SUBJECTS, PUBLIC_SUBJECTS, FEATURE_SETS, SPLIT_CONFIG,
                   COVERAGE, BOUND_POLICY, SELECTION_RULE, load_data, train_and_evaluate,
                   split_data, predict_grade, conformal_radius, regression_metrics)

SOURCE = "Cortez (2008), UCI Student Performance, DOI 10.24432/C5TG7T, CC BY 4.0"
ARTIFACT_DIR = ROOT / "artifacts" / "v3"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path):
    with Path(path).open("rb") as stream:
        content = stream.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError("Artifact metadata exceeds the size limit.")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate artifact metadata key.")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Non-finite artifact metadata is not allowed.")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite artifact metadata is not allowed.")
        return parsed
    try:
        result = json.loads(content, object_pairs_hook=unique_object, parse_constant=invalid_constant,
                            parse_float=finite_float)
    except RecursionError as exc:
        raise ValueError("Artifact metadata is nested too deeply.") from exc
    if not isinstance(result, dict):
        raise ValueError("Artifact metadata must be a JSON object.")
    return result


def _bundle_directory(artifact_dir, subject, identifier):
    if (not isinstance(subject, str) or subject not in SUBJECTS or not isinstance(identifier, str)
            or re.fullmatch(r"[0-9a-f]{32}", identifier) is None):
        raise ValueError("Invalid bundle subject or identity.")
    root = Path(artifact_dir).resolve()
    directory = root / subject / identifier
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Artifact paths must stay inside the artifact directory.")
    return directory


def code_fingerprint():
    return hashlib.sha256(b"".join((ROOT / name).read_bytes()
                                  for name in ("model.py", "model_service.py", "validation.py"))).hexdigest()


def finalize_bundle(fitted, metrics, results, *, subject, fingerprint, source):
    identity = uuid.uuid4().hex
    metrics.update(subject=subject, subject_label=SUBJECTS[subject], source=source,
                   dataset_sha256=fingerprint, sklearn_version=sklearn.__version__,
                   code_sha256=code_fingerprint(), trained_at=datetime.now(timezone.utc).isoformat(),
                   bundle_id=identity)
    fitted.bundle_id_ = identity
    results = results.assign(bundle_id=identity, subject=SUBJECTS[subject],
                             feature_set=metrics["feature_set"], predictor=metrics["selected_model"],
                             grade_scale="0-20", prediction_timing=metrics["prediction_timing"])
    return fitted, metrics, results


def session_bundle_compatible(bundle):
    try:
        fitted, metrics, _ = bundle
        return (metrics["schema_version"] == SCHEMA_VERSION and metrics["code_sha256"] == code_fingerprint()
                and metrics["sklearn_version"] == sklearn.__version__
                and metrics["features"] == FEATURE_SETS[metrics["feature_set"]]
                and fitted.bundle_id_ == metrics["bundle_id"])
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def _subject(subject):
    if not isinstance(subject, str) or subject not in PUBLIC_SUBJECTS:
        raise ValueError("No bundled dataset for this subject. Upload subject training records in the app.")


def _active_directory(subject, artifact_dir):
    _subject(subject)
    pointer = Path(artifact_dir) / subject / "latest.json"
    if not pointer.resolve().is_relative_to(Path(artifact_dir).resolve()):
        raise ValueError("Artifact pointer must stay inside the artifact directory.")
    return _bundle_directory(artifact_dir, subject, _read_json(pointer)["bundle_id"])


def bundle_signature(subject, artifact_dir=ARTIFACT_DIR, data_dir=ROOT / "data"):
    _subject(subject)
    paths = [Path(data_dir) / f"student-{subject}.csv", Path(artifact_dir) / subject / "latest.json"]
    try:
        directory = _active_directory(subject, artifact_dir)
        paths += [directory / name for name in ("manifest.json", "metrics.json", "test_predictions.csv", "model.joblib")]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return (SCHEMA_VERSION, sklearn.__version__, code_fingerprint(),
            tuple((str(path), file_hash(path) if path.exists() else None) for path in paths))


def save_bundle(bundle, artifact_dir=ARTIFACT_DIR):
    """Write an immutable run, then atomically publish its pointer after all files exist."""
    fitted, metrics, results = bundle
    directory = _bundle_directory(artifact_dir, metrics["subject"], metrics["bundle_id"])
    if not session_bundle_compatible(bundle):
        raise ValueError("Cannot save an incompatible predictor bundle.")
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(fitted, directory / "model.joblib")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "test_predictions.csv").write_text(results.to_csv(index=False, lineterminator="\n"),
                                                    encoding="utf-8", newline="\n")
    manifest = {"schema_version": SCHEMA_VERSION, "bundle_id": metrics["bundle_id"],
                "files": {name: file_hash(directory / name) for name in ("model.joblib", "metrics.json", "test_predictions.csv")}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary = directory.parent / f"{metrics['bundle_id']}.tmp"
    temporary.write_text(json.dumps({"bundle_id": metrics["bundle_id"]}), encoding="utf-8")
    temporary.replace(directory.parent / "latest.json")
    return directory


def _load_verified(subject, directory, data, fingerprint):
    for name, limit in (("manifest.json", 1024 * 1024), ("metrics.json", 1024 * 1024),
                        ("test_predictions.csv", 4 * 1024 * 1024), ("model.joblib", 64 * 1024 * 1024)):
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or path.stat().st_size > limit:
            raise ValueError("Artifact file is outside its run directory or exceeds its size limit.")
    manifest = _read_json(directory / "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["bundle_id"] != directory.name:
        raise ValueError("Incompatible artifact manifest.")
    for name in ("model.joblib", "metrics.json", "test_predictions.csv"):
        if file_hash(directory / name) != manifest["files"][name]:
            raise ValueError(f"Artifact integrity mismatch: {name}.")
    metrics = _read_json(directory / "metrics.json")
    if (metrics["schema_version"] != SCHEMA_VERSION or metrics["code_sha256"] != code_fingerprint()
            or metrics["dataset_sha256"] != fingerprint or metrics["subject"] != subject
            or metrics["sklearn_version"] != sklearn.__version__ or metrics["bundle_id"] != directory.name
            or metrics["features"] != FEATURE_SETS[metrics["feature_set"]]
            or metrics["training_feature_set"] != "expanded_v1"
            or metrics["split_config"] != SPLIT_CONFIG or metrics["selection_rule"] != SELECTION_RULE
            or metrics["bounding_policy"] != BOUND_POLICY or metrics["interval"]["nominal_coverage"] != COVERAGE):
        raise ValueError("Stored predictor settings, provenance or evaluation are incompatible.")
    train, calibration, test = split_data(data)
    for name, partition in (("train", train), ("calibration", calibration), ("test", test)):
        if metrics["split_rows"][name] != partition.index.tolist() or metrics[f"{name}_rows"] != len(partition):
            raise ValueError("Stored partitions do not match the fixed split.")
    # Only locally generated, integrity-checked binaries are loaded, never uploaded pickles.
    fitted = joblib.load(directory / "model.joblib")
    if (not session_bundle_compatible((fitted, metrics, None))
            or fitted.input_features_ != metrics["features"] or fitted.feature_set_ != metrics["feature_set"]
            or fitted.candidate_ != metrics["selected_candidate"]):
        raise ValueError("Stored predictor and metadata do not match.")
    results = pd.read_csv(directory / "test_predictions.csv")
    predicted = predict_grade(fitted, test)
    radius = conformal_radius(np.abs(calibration.G3 - predict_grade(fitted, calibration)))
    if results.source_row.tolist() != test.index.tolist() or not results.bundle_id.eq(metrics["bundle_id"]).all():
        raise ValueError("Stored result identities do not match.")
    expected = {"actual_grade": test.G3, "recorded_G2": test.G2, "predicted_grade": predicted,
                "interval_lower": predicted - radius, "interval_upper": predicted + radius,
                "absolute_error": np.abs(test.G3 - predicted)}
    for name, values in expected.items():
        if not np.allclose(results[name], values, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Stored result mismatch: {name}.")
    for name, value in regression_metrics(test.G3, predicted).items():
        if not np.isclose(metrics[name], value, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Stored evaluation mismatch: {name}.")
    coverage = float(np.mean(np.abs(test.G3 - predicted) <= radius))
    if not all(np.isclose(a, b, rtol=1e-10, atol=1e-10) for a, b in
               [(radius, metrics["interval"]["radius"]), (2 * radius, metrics["interval"]["average_width"]),
                (coverage, metrics["interval"]["test_coverage"]),
                (float(np.abs(test.G3 - test.G2).mean()), metrics["g2_baseline_mae"])]):
        raise ValueError("Stored calibration or baseline evaluation does not match.")
    return fitted, metrics, results


def load_bundle(subject, artifact_dir=ARTIFACT_DIR, data_dir=ROOT / "data"):
    _subject(subject)
    data_path = Path(data_dir) / f"student-{subject}.csv"
    data = load_data(data_path)
    fingerprint = file_hash(data_path)
    try:
        bundle = _load_verified(subject, _active_directory(subject, artifact_dir), data, fingerprint)
        bundle[1]["load_notice"] = "Verified saved predictor and evaluation bundle."
        return bundle
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, ImportError, pickle.UnpicklingError) as exc:
        logging.getLogger(__name__).info("Preparing fresh %s bundle: %s", subject, exc)
        bundle = finalize_bundle(*train_and_evaluate(data), subject=subject, fingerprint=fingerprint, source=SOURCE)
        bundle[1]["load_notice"] = ("A compatible saved bundle was unavailable. This session prepared a new predictor and "
                                    "recomputed evaluation from the public data. Historical files were not overwritten; "
                                    "the previously inspected test split is reused.")
        return bundle
