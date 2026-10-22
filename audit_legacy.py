"""Reproduce the pre-repair procedure without altering its saved evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import types

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
LEGACY_COMMIT = "6129c10"


def main():
    source = subprocess.check_output(["git", "show", f"{LEGACY_COMMIT}:model.py"], cwd=ROOT).decode("utf-8")
    legacy = types.ModuleType("legacy_student_model")
    legacy.__file__ = str(ROOT / "model.py")
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    audit = {"audited_at": datetime.now(timezone.utc).isoformat(), "source_commit": LEGACY_COMMIT,
             "sklearn_version": sklearn.__version__, "subjects": {},
             "limitation": "Previously inspected test records; reproduction is not independent confirmation."}
    for subject in ("mat", "por"):
        prefix = "" if subject == "mat" else "por_"
        path = ROOT / "data" / f"student-{subject}.csv"
        data = legacy.load_data(path)
        old_metrics = json.loads((ROOT / "artifacts" / f"{prefix}metrics.json").read_text())
        saved_results = pd.read_csv(ROOT / "artifacts" / f"{prefix}test_predictions.csv")
        saved = joblib.load(ROOT / "artifacts" / f"{prefix}model.joblib")
        fit, calibration, test = legacy.split_data(data)
        pred = saved.predict(test[legacy.FEATURES])
        cal_errors = np.abs(calibration.G3 - saved.predict(calibration[legacy.FEATURES]))
        test_errors = np.abs(test.G3 - pred)
        radius = legacy.conformal_radius(cal_errors)
        _, reproduced, results = legacy.train_and_evaluate(data)
        audit["subjects"][subject] = {
            "historical_artifact_sha256": {name: hashlib.sha256((ROOT / "artifacts" / f"{prefix}{name}").read_bytes()).hexdigest()
                                          for name in ("model.joblib", "metrics.json", "test_predictions.csv")},
            "data_hash_matches_saved": hashlib.sha256(path.read_bytes()).hexdigest() == old_metrics["dataset_sha256"],
            "saved_library_matches_current": old_metrics["sklearn_version"] == sklearn.__version__,
            "saved_test_order_matches": saved_results.source_row.tolist() == test.index.tolist(),
            "saved_prediction_max_difference": float(np.max(np.abs(pred - saved_results.predicted_grade))),
            "retrained_prediction_max_difference": float(np.max(np.abs(results.predicted_grade - saved_results.predicted_grade))),
            "saved_mae_recomputed": float(test_errors.mean()), "saved_radius_recomputed": radius,
            "saved_test_coverage_recomputed": float((test_errors <= radius).mean()),
            "saved_interval_width_recomputed": 2 * radius,
            "calibration_diagnostics": {"rows": len(calibration), "covered": int((cal_errors <= radius).sum()),
                                        "zero_final_grades": int(calibration.G3.eq(0).sum()),
                                        "mean_error": float(cal_errors.mean()), "max_error": float(cal_errors.max())},
            "test_diagnostics": {"rows": len(test), "covered": int((test_errors <= radius).sum()),
                                  "zero_final_grades": int(test.G3.eq(0).sum()),
                                  "mean_error": float(test_errors.mean()), "max_error": float(test_errors.max())},
            "reproduced_metrics": reproduced,
        }
        print(subject, json.dumps({key: value for key, value in audit["subjects"][subject].items()
                                   if key not in ("reproduced_metrics", "historical_artifact_sha256")}))
    output = ROOT / "artifacts" / "audits"
    output.mkdir(exist_ok=True)
    path = output / f"legacy-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(path)


if __name__ == "__main__":
    main()
