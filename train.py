"""Train with `python train.py`; write a model and a holdout evaluation report."""

import json
import hashlib
from datetime import datetime, timezone

import joblib
import sklearn

from download_data import download
from model import ROOT, PUBLIC_SUBJECTS, load_data, train_and_evaluate


def main():
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    for subject, label in PUBLIC_SUBJECTS.items():
        data_path = ROOT / "data" / f"student-{subject}.csv"
        if not data_path.exists():
            download(data_path, subject=subject)
        model, metrics, predictions = train_and_evaluate(load_data(data_path))
        metrics.update(subject=subject, subject_label=label, trained_at=datetime.now(timezone.utc).isoformat(),
                       sklearn_version=sklearn.__version__, dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
        prefix = "" if subject == "mat" else "por_"
        # Write each file to a temporary path first so a running app never reads a partial file.
        model_path = output / f"{prefix}model.joblib"
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, temporary)
        temporary.replace(model_path)
        for filename, content in [(f"{prefix}metrics.json", json.dumps(metrics, indent=2)),
                                  (f"{prefix}test_predictions.csv", predictions.to_csv(index=False, lineterminator="\n"))]:
            path = output / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
        print(f"{label}: {metrics['rows']} rows; {metrics['selected_model']}; MAE={metrics['mae']:.3f}; "
              f"80% interval test coverage={metrics['interval']['test_coverage']:.1%}", flush=True)
    print("Ready. Launch with: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
