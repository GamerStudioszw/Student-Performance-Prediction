"""Train with `python train.py`; write a model and a holdout evaluation report."""

import json
from datetime import datetime, timezone

import joblib
import sklearn

from download_data import download
from model import ROOT, load_data, train_and_evaluate


def main():
    data_path = ROOT / "data" / "student-mat.csv"
    if not data_path.exists():
        download(data_path)
    model, metrics, predictions = train_and_evaluate(load_data(data_path))
    metrics["trained_at"] = datetime.now(timezone.utc).isoformat()
    metrics["sklearn_version"] = sklearn.__version__
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Ready. Launch with: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
