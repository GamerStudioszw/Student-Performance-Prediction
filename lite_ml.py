"""Ridge regression with training-only scaling, using Python's standard library."""

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
FEATURES = ["G1", "G2", "studytime", "failures", "schoolsup", "famsup"]
LIMITS = {"G1": (0, 20), "G2": (0, 20), "studytime": (1, 4), "failures": (0, 3)}


def features(record):
    values = []
    for name, (low, high) in LIMITS.items():
        try:
            value = float(record[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{name} is required and must be a whole number.") from None
        if not math.isfinite(value) or not value.is_integer() or not low <= value <= high:
            raise ValueError(f"{name} must be a whole number from {low} to {high}.")
        values.append(value)
    for name in FEATURES[4:]:
        value = str(record.get(name, "")).strip().lower()
        if value not in ("yes", "no"):
            raise ValueError(f"{name} must be yes or no.")
        values.append(float(value == "yes"))
    return values


def solve(matrix, target):
    """Solve a small linear system by Gaussian elimination with partial pivoting."""
    n = len(target)
    augmented = [list(row) + [value] for row, value in zip(matrix, target)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        if abs(divisor) < 1e-12:
            raise ValueError("Singular regression system.")
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def fit(x, y, alpha):
    if alpha is None:
        return {"kind": "mean", "value": mean(y)}
    means = [mean(column) for column in zip(*x)]
    scales = [pstdev(column) or 1.0 for column in zip(*x)]
    design = [[1.0] + [(v - m) / s for v, m, s in zip(row, means, scales)] for row in x]
    size = len(design[0])
    gram = [[sum(row[i] * row[j] for row in design) for j in range(size)] for i in range(size)]
    rhs = [sum(row[i] * target for row, target in zip(design, y)) for i in range(size)]
    for i in range(1, size):
        gram[i][i] += alpha  # The intercept is intentionally not penalized.
    return {"kind": "ridge", "alpha": alpha, "means": means, "scales": scales, "weights": solve(gram, rhs)}


def predict(model, row):
    if model["kind"] == "mean":
        return model["value"]
    scaled = [1.0] + [(v - m) / s for v, m, s in zip(row, model["means"], model["scales"])]
    return sum(w * value for w, value in zip(model["weights"], scaled))


def mae(actual, predicted):
    return mean(abs(a - p) for a, p in zip(actual, predicted))


def train(path=ROOT / "data" / "student-mat.csv"):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    x = [features(row) for row in rows]
    y = [float(row["G3"]) for row in rows]
    if len(rows) < 50 or any(not math.isfinite(v) or not 0 <= v <= 20 for v in y):
        raise ValueError("Training requires at least 50 valid records with final grades from 0 to 20.")
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    test_count = math.ceil(len(rows) * 0.2)
    test_ids, train_ids = indices[:test_count], indices[test_count:]
    folds = [train_ids[i::5] for i in range(5)]
    candidates = [("Mean baseline", None)] + [(f"Ridge (alpha={a:g})", a) for a in (0.1, 1.0, 10.0, 100.0)]
    comparison = []
    for name, alpha in candidates:
        scores = []
        for fold in folds:
            fold_set = set(fold)
            fit_ids = [i for i in train_ids if i not in fold_set]
            model = fit([x[i] for i in fit_ids], [y[i] for i in fit_ids], alpha)
            scores.append(mae([y[i] for i in fold], [predict(model, x[i]) for i in fold]))
        comparison.append({"model": name, "alpha": alpha, "cv_mae": mean(scores), "cv_std": pstdev(scores)})
    comparison.sort(key=lambda result: result["cv_mae"])
    selected = comparison[0]
    model = fit([x[i] for i in train_ids], [y[i] for i in train_ids], selected["alpha"])
    actual = [y[i] for i in test_ids]
    predicted = [predict(model, x[i]) for i in test_ids]
    errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    total_variation = sum((a - mean(actual)) ** 2 for a in actual)
    report = {
        "implementation": "Python standard-library ridge regression",
        "selected_model": selected["model"], "features": FEATURES,
        "rows": len(rows), "train_rows": len(train_ids), "test_rows": len(test_ids),
        "seed": 42, "cv_folds": 5,
        "mae": mae(actual, predicted), "rmse": math.sqrt(mean(errors)),
        "r2": 1 - sum(errors) / total_variation if total_variation else None,
        "baseline_mae": mae(actual, [mean(y[i] for i in train_ids)] * len(actual)),
        "comparison": comparison,
        "grade_distribution": [sum(value == grade for value in y) for grade in range(21)],
    }
    results = [{"source_row": i, "actual_grade": a, "predicted_grade": p} for i, a, p in zip(test_ids, actual, predicted)]
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "lite_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (output / "lite_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "lite_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_row", "actual_grade", "predicted_grade"])
        writer.writeheader()
        writer.writerows(results)
    return model, report


if __name__ == "__main__":
    _, report = train()
    print(json.dumps(report, indent=2))
