"""Validation, reproducible model comparison, and final-grade inference."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = ["G1", "G2", "studytime", "failures"]
CATEGORICAL = ["schoolsup", "famsup"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "G3"
RANGES = {"G1": (0, 20), "G2": (0, 20), "studytime": (1, 4), "failures": (0, 3)}


def validate_frame(frame: pd.DataFrame, *, require_target: bool = False) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Provide at least one student record.")
    clean = frame.loc[:, required].copy()
    ranges = {**RANGES, **({TARGET: (0, 20)} if require_target else {})}
    for column, (low, high) in ranges.items():
        values = pd.to_numeric(clean[column], errors="coerce")
        if not (values.notna() & np.isfinite(values) & values.between(low, high) & (values % 1 == 0)).all():
            raise ValueError(f"{column} must contain whole numbers from {low} to {high}.")
        clean[column] = values.astype(int)
    for column in CATEGORICAL:
        values = clean[column].astype("string").str.strip().str.lower()
        if values.isna().any() or not values.isin(["yes", "no"]).all():
            raise ValueError(f"{column} must contain yes or no.")
        clean[column] = values.astype(str)
    return clean


def load_data(path: Path = ROOT / "data" / "student-mat.csv") -> pd.DataFrame:
    return validate_frame(pd.read_csv(path, sep=";"), require_target=True)


def make_pipeline(estimator) -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="error", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def train_and_evaluate(frame: pd.DataFrame):
    data = validate_frame(frame, require_target=True)
    if len(data) < 50:
        raise ValueError("At least 50 records are required for model comparison.")
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    estimators = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, max_depth=10, random_state=42, n_jobs=-1
        ),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    comparison = []
    for name, estimator in estimators.items():
        scores = -cross_val_score(
            make_pipeline(estimator), x_train, y_train, cv=folds,
            scoring="neg_mean_absolute_error", n_jobs=1,
        )
        comparison.append({"model": name, "cv_mae": float(scores.mean()), "cv_std": float(scores.std())})
    comparison.sort(key=lambda row: row["cv_mae"])
    selected = comparison[0]["model"]
    fitted = make_pipeline(estimators[selected]).fit(x_train, y_train)
    # The saved model remains trained on the training partition; test records stay unseen.
    predictions = fitted.predict(x_test)
    baseline = np.full(len(y_test), y_train.mean())
    metrics = {
        "selected_model": selected,
        "rows": len(data), "train_rows": len(x_train), "test_rows": len(x_test),
        "random_seed": 42, "cv_folds": 5,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "comparison": comparison,
        "features": FEATURES,
        "target": TARGET,
    }
    results = pd.DataFrame({
        "source_row": y_test.index, "actual_grade": y_test.to_numpy(),
        "predicted_grade": predictions,
    })
    return fitted, metrics, results


def predict_grade(fitted: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    # Preserve raw predictions for consistency with evaluation; the UI flags extrapolation.
    return fitted.predict(validate_frame(frame))
