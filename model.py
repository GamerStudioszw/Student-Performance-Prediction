"""Validation, reproducible model comparison, and final-grade inference."""

from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
PUBLIC_SUBJECTS = {"mat": "Mathematics", "por": "Portuguese language"}
SUBJECTS = {**PUBLIC_SUBJECTS, "eng": "English", "phy": "Physics", "chem": "Chemistry",
            "bio": "Biology", "geo": "Geography", "hist": "History", "cs": "Computer Science"}
NUMERIC = ["G1", "G2", "studytime", "failures", "traveltime"]
CATEGORICAL = ["schoolsup", "famsup", "paid", "activities", "internet", "higher"]
FEATURES = NUMERIC + CATEGORICAL
CORE_FEATURES = ["G1", "G2", "studytime", "failures", "schoolsup", "famsup"]
FEATURE_LABELS = {
    "G1": "First period grade", "G2": "Second period grade", "studytime": "Weekly study time",
    "failures": "Previous class failures", "traveltime": "Travel time to school",
    "schoolsup": "School educational support", "famsup": "Family educational support",
    "paid": "Paid subject classes", "activities": "Extracurricular activities",
    "internet": "Internet access at home", "higher": "Plans for higher education",
}
TARGET = "G3"
RANGES = {"G1": (0, 20), "G2": (0, 20), "studytime": (1, 4), "failures": (0, 3), "traveltime": (1, 4)}


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


def make_pipeline(estimator, features=None) -> Pipeline:
    features = FEATURES if features is None else features
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), [name for name in NUMERIC if name in features]),
        ("category", OneHotEncoder(categories=[["no", "yes"] for name in CATEGORICAL if name in features],
                                   handle_unknown="error", sparse_output=False),
         [name for name in CATEGORICAL if name in features]),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def split_data(data):
    """60% fitting, 20% interval calibration, 20% untouched final evaluation."""
    development, test = train_test_split(data, test_size=0.2, random_state=42)
    train, calibration = train_test_split(development, test_size=0.25, random_state=42)
    return train, calibration, test


def conformal_radius(residuals, coverage=0.8):
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration residuals must be a nonempty vector of finite nonnegative values.")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one.")
    rank = ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise ValueError("Not enough calibration records for this coverage level.")
    return float(np.sort(values)[rank - 1])


def train_and_evaluate(frame: pd.DataFrame):
    data = validate_frame(frame, require_target=True)
    if len(data) < 50:
        raise ValueError("At least 50 records are required for model comparison.")
    train, calibration, test = split_data(data)
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]
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
    core_scores = -cross_val_score(make_pipeline(Ridge(alpha=10.0), CORE_FEATURES),
                                  x_train, y_train, cv=folds, scoring="neg_mean_absolute_error")
    selected = comparison[0]["model"]
    fitted = make_pipeline(estimators[selected]).fit(x_train, y_train)
    # The saved model remains trained on the training partition; test records stay unseen.
    predictions = fitted.predict(x_test)
    calibration_errors = np.abs(calibration[TARGET].to_numpy() - fitted.predict(calibration[FEATURES]))
    radius = conformal_radius(calibration_errors)
    lower, upper = predictions - radius, predictions + radius
    baseline = np.full(len(y_test), y_train.mean())
    errors = np.abs(y_test.to_numpy() - predictions)
    importance = permutation_importance(fitted, x_test, y_test, scoring="neg_mean_absolute_error",
                                        n_repeats=10, random_state=42, n_jobs=1)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "selected_model": selected,
        "rows": len(data), "train_rows": len(x_train), "calibration_rows": len(calibration), "test_rows": len(x_test),
        "random_seed": 42, "cv_folds": 5,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "g2_baseline_mae": float(mean_absolute_error(y_test, x_test["G2"])),
        "core_ridge_cv_mae": float(core_scores.mean()),
        "within_two_points": float(np.mean(errors <= 2)),
        "interval": {"method": "split conformal absolute residual", "nominal_coverage": 0.8,
                     "radius": radius, "test_coverage": float(np.mean((y_test >= lower) & (y_test <= upper)))},
        "feature_importance": [{"feature": name, "mae_increase": float(mean), "std": float(std)}
                               for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        "split_rows": {"train": train.index.tolist(), "calibration": calibration.index.tolist(), "test": test.index.tolist()},
        "comparison": comparison,
        "features": FEATURES,
        "target": TARGET,
    }
    results = pd.DataFrame({
        "source_row": y_test.index, "actual_grade": y_test.to_numpy(),
        "predicted_grade": predictions,
        "interval_lower": lower, "interval_upper": upper,
        "absolute_error": errors,
    })
    return fitted, metrics, results


def predict_grade(fitted: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    # Preserve raw predictions for consistency with evaluation; the UI flags extrapolation.
    return fitted.predict(validate_frame(frame))


def prediction_report(fitted, frame, metrics, threshold=10):
    if not 0 <= threshold <= 20:
        raise ValueError("Reference threshold must be between zero and twenty.")
    clean = validate_frame(frame)
    grades = predict_grade(fitted, clean)
    radius = metrics["interval"]["radius"]
    report = clean.reset_index(drop=True).copy()
    report["predicted_grade"] = grades
    report["percentage_equivalent"] = grades * 5
    report["observed_change_G1_to_G2"] = clean["G2"].to_numpy() - clean["G1"].to_numpy()
    report["predicted_change_from_G2"] = grades - clean["G2"].to_numpy()
    report["reference_threshold"] = threshold
    report["margin_to_threshold"] = grades - threshold
    report["interval_lower"] = grades - radius
    report["interval_upper"] = grades + radius
    report["threshold_comparison"] = np.where(grades >= threshold, "At or above reference", "Below reference")
    return report
