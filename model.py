"""Versioned predictors, validation-only selection and split-conformal evaluation."""
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from validation import finite_number
from sklearn.base import BaseEstimator, RegressorMixin
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
SCHEMA_VERSION = 3
PUBLIC_SUBJECTS = {"mat": "Mathematics", "por": "Portuguese language"}
SUBJECTS = {**PUBLIC_SUBJECTS, "eng": "English", "phy": "Physics", "chem": "Chemistry",
            "bio": "Biology", "geo": "Geography", "hist": "History", "cs": "Computer Science"}
NUMERIC = ["G1", "G2", "studytime", "failures", "traveltime"]
CATEGORICAL = ["schoolsup", "famsup", "paid", "activities", "internet", "higher"]
FEATURES = NUMERIC + CATEGORICAL
CORE_FEATURES = ["G1", "G2", "studytime", "failures", "schoolsup", "famsup"]
FEATURE_SETS = {"persistence_v1": ["G2"], "grades_v1": ["G1", "G2"], "expanded_v1": FEATURES}
FEATURE_LABELS = {
    "G1": "First period grade", "G2": "Second period grade", "studytime": "Weekly study time",
    "failures": "Previous class failures", "traveltime": "Travel time to school",
    "schoolsup": "School educational support", "famsup": "Family educational support",
    "paid": "Paid subject classes", "activities": "Extracurricular activities",
    "internet": "Internet access at home", "higher": "Plans for higher education",
}
TARGET = "G3"
RANGES = {"G1": (0, 20), "G2": (0, 20), "studytime": (1, 4), "failures": (0, 3), "traveltime": (1, 4)}
SPLIT_CONFIG = {"seed": 42, "test_fraction": 0.2, "calibration_fraction_of_development": 0.25,
                "cv_folds": 5, "cv_shuffle": True, "unit": "student record; not school or time"}
COVERAGE = 0.8
TIE_TOLERANCE = 0.02
SELECTION_RULE = ("Choose lowest validation MAE; within 0.02 grade points of the minimum prefer, in order: "
                  "G2 persistence, training mean, two-grade ridge, expanded ridge, two-grade forest, expanded forest. "
                  "Test errors and coverage never select the predictor or interval settings.")
LIMITATIONS = ("Random record splits do not establish performance on later cohorts or other institutions. "
               "The bundled test results informed this repair and are reused for comparison, not independent confirmation. "
               "No seed search was performed; repeated use of uploaded-history results also weakens independence.")
BOUND_POLICY = "Raw predictions and intervals; no clipping during calibration, evaluation, inference or export."


def validate_frame(frame, *, require_target=False, features=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Provide a table with unique column names.")
    features = FEATURES if features is None else list(features)
    if not features or len(features) != len(set(features)) or not set(features).issubset(FEATURES):
        raise ValueError("Unknown or duplicate predictor fields. Outcomes and metadata cannot be predictors.")
    required = features + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Provide at least one student record.")
    clean = frame.loc[:, required].copy()
    for column in required:
        if column in CATEGORICAL:
            values = clean[column].astype("string").str.strip().str.lower()
            if values.isna().any() or not values.isin(["yes", "no"]).all():
                raise ValueError(f"{column} must contain yes or no. Missing is not no.")
            clean[column] = values.astype(str)
        else:
            low, high = (0, 20) if column == TARGET else RANGES[column]
            if clean[column].map(lambda value: isinstance(value, (bool, np.bool_, complex))).any():
                raise ValueError(f"{column} must contain whole-number grades or codes, not booleans or complex values.")
            values = pd.to_numeric(clean[column], errors="coerce")
            if not (values.notna() & np.isfinite(values) & values.between(low, high) & (values % 1 == 0)).all():
                raise ValueError(f"{column} must contain whole numbers from {low} to {high}; missing values are not replaced.")
            clean[column] = values.astype(int)
    return clean


def observed_grades(frame, clean):
    """Preserve optional G1 for reporting; missing progress is not zero progress."""
    out = clean.copy()
    if "G1" not in out:
        out["G1"] = np.nan
        if "G1" in frame:
            present = frame.G1.notna() & frame.G1.astype(str).str.strip().ne("")
            if present.any():
                out.loc[present, "G1"] = validate_frame(frame.loc[present], features=["G1"]).G1.to_numpy()
    return out


def load_data(path=ROOT / "data" / "student-mat.csv"):
    return validate_frame(pd.read_csv(path, sep=";"), require_target=True)


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """No fitted weights: forecast the final grade as the recorded G2."""
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0].copy()


def make_pipeline(estimator, features=None):
    features = FEATURES if features is None else list(features)
    if not set(features).issubset(FEATURES):
        raise ValueError("Outcomes and metadata cannot be predictors.")
    if isinstance(estimator, PersistenceRegressor):
        if features != ["G2"]:
            raise ValueError("Persistence requires only G2.")
        preprocessing = ColumnTransformer([("recorded_grade", "passthrough", ["G2"])])
    else:
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), [name for name in NUMERIC if name in features]),
            ("category", OneHotEncoder(categories=[["no", "yes"] for name in CATEGORICAL if name in features],
                                       handle_unknown="error", sparse_output=False),
             [name for name in CATEGORICAL if name in features]),
        ])
    pipeline = Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])
    pipeline.input_features_ = list(features)
    return pipeline


def split_data(data):
    development, test = train_test_split(data, test_size=SPLIT_CONFIG["test_fraction"], random_state=42)
    train, calibration = train_test_split(development, test_size=0.25, random_state=42)
    return train, calibration, test


def conformal_radius(residuals, coverage=COVERAGE):
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration residuals must be a nonempty vector of finite nonnegative values.")
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one.")
    rank = ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise ValueError("No finite conformal quantile is attainable at this coverage with this calibration sample. Add records or predefine a lower coverage; do not tune against test results.")
    return float(np.sort(values)[rank - 1])


def candidates(training_feature_set):
    if training_feature_set not in FEATURE_SETS:
        raise ValueError("Unknown training feature set.")
    rows = [("g2", "G2 persistence baseline forecast", "persistence_v1", PersistenceRegressor()),
            ("mean", "Training-mean baseline forecast", "persistence_v1", DummyRegressor(strategy="mean"))]
    if training_feature_set != "persistence_v1":
        rows += [("ridge_grades", "Ridge: G1 + G2", "grades_v1", Ridge(alpha=10.0))]
    if training_feature_set == "expanded_v1":
        rows += [("ridge_expanded", "Ridge: expanded inputs", "expanded_v1", Ridge(alpha=10.0))]
    for name, schema in [("grades", "grades_v1"), ("expanded", "expanded_v1")]:
        if set(FEATURE_SETS[schema]).issubset(FEATURE_SETS[training_feature_set]):
            rows.append((f"forest_{name}", f"Random forest: {name}", schema,
                         RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_depth=10,
                                               random_state=42, n_jobs=-1)))
    return rows


def select_candidate(comparison):
    best = min(row["cv_mae"] for row in comparison)
    return min((row for row in comparison if row["cv_mae"] <= best + TIE_TOLERANCE),
               key=lambda row: row["simplicity_rank"])


def regression_metrics(actual, prediction):
    return {"mae": float(mean_absolute_error(actual, prediction)),
            "rmse": float(root_mean_squared_error(actual, prediction)),
            "r2": float(r2_score(actual, prediction))}


def train_and_evaluate(frame, *, training_feature_set="expanded_v1"):
    specs = candidates(training_feature_set)
    inputs = FEATURE_SETS[training_feature_set]
    data = validate_frame(frame, require_target=True, features=inputs).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError("At least 50 records are required for model comparison.")
    train, calibration, test = split_data(data)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(train))
    comparison = []
    for rank, (identifier, name, schema, estimator) in enumerate(specs):
        features = FEATURE_SETS[schema]
        scores = -cross_val_score(make_pipeline(estimator, features), train[features], train.G3,
                                  cv=folds, scoring="neg_mean_absolute_error", n_jobs=1)
        comparison.append({"candidate": identifier, "model": name, "feature_set": schema, "features": features,
                           "cv_mae": float(scores.mean()), "cv_std": float(scores.std()),
                           "fold_mae": scores.tolist(), "simplicity_rank": rank})
    selected = select_candidate(comparison)
    features = selected["features"]
    estimator = next(spec[3] for spec in specs if spec[0] == selected["candidate"])
    fitted = make_pipeline(estimator, features).fit(train[features], train.G3)
    fitted.schema_version_ = SCHEMA_VERSION
    fitted.feature_set_ = selected["feature_set"]
    fitted.candidate_ = selected["candidate"]
    # Selection is fixed before observing calibration residuals or test predictions.
    cal_errors = np.abs(calibration.G3.to_numpy() - predict_grade(fitted, calibration))
    radius = conformal_radius(cal_errors)
    predictions = predict_grade(fitted, test)
    lower, upper = predictions - radius, predictions + radius
    errors = np.abs(test.G3.to_numpy() - predictions)
    importance = permutation_importance(fitted, test[features], test.G3, scoring="neg_mean_absolute_error",
                                        n_repeats=10, random_state=42, n_jobs=1)
    metrics = {
        "schema_version": SCHEMA_VERSION, "selected_model": selected["model"], "selected_candidate": selected["candidate"],
        "predictor_kind": "baseline forecast" if selected["candidate"] in ("g2", "mean") else "learned model",
        "feature_set": selected["feature_set"], "training_feature_set": training_feature_set,
        "features": features, "training_features": inputs, "feature_definitions": {name: FEATURE_LABELS[name] for name in inputs},
        "rows": len(data), "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "random_seed": 42, "cv_folds": 5, "split_config": SPLIT_CONFIG, "selection_rule": SELECTION_RULE,
        "tie_tolerance": TIE_TOLERANCE, "evaluation_limitations": LIMITATIONS,
        "grade_scale": [0, 20], "prediction_timing": "After period two; recorded G2 required",
        "bounding_policy": BOUND_POLICY, "missing_data_policy": "Reject missing required inputs; no imputation",
        "preprocessing": "Fold-fitted StandardScaler and fixed yes/no one-hot categories; persistence passes through G2",
        **regression_metrics(test.G3, predictions),
        "baseline_mae": float(mean_absolute_error(test.G3, np.full(len(test), train.G3.mean()))),
        "g2_baseline_mae": float(mean_absolute_error(test.G3, test.G2)),
        "g2_test_metrics": regression_metrics(test.G3, test.G2),
        "within_two_points": float(np.mean(errors <= 2)),
        "interval": {"method": "split conformal absolute residual", "nominal_coverage": COVERAGE,
                     "rank": ceil((len(calibration) + 1) * COVERAGE), "radius": radius,
                     "calibration_rows": len(calibration), "test_rows": len(test),
                     "test_covered": int(np.sum((test.G3 >= lower) & (test.G3 <= upper))),
                     "test_coverage": float(np.mean((test.G3 >= lower) & (test.G3 <= upper))),
                     "average_width": float(np.mean(upper - lower)), "constant_width": True},
        "feature_importance": [{"feature": name, "mae_increase": float(mean), "std": float(std)}
                               for name, mean, std in zip(features, importance.importances_mean, importance.importances_std)],
        "split_rows": {"train": train.index.tolist(), "calibration": calibration.index.tolist(), "test": test.index.tolist()},
        "cv_rows": [{"fit": train.index[a].tolist(), "validation": train.index[b].tolist()} for a, b in folds],
        "comparison": comparison, "target": TARGET,
    }
    results = pd.DataFrame({"source_row": test.index, "actual_grade": test.G3.to_numpy(), "recorded_G2": test.G2.to_numpy(),
                            "predicted_grade": predictions, "interval_lower": lower, "interval_upper": upper,
                            "absolute_error": errors})
    return fitted, metrics, results


def predict_grade(fitted, frame):
    features = getattr(fitted, "input_features_", FEATURES)
    clean = validate_frame(frame, features=features)
    try:
        predictions = np.asarray(fitted.predict(clean), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The predictor returned invalid results. Reload or retrain the model.") from exc
    if predictions.shape != (len(clean),) or not np.isfinite(predictions).all():
        raise ValueError("The predictor returned invalid results. Reload or retrain the model.")
    return predictions


def prediction_report(fitted, frame, metrics, threshold=10):
    threshold = finite_number(threshold, field="Reference threshold", minimum=0, maximum=20)
    if "bundle_id" in metrics and getattr(fitted, "bundle_id_", None) != metrics["bundle_id"]:
        raise ValueError("Predictor and evaluation bundle do not match.")
    features = getattr(fitted, "input_features_", FEATURES)
    clean = validate_frame(frame, features=features)
    grades = predict_grade(fitted, clean)
    report = observed_grades(frame, clean).reset_index(drop=True)
    radius = finite_number(metrics["interval"]["radius"], field="Prediction range radius", minimum=0, maximum=np.finfo(float).max)
    report["predicted_grade"] = grades
    report["percentage_equivalent"] = grades * 5
    report["observed_change_G1_to_G2"] = report.G2 - report.G1
    report["predicted_change_from_G2"] = grades - report.G2
    report["reference_threshold"] = threshold
    report["margin_to_threshold"] = grades - threshold
    report["interval_lower"] = grades - radius
    report["interval_upper"] = grades + radius
    numeric_results = ["predicted_grade", "percentage_equivalent", "predicted_change_from_G2",
                       "margin_to_threshold", "interval_lower", "interval_upper"]
    if not np.isfinite(report[numeric_results].to_numpy(dtype=float)).all():
        raise ValueError("The predictor returned an invalid report. Reload or retrain the model.")
    report["threshold_comparison"] = np.where(grades >= threshold, "At or above reference", "Below reference")
    for name in ("bundle_id", "feature_set", "schema_version", "prediction_timing", "predictor_kind"):
        if name in metrics:
            report[name] = metrics[name]
    return report


def alert_evaluation(actual, predicted, threshold):
    """Positive = recorded G3 < reference; alert = raw predicted G3 < same reference."""
    threshold = finite_number(threshold, field="Reference threshold", minimum=0, maximum=20)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted outcomes must be equally sized vectors.")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y, alert = actual[valid] < threshold, predicted[valid] < threshold
    tp, fp = int((y & alert).sum()), int((~y & alert).sum())
    fn, tn = int((y & ~alert).sum()), int((~y & ~alert).sum())
    return {"evaluated": int(valid.sum()), "excluded_missing": int((~valid).sum()), "reference_grade": threshold,
            "actual_below": tp + fn, "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
