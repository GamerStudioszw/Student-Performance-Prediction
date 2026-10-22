import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from model import (FEATURES, ROOT, FEATURE_SETS, PersistenceRegressor, conformal_radius, load_data, make_pipeline, predict_grade,
                   prediction_report, split_data, train_and_evaluate, validate_frame, select_candidate, alert_evaluation)
from sklearn.model_selection import cross_val_score


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame([dict(G1=12, G2=13, studytime=2, failures=0, schoolsup="no", famsup="yes",
                                         traveltime=1, paid="no", activities="yes", internet="yes", higher="yes")])

    def test_validation_rejects_invalid_records(self):
        for column, value in [("G1", -1), ("G2", 21), ("studytime", 0), ("failures", 1.5), ("failures", 4),
                              ("G1", np.inf), ("G2", np.nan), ("schoolsup", "maybe"), ("famsup", None),
                              ("traveltime", 5), ("paid", "sometimes"), ("internet", None)]:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_frame(self.student.assign(**{column: value}))

    def test_missing_column_and_empty_data(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            validate_frame(self.student.drop(columns="G1"))
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_frame(self.student.iloc[:0])

    def test_target_cannot_enter_predictors(self):
        clean = validate_frame(self.student.assign(G3=20, student_name="Example"))
        self.assertEqual(list(clean.columns), FEATURES)
        self.assertNotIn("G3", FEATURES)

    def test_prediction_ignores_extra_target_and_column_order(self):
        records = pd.concat([self.student, self.student.assign(G1=8, schoolsup="yes", famsup="no")], ignore_index=True)
        fitted = make_pipeline(DummyRegressor()).fit(records, [10, 14])
        sample = self.student.assign(G3=0).loc[:, ["G3", *reversed(FEATURES)]]
        np.testing.assert_allclose(predict_grade(fitted, sample), [12])

    def test_calibration_quantile_and_invalid_residuals(self):
        self.assertEqual(conformal_radius(np.arange(1, 11)), 9)
        for values in ([], [-1], [np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                conformal_radius(values)
        with self.assertRaises(ValueError):
            conformal_radius([1], coverage=0.99)

    def test_report_distinguishes_observed_and_predicted_changes(self):
        fitted = make_pipeline(DummyRegressor(strategy="constant", constant=12)).fit(self.student, [12])
        report = prediction_report(fitted, self.student, {"interval": {"radius": 2}}, threshold=10).iloc[0]
        self.assertEqual(report["predicted_grade"], 12)
        self.assertEqual(report["percentage_equivalent"], 60)
        self.assertEqual(report["observed_change_G1_to_G2"], 1)
        self.assertEqual(report["predicted_change_from_G2"], -1)
        self.assertEqual(report["margin_to_threshold"], 2)
        self.assertEqual(report["interval_lower"], 10)
        self.assertEqual(report["interval_upper"], 14)
        with self.assertRaises(ValueError):
            prediction_report(fitted, self.student, {"interval": {"radius": 2}}, threshold=21)

    def test_calibration_and_holdout_never_fit_model(self):
        data = load_data().iloc[:100]
        train, calibration, test = split_data(data)
        sets = [set(part.index) for part in (train, calibration, test)]
        self.assertEqual([len(part) for part in sets], [60, 20, 20])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        fitted, metrics, results = train_and_evaluate(data)
        if "numeric" in fitted.named_steps["preprocessing"].named_transformers_:
            scaler = fitted.named_steps["preprocessing"].named_transformers_["numeric"]
            self.assertEqual(scaler.n_samples_seen_, len(train))
        residuals = np.abs(calibration.G3.to_numpy() - predict_grade(fitted, calibration))
        self.assertAlmostEqual(metrics["interval"]["radius"], conformal_radius(residuals))
        self.assertEqual(set(results.source_row), sets[2])
        self.assertEqual(metrics["split_rows"]["train"], train.index.tolist())
        self.assertEqual(len(metrics["feature_importance"]), len(metrics["features"]))
        self.assertEqual(metrics["interval"]["average_width"], 2 * metrics["interval"]["radius"])
        for fold in metrics["cv_rows"]:
            self.assertTrue(set(fold["fit"]) <= sets[0])
            self.assertTrue(set(fold["validation"]) <= sets[0])
            self.assertFalse(set(fold["fit"]) & set(fold["validation"]))

    def test_g2_forecast_uses_only_g2_and_missing_progress_stays_missing(self):
        fitted = make_pipeline(PersistenceRegressor(), ["G2"]).fit(pd.DataFrame({"G2": [0, 12, 20]}), [20, 0, 1])
        profile = pd.DataFrame({"G2": [0, 7, 20], "G3": [20, 20, 0], "student_id": ["a", "b", "c"]})
        np.testing.assert_array_equal(predict_grade(fitted, profile), [0, 7, 20])
        report = prediction_report(fitted, profile, {"interval": {"radius": 1}})
        self.assertTrue(report.G1.isna().all())
        self.assertTrue(report.observed_change_G1_to_G2.isna().all())
        self.assertNotIn("student_id", report)
        self.assertNotIn("G3", report)
        with self.assertRaises(ValueError):
            predict_grade(fitted, profile.drop(columns="G2"))
        with self.assertRaises(ValueError):
            predict_grade(fitted, profile.assign(G2=np.nan))
        with self.assertRaises(ValueError):
            validate_frame(profile, features=["G3"])

    def test_identical_training_only_folds_and_baseline_participation(self):
        data = load_data().iloc[:80].assign(student_id="excluded", class_name="excluded")
        calls = []
        def score(estimator, x, y, **kwargs):
            calls.append((estimator, x.copy(), y.copy(), kwargs["cv"]))
            self.assertIsInstance(estimator.named_steps["preprocessing"], __import__("sklearn.compose", fromlist=["ColumnTransformer"]).ColumnTransformer)
            return cross_val_score(estimator, x, y, **kwargs)
        with patch("model.cross_val_score", side_effect=score):
            _, metrics, _ = train_and_evaluate(data)
        self.assertEqual(len(calls), 6)
        self.assertEqual({row["candidate"] for row in metrics["comparison"]},
                         {"g2", "mean", "ridge_grades", "ridge_expanded", "forest_grades", "forest_expanded"})
        train = split_data(data)[0]
        for _, x, y, folds in calls:
            self.assertEqual(x.index.tolist(), train.index.tolist())
            self.assertEqual(y.index.tolist(), train.index.tolist())
            self.assertTrue(set(x.columns).issubset(FEATURES))
            self.assertIs(folds, calls[0][3])
        expected = [float(np.abs(train.iloc[b].G3 - train.iloc[b].G2).mean()) for _, b in calls[0][3]]
        np.testing.assert_allclose(metrics["comparison"][0]["fold_mae"], expected)
        # Fit one actual ridge fold, then verify scaler sees only that fold's fitting records.
        ridge, x, y, folds = calls[2]
        a, b = folds[0]
        ridge.fit(x.iloc[a], y.iloc[a])
        scaler = ridge.named_steps["preprocessing"].named_transformers_["numeric"]
        self.assertEqual(scaler.n_samples_seen_, len(a))
        np.testing.assert_allclose(scaler.mean_, x.iloc[a].mean())

    def test_simple_candidate_tie_rule_and_finite_quantile_boundary(self):
        rows = [{"candidate": "g2", "cv_mae": 1.019, "simplicity_rank": 0},
                {"candidate": "forest", "cv_mae": 1., "simplicity_rank": 5}]
        self.assertEqual(select_candidate(rows)["candidate"], "g2")
        rows[0]["cv_mae"] = 1.021
        self.assertEqual(select_candidate(rows)["candidate"], "forest")
        self.assertEqual(conformal_radius([0, 1, 2, 3], .8), 3)
        for coverage in (.8, .99):
            with self.assertRaisesRegex(ValueError, "finite"):
                conformal_radius([0, 1, 2], coverage)
        with self.assertRaises(ValueError):
            conformal_radius([0, 1, 2, 3], float("nan"))

    def test_alert_boundaries_missing_outcomes_and_undefined_denominators(self):
        result = alert_evaluation([9, 10, 9, 10, np.nan], [9, 9, 10, 10, 0], 10)
        self.assertEqual([result[k] for k in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertEqual(result["excluded_missing"], 1)
        no_positive = alert_evaluation([10, 20], [10, 20], 10)
        self.assertIsNone(no_positive["precision"])
        self.assertIsNone(no_positive["recall"])
        self.assertIsNone(no_positive["f1"])
        missed = alert_evaluation([9], [10], 10)
        self.assertIsNone(missed["precision"])
        self.assertEqual(missed["recall"], 0)
        self.assertEqual(missed["f1"], 0)
        self.assertEqual(alert_evaluation([], [], 0)["evaluated"], 0)
        with self.assertRaises(ValueError):
            alert_evaluation([9], [9, 9], 10)

    def test_both_subject_files_pass_validation(self):
        for subject, count in (("mat", 395), ("por", 649)):
            with self.subTest(subject=subject):
                data = load_data(ROOT / "data" / f"student-{subject}.csv")
                self.assertEqual(len(data), count)
                self.assertEqual(list(data.columns), [*FEATURES, "G3"])


if __name__ == "__main__":
    unittest.main()
