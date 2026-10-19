import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from model import (FEATURES, ROOT, conformal_radius, load_data, make_pipeline, predict_grade,
                   prediction_report, split_data, train_and_evaluate, validate_frame)


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
        scaler = fitted.named_steps["preprocessing"].named_transformers_["numeric"]
        self.assertEqual(scaler.n_samples_seen_, len(train))
        residuals = np.abs(calibration.G3.to_numpy() - predict_grade(fitted, calibration))
        self.assertAlmostEqual(metrics["interval"]["radius"], conformal_radius(residuals))
        self.assertEqual(set(results.source_row), sets[2])
        self.assertEqual(metrics["split_rows"]["train"], train.index.tolist())
        self.assertEqual(len(metrics["feature_importance"]), len(FEATURES))

    def test_both_subject_files_pass_validation(self):
        for subject, count in (("mat", 395), ("por", 649)):
            with self.subTest(subject=subject):
                data = load_data(ROOT / "data" / f"student-{subject}.csv")
                self.assertEqual(len(data), count)
                self.assertEqual(list(data.columns), [*FEATURES, "G3"])


if __name__ == "__main__":
    unittest.main()
