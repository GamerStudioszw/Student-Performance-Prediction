import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from model import FEATURES, make_pipeline, predict_grade, validate_frame


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame([dict(G1=12, G2=13, studytime=2, failures=0, schoolsup="no", famsup="yes")])

    def test_validation_rejects_invalid_records(self):
        for column, value in [("G1", -1), ("G2", 21), ("studytime", 0), ("failures", 1.5), ("failures", 4),
                              ("G1", np.inf), ("G2", np.nan), ("schoolsup", "maybe"), ("famsup", None)]:
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


if __name__ == "__main__":
    unittest.main()
