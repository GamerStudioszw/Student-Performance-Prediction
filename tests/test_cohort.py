"""Verify staff summaries, import boundaries and public demonstration isolation."""

import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from cohort import (BANDS, MAX_BYTES, STATUSES, analyse_cohort, class_summary, csv_template,
                    example_cohort, export_csv, filter_cohort, grade_distribution, read_cohort_csv, validate_cohort)
from model import make_pipeline
from model_service import load_bundle


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(student_id="001", class_name="Class A", G1=12, G2=13, studytime=2, failures=0,
                        traveltime=1, schoolsup="no", famsup="yes", paid="no", activities="yes", internet="yes", higher="yes")
        self.frame = pd.DataFrame([self.row, {**self.row, "student_id": "002", "G1": 11, "G2": 8}])
        self.metrics = {"subject": "mat", "selected_model": "Test model", "trained_at": "test",
                        "interval": {"radius": 2, "nominal_coverage": 0.8, "test_coverage": 0.7}}

    def report(self, grade=9):
        clean = validate_cohort(self.frame)
        fitted = make_pipeline(DummyRegressor(strategy="constant", constant=grade)).fit(clean, [9, 9])
        return analyse_cohort(fitted, clean, self.metrics, 10)

    def test_import_preserves_codes_and_excludes_optional_outcome_from_features(self):
        payload = self.frame.assign(G3=[14, np.nan], student_name="Ignored name").to_csv(index=False).encode()
        cohort = read_cohort_csv(payload)
        self.assertEqual(cohort.student_id.tolist(), ["001", "002"])
        self.assertEqual(cohort.actual_grade.iloc[0], 14)
        self.assertTrue(pd.isna(cohort.actual_grade.iloc[1]))
        self.assertNotIn("student_name", cohort)
        self.assertNotIn("G3", cohort)
        self.assertEqual(len(read_cohort_csv(csv_template())), 1)

    def test_rejects_invalid_records_and_oversized_files(self):
        invalid = [self.frame.drop(columns="G2"), self.frame.assign(student_id="same"),
                   self.frame.assign(class_name=""), self.frame.assign(G1=21),
                   self.frame.assign(G3=21), self.frame.assign(student_id="bad\ncode")]
        for frame in invalid:
            with self.subTest(columns=frame.columns.tolist()), self.assertRaises(ValueError):
                validate_cohort(frame)
        for payload in (b"", b"\xff\xfe", b"a" * (MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                read_cohort_csv(payload)

    def test_threshold_flags_and_summary_totals(self):
        report = self.report()
        self.assertEqual(report.review_status.tolist(), ["Below reference"] * 2)
        summary = class_summary(report).iloc[0]
        self.assertEqual(summary.students, 2)
        self.assertEqual(summary.second_period_mean, 10.5)
        self.assertEqual(summary.below_reference, 2)
        self.assertEqual(summary.observed_declines, 1)
        self.assertEqual(self.report(11).review_status.iloc[0], "Near reference")
        self.assertEqual(self.report(15).review_status.iloc[0], "Above reference")
        self.assertEqual(self.report(21).review_status.iloc[0], "Outside grade scale")

    def test_filters_and_distribution_preserve_counts(self):
        report = self.report(21)
        self.assertEqual(len(filter_cohort(report, ["Class A"], STATUSES, "001")), 1)
        self.assertTrue(filter_cohort(report, [], STATUSES).empty)
        self.assertTrue(filter_cohort(report, ["Class A"], STATUSES, ".*").empty)
        distribution = grade_distribution(report)
        self.assertEqual(distribution.groupby("Series").Students.sum().tolist(), [2, 2])
        self.assertEqual(distribution[distribution["Grade band"] == BANDS[-1]].Students.sum(), 2)

    def test_csv_export_neutralizes_formulas(self):
        payload = export_csv(pd.DataFrame({"student_id": ["=1+1", "@name", "+2", " -3"], "grade": [-1, 2, 3, 4]})).decode("utf-8-sig")
        self.assertIn("'=1+1", payload)
        self.assertIn("'@name", payload)
        self.assertIn("'+2", payload)
        self.assertIn("' -3", payload)
        self.assertIn(",-1", payload)

    def test_public_example_contains_only_held_out_records(self):
        _, metrics, _ = load_bundle("mat")
        example = example_cohort("mat", metrics)
        row_numbers = {int(code.split("-")[1]) - 1 for code in example.student_id}
        self.assertEqual(row_numbers, set(metrics["split_rows"]["test"]))
        self.assertFalse(row_numbers & set(metrics["split_rows"]["train"]))
        self.assertFalse(row_numbers & set(metrics["split_rows"]["calibration"]))


if __name__ == "__main__":
    unittest.main()
