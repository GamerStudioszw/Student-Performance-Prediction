"""Validated cohort inputs and staff-facing summaries, independent of the UI."""

from io import BytesIO
import csv

import numpy as np
import pandas as pd

from model import FEATURES, ROOT, SUBJECTS, prediction_report, validate_frame

MAX_ROWS = 5000
MAX_BYTES = 2 * 1024 * 1024
STATUSES = ["Outside grade scale", "Below reference", "Near reference", "Above reference"]
BANDS = ["Below 0", "0–<5", "5–<10", "10–<15", "15–20", "Above 20"]


def validate_cohort(frame):
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Upload no more than {MAX_ROWS:,} student records at a time.")
    missing = sorted(set(["student_id", "class_name", *FEATURES]) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    clean = validate_frame(frame).reset_index(drop=True)
    for name in ("student_id", "class_name"):
        values = frame[name].reset_index(drop=True).astype("string").str.strip()
        if values.isna().any() or values.eq("").any() or values.str.len().gt(80).any():
            raise ValueError(f"{name} must contain a nonempty code or label of at most 80 characters.")
        if values.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
            raise ValueError(f"{name} must not contain control characters.")
        clean[name] = values.astype(str)
    if clean.student_id.duplicated().any():
        raise ValueError("student_id must be unique within the file. Remove repeated student records.")
    if "G3" in frame:
        final = frame["G3"].reset_index(drop=True)
        present = final.notna() & final.astype(str).str.strip().ne("")
        clean["actual_grade"] = np.nan
        if present.any():
            checked = validate_frame(frame.reset_index(drop=True).loc[present], require_target=True)
            clean.loc[present, "actual_grade"] = checked["G3"].to_numpy()
    return clean


def read_cohort_csv(content):
    if len(content) > MAX_BYTES:
        raise ValueError("Use a CSV file smaller than 2 MB.")
    try:
        frame = pd.read_csv(BytesIO(content), sep=None, engine="python", encoding="utf-8-sig",
                            dtype={"student_id": "string", "class_name": "string"}, nrows=MAX_ROWS + 1)
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise ValueError("The file could not be read. Use a UTF-8 CSV with the template's column names.") from exc
    return validate_cohort(frame)


def example_cohort(subject, metrics):
    """Only untouched test records are used for the historical demonstration."""
    frame = pd.read_csv(ROOT / "data" / f"student-{subject}.csv", sep=";").loc[metrics["split_rows"]["test"]].copy()
    frame["student_id"] = [f"{subject.upper()}-{index + 1:04d}" for index in frame.index]
    frame["class_name"] = "School " + frame["school"]
    return validate_cohort(frame)


def analyse_cohort(fitted, frame, metrics, threshold):
    report = prediction_report(fitted, frame, metrics, threshold)
    report["student_id"] = frame["student_id"].to_numpy()
    report["class_name"] = frame["class_name"].to_numpy()
    if "actual_grade" in frame:
        report["actual_grade"] = frame["actual_grade"].to_numpy()
    grade = report["predicted_grade"]
    report["review_status"] = np.select(
        [(grade < 0) | (grade > 20), grade < threshold,
         (report["interval_lower"] <= threshold) & (report["interval_upper"] >= threshold)],
        STATUSES[:3], default=STATUSES[3])
    report["subject"] = SUBJECTS[metrics["subject"]]
    report["model"] = metrics["selected_model"]
    report["model_trained_at"] = metrics["trained_at"]
    report["nominal_interval_coverage"] = metrics["interval"]["nominal_coverage"]
    report["reference_test_coverage"] = metrics["interval"]["test_coverage"]
    return report


def filter_cohort(report, classes, statuses, query=""):
    selected = report[report.class_name.isin(classes) & report.review_status.isin(statuses)]
    if query.strip():
        selected = selected[selected.student_id.str.contains(query.strip(), case=False, regex=False)]
    return selected.copy()


def class_summary(report):
    return report.assign(below_reference=report.margin_to_threshold < 0,
                         declining=report.observed_change_G1_to_G2 < 0).groupby("class_name", sort=True).agg(
        students=("student_id", "size"), first_period_mean=("G1", "mean"),
        second_period_mean=("G2", "mean"), estimated_final_mean=("predicted_grade", "mean"),
        below_reference=("below_reference", "sum"), observed_declines=("declining", "sum"),
    ).reset_index()


def grade_distribution(report):
    rows = []
    for column, series in (("G2", "Second period (observed)"), ("predicted_grade", "Final grade (estimated)")):
        values = report[column]
        bands = np.select([values < 0, values < 5, values < 10, values < 15, values <= 20], BANDS[:5], default=BANDS[5])
        counts = pd.Series(bands).value_counts().reindex(BANDS, fill_value=0)
        rows.extend({"Grade band": band, "Series": series, "Students": int(count)} for band, count in counts.items())
    return pd.DataFrame(rows)


def export_csv(frame):
    # Neutralize spreadsheet formulas in externally supplied IDs/class labels.
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]):
        safe[column] = safe[column].map(lambda value: "'" + value if isinstance(value, str)
                                       and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return safe.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def csv_template():
    return export_csv(pd.DataFrame([dict(student_id="S001", class_name="Class A", G1=12, G2=13,
        studytime=2, failures=0, traveltime=1, schoolsup="no", famsup="yes", paid="no",
        activities="yes", internet="yes", higher="yes")]))
