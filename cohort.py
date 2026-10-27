"""Validated cohort inputs and staff-facing summaries, independent of the UI."""

from io import StringIO
import csv
import unicodedata

import numpy as np
import pandas as pd

from model import FEATURES, ROOT, SUBJECTS, prediction_report, validate_frame, observed_grades
from validation import safe_text

MAX_ROWS = 5000
MAX_BYTES = 2 * 1024 * 1024
MAX_COLUMNS = 64
MAX_CELL_CHARS = 1000
STATUSES = ["Outside grade scale", "Below reference", "Near reference", "Above reference"]
BANDS = ["Below 0", "0–<5", "5–<10", "10–<15", "15–20", "Above 20"]


def student_details(student_name="", student_id="", class_name=""):
    """Optional single-student labels; these never enter the predictor."""
    return {key: safe_text(value, field=key, minimum=0, maximum=80)
            for key, value in {"student_name": student_name, "student_id": student_id,
                               "class_name": class_name}.items()}


def identity_columns(identifier):
    options = {"student_name": ["student_name"], "student_id": ["student_id"],
               "both": ["student_name", "student_id"]}
    if not isinstance(identifier, str) or identifier not in options:
        raise ValueError("Choose student names, student IDs, or both.")
    return options[identifier]


def validate_cohort(frame, features=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Provide a table with unique column names.")
    features = FEATURES if features is None else list(features)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Upload no more than {MAX_ROWS:,} student records at a time.")
    missing = sorted(set(["class_name", *features]) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    clean = observed_grades(frame, validate_frame(frame, features=features)).reset_index(drop=True)
    for name in ("student_id", "student_name", "class_name"):
        values = (frame[name].reset_index(drop=True).astype("string").fillna("") if name in frame
                  else pd.Series("", index=clean.index, dtype="string"))
        clean[name] = values.map(lambda value: safe_text(value, field=name, minimum=1 if name == "class_name" else 0))
    has_id = clean.student_id.ne("")
    if (clean.student_name.eq("") & ~has_id).any():
        raise ValueError("Each student needs a student_name or student_id. Include either column, or both.")
    if clean.loc[has_id, "student_id"].duplicated().any():
        raise ValueError("student_id must be unique within the file. Remove repeated student records.")
    # Do not invent persistent IDs from row positions or treat names as globally unique.
    normalized = clean[["student_name", "class_name"]].map(
        lambda value: " ".join(unicodedata.normalize("NFKC", value).casefold().split()))
    ambiguous = normalized.duplicated(keep=False) & clean.student_name.ne("")
    if (ambiguous & ~has_id).any():
        raise ValueError("Students sharing a name in the same class need distinct student_id values. Remove accidental duplicate rows.")
    if "G3" in frame:
        final = frame["G3"].reset_index(drop=True)
        present = final.notna() & final.astype(str).str.strip().ne("")
        clean["actual_grade"] = np.nan
        if present.any():
            checked = validate_frame(frame.reset_index(drop=True).loc[present], require_target=True, features=features)
            clean.loc[present, "actual_grade"] = checked["G3"].to_numpy()
    return clean


def parse_csv(content):
    """Parse once without header mangling, implicit indexes or type guessing."""
    if not isinstance(content, bytes):
        raise ValueError("Provide a UTF-8 CSV file as bytes.")
    if len(content) > MAX_BYTES:
        raise ValueError("Use a CSV file smaller than 2 MB.")
    try:
        text = content.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("CSV files must not contain null bytes.")
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
        reader = csv.reader(StringIO(text, newline=""), dialect, strict=True)
        header = next(reader)
        if not 1 <= len(header) <= MAX_COLUMNS:
            raise ValueError(f"Use at most {MAX_COLUMNS} CSV columns.")
        for name in header:
            if safe_text(name, field="Column name") != name:
                raise ValueError("Column names must not have leading or trailing spaces.")
        if len(set(name.casefold() for name in header)) != len(header):
            raise ValueError("CSV column names must be unique; duplicate headers are not allowed.")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(rows) >= MAX_ROWS:
                raise ValueError(f"Upload no more than {MAX_ROWS:,} student records at a time.")
            if len(row) != len(header):
                raise ValueError(f"CSV row {row_number} must have exactly {len(header)} fields.")
            if any(len(value) > MAX_CELL_CHARS for value in row):
                raise ValueError(f"CSV row {row_number} contains a field longer than {MAX_CELL_CHARS} characters.")
            rows.append(row)
    except (UnicodeError, csv.Error, IndexError, StopIteration) as exc:
        raise ValueError("The file could not be read. Use a UTF-8 CSV with the template's column names.") from exc
    # Preserve literal identifiers such as NA and 001; only empty cells are missing.
    return pd.DataFrame(rows, columns=header, dtype="string").replace("", pd.NA)


def read_cohort_csv(content, features=None):
    return validate_cohort(parse_csv(content), features)


def example_cohort(subject, metrics):
    """Only untouched test records are used for the historical demonstration."""
    frame = pd.read_csv(ROOT / "data" / f"student-{subject}.csv", sep=";").loc[metrics["split_rows"]["test"]].copy()
    frame["student_id"] = [f"{subject.upper()}-{index + 1:04d}" for index in frame.index]
    frame["class_name"] = "School " + frame["school"]
    return validate_cohort(frame, metrics["features"])


def analyse_cohort(fitted, frame, metrics, threshold):
    report = prediction_report(fitted, frame, metrics, threshold)
    report["student_id"] = frame["student_id"].to_numpy()
    report["student_name"] = frame.get("student_name", pd.Series("", index=frame.index)).to_numpy()
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
        matches = selected.student_id.str.contains(query.strip(), case=False, regex=False)
        if "student_name" in selected:
            matches |= selected.student_name.str.contains(query.strip(), case=False, regex=False)
        selected = selected[matches]
    return selected.copy()


def class_summary(report):
    return report.assign(below_reference=report.margin_to_threshold < 0,
                         declining=report.observed_change_G1_to_G2 < 0).groupby("class_name", sort=True).agg(
        students=("student_id", "size"), first_period_mean=("G1", "mean"),
        second_period_mean=("G2", "mean"), estimated_final_mean=("predicted_grade", "mean"),
        below_reference=("below_reference", "sum"), observed_declines=("declining", "sum"),
        progress_available=("observed_change_G1_to_G2", "count"),
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


def csv_template(features=None, identifier="student_id"):
    features = FEATURES if features is None else list(features)
    example = pd.DataFrame([dict(student_id="S001", student_name="Example Student", class_name="Class A", G1=12, G2=13,
        studytime=2, failures=0, traveltime=1, schoolsup="no", famsup="yes", paid="no",
        activities="yes", internet="yes", higher="yes")])
    return export_csv(example[[*identity_columns(identifier), "class_name", *features]])
