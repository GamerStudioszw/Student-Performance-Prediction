"""Local Streamlit interface for the student performance ML project."""

import logging

import pandas as pd
import streamlit as st

from model import ROOT, FEATURES, FEATURE_LABELS, SUBJECTS, PUBLIC_SUBJECTS, load_data, prediction_report
from model_service import bundle_signature, load_bundle
from dashboard import render_dashboard
from auth_ui import require_login
from subject_training import render_subject_training
from input_ui import profile_fields
from quality_ui import render_model_quality
from cohort import export_csv

st.set_page_config(page_title="StudentInsight | Staff analytics", page_icon="🎓", layout="wide")
require_login()
st.caption("STAFF WORKSPACE · STUDENT ANALYTICS")
st.title("Student Performance Analytics")
st.write("Review class progress, compare groups, and plan student support.")
subject = st.sidebar.selectbox("Subject", list(SUBJECTS), format_func=SUBJECTS.get, key="subject")
if subject not in SUBJECTS:
    st.error("Select a supported subject.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_subject_model(subject, signature):
    # Only public reference models are shared across sessions; uploaded classes are not cached here.
    return load_bundle(subject)


history = None
if subject in PUBLIC_SUBJECTS:
    try:
        with st.spinner("Preparing the analytics workspace. The first visit may take a moment…"):
            model, metrics, model_results = load_subject_model(subject, bundle_signature(subject))
    except Exception:
        logging.getLogger(__name__).exception("Could not prepare the subject model")
        st.error("The analytics workspace could not be prepared. Please retry or contact the app administrator.")
        if st.button("Retry loading"):
            load_subject_model.clear()
            st.rerun()
        st.stop()
else:
    workspace = render_subject_training(subject)
    if workspace is None:
        st.stop()
    model, metrics, model_results = workspace["bundle"]
    history = workspace["history"]
features = metrics["features"]
if "load_notice" in metrics and "recomputed" in metrics["load_notice"]:
    st.info(metrics["load_notice"])
with st.sidebar:
    st.header("Model context")
    st.write(f"**Subject:** {SUBJECTS[subject]}\n\n**Target:** Final grade, G3 (0–20)\n\n**Prediction point:** After period two")
    if history is None:
        st.caption("An educational demonstration based on historical records from two Portuguese schools. Local validation is needed before using it with another school.")
        st.link_button("View the UCI dataset", "https://archive.ics.uci.edu/dataset/320/student+performance")
    else:
        st.caption("Trained only on your uploaded records for this subject. Random held-out evaluation does not establish performance on future classes or other institutions. This model is available for this session only.")
    st.divider()
    st.write(f"**Selected predictor:** {metrics['selected_model']}")
    st.caption(metrics["predictor_kind"] + ". Selected using training-only cross-validation.")
    st.caption("Required: " + ", ".join(features) + ". After period two.")
    st.caption("Uploaded records and private models are lost on sign-out, session loss or server restart.")
    st.caption(f"{metrics['rows']} subject records · {len(features)} required inputs")

dashboard_tab, predict_tab, evaluation_tab, data_tab = st.tabs(["Class dashboard", "Student analysis", "Model quality", "Data guide"])

with dashboard_tab:
    render_dashboard(model, metrics, subject, history=history)

with predict_tab:
    st.subheader("Student learning profile")
    st.caption("Use recorded values available after period two. Each subject has a separately selected predictor.")
    st.caption("Required inputs: " + ", ".join(features) + ". No missing input is invented or imputed.")
    with st.form("prediction"):
        student = profile_fields(features)
        threshold = st.number_input("Reference grade for comparison", 0, 20, 10,
                                    help="A threshold for interpreting the estimate. It is not a model input or an official pass rule.")
        submitted = st.form_submit_button("Predict final grade", type="primary", width="stretch")
    if submitted:
        try:
            report = prediction_report(model, student, metrics, threshold)
        except ValueError as exc:
            st.session_state.pop("grade_report", None)
            st.error(str(exc))
            st.stop()
        report["subject"] = SUBJECTS[subject]
        report["model"] = metrics["selected_model"]
        report["nominal_interval_coverage"] = metrics["interval"]["nominal_coverage"]
        report["measured_test_coverage"] = metrics["interval"]["test_coverage"]
        st.session_state["grade_report"] = {"subject": subject, "bundle_id": metrics["bundle_id"], "report": report}
    saved = st.session_state.get("grade_report")
    if saved and saved["subject"] == subject and saved.get("bundle_id") == metrics["bundle_id"]:
        report = saved["report"]
        result = report.iloc[0]
        grade = float(result["predicted_grade"])
        cards = st.columns(3)
        cards[0].metric("Estimated final grade", f"{grade:.1f} / 20")
        cards[1].metric("Percentage equivalent", f"{result['percentage_equivalent']:.1f}%")
        cards[2].metric("Estimated change from G2", f"{result['predicted_change_from_G2']:+.1f} points")
        st.caption("The percentage is the predicted grade converted to a 0–100 scale. It is not a probability of passing. Results describe your last submitted profile.")
        st.subheader("Progress and reference grade")
        progress, comparison = st.columns(2)
        progress.metric("Observed change: G1 to G2", "Unavailable" if pd.isna(result["observed_change_G1_to_G2"]) else f"{result['observed_change_G1_to_G2']:+.0f} points")
        comparison.metric(f"Margin to reference ({result['reference_threshold']:.0f}/20)", f"{result['margin_to_threshold']:+.1f} points")
        st.write(f"**{result['threshold_comparison']}** — this comparison uses the point estimate and is not a pass/fail classification model.")
        st.caption("Progress is recorded G2 minus recorded G1. A missing G1 leaves progress unavailable; it is not filled with zero.")
        st.subheader("Prediction range")
        st.write(f"**{result['interval_lower']:.1f} to {result['interval_upper']:.1f} / 20** · nominal 80% prediction range")
        st.caption(f"Calibrated on {metrics['calibration_rows']} separate records. Actual coverage on {metrics['test_rows']} held-out records: "
                   f"{metrics['interval']['test_coverage']:.1%}. Coverage describes groups of similar records, not an 80% guarantee for this student or another school. Ranges are unbounded and can extend outside 0–20.")
        st.caption(f"Fixed interval width: {metrics['interval']['average_width']:.2f} points for every student using this predictor. This is not personalised uncertainty.")
        if metrics["interval"]["test_coverage"] < 0.8:
            st.warning("The measured test coverage was below the nominal 80% level. Treat this range cautiously.")
        if result["interval_lower"] <= result["reference_threshold"] <= result["interval_upper"]:
            st.info("The reference grade falls within the prediction range, so the outcome relative to that grade is uncertain.")
        st.caption(f"Held-out mean absolute error: {metrics['mae']:.2f} grade points. This is average error, not an individual confidence interval.")
        if grade < 0 or grade > 20:
            st.warning("The model estimate is outside the valid 0–20 grade scale. Treat this input as an extrapolation.")
        st.info("Use the estimate to support a conversation about learning needs. Changing an input does not establish that it causes a grade change.")
        st.download_button("Download this prediction report", export_csv(report),
                           f"{subject}_prediction_report.csv", "text/csv")

with evaluation_tab:
    render_model_quality(metrics, model_results)

with data_tab:
    st.subheader("The training dataset")
    st.caption(f"Source: {metrics['source']} · Schema: {metrics['feature_set']} · Bundle: {metrics['bundle_id']}")
    st.caption(metrics["missing_data_policy"] + ". Grade scale 0–20; after period two.")
    try:
        data = (load_data(ROOT / "data" / f"student-{subject}.csv") if history is None
                else history[metrics["training_features"]].assign(G3=history.actual_grade.astype(int)))
    except FileNotFoundError:
        st.warning("The reference dataset is unavailable. Please contact the app administrator.")
    else:
        st.write(f"{len(data)} {SUBJECTS[subject].lower()} records · {len(metrics['training_features'])} training inputs; {len(features)} active predictor inputs · final grade G3 as the target")
        if history is None:
            st.caption("The project includes 395 mathematics and 649 Portuguese-language subject records. Some students occur in both files; 1,044 subject records does not mean 1,044 unique students. Models and evaluation splits stay separate by subject.")
        else:
            st.caption(f"Source: your uploaded {SUBJECTS[subject]} history. No records from another subject were added. Split by student record into fitting, calibration and test partitions; classes and time periods are not held out as groups.")
        quality = st.columns(3)
        quality[0].metric("Missing predictor/target values", int(data.isna().sum().sum()))
        quality[1].metric("Selected input fields", len(features))
        if history is None:
            quality[2].metric("Source schools", 2)
        else:
            quality[2].metric("Source classes / groups", history.class_name.nunique())
        distribution = data["G3"].value_counts().reindex(range(21), fill_value=0).rename_axis("Final grade").to_frame("Students")
        st.bar_chart(distribution)
        st.dataframe(data, hide_index=True, width="stretch")
    if history is None:
        st.caption("Source: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. DOI: 10.24432/C5TG7T. CC BY 4.0.")
    st.write("This predictor estimates final performance after period two using the listed required inputs. It is not an early-enrollment prediction system. G2 must be recorded. Verify that every input was known by the end of period two, before the final outcome was known.")
    st.subheader("What would make the dataset stronger?")
    st.write("Collect recent, consented records from your target schools across several cohorts, using consistent subject and grading definitions. Timestamp attendance, assignment scores, and engagement before the prediction date, then record the final result later. Evaluate on later cohorts and schools excluded from training.")
    st.caption("The bundled source has no reliable prediction-date snapshot for absences, so attendance is not added as an input. More columns or another subject do not by themselves demonstrate better generalisation.")
