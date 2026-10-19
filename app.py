"""Local Streamlit interface for the student performance ML project."""

import logging

import pandas as pd
import streamlit as st

from model import ROOT, FEATURES, FEATURE_LABELS, SUBJECTS, PUBLIC_SUBJECTS, load_data, prediction_report
from model_service import bundle_signature, load_bundle
from dashboard import render_dashboard
from auth_ui import require_login
from subject_training import render_subject_training

st.set_page_config(page_title="StudentInsight | Staff analytics", page_icon="🎓", layout="wide")
require_login()
st.caption("STAFF WORKSPACE · STUDENT ANALYTICS")
st.title("Student Performance Analytics")
st.write("Review class progress, compare groups, and plan student support.")
subject = st.sidebar.selectbox("Subject", list(SUBJECTS), format_func=SUBJECTS.get, key="subject")


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
with st.sidebar:
    st.header("Model context")
    st.write(f"**Subject:** {SUBJECTS[subject]}\n\n**Target:** Final grade, G3 (0–20)\n\n**Prediction point:** After period two")
    if history is None:
        st.caption("An educational demonstration based on historical records from two Portuguese schools. Local validation is needed before using it with another school.")
        st.link_button("View the UCI dataset", "https://archive.ics.uci.edu/dataset/320/student+performance")
    else:
        st.caption("Trained only on your uploaded records for this subject. Random held-out evaluation does not establish performance on future classes or other institutions. This model is available for this session only.")
    st.divider()
    st.write(f"**Selected model:** {metrics['selected_model']}")
    st.caption("Selected by five-fold cross-validation on the training partition.")
    st.caption(f"{metrics['rows']} subject records · {len(FEATURES)} inputs")

dashboard_tab, predict_tab, evaluation_tab, data_tab = st.tabs(["Class dashboard", "Student analysis", "Model quality", "Data guide"])

with dashboard_tab:
    render_dashboard(model, metrics, subject, history=history)

with predict_tab:
    st.subheader("Student learning profile")
    st.caption("Use values available after period two. Each subject has its own independently trained model.")
    with st.form("prediction"):
        left, right = st.columns(2)
        with left:
            g1 = st.number_input("First period grade (G1)", 0, 20, 12, key="profile_G1")
            g2 = st.number_input("Second period grade (G2)", 0, 20, 12, key="profile_G2")
            studytime = st.selectbox(
                "Weekly study time", [1, 2, 3, 4], index=1,
                format_func=lambda value: {1: "Under 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "Over 10 hours"}[value],
            )
            traveltime = st.selectbox("Travel time to school", [1, 2, 3, 4],
                                     format_func=lambda v: {1: "Under 15 minutes", 2: "15–30 minutes",
                                                            3: "30–60 minutes", 4: "Over 60 minutes"}[v])
            paid = st.selectbox("Paid extra classes in this subject", ["no", "yes"])
            activities = st.selectbox("Extracurricular activities", ["yes", "no"])
        with right:
            failures = st.selectbox("Previous class failures", [0, 1, 2, 3],
                                    help="The supported input range is 0–3 previous failures.")
            schoolsup = st.selectbox("Extra educational support at school", ["no", "yes"])
            famsup = st.selectbox("Family educational support", ["yes", "no"])
            internet = st.selectbox("Internet access at home", ["yes", "no"])
            higher = st.selectbox("Plans to pursue higher education", ["yes", "no"])
        threshold = st.number_input("Reference grade for comparison", 0, 20, 10,
                                    help="A threshold for interpreting the estimate. It is not a model input or an official pass rule.")
        submitted = st.form_submit_button("Predict final grade", type="primary", width="stretch")
    if submitted:
        student = pd.DataFrame([{"G1": g1, "G2": g2, "studytime": studytime, "failures": failures,
                                 "schoolsup": schoolsup, "famsup": famsup, "traveltime": traveltime,
                                 "paid": paid, "activities": activities, "internet": internet, "higher": higher}])
        report = prediction_report(model, student, metrics, threshold)
        report["subject"] = SUBJECTS[subject]
        report["model"] = metrics["selected_model"]
        report["nominal_interval_coverage"] = metrics["interval"]["nominal_coverage"]
        report["measured_test_coverage"] = metrics["interval"]["test_coverage"]
        st.session_state["grade_report"] = {"subject": subject, "trained_at": metrics["trained_at"], "report": report}
    saved = st.session_state.get("grade_report")
    if saved and saved["subject"] == subject and saved["trained_at"] == metrics["trained_at"]:
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
        progress.metric("Observed change: G1 to G2", f"{result['observed_change_G1_to_G2']:+.0f} points")
        comparison.metric(f"Margin to reference ({result['reference_threshold']:.0f}/20)", f"{result['margin_to_threshold']:+.1f} points")
        st.write(f"**{result['threshold_comparison']}** — this comparison uses the point estimate and is not a pass/fail classification model.")
        st.subheader("Prediction range")
        st.write(f"**{result['interval_lower']:.1f} to {result['interval_upper']:.1f} / 20** · nominal 80% prediction range")
        st.caption(f"Calibrated on {metrics['calibration_rows']} separate records. Actual coverage on {metrics['test_rows']} held-out records: "
                   f"{metrics['interval']['test_coverage']:.1%}. Coverage describes groups of similar records, not an 80% guarantee for this student or another school. Ranges are unbounded and can extend outside 0–20.")
        if metrics["interval"]["test_coverage"] < 0.8:
            st.warning("The measured test coverage was below the nominal 80% level. Treat this range cautiously.")
        if result["interval_lower"] <= result["reference_threshold"] <= result["interval_upper"]:
            st.info("The reference grade falls within the prediction range, so the outcome relative to that grade is uncertain.")
        st.caption(f"Held-out mean absolute error: {metrics['mae']:.2f} grade points. This is average error, not an individual confidence interval.")
        if grade < 0 or grade > 20:
            st.warning("The model estimate is outside the valid 0–20 grade scale. Treat this input as an extrapolation.")
        st.info("Use the estimate to support a conversation about learning needs. Changing an input does not establish that it causes a grade change.")
        st.download_button("Download this prediction report", report.to_csv(index=False),
                           f"{subject}_prediction_report.csv", "text/csv")

with evaluation_tab:
    st.subheader("Performance on unseen records")
    cards = st.columns(4)
    cards[0].metric("Mean absolute error ↓", f"{metrics['mae']:.2f}")
    cards[1].metric("Root mean squared error ↓", f"{metrics['rmse']:.2f}")
    cards[2].metric("R² ↑", f"{metrics['r2']:.3f}")
    cards[3].metric("Baseline MAE ↓", f"{metrics['baseline_mae']:.2f}")
    st.caption(f"{metrics['train_rows']} fitting records · {metrics['calibration_rows']} calibration records · {metrics['test_rows']} held-out test records · fixed random seed 42. Errors are measured on the 0–20 scale.")
    extra = st.columns(3)
    extra[0].metric("Predictions within ±2 points", f"{metrics['within_two_points']:.1%}")
    extra[1].metric("Prediction range coverage", f"{metrics['interval']['test_coverage']:.1%}")
    extra[2].metric("G2-only baseline MAE", f"{metrics['g2_baseline_mae']:.2f}")
    st.caption("The G2-only baseline uses the second-period grade as the final estimate. Range coverage is measured on the test set; its nominal target is 80%.")
    if metrics["mae"] > metrics["g2_baseline_mae"]:
        st.info("Simply reusing the second-period grade had lower test error than the selected model on this split. The more detailed model has not demonstrated an improvement over that simple benchmark.")
    st.write("The mean baseline predicts the training-set average for every student. R² compares prediction errors with variation in the test grades; it can be negative.")
    st.subheader("Model selection")
    comparison = pd.DataFrame(metrics["comparison"]).rename(columns={"model": "Model", "cv_mae": "CV mean absolute error", "cv_std": "CV standard deviation"})
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("Lower cross-validation error is better. The test partition was reserved until model selection was complete. Preprocessing is fitted separately inside each training fold.")
    ridge_cv = next(row["cv_mae"] for row in metrics["comparison"] if row["model"] == "Ridge regression")
    st.write(f"**Do the additional inputs help?** On the same fitting records and folds, six-input ridge CV MAE was "
             f"{metrics['core_ridge_cv_mae']:.3f}; eleven-input ridge CV MAE was {ridge_cv:.3f}. Lower is better.")
    results = model_results.copy()
    st.subheader("Actual and predicted test grades")
    st.scatter_chart(results, x="actual_grade", y="predicted_grade", x_label="Actual final grade", y_label="Predicted final grade")
    st.download_button("Download held-out predictions", results.to_csv(index=False), "test_predictions.csv", "text/csv")
    st.subheader("Which inputs does the model rely on?")
    importance = pd.DataFrame(metrics["feature_importance"])
    importance["feature"] = importance["feature"].map(FEATURE_LABELS)
    importance = importance.rename(columns={"feature": "Input", "mae_increase": "MAE increase when shuffled", "std": "Repeat standard deviation"})
    st.dataframe(importance.sort_values("MAE increase when shuffled", ascending=False), hide_index=True, width="stretch")
    st.caption("Permutation importance on held-out records: higher positive values mean shuffling the input worsened predictions. Negative values can occur. Correlated inputs share information. This is global model reliance, not a causal effect or an explanation of one student's grade.")

with data_tab:
    st.subheader("The training dataset")
    try:
        data = (load_data(ROOT / "data" / f"student-{subject}.csv") if history is None
                else history[FEATURES].assign(G3=history.actual_grade.astype(int)))
    except FileNotFoundError:
        st.warning("The reference dataset is unavailable. Please contact the app administrator.")
    else:
        st.write(f"{len(data)} {SUBJECTS[subject].lower()} records · {len(FEATURES)} predictors · final grade G3 as the target")
        if history is None:
            st.caption("The project includes 395 mathematics and 649 Portuguese-language subject records. Some students occur in both files; 1,044 subject records does not mean 1,044 unique students. Models and evaluation splits stay separate by subject.")
        else:
            st.caption(f"Source: your uploaded {SUBJECTS[subject]} history. No records from another subject were added. Split by student record into fitting, calibration and test partitions; classes and time periods are not held out as groups.")
        quality = st.columns(3)
        quality[0].metric("Missing predictor/target values", int(data.isna().sum().sum()))
        quality[1].metric("Selected input fields", len(FEATURES))
        if history is None:
            quality[2].metric("Source schools", 2)
        else:
            quality[2].metric("Source classes / groups", history.class_name.nunique())
        distribution = data["G3"].value_counts().reindex(range(21), fill_value=0).rename_axis("Final grade").to_frame("Students")
        st.bar_chart(distribution)
        st.dataframe(data, hide_index=True, width="stretch")
    if history is None:
        st.caption("Source: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. DOI: 10.24432/C5TG7T. CC BY 4.0.")
    st.write("This model estimates final performance after period two using G1, G2 and learning context. It is not an early-enrollment prediction system. Verify that every input was recorded by the end of period two, before the final outcome was known.")
    st.subheader("What would make the dataset stronger?")
    st.write("Collect recent, consented records from your target schools across several cohorts, using consistent subject and grading definitions. Timestamp attendance, assignment scores, and engagement before the prediction date, then record the final result later. Evaluate on later cohorts and schools excluded from training.")
    st.caption("The bundled source has no reliable prediction-date snapshot for absences, so attendance is not added as an input. More columns or another subject do not by themselves demonstrate better generalisation.")
