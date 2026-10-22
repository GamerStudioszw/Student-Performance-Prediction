"""Model evaluation, explicitly separate from a teacher's uploaded class outcomes."""
import json
import pandas as pd
import streamlit as st

from cohort import export_csv
from evaluation_ui import render_alert_evaluation
from model import FEATURE_LABELS


def render_model_quality(metrics, results):
    st.subheader("Performance on reserved test records")
    st.caption(metrics["evaluation_limitations"])
    st.write(f"**Selected predictor:** {metrics['selected_model']} · {metrics['predictor_kind']}")
    cards = st.columns(4)
    for column, name, value in zip(cards, ["Mean absolute error ↓", "Root mean squared error ↓", "R² ↑", "Training-mean baseline MAE ↓"],
                                   [metrics["mae"], metrics["rmse"], metrics["r2"], metrics["baseline_mae"]]):
        column.metric(name, f"{value:.3f}")
    st.caption("MAE is average absolute error in grade points. RMSE gives more weight to large errors. "
               "R² compares squared errors with test-grade variation; it can be negative and is not an accuracy percentage.")
    st.caption(f"{metrics['train_rows']} fitting records · {metrics['calibration_rows']} calibration records · "
               f"{metrics['test_rows']} test records · seed 42 · grade scale 0–20. No test records fit this predictor.")
    st.dataframe(pd.DataFrame([
        {"Predictor": metrics["selected_model"], "MAE": metrics["mae"], "RMSE": metrics["rmse"], "R²": metrics["r2"]},
        {"Predictor": "G3 = G2 baseline", **{k.upper() if k != "r2" else "R²": v for k, v in metrics["g2_test_metrics"].items()}}
    ]), hide_index=True, width="stretch")
    st.caption("Both rows use exactly the same reserved test records. The test comparison did not select the active predictor.")
    if metrics["selected_candidate"] == "g2":
        st.info("The G2 persistence baseline won validation selection. Its forecast is exactly the recorded G2; there are no learned prediction weights.")
    elif metrics["mae"] > metrics["g2_baseline_mae"]:
        st.info("G2 persistence had lower test MAE. The predictor remains selected by validation, not by these test results.")
    st.subheader("Prediction interval evaluation")
    interval = metrics["interval"]
    columns = st.columns(3)
    columns[0].metric("Nominal interval coverage", f"{interval['nominal_coverage']:.0%}")
    columns[1].metric("Observed test coverage", f"{interval['test_coverage']:.1%}")
    columns[2].metric("Average interval width", f"{interval['average_width']:.2f} points")
    st.write(f"Calibration sample: {interval['calibration_rows']}; test sample: {interval['test_rows']}; "
             f"test outcomes covered: {interval['test_covered']}/{interval['test_rows']}.")
    st.caption(f"A single absolute-error threshold gives the same width for every student using this predictor. "
               f"The calibration order statistic is rank {interval['rank']}. Widths are not personalised uncertainty. "
               "Realised test coverage may differ from the nominal target; it does not give a student's chance of passing. "
               "Coverage settings are fixed before evaluating test outcomes.")
    st.caption(metrics["bounding_policy"])
    if interval["test_coverage"] < interval["nominal_coverage"]:
        st.warning("Observed coverage is below the nominal target. The interval has not been widened to fit test outcomes.")
    st.metric("Predictions within ±2 points", f"{metrics['within_two_points']:.1%}")
    st.subheader("Validation model comparison")
    comparison = pd.DataFrame(metrics["comparison"])
    comparison["features"] = comparison.features.map(", ".join)
    st.dataframe(comparison[["model", "feature_set", "features", "cv_mae", "cv_std"]].rename(columns={
        "model": "Candidate", "feature_set": "Feature schema", "features": "Predictor inputs", "cv_mae": "CV mean MAE",
        "cv_std": "CV MAE standard deviation"}), hide_index=True, width="stretch")
    st.caption(metrics["selection_rule"])
    st.caption("Identical five folds and MAE scoring for all eligible candidates. Scaling is fitted inside each fold. "
               "Feature sets and modest algorithm settings were specified before comparison. CV variation is not a confidence interval.")
    st.subheader("Recorded G3 and predicted G3")
    st.scatter_chart(results, x="actual_grade", y="predicted_grade", x_label="Recorded final grade (G3)", y_label="Predicted final grade (G3)")
    st.download_button("Download held-out predictions", export_csv(results), f"{metrics['subject']}_test_predictions.csv", "text/csv")
    st.download_button("Download model evaluation and provenance", json.dumps(metrics, indent=2),
                       f"{metrics['subject']}_evaluation.json", "application/json")
    st.subheader("Which inputs does the predictor rely on?")
    importance = pd.DataFrame(metrics["feature_importance"])
    importance["feature"] = importance.feature.map(FEATURE_LABELS)
    st.dataframe(importance.rename(columns={"feature": "Input", "mae_increase": "MAE increase when shuffled",
                                            "std": "Repeat standard deviation"}), hide_index=True, width="stretch")
    st.caption("Shuffling inputs on test records measures model reliance, not causal effects, guaranteed interventions or "
               "an individual explanation. No field is removed because of this table. G2 − G1 is a derived progress measure, not independent information.")
    threshold = st.number_input("Test alert reference grade", 0, 20, 10, key="test_alert_threshold")
    render_alert_evaluation(results.actual_grade, results.predicted_grade, threshold,
                            source="Reserved model test records only; not metrics from the uploaded class.",
                            key=f"model_alerts_{metrics['subject']}", bundle_id=metrics["bundle_id"])
