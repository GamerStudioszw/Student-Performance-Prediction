"""Local Streamlit interface for the student performance ML project."""

import json

import joblib
import pandas as pd
import streamlit as st

from model import ROOT, load_data, predict_grade
from auth_ui import require_login

st.set_page_config(page_title="Student Performance | ML", page_icon="🎓", layout="wide")
require_login()
st.caption("EDUCATION ANALYTICS · MACHINE LEARNING PROJECT")
st.title("Student Performance Prediction")
st.write("Estimate a student's final mathematics grade using their first two period grades and learning context.")

artifact_dir = ROOT / "artifacts"
required = [artifact_dir / name for name in ("model.joblib", "metrics.json", "test_predictions.csv")]
if not all(path.exists() for path in required):
    st.info("Train the model first, then refresh this page.")
    st.code("python train.py", language="bash")
    st.stop()


@st.cache_resource
def load_model(modified_ns):
    return joblib.load(artifact_dir / "model.joblib")


model = load_model(required[0].stat().st_mtime_ns)
metrics = json.loads(required[1].read_text(encoding="utf-8"))
with st.sidebar:
    st.header("About this project")
    st.write("**Subject:** Mathematics\n\n**Target:** Final grade, G3 (0–20)\n\n**Prediction point:** After period two")
    st.caption("An educational demonstration based on historical records from two Portuguese schools. Local validation is needed before using it with another school.")
    st.link_button("View the UCI dataset", "https://archive.ics.uci.edu/dataset/320/student+performance")
    st.divider()
    st.write(f"**Selected model:** {metrics['selected_model']}")
    st.caption("Selected by five-fold cross-validation on the training partition.")

predict_tab, evaluation_tab, data_tab = st.tabs(["Predict a grade", "Model evaluation", "Explore the data"])

with predict_tab:
    st.subheader("Student learning profile")
    st.caption("Use values available after period two. This is a grade estimate, not a probability of passing.")
    with st.form("prediction"):
        left, right = st.columns(2)
        with left:
            g1 = st.number_input("First period grade (G1)", 0, 20, 12)
            g2 = st.number_input("Second period grade (G2)", 0, 20, 12)
            studytime = st.selectbox(
                "Weekly study time", [1, 2, 3, 4], index=1,
                format_func=lambda value: {1: "Under 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "Over 10 hours"}[value],
            )
        with right:
            failures = st.selectbox("Previous class failures", [0, 1, 2, 3],
                                    help="These are the values observed in the mathematics training dataset.")
            schoolsup = st.selectbox("Extra educational support at school", ["no", "yes"])
            famsup = st.selectbox("Family educational support", ["yes", "no"])
        submitted = st.form_submit_button("Predict final grade", type="primary", width="stretch")
    if submitted:
        student = pd.DataFrame([{"G1": g1, "G2": g2, "studytime": studytime, "failures": failures,
                                 "schoolsup": schoolsup, "famsup": famsup}])
        grade = float(predict_grade(model, student)[0])
        st.metric("Estimated final grade", f"{grade:.1f} / 20")
        st.caption(f"On held-out records, this model's mean absolute error was {metrics['mae']:.2f} grade points. This is an average error, not an individual confidence interval.")
        if grade < 0 or grade > 20:
            st.warning("The model estimate is outside the valid 0–20 grade scale. Treat this input as an extrapolation.")
        st.info("Use the estimate to support a conversation about learning needs. Changing an input does not establish that it causes a grade change.")

with evaluation_tab:
    st.subheader("Performance on unseen records")
    cards = st.columns(4)
    cards[0].metric("Mean absolute error ↓", f"{metrics['mae']:.2f}")
    cards[1].metric("Root mean squared error ↓", f"{metrics['rmse']:.2f}")
    cards[2].metric("R² ↑", f"{metrics['r2']:.3f}")
    cards[3].metric("Baseline MAE ↓", f"{metrics['baseline_mae']:.2f}")
    st.caption(f"{metrics['train_rows']} training records · {metrics['test_rows']} held-out test records · fixed random seed 42. Errors are measured on the 0–20 scale.")
    st.write("The mean baseline predicts the training-set average for every student. R² compares prediction errors with variation in the test grades; it can be negative.")
    st.subheader("Model selection")
    comparison = pd.DataFrame(metrics["comparison"]).rename(columns={"model": "Model", "cv_mae": "CV mean absolute error", "cv_std": "CV standard deviation"})
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("Lower cross-validation error is better. The test partition was reserved until model selection was complete. Preprocessing is fitted separately inside each training fold.")
    results = pd.read_csv(required[2])
    st.subheader("Actual and predicted test grades")
    st.scatter_chart(results, x="actual_grade", y="predicted_grade", x_label="Actual final grade", y_label="Predicted final grade")
    st.download_button("Download held-out predictions", results.to_csv(index=False), "test_predictions.csv", "text/csv")

with data_tab:
    st.subheader("The training dataset")
    try:
        data = load_data()
    except FileNotFoundError:
        st.warning("Dataset not found. Run python download_data.py to restore it.")
    else:
        st.write(f"{len(data)} mathematics records · six predictors · final grade G3 as the target")
        distribution = data["G3"].value_counts().reindex(range(21), fill_value=0).rename_axis("Final grade").to_frame("Students")
        st.bar_chart(distribution)
    st.dataframe(data, hide_index=True, width="stretch")
    st.caption("Source: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. DOI: 10.24432/C5TG7T. CC BY 4.0.")
    st.write("G1 and G2 are earlier grades and are strongly related to G3. This model therefore estimates performance after period two. It is not an early-enrollment prediction system. Support variables were collected in questionnaires; their exact measurement timing should be verified for prospective use.")
