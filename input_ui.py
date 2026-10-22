"""Prediction fields follow the active feature schema; optional progress stays missing."""
import pandas as pd
import streamlit as st

from model import FEATURE_LABELS, CATEGORICAL


def profile_fields(features):
    values = {}
    left, right = st.columns(2)
    with left:
        g1 = st.number_input("First period grade (G1)", 0, 20, 12 if "G1" in features else None, key="profile_G1",
                             help="Required by this predictor." if "G1" in features else "Optional: used only to show G2 − G1 progress. Leave blank if unknown.")
        if g1 is not None:
            values["G1"] = g1
        values["G2"] = st.number_input("Second period grade (G2)", 0, 20, 12, key="profile_G2",
                                       help="Required recorded grade after period two; never replaced with a default in uploaded files.")
        if "studytime" in features:
            values["studytime"] = st.selectbox("Weekly study time", [1, 2, 3, 4], index=1,
                                               format_func=lambda v: {1: "Under 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "Over 10 hours"}[v])
        if "traveltime" in features:
            values["traveltime"] = st.selectbox("Travel time to school", [1, 2, 3, 4],
                                                format_func=lambda v: {1: "Under 15 minutes", 2: "15–30 minutes", 3: "30–60 minutes", 4: "Over 60 minutes"}[v])
    with right:
        if "failures" in features:
            values["failures"] = st.selectbox("Previous class failures", [0, 1, 2, 3])
        for name in CATEGORICAL:
            if name in features:
                values[name] = st.selectbox(FEATURE_LABELS[name], ["no", "yes"], key=f"profile_{name}")
    return pd.DataFrame([values])
