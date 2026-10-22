"""Shared wording for alert counts; caller identifies the population explicitly."""
import pandas as pd
import streamlit as st

from cohort import export_csv
from model import alert_evaluation


def render_alert_evaluation(actual, predicted, threshold, *, source, key, bundle_id):
    st.markdown("#### Below-reference alert evaluation")
    st.caption(source)
    result = alert_evaluation(actual, predicted, threshold)
    st.caption(f"Positive outcome: recorded G3 < {threshold}/20. Alert: predicted G3 < {threshold}/20. "
               "Equality is not below reference. These rules are separate from interval-based review categories.")
    cards = st.columns(3)
    for column, name in zip(cards, ["precision", "recall", "f1"]):
        value = result[name]
        column.metric(f"Alert {name}", "Unavailable" if value is None else f"{value:.1%}")
    st.write(f"Evaluated: {result['evaluated']} · Actual below reference: {result['actual_below']} · "
             f"Alerts: {result['alerts']} · Missing outcomes/predictions excluded: {result['excluded_missing']}")
    st.dataframe(pd.DataFrame([[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
                             index=["Recorded G3 at/above reference", "Recorded G3 below reference"],
                             columns=["No alert", "Alert"]), width="stretch")
    st.caption("Precision: fraction of alerts with recorded outcomes below reference. Recall: fraction of recorded "
               "below-reference outcomes alerted. F1 combines precision and recall. Unavailable means the metric's "
               "denominator is zero; it is not reported as perfect performance or as zero.")
    report = pd.DataFrame([{**result, "source": source, "bundle_id": bundle_id}])
    st.download_button("Download alert evaluation", export_csv(report), f"{key}_alerts.csv", "text/csv", key=key)
