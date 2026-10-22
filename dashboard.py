"""Educator dashboard: cohort charts, review tables and downloadable summaries."""

import hashlib

import altair as alt
import pandas as pd
import streamlit as st

from cohort import (BANDS, STATUSES, analyse_cohort, class_summary, csv_template, example_cohort,
                    export_csv, filter_cohort, grade_distribution, read_cohort_csv)
from model import FEATURE_LABELS, FEATURES
from evaluation_ui import render_alert_evaluation

COLORS = ["#913b79", "#b24431", "#a16b00", "#087f8c"]


def render_dashboard(model, metrics, subject, *, history=None):
    st.subheader("Class performance overview")
    st.caption("For teachers, lecturers, principals, and heads of department: review progress, compare groups, and identify students for a follow-up conversation.")
    features = metrics["features"]
    example_label = "Historical example" if history is None else "Held-out history"
    source = st.radio("Class data", [example_label, "Upload class CSV"], horizontal=True,
                      key="cohort_source" if history is None else f"cohort_source_{subject}")
    if source == "Upload class CSV":
        st.download_button("Download class CSV template", csv_template(features), "class_template.csv", "text/csv")
        with st.expander("How to prepare your class file"):
            st.write("Use one row per student for the selected subject, with a unique student code and class or group label. Grades use the 0–20 scale. Validate the model on later cohorts before relying on estimates for your institution.")
            st.write(f"Required prediction columns for {metrics['feature_set']}: " + ", ".join(features))
            st.dataframe(pd.DataFrame({"Column": features, "Meaning": [FEATURE_LABELS[name] for name in features]}),
                         hide_index=True, width="stretch")
            st.caption("Include student_id and class_name as metadata. G1 is optional when not required by the predictor; "
                       "a missing G1 leaves progress unavailable. Optional G3 is a recorded outcome, never an input. "
                       "G1/G2/G3: whole numbers 0–20; studytime/traveltime: codes 1–4; failures: 0–3; support fields: yes/no. "
                       "The template row is illustrative; replace it with actual records. No required field is filled automatically.")
            st.caption("Class files are processed in your session and are not used to retrain the model. Limit: 5,000 rows / 2 MB. Only upload records you are authorised to use; student codes are enough.")
        upload = st.file_uploader("Upload a class file for this subject", type="csv", key=f"cohort_file_{subject}")
        if upload is None:
            st.info("Upload a class CSV to see your class dashboard.")
            return
        try:
            cohort = read_cohort_csv(upload.getvalue(), features)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.caption(f"UPLOADED CLASS DATA · {len(cohort):,} records · selected subject: {metrics['subject_label']}")
        if history is not None:
            overlap = cohort.student_id.isin(history.student_id).sum()
            if overlap:
                st.warning(f"{overlap} student codes also occur in the training history. These are not an independent evaluation cohort; use new students to assess generalisation.")
    else:
        if history is None:
            cohort = example_cohort(subject, metrics)
            st.caption(f"HISTORICAL PUBLIC EXAMPLE · {len(cohort)} held-out subject records · groups are the source schools, not classes at your institution.")
        else:
            cohort = history.loc[metrics["split_rows"]["test"]].copy().reset_index(drop=True)
            st.caption(f"HELD-OUT UPLOADED HISTORY · {len(cohort)} {metrics['subject_label']} records · these students were excluded from fitting and calibration. Select Upload class CSV to analyse a new class.")

    controls = st.columns([2, 1])
    with controls[0]:
        options = sorted(cohort.class_name.unique().tolist())
        group_signature = hashlib.sha256("\0".join(options).encode("utf-8")).hexdigest()[:12]
        groups = st.multiselect("Classes / groups", options, default=options,
                                key=f"cohort_groups_{subject}_{source}_{group_signature}")
    with controls[1]:
        threshold = st.number_input("Review reference grade", min_value=0, max_value=20, value=10, key="cohort_threshold",
                                    help="A staff-selected threshold for review, not an official pass rule or a probability.")
    report = analyse_cohort(model, cohort, metrics, threshold)
    report["data_source"] = source
    report = filter_cohort(report, groups, STATUSES)
    if report.empty:
        st.info("Select at least one class or group to view the dashboard.")
        return
    cards = st.columns(4)
    cards[0].metric("Students in view", f"{len(report):,}")
    cards[1].metric("Second-period average", f"{report.G2.mean():.1f} / 20")
    cards[2].metric("Estimated final average", f"{report.predicted_grade.mean():.1f} / 20")
    cards[3].metric("Estimates below reference", int((report.margin_to_threshold < 0).sum()))
    st.caption(f"{int((report.observed_change_G1_to_G2 < 0).sum())} observed grade declines among "
               f"{int(report.observed_change_G1_to_G2.notna().sum())} students with both G1 and G2. "
               "G2 − G1 describes recorded progress, not an additional independent predictor. Missing progress is not zero. "
               "Review categories are rule-based prompts for staff judgement.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Grade distribution")
        chart = alt.Chart(grade_distribution(report)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Grade band:N", sort=BANDS, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("Students:Q", axis=alt.Axis(tickMinStep=1)), xOffset="Series:N",
            color=alt.Color("Series:N", scale=alt.Scale(range=["#183e50", "#38a4a6"]), legend=alt.Legend(orient="bottom")),
            tooltip=["Grade band:N", "Series:N", "Students:Q"]).properties(height=270)
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### Review overview")
        counts = report.review_status.value_counts().reindex(STATUSES, fill_value=0).rename_axis("Status").reset_index(name="Students")
        chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Status:N", sort=STATUSES, title=None), x=alt.X("Students:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Status:N", scale=alt.Scale(domain=STATUSES, range=COLORS), legend=None),
            tooltip=["Status:N", "Students:Q"]).properties(height=270)
        st.altair_chart(chart, width="stretch")
        st.caption("Near reference: the point estimate is at/above the reference, but its prediction range crosses it. Outside grade scale: an estimate below 0 or above 20 needs review.")

    summary = class_summary(report)
    st.markdown("#### Class / group comparison")
    chart_groups = summary.sort_values("students", ascending=False).head(20)
    if len(summary) > 20:
        st.caption("The chart shows the 20 largest groups. All selected groups are included in the table and export.")
    averages = chart_groups.melt(id_vars="class_name", value_vars=["second_period_mean", "estimated_final_mean"], var_name="Series", value_name="Grade")
    averages["Series"] = averages.Series.map({"second_period_mean": "Second period (observed)", "estimated_final_mean": "Final grade (estimated)"})
    chart = alt.Chart(averages).mark_bar().encode(x=alt.X("class_name:N", title="Class / group", axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("Grade:Q", title="Mean grade / 20"), xOffset="Series:N",
        color=alt.Color("Series:N", scale=alt.Scale(range=["#183e50", "#38a4a6"]), legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("class_name:N", title="Group"), "Series:N", alt.Tooltip("Grade:Q", format=".2f")]).properties(height=260)
    st.altair_chart(chart, width="stretch")
    st.dataframe(summary.rename(columns={"class_name": "Class / group", "students": "Students", "first_period_mean": "G1 average",
        "second_period_mean": "G2 average", "estimated_final_mean": "Estimated final average", "below_reference": "Below reference",
        "observed_declines": "Observed grade declines", "progress_available": "Records with G1 and G2"}).round(2), hide_index=True, width="stretch")

    st.markdown("#### Student review table")
    filters = st.columns(2)
    with filters[0]:
        statuses = st.multiselect("Review status", STATUSES, default=STATUSES, key="cohort_statuses")
    with filters[1]:
        query = st.text_input("Find student code", placeholder="e.g. S001", key="cohort_search")
    visible = filter_cohort(report, groups, statuses, query).sort_values(["predicted_grade", "student_id"])
    if visible.empty:
        st.info("No student records match these table filters. The charts above still describe the selected groups.")
    else:
        columns = ["student_id", "class_name", "G1", "G2", "predicted_grade", "observed_change_G1_to_G2",
                   "interval_lower", "interval_upper", "review_status"]
        if "actual_grade" in visible:
            columns.append("actual_grade")
        st.dataframe(visible[columns].rename(columns={"student_id": "Student code", "class_name": "Class / group",
            "predicted_grade": "Estimated final grade", "observed_change_G1_to_G2": "Observed change",
            "interval_lower": "Range lower", "interval_upper": "Range upper", "review_status": "Review status",
            "actual_grade": "Recorded final grade"}).round(2), hide_index=True, width="stretch", height=350)
        st.caption(f"Showing {len(visible):,} of {len(report):,} records in selected groups, ordered by estimated grade. Recorded final grades, where supplied, are never prediction inputs.")
    downloads = st.columns(2)
    with downloads[0]:
        st.download_button("Download filtered student report", export_csv(visible), f"{subject}_student_review.csv", "text/csv", disabled=visible.empty)
    with downloads[1]:
        export = summary.assign(subject=metrics["subject_label"], reference_grade=threshold,
                                data_source=source, model_trained_at=metrics["trained_at"], bundle_id=metrics["bundle_id"],
                                feature_set=metrics["feature_set"], grade_scale="0-20", prediction_timing=metrics["prediction_timing"])
        st.download_button("Download class summary", export_csv(export), f"{subject}_class_summary.csv", "text/csv")
    st.caption(f"Prediction ranges target 80% coverage on comparable records; measured held-out test coverage was {metrics['interval']['test_coverage']:.1%}. This does not establish coverage for future classes or other institutions.")
    st.caption(f"Fixed interval width: {metrics['interval']['average_width']:.2f} points for every student using this predictor. "
               "This is not personalised uncertainty or a pass probability.")
    if "actual_grade" in report:
        population = ("Uploaded class outcomes in selected groups; descriptive class evaluation, separate from reserved model-test metrics."
                      if source == "Upload class CSV" else "Historical held-out outcomes in selected groups; not live class records.")
        render_alert_evaluation(report.actual_grade, report.predicted_grade, threshold, source=population,
                                key=f"cohort_alerts_{subject}", bundle_id=metrics["bundle_id"])
