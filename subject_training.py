"""Subject-specific training from validated history, kept only in the staff session."""

import hashlib

import pandas as pd

from cohort import export_csv, parse_csv, validate_cohort
from validation import safe_text
from model import FEATURES, FEATURE_SETS, PUBLIC_SUBJECTS, SUBJECTS, train_and_evaluate
from model_service import finalize_bundle, session_bundle_compatible


def training_template(training_feature_set="expanded_v1"):
    # Header only: do not supply invented historical outcomes as training data.
    return export_csv(pd.DataFrame(columns=["subject", "student_id", "class_name", *FEATURE_SETS[training_feature_set], "G3"]))


def read_training_csv(content, subject, training_feature_set="expanded_v1"):
    if not isinstance(subject, str) or subject not in SUBJECTS or subject in PUBLIC_SUBJECTS:
        raise ValueError("Select one of the subjects that requires uploaded training records.")
    if not isinstance(training_feature_set, str) or training_feature_set not in FEATURE_SETS:
        raise ValueError("Unknown training feature set.")
    raw = parse_csv(content)
    cohort = validate_cohort(raw, FEATURE_SETS[training_feature_set])
    if "subject" not in raw:
        raise ValueError(f"Add a subject column containing {subject} or {SUBJECTS[subject]} on every row.")
    labels = raw.subject.str.strip().str.casefold()
    if not labels.isin([subject.casefold(), SUBJECTS[subject].casefold()]).all():
        raise ValueError(f"Every row must be for {SUBJECTS[subject]} ({subject}). Use separate files for other subjects.")
    if "actual_grade" not in cohort or cohort.actual_grade.isna().any():
        raise ValueError("Training history needs a known final grade (G3) on every row.")
    if len(cohort) < 50:
        raise ValueError("Provide at least 50 distinct students with completed final grades for this subject.")
    if cohort.actual_grade.nunique() < 2:
        raise ValueError("Training history must include at least two different final grades.")
    return cohort


def train_subject(content, subject, training_feature_set="expanded_v1", source="Staff-provided historical records"):
    source = safe_text(source, field="Training data source", maximum=200)
    history = read_training_csv(content, subject, training_feature_set)
    training = history[FEATURE_SETS[training_feature_set]].assign(G3=history.actual_grade.astype(int))
    bundle = finalize_bundle(*train_and_evaluate(training, training_feature_set=training_feature_set),
                             subject=subject, fingerprint=hashlib.sha256(content).hexdigest(), source=source)
    return {"bundle": bundle, "history": history}


def render_subject_training(subject):
    """Never put private training records or fitted models in Streamlit's shared cache."""
    import streamlit as st

    key = f"trained_subject_{subject}"
    saved = st.session_state.get(key)
    if saved is not None and not session_bundle_compatible(saved["bundle"]):
        st.warning("The previous session model uses an incompatible version. Re-upload or retrain the selected history before predicting.")
        saved = None
    revision_key = f"training_revision_{subject}"
    with st.expander(f"{SUBJECTS[subject]} training records", expanded=saved is None):
        st.write(f"Build a separate {SUBJECTS[subject]} model from completed student records for this subject.")
        st.caption("Use 50–5,000 distinct students, up to 2 MB. Fifty is the technical minimum, not evidence of a reliable model. More representative records and evaluation on later cohorts are needed.")
        mode = st.selectbox("Training input set", ["grades_v1", "expanded_v1", "persistence_v1"],
                            format_func=lambda value: {"grades_v1": "G1 and G2: compare baselines, ridge and forest",
                                                       "expanded_v1": "Expanded: compare all six candidates",
                                                       "persistence_v1": "G2 only: compare persistence and mean baselines"}[value],
                            key=f"training_mode_{subject}")
        if mode not in FEATURE_SETS:
            st.error("Select a supported training input set.")
            return None
        source = st.text_input("Training data source", value="Staff-provided historical records", max_chars=200,
                               key=f"training_source_{subject}")
        st.download_button("Download training CSV template", training_template(mode),
                           f"{subject}_training_template.csv", "text/csv", key=f"training_template_{subject}")
        st.write(f"Fill `subject` with **{subject}** or **{SUBJECTS[subject]}** on every row. Include `student_id`, `class_name`, "
                 f"{', '.join(FEATURE_SETS[mode])}, and completed final grades in `G3`.")
        st.caption("Only candidates supported by the selected input set are compared, on identical rows and folds. "
                   "No missing fields are invented or imputed. The winning predictor may require fewer inputs for new students.")
        st.caption("G1, G2 and G3: whole numbers 0–20. Use a consistent grading scale; do not enter marks out of 100 directly. Use one record per student and subject; repeated student codes are rejected.")
        st.dataframe(pd.DataFrame({
            "Input": ["studytime", "failures", "traveltime", "schoolsup, famsup, paid, activities, internet, higher"],
            "Values": ["1: <2h; 2: 2–5h; 3: 5–10h; 4: >10h per week", "0–3 previous failures",
                       "1: <15m; 2: 15–30m; 3: 30–60m; 4: >60m", "yes or no"]
        }), hide_index=True, width="stretch")
        st.caption("Use inputs known by the end of period two. Upload only records you are authorised to use, with student codes instead of names. Training data and models stay in this signed-in session; sign-out, session loss or a server restart requires re-uploading. They are not saved to Git or shared with other sessions.")
        upload = st.file_uploader(f"Upload completed {SUBJECTS[subject]} records", type="csv",
                                  key=f"training_file_{subject}_{st.session_state.get(revision_key, 0)}")
        if saved and st.button("Clear this subject model", key=f"clear_model_{subject}"):
            del st.session_state[key]
            st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
            st.session_state.pop("grade_report", None)
            st.rerun()
        if upload is not None:
            content = upload.getvalue()
            try:
                history = read_training_csv(content, subject, mode)
            except ValueError as exc:
                st.error(str(exc))
                return None
            fingerprint = hashlib.sha256(content).hexdigest()
            current = (saved is not None and saved["bundle"][1]["dataset_sha256"] == fingerprint
                       and saved["bundle"][1]["training_feature_set"] == mode
                       and saved["bundle"][1]["source"] == source)
            if not current:
                st.write(f"{len(history):,} valid {SUBJECTS[subject]} records ready for training.")
                if st.button(f"Train {SUBJECTS[subject]} model", type="primary", key=f"train_{subject}"):
                    try:
                        with st.spinner(f"Training and evaluating {SUBJECTS[subject]}…"):
                            saved = train_subject(content, subject, mode, source)
                    except ValueError as exc:
                        st.error(str(exc))
                        return None
                    st.session_state[key] = saved
                    st.success("Model ready. Review its held-out quality report before using estimates.")
                else:
                    st.info("Train the selected file to open this subject's analytics.")
                    return None
        if saved is None:
            st.info(f"Upload completed {SUBJECTS[subject]} records to unlock class charts, student predictions and model quality reports. No reference dataset is bundled for this subject.")
        else:
            st.caption(f"Active model: {saved['bundle'][1]['rows']:,} {SUBJECTS[subject]} records. Training, calibration and test partitions remain separate.")
    return saved
