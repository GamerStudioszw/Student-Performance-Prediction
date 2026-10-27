# Student Performance Prediction using Machine Learning

A staff analytics app for teachers, lecturers, principals, headmasters, and heads of department. It provides class dashboards, grade charts, student review tables, group comparisons, CSV imports, and downloadable reports. Its demonstration models estimate final mathematics or Portuguese-language grades (G3, 0–20) from earlier grades and learning context.

The subject selector also includes **English, Physics, Chemistry, Biology, Geography, History, and Computer Science**. These seven subjects require your own completed subject records before predictions are available. They do not reuse or relabel a mathematics model.

## Academic milestone submission

The submission for **Bindura University of Science Education**, **Network Engineering**, module **Artificial Intelligence and Intelligent System**, supervised by **Mr C. Zano**, uses one chapter per milestone:

1. Proposal: problem, objectives, methodology, literature and proposed team roles.
2. Design: architecture, data pipeline, model selection and ethics.
3. Prototype: implementation, validation comparison and measured evaluation.
4. Integration: staff app, notebooks, Drive support, tests and operation.
5. Final: conclusions, handover and assessment readiness.

The [Word technical report](documents/milestones/StudentInsight_Five_Chapter_Report.docx), [12-slide PowerPoint presentation](documents/milestones/StudentInsight_Milestone_Presentation.pptx), and [15-minute demonstration guide](documents/milestones/StudentInsight_Demo_Guide.docx) document code baseline `88b1003` and the full **56-test passing suite** checked on 26 October 2026. The report supersedes the original proposal's scope for this submission; the earlier proposal is preserved.

The [rendered report PDF](documents/milestones/StudentInsight_Five_Chapter_Report.pdf) is **15 pages including cover and references**, within the 20-page limit. All report and guide pages and all 12 slides were visually reviewed. PowerPoint package, layout, font, native-table and embedded-chart checks passed; these checks do not certify rendering in every Office version.

Complete the group number, member names, student numbers and role allocations before submission. Rehearsal, the assessed live demonstration, supervisor review, and real Google authorization remain user activities. The materials distinguish implementation evidence from independent institutional validation and do not claim those activities have occurred.

## Google Colab and Jupyter Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamerStudioszw/Student-Performance-Prediction/blob/main/notebooks/StudentInsight.ipynb)

Open [`notebooks/StudentInsight.ipynb`](notebooks/StudentInsight.ipynb) to explore the same training, predictions and evaluation used by the app. It trains Mathematics and Portuguese, shows cross-validation comparisons and grade charts, evaluates intervals and below-reference alerts, and exports reports. CPU is sufficient. Google Drive access is optional and uses Google's authorization prompt; no API key is required.

**Google Colab:** use the button above, connect to a runtime, and run the cells in order. The first cell clones this GitHub repository and installs its requirements. The current `main` branch must contain the notebook and repaired model code. An existing runtime checkout is reused without overwriting edits; start a fresh runtime for the latest version. If dependency installation asks for a restart, restart the runtime and run the cells again. Set `DOWNLOAD_IN_COLAB = True` in the export cell to download the experiment ZIP. Colab's runtime files are temporary, and a notebook copy in Drive does not preserve them automatically. See [Google's Colab documentation](https://research.google.com/colaboratory/faq.html).

**Connect Google Drive in Colab:** in notebook section **1a**, change `USE_GOOGLE_DRIVE = False` to `True`, run the cell and authorize your Google account. The notebook creates `My Drive/StudentInsight/datasets` and `My Drive/StudentInsight/reports`. In section **5a**, optionally set `DRIVE_CLASS_CSV` to the name of a class CSV in the datasets folder. The existing cohort validator checks the active predictor's required inputs; this predicts a class without retraining or mixing class results into the historical test evaluation. Section **6a** copies the experiment ZIP, including any class predictions, to the reports folder and verifies its contents. Each run has a unique name; existing reports are never overwritten. Set `SAVE_REPORTS_TO_DRIVE = False` to keep reports local.

Drive mounting makes your Drive accessible to code in the notebook, as described in [Google's FAQ](https://research.google.com/colaboratory/faq.html). This workflow accesses only the selected CSV and the StudentInsight folders. No account credentials are committed. Authorization must be completed in your Colab session. The Streamlit app and its accounts are not automatically connected to Drive. Local Jupyter can instead use an existing Google Drive for desktop folder through `LOCAL_DRIVE_FOLDER`; it does not use Colab's mount API.

**Local Jupyter Notebook:** from the project folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-notebook.ps1
```

For JupyterLab, append `-Lab`. The launcher installs notebook dependencies in `.venv`, registers **StudentInsight (.venv)** inside that environment, and opens the notebook on localhost. Select that kernel in Jupyter or select `.venv\Scripts\python.exe` in the VS Code notebook kernel picker. This avoids installing packages into one Python environment and running another.

Manual setup, following the [Jupyter installation guide](https://jupyter.org/install):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name studentinsight --display-name "StudentInsight (.venv)"
.\.venv\Scripts\python.exe -m notebook notebooks/StudentInsight.ipynb
```

The notebook creates new experiment exports under `notebook-results/`, excluded from Git. Its default workflow leaves application artifact pointers unchanged. The optional `SAVE_APPLICATION_BUNDLE` cell explicitly saves a new local bundle through `model_service.py`; in Colab this affects only that runtime. Neither platform automatically synchronizes accounts, private student records or deployed models. Continue using Streamlit for the staff login/dashboard. Clear notebook outputs before committing notebooks containing private data; the supplied notebook is committed without outputs.

Notebook verification on 26 October 2026: every cell executed in the local project kernel; both subjects' predictions and export bundle identities checked; both chart HTML outputs generated; original evaluation files and active artifact pointers unchanged. Jupyter's local server and authenticated ZIP download also passed. The hosted Colab runtime has not been exercised from a Google account.

Drive integration verification: five file-operation tests passed, and the complete notebook executed both with Drive disabled and with a temporary connected folder. The latter checked class CSV validation, unchanged source data and correct predictions inside the saved ZIP. Temporary example rows were used only to test software behaviour. These tests do not exercise Google's authorization or remote synchronization. `notebook_storage.py` supplies the bounded CSV reader and verified, non-overwriting archive copy.

## Open the login workspace

The Streamlit app includes a responsive StudentInsight sign-in screen, staff registration, and a protected prediction workspace. With the existing project environment installed, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open **http://127.0.0.1:8502**. The app always starts with **Sign in**. Existing users enter their username and password. New staff choose **Sign up**, enter their name, a unique username, and a passphrase of at least 15 characters, then sign in. **Sign up** stays available after other accounts have been created. The registration screen includes an **Already have an account? Sign in** route. There are no default credentials. Use **Sign out** in the sidebar to end your session.

Accounts belong to the running installation's account database. A local account is not automatically available on a separately hosted copy, because account databases are excluded from Git. Existing single-owner databases are upgraded automatically in one transaction, preserving the original username, password hash, and active sessions. Staff can register additional accounts without replacing the original account.

Accounts persist in `.local/accounts.sqlite3`, which is excluded from Git. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes with 600,000 iterations, following the [OWASP password storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Session tokens are also hashed in the database and expire after eight hours. Five failed attempts pause sign-in for that username for one minute across browser sessions; other accounts can still sign in. A page refresh or a new browser tab may require signing in again because authentication is held in the Streamlit session. Tests use temporary databases and never create or change your real account.

Registration is open to anyone who can reach the app. All accounts have the same analytics features; uploads, privately trained models, and prediction results remain in their browser session and are cleared on sign-out. There is no email recovery, role management, or account approval workflow. The login protects `app.py`; the separate lightweight demo below remains unauthenticated. The source CSV and model artifacts are local project files, not encrypted by the login feature. Hosted installations need persistent storage for the account database; `STUDENT_AUTH_DB` can select its location.

## Security verification

See [SECURITY.md](SECURITY.md) for enforced input limits, authentication safeguards, artifact trust boundaries, and deployment limitations. Run `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`, then `.\.venv\Scripts\python.exe verify.py` for the complete test suite, dependency compatibility checks, and strict online vulnerability audit. Any failure returns a nonzero exit status; detailed evidence is saved under `.local/verification/`. GitHub runs the same verification on pushes to `main` and pull requests.

The CSV importer rejects duplicate headers and uneven rows instead of guessing their meaning. Uploads are limited to 2 MiB and 5,000 records. Account creation is limited to ten attempts per installation per minute; each account can retain up to ten active sessions. Registration stays available for additional staff.

## Staff dashboard

After signing in, staff land on **Class dashboard**. The four work areas are Class dashboard, Student analysis, Model quality, and Data guide. The subject selector is in the sidebar.

- Overview cards show the number of students in selected groups, their second-period average, estimated final average, and count of estimates below a staff-selected reference grade.
- Charts compare observed G2 and estimated final-grade distributions, review categories, and group averages. The group chart shows up to 20 groups; the table and exports include every selected group.
- The review table includes student codes, class/group, G1, G2, estimated final grade, observed progress, prediction-range bounds, review status, and recorded final grade when available. Filter by group, status, or student code.
- Download the filtered student report or a summary of all selected groups. Table filters do not change the overview charts; group filters do. Exported reports identify the subject, model timestamp, and whether they use example or uploaded data.

The default **Historical example** uses only the selected model's held-out test records (79 mathematics or 130 Portuguese-language records). Its group labels are the two source schools, not invented classes. Training and calibration records are excluded from this example dashboard. These public records are clearly labelled and do not represent the staff member's institution.

For your own cohort, choose **Upload class CSV** and download the template. Supply one row per student for the selected subject, using `student_id`, `class_name`, and only the active predictor's required input columns, shown in the interface and template. For the current bundled predictors this is just `G2`. Optional `G1` enables recorded progress; blank G1 leaves progress unavailable. Use anonymous student codes where possible. The import accepts UTF-8 comma- or semicolon-delimited CSV, up to 5,000 records and 2 MB. It validates numeric ranges, yes/no fields, duplicate student codes, and required columns. `G3` may optionally contain a known final grade or be blank; it is displayed but never passed to the model. Extra fields such as student names are ignored. Invalid files show an actionable error and do not leave a stale dashboard visible.

Uploaded class records are processed in the signed-in session and are not stored in the account database, shared in the global model cache, or used to retrain models. Separate training-history uploads for the seven additional subjects are described below. Signing out clears the session. Staff explicitly download any reports they want to retain. CSV exports neutralize spreadsheet formulas in supplied text labels.

Review categories are **Outside grade scale** (estimate outside 0–20), **Below reference** (point estimate below the selected grade), **Near reference** (point estimate at/above the reference with a range crossing it), and **Above reference**. They are prompts for a staff conversation, not calibrated risk probabilities, official pass rules, or automatic student decisions. The public model uses secondary-school data; lecturer-facing use does not establish validity for university courses.

## Additional subjects

Select English (`eng`), Physics (`phy`), Chemistry (`chem`), Biology (`bio`), Geography (`geo`), History (`hist`), or Computer Science (`cs`) in the sidebar:

1. Download the header-only **training CSV template**. Supply actual completed records for that subject, with `subject`, a unique `student_id`, `class_name`, the selected training inputs and final grade `G3`. Choose `grades_v1` (G1 and G2, default), `expanded_v1` (all 11 fields), or `persistence_v1` (G2 only). Expanded comparison remains available when genuine support fields exist; missing fields are never manufactured or imputed. Every `subject` value must match the selected code or full subject name. Mixed subjects, missing outcomes and duplicate student codes are rejected.
2. Upload 50–5,000 distinct students (up to 2 MB), then choose **Train [subject] model**. Grades must be whole numbers from 0 to 20 and final grades must contain at least two distinct values. The interface explains all category codes. Fifty records is only a technical minimum; it does not establish that the model is reliable. Do not enter marks out of 100 directly or invent missing learning-context values.
3. Review **Model quality**: training-only cross-validation selects the estimator, a separate calibration partition sets the prediction range, and a held-out partition reports error, coverage and the G2-only benchmark. Splits are by unique student record, not by class, school or time; a later-cohort evaluation is still necessary.
4. The **Held-out history** dashboard shows only that subject's test records. It provides grade distributions, group comparisons, review filters, tables and CSV reports. Choose **Upload class CSV** to analyse a new class, or use **Student analysis** for one student. An overlap warning identifies student codes already present in the training history.

Each subject's model, history and evaluation remain separate in the signed-in session. No uploaded history or trained private model enters the shared public-model cache, account database or Git repository. You can switch subjects without retraining during the session, or use **Clear this subject model** to remove one. Signing out, losing the session or restarting the server requires re-uploading and retraining. Keep the original CSV securely if you need it again. Changing the selected training file hides the previous model's dashboard until the new file is validated and trained.

No reference training datasets are bundled for these seven subjects, and no accuracy claims are made for them. Their usefulness depends on the representativeness and measurement timing of the records supplied. The lightweight demo remains mathematics-only.

## Hosted startup

Point the hosting service at `app.py` and install `requirements.txt`. Binary `.joblib` files remain excluded from Git. On first access to mathematics or Portuguese, the app loads a compatible trusted artifact bundle if available; otherwise it trains and evaluates from the bundled public CSV and caches the entire bundle in process memory. Staff see a preparation message and an explicit notice if evaluation was recomputed. Historical evaluations are not overwritten. No writable artifact folder or external data download is required. Missing, incompatible, or outdated public models trigger the same recovery, including after a library-version change. Uploaded classes never enter this training process. The other seven subjects use the session-specific training workflow above.

The cache is keyed by subject, schema and scikit-learn version, source-data and model-code hashes, and active artifact content hashes. It is shared only for the public model bundle. The signed-in account and uploaded student data are session-specific. `python train.py` remains available to administrators who want to prepare model artifacts in advance.

## Current predictor selection (schema 3)

All candidates use identical five-fold splits within the fitting partition and the same mean absolute error (MAE) scoring. The fixed outer split remains 60% fitting, 20% calibration and 20% test, using seed 42. Numeric scaling is fitted inside each fold; yes/no categories are explicitly defined. Student codes, class labels and G3 are never predictors. No imputation is performed. G2 is always required after period two; a G1-only early-warning model would need its own training and evaluation.

The candidates are G2 persistence, training-set mean, ridge on G1/G2, ridge on the 11 expanded fields, random forest on G1/G2, and random forest on the expanded fields. Ridge uses alpha 10; forests use 250 trees, minimum leaf size 3 and maximum depth 10. These modest fixed settings were not tuned against test outcomes. Reduced-input training compares the supported candidates on the same records rather than filling absent columns or dropping different rows for different candidates.

The selection rule is fixed: find the lowest validation MAE; candidates within **0.02 grade points** of that minimum are effectively tied. Prefer, in order, G2 persistence (no learned weights), training mean, two-grade ridge, expanded ridge, two-grade forest, expanded forest. This small absolute tolerance is a transparent simplicity policy, not a statistical significance test. No test metric enters selection.

Reproduced validation MAE (mean ± standard deviation across the same five folds):

| Candidate | Versioned feature set | Mathematics | Portuguese |
| --- | --- | ---: | ---: |
| G2 persistence baseline | `persistence_v1`: G2 | 0.9551 ± 0.3000 | 0.7970 ± 0.1471 |
| Training-mean baseline | `persistence_v1`: G2 required for timing, unused in the constant forecast | 3.5402 ± 0.0566 | 2.5020 ± 0.1496 |
| Ridge, two grades | `grades_v1`: G1, G2 | 1.1667 ± 0.2567 | 0.8266 ± 0.1309 |
| Ridge, expanded | `expanded_v1`: existing 11 fields | 1.1963 ± 0.2301 | 0.8386 ± 0.1241 |
| Forest, two grades | `grades_v1` | 1.2226 ± 0.3070 | 0.8664 ± 0.1265 |
| Forest, expanded | `expanded_v1` | 1.2782 ± 0.2753 | 0.9143 ± 0.0903 |

**G2 persistence wins for both bundled subjects.** The active point forecast is exactly G3 = G2. Forms and class CSV templates require only G2; G1 remains optional for progress. An unknown G1 leaves progress blank, and group summaries count how many records have both grades. The six comparison candidates remain available for expanded training. No field was removed because of low permutation importance.

## Reproduction and interval findings (22 October 2026)

`audit_legacy.py` loads the training implementation from Git commit `6129c10`, checks the original data/model/metrics/result files and retrains that exact procedure. It writes a new timestamped audit under `artifacts/audits/`; it never replaces the original artifacts. Both data fingerprints and library versions matched. Recomputed saved-model predictions and retrained predictions agreed with the saved CSVs to about 3.6e-15. The reported old errors and interval coverage were reproduced.

| Same reserved test records | Mathematics before | Mathematics now | Portuguese before | Portuguese now |
| --- | ---: | ---: | ---: | ---: |
| Predictor | Expanded ridge | G2 baseline | Expanded ridge | G2 baseline |
| MAE | 1.4145 | 1.0000 | 0.7555 | 0.6769 |
| RMSE | 2.1948 | 2.0715 | 1.1591 | 1.1701 |
| R² | 0.7651 | 0.7907 | 0.8622 | 0.8596 |
| Nominal interval coverage | 80% | 80% | 80% | 80% |
| Observed test coverage | 54/79 (68.4%) | 66/79 (83.5%) | 103/130 (79.2%) | 122/130 (93.8%) |
| Average interval width, grade points | 2.8963 | 2.0000 | 2.0134 | 2.0000 |
| Calibration / test sample size | 79 / 79 | 79 / 79 | 130 / 130 | 130 / 130 |

The new predictor improves MAE on these records; Portuguese RMSE and R² are slightly worse. This does not establish a learned-model improvement: the selected predictor is a baseline. **These test results had already informed the repair. Reusing the split supports a controlled comparison, not independent confirmation.** No favourable seed was searched. New institution-specific and later-cohort data are required for independent evaluation.

The old interval formula was correct: the absolute calibration residual at rank `ceil((n+1)*0.8)` becomes the radius. The predictor is fixed before calibration; training, calibration and test rows are disjoint. Mathematics' old radius was 1.4481: it covered 64/79 calibration outcomes but only 54/79 test outcomes. Mean absolute error was 1.1356 in calibration versus 1.4145 in test. This demonstrates a less favourable realised test residual distribution, not an identified quantile bug or proof of a particular causal explanation. Intervals have not been widened to fit test coverage.

Both new radii are 1.0, derived solely from their calibration records. Integer errors and ties can yield coverage above the nominal target. One absolute residual threshold means **constant interval width** for each predictor, not personalised uncertainty. Raw predictions and interval bounds are used consistently in calibration, metrics, charts and exports; no clipping is applied. If the finite-sample rank exceeds the available calibration count, training fails explicitly rather than silently capping the quantile. An 80% target is neither a student's probability of passing nor guaranteed coverage in another institution.

Methods: [scikit-learn cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html) and [Angelopoulos and Bates on conformal prediction](https://arxiv.org/abs/2107.07511).

## Alert evaluation and export provenance

Recorded G1/G2, predicted G3, recorded G3 (when available), intervals, rule-based review categories and staff reference grades are labelled separately. The positive outcome is **recorded G3 strictly below the selected reference grade**; an alert is **predicted G3 strictly below that same grade**. Equality is not below reference. Alerts do not use interval overlap; interval overlap only contributes to the separate Near reference review category.

The UI reports precision, recall, F1, true/false positives and negatives, evaluated counts, actual below-reference counts, alert counts and excluded missing outcomes. A zero denominator displays **Unavailable** and exports a blank value; a defined zero score remains zero. The confusion matrix uses recorded outcomes as rows and alert/no alert as columns. Model-quality alert metrics use reserved test records. Dashboard alert metrics use only selected groups from the labelled historical example or uploaded class and never replace the model's saved evaluation.

Prediction and cohort exports include the active bundle ID, feature schema, subject context and prediction timing. Saved test predictions carry the same bundle ID. The model evaluation JSON includes CV fold errors and record partitions, provenance, calibration rank and coverage, preprocessing policy and limitations. Grade percentages are conversions, not pass probabilities; R² is not an accuracy percentage. Permutation importance describes predictor reliance, not causal effects or guaranteed interventions.

## Artifact consistency

`train.py` explicitly creates a new immutable run under `artifacts/v3/<subject>/<bundle_id>/`. A manifest hashes the model binary, metrics and test predictions; a per-subject pointer is published only after every file is complete. Legacy files are retained. Binaries remain excluded from Git. The loader checks feature and code versions, library version, data hash, fixed split, selection rule, calibration settings and model/bundle identity. It also recomputes saved test predictions, error metrics and the calibration radius against the active data before accepting a bundle.

If a compatible bundle is absent or rejected, the app creates a whole new bundle in memory and displays a recomputation notice. It does not overwrite historical evidence or combine new predictions with old intervals. Private uploaded models remain outside the shared public cache. Old session bundles are invalidated after incompatible code/schema changes; users must retrain them. No permanent-history or multi-user changes were made.

## Dataset compatibility and future collection

The downloaded, untracked `data/halabja-grades.csv` remains untouched and is not used for prediction. Inspection found 16,823 data rows (the publisher reports 16,824 entries), 20 columns and 6,799 serial identifiers, with 40 duplicate student/year pairs. It contains annual multi-subject marks on a 0–100 scale, including decimals and many missing subject values. It has no direct period-one/period-two/final target definition matching G1/G2/G3 and lacks the expanded support fields. School-level Computer grades are not automatically equivalent to our Computer Science curriculum. Annual years must not be relabelled as grading periods. Its [source](https://zenodo.org/records/16623461), timing, duplicates, target and scale need a separate integration design; no invented mappings were made.

Future attendance/assignment features require genuine historical snapshots at or before the prediction cutoff. Suggested collection fields (not current mandatory predictors):

| Field | Definition |
| --- | --- |
| student_id, subject, class_id, school_id, cohort_year | Stable metadata for linkage and grouped/temporal evaluation |
| prediction_cutoff, recorded_at | Cutoff after period two, and timestamp the snapshot was actually available |
| attendance_sessions_possible, attendance_sessions_present | Counts up to the cutoff; missing is not zero |
| assignments_due, assignments_submitted, assignments_late | Counts for assignments due by the cutoff; define late consistently |
| assignment_points_earned, assignment_points_possible | Same cutoff and included-assignment definition; document denominators |
| G1, G2, grade_scale | Recorded earlier grades with their assessment dates |
| final_grade, final_grade_date | Later observed outcome, excluded from input snapshots |

Derived attendance/completion rates need nonzero denominators. These variables should enter a future version only after timing and data sufficiency are verified, with preprocessing fitted inside training folds and evaluation on later cohorts or excluded schools. Synthetic fixtures may test software behaviour but cannot support predictive-quality claims.

## Historical schema 2 evaluation (preserved evidence)

The historical expanded training procedure used **11 inputs**: first- and second-period grades, study time, previous class failures, travel time to school, school support, family support, paid subject classes, extracurricular activities, home internet access, and plans for higher education. The reference grade is a user-selected comparison threshold, not a predictor.

Reports include the final-grade estimate, its percentage equivalent, observed G1-to-G2 progress, predicted change from G2, margin to the reference grade, and a nominal 80% prediction range. Percentages and threshold comparisons are derived from the grade estimate; they are not separately trained outcomes or probabilities of passing. A downloadable CSV contains the submitted inputs and results. Results remain visible across reruns, are isolated by subject/model version, and are cleared on logout.

Both original UCI subject files are bundled: **395 mathematics records** and **649 Portuguese-language records**. These are 1,044 subject records, not 1,044 unique students: some students occur in both files. Each subject has a separate model, calibration sample, and test set. Neither subject supplies training records to the other's model. This expands subject coverage without claiming that more rows in a different subject improve mathematics predictions. Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), CC BY 4.0.

The historical schema 2 training procedure reserved 20% for final evaluation, then takes 25% of the remaining records for interval calibration (approximately 60% fitting, 20% calibration, 20% test overall). Mean baseline, ridge regression and random forest were compared using five-fold CV on fitting records only. A six-input ridge benchmark used the same folds as the expanded ridge model. G2 persistence was evaluated only after selection, which prevented it from winning. Preprocessing is fitted inside each fold. Calibration and test records are never used to fit or select the saved model.

Prediction ranges use split conformal absolute residuals: for `n` calibration errors and nominal coverage `c=0.8`, use the `ceil((n+1)*c)`-th smallest error as the radius around the point estimate. These unbounded ranges may extend beyond 0–20. Coverage depends on exchangeable records from the same population; it is a marginal target, not an individual guarantee or a claim about a different school. See [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

Measured results for the expanded model (seed 42, ridge selected for both subjects):

| Measure | Mathematics | Portuguese language |
| --- | ---: | ---: |
| Fitting / calibration / test records | 237 / 79 / 79 | 389 / 130 / 130 |
| Test mean absolute error | 1.415 | 0.755 |
| G2-only test MAE (reuse the second-period grade) | 1.000 | 0.677 |
| Six-input ridge CV MAE | 1.186 | 0.827 |
| Eleven-input ridge CV MAE | 1.196 | 0.839 |
| Actual coverage of nominal 80% range | 68.4% | 79.2% |

**The extra inputs did not improve ridge CV error on these splits, and the G2-only benchmark had lower test error than the selected model.** The mathematics prediction range also substantially missed its nominal coverage target. These historical findings motivated the schema 3 repair described above. The earlier mathematics model's MAE was 1.363, but that model used 316 fitting records without a calibration partition, so the old and new fitting procedures are not equivalent. More inputs and richer reports are implemented; stronger predictive accuracy has not been established.

The evaluation page also reports RMSE, R², the fraction of errors within two grade points, and held-out permutation importance with repeat standard deviations. Importance describes model reliance, not causal effects or an individual explanation. Metrics record dataset SHA-256 hashes, row partitions, model/library version, and training timestamp for reproducibility. The original schema 2 files keep the existing mathematics names and `por_` Portuguese prefix. The active schema 3 evaluations are separate immutable runs under `artifacts/v3/`.

For a stronger school-grade dataset, collect recent records from the target schools across multiple cohorts. Record student/group identifiers for leakage-safe evaluation, subject and grading definitions, prediction dates, attendance up to that date, assignment marks, engagement, and later final outcomes. Test on later cohorts and schools excluded from fitting. Do not combine unrelated outcomes or duplicate rows to inflate the sample size. The bundled absences field has no verified period-two cutoff, so it is excluded.

A different task could use [UCI's Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success): 4,424 higher-education students, 36 features, and dropout/enrolled/graduate outcomes. That would require a separate classification workflow and is not included in this school-grade implementation.

## Start the app — no packages needed

From this folder in PowerShell:

```powershell
python lightweight_app.py
```

Open **http://127.0.0.1:8501**. The app trains at startup using Python's standard library. The dataset is included, so this version runs without internet or package installation. If port 8501 is occupied, use `python lightweight_app.py --port 8502` and open that port instead. Press Ctrl+C in the terminal to stop the server. Requires Python 3.10 or later.

If your PowerShell configuration permits local scripts, `./run.ps1` launches the same app. To train without starting the web server, run `python lite_ml.py`.

## Measured results

The dependency-free implementation selects ridge regression with alpha = 1 using five-fold cross-validation. It fits on 316 training records and evaluates on 79 held-out records.

| Test metric | Value |
| --- | ---: |
| Mean absolute error (MAE) | 1.346 grade points |
| Root mean squared error (RMSE) | 2.082 grade points |
| R² | 0.738 |
| Mean-baseline MAE | 2.999 grade points |

These are measured results, not a percentage accuracy or a guarantee for an individual student. Full-precision results are saved in `artifacts/lite_metrics.json`.

## Lightweight implementation methodology

1. Validate six predictors: `G1`, `G2`, `studytime`, `failures`, `schoolsup`, and `famsup`. The final grade `G3` is never an input.
2. Shuffle records with seed 42 and hold out 20% for final testing.
3. Compare a mean baseline and ridge regression with alpha values 0.1, 1, 10, and 100 using five training folds. Each fold fits its own scaling parameters.
4. Select the lowest cross-validation MAE, fit on all training records, and evaluate on the untouched test partition.
5. Save model coefficients and scaling parameters as JSON, the evaluation report, and test predictions. Test records are not used to refit the saved model.

Ridge regression minimizes squared prediction errors plus an L2 penalty on coefficients. The intercept is not penalized. The implementation standardizes each input and solves the regularized normal equations using Gaussian elimination with partial pivoting. Binary support variables are encoded as 0/1. A test with a known linear relationship checks that the implementation learns the expected relationship.

MAE and RMSE measure error in grade points; lower is better. R² compares squared error with variation around the test-set mean and can be negative. Cross-validation standard deviation is variation between folds, not a confidence interval. Raw predictions are preserved for consistency with evaluation; the app flags estimates outside 0–20.

## Verification

```powershell
python -m unittest discover -s tests -p test_lite.py -v
```

Six passing tests cover the linear solver, recovery of a known relationship, invalid inputs and target exclusion, reproducible training, HTTP page/evaluation responses, and valid/invalid prediction requests. A connected browser was unavailable during development, so a visual browser check was not completed.

## Optional scikit-learn and Streamlit version

The staff workspace described above uses scikit-learn, Streamlit, and Altair. Its four tabs provide cohort analytics, individual analysis, model quality, and a data guide.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 60 -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Alternatively, `./run.ps1 -Full` installs dependencies and launches Streamlit on port 8502. The app verifies or prepares its bundle; run `train.py` explicitly to save new immutable evaluation evidence. Python 3.11 or later is recommended for this optional version. Dependencies are installed in the project's `.venv`; select `.venv\Scripts\python.exe` as your IDE interpreter. Package imports, `pip check`, and model training passed on Python 3.14.7.

Once installed and trained, run its tests separately:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_auth.py -v
```

Run the complete suite with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Tests cover both prediction implementations, staff dashboard filters, class uploads, group summaries, export safety, subject switching, report calculations, calibration quantiles, disjoint fitting/calibration/test partitions, both source files, first-time setup, failed login, restricted access, prediction after login, logout, session revocation and expiry, credential storage, and login throttling. Deployment tests also open the staff dashboard with an empty artifact directory and rebuild models with incompatible metadata. Browser preview tools were unavailable during development; visual browser verification remains outstanding.

Repair verification on 22 October 2026: all 51 tests passed. This includes Streamlit AppTest smoke checks for authentication, both bundled subjects, individual prediction, minimal class uploads, custom-subject training, quality views and exported prediction/alert data. Historical artifact checksums remained unchanged. Visual browser verification and validation of the deployed Streamlit site were not performed for this repair.

The two implementations use different random-number generators for their splits. Their metrics and model files are separate and should not be treated as a direct algorithm comparison.

## Main files

| File | Purpose |
| --- | --- |
| `lightweight_app.py` | Local HTTP server and prediction endpoint |
| `lite_ml.py` | Dependency-free ridge regression, cross-validation, and evaluation |
| `web/index.html` | Responsive browser interface |
| `download_data.py` | Restores the original UCI subject CSV files when needed |
| `data/student-mat.csv`, `data/student-por.csv` | Unmodified subject datasets, semicolon-delimited |
| `artifacts/metrics.json`, `artifacts/por_metrics.json` | Preserved historical schema 2 evaluations |
| `artifacts/v3/` | Active immutable schema 3 runs, integrity manifests and per-subject pointers |
| `artifacts/audits/`, `audit_legacy.py` | Historical reproduction, checksums and repair evidence |
| `artifacts/lite_model.json` | Trained model coefficients and preprocessing parameters |
| `artifacts/lite_metrics.json` | Measured evaluation results |
| `artifacts/lite_test_predictions.csv` | Held-out labels and predictions; source_row is the zero-based data-row index |
| `tests/test_lite.py` | Model and HTTP integration tests |
| `run.ps1` | Launch helper; `-Full` selects the optional version |
| `app.py`, `model.py`, `train.py` | Optional Streamlit/scikit-learn implementation |
| `auth.py`, `auth_ui.py` | Persistent local account storage, session checks, and login screen |
| `model_service.py` | Compatible artifact loading and automatic in-memory model preparation |
| `subject_training.py` | Validated per-subject history uploads, separate training and session-only storage |
| `input_ui.py`, `quality_ui.py`, `evaluation_ui.py` | Schema-specific forms, model evaluation and labelled alert metrics |
| `cohort.py`, `dashboard.py` | Cohort validation, staff summaries, charts, review tables, and exports |
| `assets/app.css` | Responsive login design and matching workspace styling |
| `tests/test_auth.py`, `tests/test_app.py` | Isolated authentication and Streamlit interaction tests |

## Dataset and interpretation

Source: Cortez, P. (2008). **Student Performance** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. [Dataset documentation](https://archive.ics.uci.edu/dataset/320/student+performance). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Both bundled subject CSV files are unmodified copies of the source. They are modeled independently because some students appear in both subjects.

The data contains historical records from two Portuguese schools. G1 and G2 are earlier grades strongly associated with the final grade. The active Streamlit predictors require G2; G1 is optional for reporting progress. The separate lightweight implementation requires both grades. All current predictions are intended for after period two, not enrollment. Questionnaire timing should be verified before prospective use of support variables. Absences are omitted because their aggregation window may extend beyond the prediction point. Names and demographic attributes are not model inputs.

Study-time codes: 1 = under 2 hours/week, 2 = 2–5, 3 = 5–10, 4 = over 10. Failure values are limited to 0–3, the codes observed in the mathematics file; consult the original documentation before collecting new data. Grades are whole numbers from 0 to 20; support fields accept `yes` or `no`.

A random holdout from this dataset does not establish performance for another school, future cohorts, or individual student groups. Validate with suitable local and temporal holdouts before practical use. Grade estimates are not pass probabilities; the expanded app's prediction ranges have the limitations described above. Associations do not show that changing an input causes a grade change. Submitted profiles are held in session memory for the report, not saved to the account database. Reports are written to a file only when the user downloads them.
