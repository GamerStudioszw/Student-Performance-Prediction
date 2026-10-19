# Student Performance Prediction using Machine Learning

A local web app that estimates final mathematics or Portuguese-language grades (G3, 0–20) from earlier grades and learning context. Includes real public data, reproducible training, model comparison, held-out evaluation, and downloadable prediction reports.

## Open the login workspace

The Streamlit app now includes a responsive StudentInsight sign-in screen, one-time owner account setup, and a protected prediction workspace. With the existing project environment installed, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open **http://127.0.0.1:8502**. The app always starts with **Sign in**. Existing users enter their username and password. If this installation has no account yet, choose **Create account**, enter your name, a username, and a passphrase of at least 15 characters, then sign in. The creation screen includes an **Already have an account? Sign in** route. There are no default credentials. Use **Sign out** in the sidebar to end your session.

Accounts belong to the running installation's account database. A local account is not automatically available on a separately hosted copy, because account databases are excluded from Git. If that copy has no saved account, an attempted sign-in explains this and offers initial account setup. The app still supports one owner account per installation; this navigation change does not add public multi-user registration.

The owner account persists in `.local/accounts.sqlite3`, which is excluded from Git. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes with 600,000 iterations, following the [OWASP password storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Session tokens are also hashed in the database and expire after eight hours. Five failed attempts pause sign-in for one minute across all sessions. A page refresh or a new browser tab may require signing in again because authentication is held in the Streamlit session. Tests use temporary databases via `STUDENT_AUTH_DB` and never create or change your real account.

This is a local, single-owner workspace bound to `127.0.0.1`. First-run setup closes once the account exists. There is no email recovery or public registration. The login protects `app.py`; the separate lightweight demo below remains unauthenticated. The source CSV and model artifacts are local project files, not encrypted by the login feature.

## Expanded school-grade model

The Streamlit workspace now supports **11 inputs**: first- and second-period grades, study time, previous class failures, travel time to school, school support, family support, paid subject classes, extracurricular activities, home internet access, and plans for higher education. The reference grade is a user-selected comparison threshold, not a predictor.

Reports include the final-grade estimate, its percentage equivalent, observed G1-to-G2 progress, predicted change from G2, margin to the reference grade, and a nominal 80% prediction range. Percentages and threshold comparisons are derived from the grade estimate; they are not separately trained outcomes or probabilities of passing. A downloadable CSV contains the submitted inputs and results. Results remain visible across reruns, are isolated by subject/model version, and are cleared on logout.

Both original UCI subject files are bundled: **395 mathematics records** and **649 Portuguese-language records**. These are 1,044 subject records, not 1,044 unique students: some students occur in both files. Each subject has a separate model, calibration sample, and test set. Neither subject supplies training records to the other's model. This expands subject coverage without claiming that more rows in a different subject improve mathematics predictions. Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), CC BY 4.0.

Run `python train.py` using the project environment to train both models. The training procedure reserves 20% for final evaluation, then takes 25% of the remaining records for interval calibration (approximately 60% fitting, 20% calibration, 20% test overall). Mean baseline, ridge regression and random forest are compared using five-fold CV on fitting records only. A six-input ridge benchmark uses the same folds as the expanded ridge model. Preprocessing is fitted inside each fold. Calibration and test records are never used to fit or select the saved model.

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

**The extra inputs did not improve ridge CV error on these splits, and the G2-only benchmark had lower test error than the selected model.** The mathematics prediction range also substantially missed its nominal coverage target. The UI displays these limitations. The earlier mathematics model's MAE was 1.363, but that model used 316 fitting records without a calibration partition, so the old and new fitting procedures are not equivalent. More inputs and richer reports are implemented; stronger predictive accuracy has not been established.

The evaluation page also reports RMSE, R², the fraction of errors within two grade points, and held-out permutation importance with repeat standard deviations. Importance describes model reliance, not causal effects or an individual explanation. Metrics record dataset SHA-256 hashes, row partitions, model/library version, and training timestamp for reproducibility. Files use the existing names for mathematics and the `por_` prefix for Portuguese language.

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

The expanded login workspace described above uses scikit-learn and Streamlit. Its interface provides prediction, model evaluation, and data exploration tabs, with a subject selector.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 60 -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Alternatively, `./run.ps1 -Full` installs dependencies, trains if needed, and launches Streamlit on port 8502. Python 3.11 or later is recommended for this optional version. Dependencies are installed in the project's `.venv`; select `.venv\Scripts\python.exe` as your IDE interpreter. Package imports, `pip check`, and model training passed on Python 3.14.7.

Once installed and trained, run its tests separately:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_auth.py -v
```

Run the complete suite with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Tests cover both prediction implementations, subject switching, report calculations, calibration quantiles, disjoint fitting/calibration/test partitions, both source files, first-time setup, failed login, restricted access, prediction after login, logout, session revocation and expiry, credential storage, and login throttling. Browser preview tools were unavailable during development; visual browser verification remains outstanding.

The two implementations use different random-number generators for their splits. Their metrics and model files are separate and should not be treated as a direct algorithm comparison.

## Main files

| File | Purpose |
| --- | --- |
| `lightweight_app.py` | Local HTTP server and prediction endpoint |
| `lite_ml.py` | Dependency-free ridge regression, cross-validation, and evaluation |
| `web/index.html` | Responsive browser interface |
| `download_data.py` | Restores the original UCI subject CSV files when needed |
| `data/student-mat.csv`, `data/student-por.csv` | Unmodified subject datasets, semicolon-delimited |
| `artifacts/metrics.json`, `artifacts/por_metrics.json` | Expanded models' evaluation, calibration, provenance, and split records |
| `artifacts/lite_model.json` | Trained model coefficients and preprocessing parameters |
| `artifacts/lite_metrics.json` | Measured evaluation results |
| `artifacts/lite_test_predictions.csv` | Held-out labels and predictions; source_row is the zero-based data-row index |
| `tests/test_lite.py` | Model and HTTP integration tests |
| `run.ps1` | Launch helper; `-Full` selects the optional version |
| `app.py`, `model.py`, `train.py` | Optional Streamlit/scikit-learn implementation |
| `auth.py`, `auth_ui.py` | Persistent local account storage, session checks, and login screen |
| `assets/app.css` | Responsive login design and matching workspace styling |
| `tests/test_auth.py`, `tests/test_app.py` | Isolated authentication and Streamlit interaction tests |

## Dataset and interpretation

Source: Cortez, P. (2008). **Student Performance** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. [Dataset documentation](https://archive.ics.uci.edu/dataset/320/student+performance). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Both bundled subject CSV files are unmodified copies of the source. They are modeled independently because some students appear in both subjects.

The data contains historical records from two Portuguese schools. G1 and G2 are earlier grades strongly associated with the final grade. Predictions require both and are intended for after period two, not enrollment. Questionnaire timing should be verified before prospective use of support variables. Absences are omitted because their aggregation window may extend beyond the prediction point. Names and demographic attributes are not model inputs.

Study-time codes: 1 = under 2 hours/week, 2 = 2–5, 3 = 5–10, 4 = over 10. Failure values are limited to 0–3, the codes observed in the mathematics file; consult the original documentation before collecting new data. Grades are whole numbers from 0 to 20; support fields accept `yes` or `no`.

A random holdout from this dataset does not establish performance for another school, future cohorts, or individual student groups. Validate with suitable local and temporal holdouts before practical use. Grade estimates are not pass probabilities; the expanded app's prediction ranges have the limitations described above. Associations do not show that changing an input causes a grade change. Submitted profiles are held in session memory for the report, not saved to the account database. Reports are written to a file only when the user downloads them.
