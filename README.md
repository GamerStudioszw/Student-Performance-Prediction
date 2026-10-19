# Student Performance Prediction using Machine Learning

A working local web app that estimates final mathematics grades (G3, 0–20) from earlier grades and learning context. Includes real public data, reproducible training, model comparison, held-out evaluation, a prediction form, and downloadable test predictions.

## Open the login workspace

The Streamlit app now includes a responsive StudentInsight sign-in screen, one-time owner account setup, and a protected prediction workspace. With the existing project environment installed, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open **http://127.0.0.1:8502**. On your first visit, create your account using your name, a username, and a passphrase of at least 15 characters. Then sign in. There are no default credentials. Later visits show the login form; use **Sign out** in the sidebar to end your session.

The owner account persists in `.local/accounts.sqlite3`, which is excluded from Git. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes with 600,000 iterations, following the [OWASP password storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Session tokens are also hashed in the database and expire after eight hours. Five failed attempts pause sign-in for one minute across all sessions. A page refresh or a new browser tab may require signing in again because authentication is held in the Streamlit session. Tests use temporary databases via `STUDENT_AUTH_DB` and never create or change your real account.

This is a local, single-owner workspace bound to `127.0.0.1`. First-run setup closes once the account exists. There is no email recovery or public registration. The login protects `app.py`; the separate lightweight demo below remains unauthenticated. The source CSV and model artifacts are local project files, not encrypted by the login feature.

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

## Methodology

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

The project also includes a separate implementation comparing a mean baseline, ridge regression, and random forest with scikit-learn. Its interface provides prediction, model evaluation, and data exploration tabs.

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

Run the complete suite with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. The 17 tests cover both prediction implementations, first-time setup, failed login, restricted access, prediction after login, logout, session revocation and expiry, credential storage, and login throttling. Browser preview tools were unavailable during the login update; responsive CSS is implemented, but visual browser verification remains outstanding.

The two implementations use different random-number generators for their splits. Their metrics and model files are separate and should not be treated as a direct algorithm comparison.

## Main files

| File | Purpose |
| --- | --- |
| `lightweight_app.py` | Local HTTP server and prediction endpoint |
| `lite_ml.py` | Dependency-free ridge regression, cross-validation, and evaluation |
| `web/index.html` | Responsive browser interface |
| `download_data.py` | Restores the original UCI mathematics CSV when needed |
| `data/student-mat.csv` | Unmodified source dataset, semicolon-delimited |
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

Source: Cortez, P. (2008). **Student Performance** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. [Dataset documentation](https://archive.ics.uci.edu/dataset/320/student+performance). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled CSV is an unmodified copy of the mathematics file. The Portuguese-language file is not combined with it because some students appear in both subjects.

The data contains historical records from two Portuguese schools. G1 and G2 are earlier grades strongly associated with the final grade. Predictions require both and are intended for after period two, not enrollment. Questionnaire timing should be verified before prospective use of support variables. Absences are omitted because their aggregation window may extend beyond the prediction point. Names and demographic attributes are not model inputs.

Study-time codes: 1 = under 2 hours/week, 2 = 2–5, 3 = 5–10, 4 = over 10. Failure values are limited to 0–3, the codes observed in the mathematics file; consult the original documentation before collecting new data. Grades are whole numbers from 0 to 20; support fields accept `yes` or `no`.

A random holdout from this dataset does not establish performance for another school, future cohorts, or individual student groups. Validate with suitable local and temporal holdouts before practical use. Estimates are not pass probabilities or individual confidence intervals. Associations do not show that changing an input causes a grade change. The local prediction endpoint processes form submissions in memory without saving student inputs.
