# Security and validation

StudentInsight validates requests on the server. Browser field limits are usability controls, not the security boundary. Registration remains open as requested; every account has the same analytics access. Authentication does not verify a person's employment or institutional identity.

## Enforced boundaries

| Area | Enforced behavior |
| --- | --- |
| Accounts | Unique normalized usernames, validated names, password length 15–128, no plaintext password storage, salted PBKDF2-HMAC-SHA256 with 600,000 iterations. Malformed types and invalid Unicode fail before hashing. |
| Abuse limits | Five failed logins per username cause a 60-second cooldown. Unknown usernames share one bounded bucket. Registration permits ten validated attempts per installation per minute, including duplicates. |
| Sessions | Hashed random tokens, eight-hour expiry, at most ten active sessions per account. The oldest session is revoked when the limit is exceeded. Sign-out revokes its token and clears uploaded data and results from that browser session. |
| Account upgrades | The original owner and sessions migrate in one SQLite transaction. Failed migrations roll back. Unique database constraints handle concurrent registrations. |
| CSV uploads | UTF-8, at most 2 MiB, 5,000 rows, 64 columns and 1,000 characters per cell. Duplicate or blank headers, ragged rows, null bytes and malformed quoting are rejected. No automatic header renaming, implicit indexes or grade imputation. |
| Student records | Required model inputs, unique student codes, labels up to 80 characters, no control or invisible formatting characters. Grades must be whole numbers 0–20; booleans, infinities, missing required grades and fractional values are rejected. |
| Training | Separate subject validation, known final outcomes, at least 50 distinct student identifiers, at least two final-grade values, bounded source descriptions. Uploaded histories and models are session-local. |
| Predictions | Validated reference threshold and interval radius; finite, correctly sized output vectors. Invalid submissions clear the previous prediction. Estimates outside 0–20 remain explicitly flagged, rather than silently clipped. |
| Exports | Formula-like text cells are prefixed so spreadsheet programs do not treat supplied identifiers as formulas. |
| Artifacts | Validated subject and run IDs, bounded metadata, unique JSON keys, contained paths, file-size limits, hashes and provenance checks before loading the local binary. Evaluation results are recomputed for consistency. Corrupt bundles are rebuilt without overwriting old evidence. |
| Downloads and Drive | Bounded network and ZIP-member reads. Drive CSV paths must remain inside the selected workspace. Existing report archives cannot be overwritten. |
| Web entry points | Streamlit CORS/XSRF protections enabled, 2 MiB upload limit, browser tracebacks hidden. The separate loopback demo checks Host/Origin, rejects duplicate JSON fields and oversized requests, and sets socket timeouts and no-store responses. |

## Repeatable verification

Install development requirements and run the verification entry point using the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe verify.py
```

The command runs dependency compatibility checks, all unit/integration/adversarial tests, and a strict online dependency-advisory audit. Any failed check or timeout produces a nonzero exit status. Logs and machine-readable results go under `.local/verification/`, excluded from Git. An unavailable advisory service is a failed check, not a clean security result. The GitHub workflow runs these checks for pushes and pull requests; repository administrators must separately configure branch protection if merges must require them.

Tests cover concurrent duplicate sign-ups, invalid credential types, oversized input, cooldown persistence, migration rollback, malformed session tokens, database errors, stale-result removal, malicious CSV structures, seeded fuzz cases, model output corruption, path traversal, checksum corruption, decompression limits, and HTTP origin/host rejection. Test records are synthetic software fixtures, never claimed as research evidence. Tests use temporary account databases.

Local verification on **27 October 2026**: all **77 tests passed**, including **250 seeded malformed-CSV cases**. The 15 adversarial tests also passed after the final numeric-overflow checks were added. `pip check` passed and the strict audit reported **zero known dependency vulnerabilities** after updating Streamlit, Pillow and certifi. This is a point-in-time automated result, not a guarantee against undiscovered vulnerabilities. Hosted deployment and an independent penetration test were not exercised in this run.

## Deployment boundaries

Use HTTPS and persistent account storage when hosting. Keep database files and backups outside public/static directories and restrict OS permissions to the service account. Do not commit credentials or student records. A hosted restart can lose accounts if its filesystem is ephemeral; `STUDENT_AUTH_DB` selects a persistent database path.

Model files use Python serialization. Hash verification detects corruption; it is **not** a signature and does not make an untrusted pickle safe. Only the administrator-controlled artifact directory may supply binaries. Never copy a user's uploaded pickle or joblib file into it. A person who can modify both a model and its manifest must already be treated as trusted code access.

The app does not provide MFA, email ownership verification, role-based institution boundaries, distributed abuse protection, a tamper-evident audit trail, or a hardened production HTTP server for the lightweight demo. The local registration budget is a resource safeguard, not protection against a distributed denial-of-service attack. Session data is separated, but there is no institution-level access model. These checks do not constitute an independent penetration test or prove that a dataset is authentic, representative or free of leakage. Fifty records is a technical minimum, not evidence of model reliability.

Validation follows the [OWASP Input Validation guidance](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html) and [File Upload guidance](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html). Dependency advisories are checked with [PyPA pip-audit](https://github.com/pypa/pip-audit).
