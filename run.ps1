param([switch]$Full)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Full) {
    python lightweight_app.py
    exit $LASTEXITCODE
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $projectPython -m pip install --timeout 60 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection.' }
# The app verifies versioned bundles and visibly prepares an in-memory bundle if needed.
# Run train.py explicitly to save a new immutable evaluation; historical reports stay intact.
& $projectPython -m streamlit run app.py --server.port 8502
