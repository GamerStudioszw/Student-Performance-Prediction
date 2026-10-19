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
$artifactsReady = $true
foreach ($prefix in @('', 'por_')) {
    foreach ($filename in @('model.joblib', 'metrics.json', 'test_predictions.csv')) {
        if (-not (Test-Path -LiteralPath (Join-Path 'artifacts' ($prefix + $filename)))) { $artifactsReady = $false }
    }
    $metricsPath = Join-Path 'artifacts' ($prefix + 'metrics.json')
    if (Test-Path -LiteralPath $metricsPath) {
        try {
            $report = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
            if ($report.schema_version -ne 2) { $artifactsReady = $false }
        } catch { $artifactsReady = $false }
    }
}
if (-not $artifactsReady) {
    & $projectPython train.py
    if ($LASTEXITCODE -ne 0) { throw 'Model training failed.' }
}
& $projectPython -m streamlit run app.py --server.port 8502
