param([switch]$Lab)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $projectPython -m pip install -r requirements-notebooks.txt
if ($LASTEXITCODE -ne 0) { throw 'Notebook dependency installation failed.' }
# Keep the kernel in this virtual environment; do not replace the user's global kernel.
& $projectPython -m ipykernel install --sys-prefix --name studentinsight --display-name 'StudentInsight (.venv)'
if ($LASTEXITCODE -ne 0) { throw 'Could not register the StudentInsight kernel.' }
if ($Lab) {
    & $projectPython -m jupyterlab notebooks/StudentInsight.ipynb --ServerApp.ip=127.0.0.1
} else {
    & $projectPython -m notebook notebooks/StudentInsight.ipynb --ServerApp.ip=127.0.0.1
}
exit $LASTEXITCODE
