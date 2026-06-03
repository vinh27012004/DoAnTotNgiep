# Chay demo app tren Windows PowerShell
# Cach dung:  .\demo\run_demo.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# Cai thu vien neu thieu
& $py -c "import streamlit, plotly" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dang cai streamlit + plotly..." -ForegroundColor Yellow
    & $py -m pip install streamlit plotly
}

# Train model neu chua co artifacts
if (-not (Test-Path (Join-Path $root "demo\artifacts\model_VN.pkl"))) {
    Write-Host "Chua co model - dang train (1-2 phut)..." -ForegroundColor Yellow
    & $py (Join-Path $root "demo\train_models.py")
}

Write-Host "Khoi dong app tai http://localhost:8501 ..." -ForegroundColor Green
& $py -m streamlit run (Join-Path $root "demo\app.py")
