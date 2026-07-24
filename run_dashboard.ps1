param(
    [int]$Port = 8503
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\kjy26\miniconda3\envs\KMAP\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "KMAP Python을 찾지 못했습니다: $Python"
}

Set-Location -LiteralPath $AppRoot
& $Python -m streamlit run app.py --server.port $Port --server.address localhost
