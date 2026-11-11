param([string]$PipelineScript, [string]$AtlasPath)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
if ($PipelineScript) { $env:PACSCAN_BRAINTENSOR_SCRIPT = $PipelineScript }
if ($AtlasPath) { $env:PACSCAN_PD25_ATLAS = $AtlasPath }
Write-Host "Start with: python -m streamlit run app.py"
Write-Host "Full preprocessing additionally needs the research pipeline, FSL BET, ANTsPy and PD25 atlas."
