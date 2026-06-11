param(
    [string]$EnvName = $(if ($env:LSD_APP_CONDA_ENV) { $env:LSD_APP_CONDA_ENV } else { "lsdetector" }),
    [string]$Port = $(if ($env:LSD_APP_PORT) { $env:LSD_APP_PORT } else { "8501" }),
    [string]$Address = $(if ($env:LSD_APP_ADDRESS) { $env:LSD_APP_ADDRESS } else { "127.0.0.1" })
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Resolve-Path (Join-Path $ScriptDir "..")
$EnvFile = if ($env:LSD_APP_ENV_FILE) { $env:LSD_APP_ENV_FILE } else { Join-Path $AppRoot "environment.yml" }

$Conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $Conda) {
    $Candidates = @(
        "$HOME\anaconda3\Scripts\conda.exe",
        "$HOME\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            $CondaPath = $Candidate
            break
        }
    }
} else {
    $CondaPath = $Conda.Source
}

if (-not $CondaPath) {
    throw "conda was not found. Install Anaconda or Miniconda, then rerun this script."
}

$EnvNames = & $CondaPath env list | ForEach-Object { ($_ -split "\s+")[0] }
if ($EnvNames -contains $EnvName) {
    Write-Host "[LSDetector] Reusing conda environment: $EnvName"
} else {
    Write-Host "[LSDetector] Creating conda environment: $EnvName"
    Write-Host "[LSDetector] Environment file: $EnvFile"
    & $CondaPath env create -n $EnvName -f $EnvFile
}

Write-Host "[LSDetector] Starting Streamlit on ${Address}:${Port}"
Set-Location $AppRoot
$env:PYTHONPATH = "$AppRoot" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
& $CondaPath run --no-capture-output -n $EnvName streamlit run "$AppRoot\lsd_app\web\app.py" --server.address $Address --server.port $Port --server.headless true
