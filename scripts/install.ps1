# OpenHybridHarness installer for a single NVIDIA Windows machine.
#   scripts\install.ps1            full install (CUDA PyTorch + diffusers for image models)
#   scripts\install.ps1 -TextOnly  text models only; skips the multi-GB PyTorch download
param([switch]$TextOnly)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Test-Tool([string]$name, [string]$link) {
    if (Get-Command $name -ErrorAction SilentlyContinue) { return $true }
    Write-Host "Missing: $name. Get it from $link" -ForegroundColor Red
    return $false
}

$ok = $true
$ok = (Test-Tool "nvidia-smi" "https://www.nvidia.com/Download/index.aspx") -and $ok
$ok = (Test-Tool "ollama" "https://ollama.com") -and $ok
$ok = (Test-Tool "node" "https://nodejs.org") -and $ok
$ok = (Test-Tool "python" "https://www.python.org/downloads/") -and $ok
if (-not $ok) { Write-Host "Install what is missing, then run this again." -ForegroundColor Red; exit 1 }

$nodeMajor = [int]((node --version).TrimStart("v").Split(".")[0])
if ($nodeMajor -lt 24) { Write-Host "Node 24 or newer is required (found $nodeMajor). https://nodejs.org" -ForegroundColor Red; exit 1 }
$pyVersion = [version](python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ($pyVersion -lt [version]"3.13") { Write-Host "Python 3.13 or newer is required (found $pyVersion). https://www.python.org/downloads/" -ForegroundColor Red; exit 1 }

Set-Location $root

# Native commands do not throw in Windows PowerShell 5.1: check each one, and stop at the first failure.
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed: $step (exit code $LASTEXITCODE). Fix that and run this again." -ForegroundColor Red
        exit 1
    }
}

python -m venv gpu-service/.venv
Assert-Ok "creating the Python environment (python -m venv)"
$venvPython = Join-Path $root "gpu-service\.venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
Assert-Ok "upgrading pip"
if ($TextOnly) {
    & $venvPython -m pip install -e "gpu-service[dev]"
    Assert-Ok "installing the GPU service (pip install -e gpu-service[dev])"
} else {
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    Assert-Ok "installing CUDA PyTorch (pip install torch)"
    & $venvPython -m pip install -e "gpu-service[gpu,dev]"
    Assert-Ok "installing the GPU service (pip install -e gpu-service[gpu,dev])"
}

npm --prefix orchestrator install
Assert-Ok "installing the orchestrator (npm install)"

ollama pull llama3.1:8b
Assert-Ok "downloading the model (ollama pull llama3.1:8b; is Ollama running? try 'ollama serve')"

# A 64-char lowercase hex token, the same format the orchestrator generates.
$envFile = Join-Path $root ".env"
$hasToken = (Test-Path $envFile) -and (Select-String -Path $envFile -Pattern "^\s*OHH_GPU_TOKEN=\S" -Quiet)
if (-not $hasToken) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
    $prefix = ""
    if ((Test-Path $envFile) -and ((Get-Content $envFile -Raw) -match "[^\n]\z")) { $prefix = "`n" }
    [System.IO.File]::AppendAllText($envFile, "${prefix}OHH_GPU_TOKEN=$token`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Created a GPU service token in .env"
}

Write-Host ""
Write-Host "Installed. Start it with:  npm start" -ForegroundColor Green
