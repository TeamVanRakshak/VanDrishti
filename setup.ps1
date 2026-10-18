$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Prefer a user-installed Python; the bundled runtime supports this workspace too.
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $basePython = $bundledPython
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = (& py -3 -c 'import sys; print(sys.executable)')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $basePython = (Get-Command python).Source
} else {
    throw 'Install Python 3.12 or 3.13 from python.org, then run this script again.'
}
& $basePython -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Please use Python 3.10 through 3.13"'
if ($LASTEXITCODE -ne 0) { throw 'A working Python 3.10-3.13 installation is required.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $basePython -m venv --without-pip .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
}
$envPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
# --python lets a bundled Python with pip manage a venv without ensurepip.
& $basePython -m pip --python $envPython install --no-compile torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed. Check your internet connection, and keep the project in a short folder path (Windows limits paths to 260 characters unless long paths are enabled).' }
& $basePython -m pip --python $envPython install --no-compile -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $envPython download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Check your network connection and rerun setup.ps1.' }
& $envPython evaluation\eval_set.py fetch
if ($LASTEXITCODE -ne 0) { throw 'Evaluation set download failed. Check your network connection and rerun setup.ps1.' }
Write-Host 'VanRakshak is ready. Double-click start.cmd.' -ForegroundColor Green
